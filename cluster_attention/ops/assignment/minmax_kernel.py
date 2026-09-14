import triton
import triton.language as tl
import torch
#from triton.experimental import gluon

####################
## Kernel wrapper ##
####################

def launch_minmax_kernel(
    qc, kc, last_kc_frac, 
    mins, maxes, denominators,
    QC_BLOCK_SIZE=64, KC_BLOCK_SIZE=128
):
    B, H, n_qc, D = qc.shape
    BH = B*H
    n_kc = kc.shape[2]

    # grid that groups queries from the same head and batch, as in FlashAttention for bidirectional attention
    # no gain from this when all keys fit in cache, but helps keep keys warm when they do not fit
    n_q_blocks = triton.cdiv(n_qc, QC_BLOCK_SIZE)
    grid = (n_q_blocks * BH,)
    minmax_kernel[grid](
        qc, kc, last_kc_frac,
        mins, maxes, denominators,
        n_qc, n_kc, n_q_blocks,
        QC_BLOCK_SIZE=QC_BLOCK_SIZE, KC_BLOCK_SIZE=KC_BLOCK_SIZE, 
        D=D,
    )
    return mins, maxes, denominators

############
## Kernel ##
############

@triton.jit(do_not_specialize=["n_qc", "n_kc", "last_kc_frac", "n_q_blocks"])
#@gluon.jit(do_not_specialize=["n_qc", "n_kc", "n_q_blocks"])
def minmax_kernel(
    qc_ptr, kc_ptr, last_kc_frac, 
    min_ptr, max_ptr, denominator_ptr,
    n_qc, n_kc, n_q_blocks,
    QC_BLOCK_SIZE: tl.constexpr,
    KC_BLOCK_SIZE: tl.constexpr,
    D: tl.constexpr,
):
    linear_id = tl.program_id(0)
    pid_q = linear_id % n_q_blocks
    pid_bh = linear_id // n_q_blocks
    rsqrt_d = 1.0 / tl.sqrt(float(D))

    q_start = pid_q *QC_BLOCK_SIZE
    q_offs = q_start +tl.arange(0, QC_BLOCK_SIZE)
    q_mask = q_offs < n_qc
    d_offs = tl.arange(0, D)

    # Load qc block [QC_BLOCK_SIZE, D]
    qc_block = tl.load(
        qc_ptr +pid_bh*n_qc*D +q_offs[:, None]*D +d_offs[None, :],
        mask=q_mask[:, None],
        other=0.0,
        eviction_policy="evict_first"  # do not want it to kick out keys which are accessed many times!
    )

    row_min = tl.full([QC_BLOCK_SIZE], float("inf"), dtype=tl.float32)
    row_max = tl.full([QC_BLOCK_SIZE], float("-inf"), dtype=tl.float32)
    row_denominator = tl.zeros([QC_BLOCK_SIZE], dtype=tl.float32)

    for kc_start in range(0, n_kc, KC_BLOCK_SIZE):
        kc_offs = kc_start +tl.arange(0, KC_BLOCK_SIZE)
        kc_mask = kc_offs < n_kc

        # Load kc block [KC_BLOCK_SIZE, D]
        kc_block = tl.load(
            kc_ptr +pid_bh*n_kc*D +kc_offs[:, None]*D +d_offs[None, :],
            mask=kc_mask[:, None],
            other=0.0,
        )

        # [QC_BLOCK_SIZE, KC_BLOCK_SIZE], fp32 accumulation
        dots = tl.dot(qc_block, tl.trans(kc_block), out_dtype=tl.float32) * rsqrt_d
        dots = tl.where(kc_offs[None, :] == n_kc -1, dots +tl.log(last_kc_frac), dots)

        # Mask out-of-bounds K positions
        dots_min = tl.where(kc_mask[None, :], dots, float("inf"))
        dots_max = tl.where(kc_mask[None, :], dots, float("-inf"))

        row_min = tl.minimum(row_min, tl.min(dots_min, axis=1))
        #row_max = tl.maximum(row_max, tl.max(dots_max, axis=1))
        new_max = tl.maximum(row_max, tl.max(dots_max, axis=1))
        # from 0.12 -> 0.17 ms
        row_denominator = row_denominator * tl.exp(row_max -new_max) +tl.sum(tl.exp(dots_max -new_max[:, None]), axis=1)
        row_max = new_max

    # expanding ranges slightly to account for numerical variance (so edge values do not fall out next run), check correctness
    eps = 1e-5 #* tl.maximum(1.0, tl.maximum(tl.abs(row_min), tl.abs(row_max)))  # scale dependent?
    row_min -= eps
    row_max += eps
    row_denominator *= tl.exp(-eps)

    base = pid_bh*n_qc +q_offs
    tl.store(min_ptr +base, row_min, mask=q_mask)
    tl.store(max_ptr +base, row_max, mask=q_mask)
    tl.store(denominator_ptr +base, row_denominator, mask=q_mask)
