import torch
import triton
import triton.language as tl
import triton.profiler.language as pl
pl.enable_semantic("triton")  # usually disabled due to potential for errors so beware

####################
## Kernel wrapper ##
####################

# benchmark case: qc has shape (1, 16, 15000, 64), ks has (1, 16, 7500, 64). one run takes 30 ms.
def launch_lut_kernel(qc, kc, thresholds, last_kc_frac, lut_workspace, QC_BLOCK_SIZE=64, KC_BLOCK_SIZE=128):
    B, H, n_qc, D = qc.shape
    BH = B*H
    n_kc = kc.shape[2]

    #lut = torch.empty(BH, n_qc, n_kc, dtype=torch.int32, device=qc.device)
    valid_block_num = torch.empty(BH, n_qc, dtype=torch.int32, device=qc.device)

    n_q_blocks = triton.cdiv(n_qc, QC_BLOCK_SIZE)
    grid = (n_q_blocks * BH,)
    lut_kernel[grid](
        qc, kc, thresholds, last_kc_frac,
        lut_workspace, valid_block_num,
        n_qc, n_kc, n_q_blocks,
        QC_BLOCK_SIZE=QC_BLOCK_SIZE,
        KC_BLOCK_SIZE=KC_BLOCK_SIZE,
        D=D,
    )
    return lut_workspace, valid_block_num

############
## Kernel ##
############

@triton.jit
def combine_max(a, b):
    return tl.maximum(a, b)

@triton.jit(do_not_specialize=["n_qc", "n_kc", "last_kc_frac", "n_q_blocks"])
def lut_kernel(
    qc_ptr, kc_ptr, threshold_ptr, last_kc_frac,
    lut_ptr, count_ptr,
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

    qc_block = tl.load(
        qc_ptr +pid_bh*n_qc*D +q_offs[:, None]*D +d_offs[None, :],
        mask=q_mask[:, None],
        other=0.0,
        eviction_policy="evict_first",
    )
    # per-query threshold, out-of-bounds rows get +inf so they keep nothing
    threshold = tl.load(threshold_ptr +pid_bh*n_qc +q_offs, mask=q_mask, other=float("inf"))

    lut_cursor = tl.zeros([QC_BLOCK_SIZE], dtype=tl.int32)
    last_idx = tl.zeros([QC_BLOCK_SIZE], dtype=tl.int32)
    block_range = tl.arange(0, KC_BLOCK_SIZE)
    shift_idx = tl.maximum(block_range -1, 0)[None, :].broadcast_to(QC_BLOCK_SIZE, KC_BLOCK_SIZE)
    lut_base = lut_ptr +(pid_bh*n_qc +q_offs[:, None])*n_kc

    # 1D example (one row of the tile) with values:
    # - KC_BLOCK_SIZE = 4
    # - lut_cursor = 2
    # - last_idx = 2
    # - start = 4
    # Parenthesis mark unkept values.

    for kc_start in range(0, n_kc, KC_BLOCK_SIZE):
        kc_offs = kc_start +block_range  # [4, 5, 6, 7]
        kc_mask = kc_offs < n_kc

        with pl.scope("dots"):
            kc_block = tl.load(
                kc_ptr +pid_bh*n_kc*D +kc_offs[:, None]*D +d_offs[None, :],
                mask=kc_mask[:, None],
                other=0.0,
            )

            dots = tl.dot(qc_block, tl.trans(kc_block), out_dtype=tl.float32) * rsqrt_d
            dots = tl.where(kc_offs[None, :] == n_kc -1, dots +tl.log(last_kc_frac), dots)
            dots = tl.where(kc_mask[None, :], dots, float("-inf"))

        with pl.scope("1"):
            # 1. Compute mask
            keep = dots > threshold[:, None]  # [0, 1, 0, 1]

        with pl.scope("2"):
            # 2. Compute positions to write in for values above threshold
            relative_destinations = tl.cumsum(keep.to(tl.int32), axis=1) -1  # [(-1), 0, (0), 1]
            destinations = lut_cursor[:, None] +relative_destinations  # [(1), 2, (2), 3], we now know where the  (in delta form) should be written!

        with pl.scope("3"):
            # 3. Compute deltas
            masked_kc_idx = tl.where(keep, kc_offs[None, :], last_idx[:, None])  # [(2), 5, (2), 7]
            running_max = tl.associative_scan(masked_kc_idx, 1, combine_max)  # [(2), 5, (5), 7]
            shifted = tl.gather(running_max, shift_idx, 1)  # [(2), 2, (5), 5]
            shifted = tl.where(block_range[None, :] == 0, last_idx[:, None], shifted)  # [(2), 2, (5), 5], does stuff if the first kc_idx should be kept
            deltas = running_max -shifted  # [(0), 3, (0), 2], we now know what deltas should be written!

        with pl.scope("4"):
            # 4. Write deltas into the right positions
            tl.store(lut_base +destinations, deltas, mask=keep)

        # Update for next block
        lut_cursor += tl.sum(keep.to(tl.int32), axis=1)
        last_idx = tl.max(tl.where(keep, kc_offs[None, :], last_idx[:, None]), axis=1)

    tl.store(count_ptr +pid_bh*n_qc +q_offs, lut_cursor, mask=q_mask)