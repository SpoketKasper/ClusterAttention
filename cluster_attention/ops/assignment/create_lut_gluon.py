import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
import triton.profiler.language as pl

def launch_lut_kernel(
    qc, kc, thresholds, last_kc_frac,
    lut_workspace, valid_block_num,
    QC_BLOCK_SIZE=64, KC_BLOCK_SIZE=128,
    WARPS_M=4, WARPS_N=1
):
    B, H, n_qc, D = qc.shape
    BH = B * H
    n_kc = kc.shape[2]

    n_q_blocks = triton.cdiv(n_qc, QC_BLOCK_SIZE)
    grid = (BH * n_q_blocks,)
    lut_kernel[grid](
        qc, kc,
        thresholds, lut_workspace, valid_block_num,
        last_kc_frac,
        n_qc, n_kc, n_q_blocks,
        D=D,
        QC_BLOCK_SIZE=QC_BLOCK_SIZE, KC_BLOCK_SIZE=KC_BLOCK_SIZE,
        WARPS_M=WARPS_M, WARPS_N=WARPS_N, num_warps=WARPS_M * WARPS_N,
    )
    return lut_workspace, valid_block_num

# Small associative helpers
@gluon.jit
def _combine_add(a, b):
    return a + b
@gluon.jit
def _combine_max(a, b):
    return ttgl.maximum(a, b)

# could use half the smem if not using a double width buffer, maybe better occupancy!
@gluon.jit(do_not_specialize=["last_kc_frac", "n_qc", "n_kc", "n_q_blocks"])
def lut_kernel(
    # global
    qc_ptr, kc_ptr, 
    
    # per query-centroid
    threshold_ptr, 
    lut_ptr, count_ptr,

    # other
    last_kc_frac,
    n_qc, n_kc, n_q_blocks,
    D: ttgl.constexpr,

    # config
    QC_BLOCK_SIZE: ttgl.constexpr,
    KC_BLOCK_SIZE: ttgl.constexpr,
    WARPS_M: ttgl.constexpr,
    WARPS_N: ttgl.constexpr,
):
    log_last_kc_frac = tl.log(last_kc_frac)

    #############
    ## Layouts ##
    #############

    # ported from accumulate_kernel
    warps_per_cta: ttgl.constexpr = [WARPS_M, WARPS_N]
    acc_layout: ttgl.constexpr = ttgl.NVMMADistributedLayout(
        version=[2, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[16, 8],
    )
    lhs_layout: ttgl.constexpr = ttgl.DotOperandLayout(
        parent=acc_layout,
        operand_index=0,
        k_width=8,
    )
    rhs_layout: ttgl.constexpr = ttgl.DotOperandLayout(
        parent=acc_layout,
        operand_index=1,
        k_width=8,
    )
    THREADS_D: ttgl.constexpr = D // 8
    qc_load_layout: ttgl.constexpr = ttgl.BlockedLayout(
        [1, 8], [32 // THREADS_D, THREADS_D], [WARPS_M, WARPS_N], [1, 0]
    )
    kct_load_layout: ttgl.constexpr = ttgl.BlockedLayout(
        [8, 1], [THREADS_D, 32 // THREADS_D], [WARPS_N, WARPS_M], [0, 1]
    )

    # New layouts
    # Layout for everything after the dot: scans, compaction, output store.
    # Each thread holds 4 consecutive int32/fp32 along axis 1 -> 16B chunks.
    scan_layout: ttgl.constexpr = ttgl.BlockedLayout(
        [1, 4], [1, 32], [WARPS_M, WARPS_N], [1, 0]
    )
    row_layout: ttgl.constexpr = ttgl.SliceLayout(1, scan_layout)
    col_layout: ttgl.constexpr = ttgl.SliceLayout(0, scan_layout)

    smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    #################
    ## IDs/offsets ##
    #################

    linear_id = ttgl.program_id(0)
    pid_q = linear_id % n_q_blocks
    pid_bh = linear_id // n_q_blocks
    rsqrt_d: ttgl.constexpr = 1.0 / (D ** 0.5)

    q_offs = pid_q*QC_BLOCK_SIZE + ttgl.arange(0, QC_BLOCK_SIZE, layout=row_layout)
    q_mask = q_offs < n_qc

    kct_d = ttgl.arange(0, D, layout=ttgl.SliceLayout(1, kct_load_layout))
    kct_k = ttgl.arange(0, KC_BLOCK_SIZE, layout=ttgl.SliceLayout(0, kct_load_layout))

    ####################
    ## One-time loads ##
    ####################

    q_offs_l = (
        pid_q*QC_BLOCK_SIZE
        + ttgl.arange(0, QC_BLOCK_SIZE, layout=ttgl.SliceLayout(1, qc_load_layout))
    )
    d_offs_l = ttgl.arange(0, D, layout=ttgl.SliceLayout(0, qc_load_layout))
    qc_block = ttgl.load(
        qc_ptr + pid_bh*n_qc*D + q_offs_l[:, None]*D + d_offs_l[None, :],
        mask=(q_offs_l < n_qc)[:, None],
        other=0.0,
    )
    qc_block = ttgl.convert_layout(qc_block, lhs_layout)

    threshold = ttgl.load(
        threshold_ptr + pid_bh*n_qc + q_offs, mask=q_mask, other=float("inf")
    )  # query-centroids that are padding get 'inf' threshold so they are skipped

    ###########################
    ## Allocations and inits ##
    ###########################

    lut_cursor = ttgl.zeros([QC_BLOCK_SIZE], ttgl.int32, row_layout)
    last_idx = ttgl.zeros([QC_BLOCK_SIZE], ttgl.int32, row_layout)

    block_range = ttgl.arange(0, KC_BLOCK_SIZE, layout=col_layout)
    shift_idx = ttgl.maximum(block_range - 1, 0)[None, :].broadcast_to(
        QC_BLOCK_SIZE, KC_BLOCK_SIZE
    )
    lut_base = lut_ptr + n_kc*(pid_bh*n_qc + q_offs[:, None])

    smem = ttgl.allocate_shared_memory(  # smem allocation for buffering deltas
        ttgl.int32, [QC_BLOCK_SIZE, 2*KC_BLOCK_SIZE], smem_layout
    )

    #########################
    ## Loop over kc-blocks ##
    #########################

    # 1D example (one row of the tile) with values:
    # - KC_BLOCK_SIZE = 4
    # - lut_cursor = 2
    # - last_idx = 2
    # - start = 4
    # Parenthesis mark unkept values. ? marks uninitialized memory or leftover values.

    for kc_start in range(0, n_kc, KC_BLOCK_SIZE):
        kc_offs = kc_start + block_range  # [4, 5, 6, 7]
        kc_mask = kc_offs < n_kc

        with pl.scope("dots"):
            kct_offs = kc_start + kct_k
            kc_block_t = ttgl.load(
                kc_ptr + pid_bh*n_kc*D + kct_offs[None, :]*D + kct_d[:, None],
                mask=(kct_offs < n_kc)[None, :],
                other=0.0,
            )
            kc_block_t = ttgl.convert_layout(kc_block_t, rhs_layout)

            acc = ttgl.zeros([QC_BLOCK_SIZE, KC_BLOCK_SIZE], ttgl.float32, acc_layout)
            dots = ttgl.nvidia.ampere.mma_v2(qc_block, kc_block_t, acc) * rsqrt_d

            # convert to scan_layout for the rest of the work
            dots = ttgl.convert_layout(dots, scan_layout)
            dots = ttgl.where(
                kc_offs[None, :]==n_kc - 1, dots+log_last_kc_frac, dots
            )  # adjust for potentially smaller last cluster
            dots = ttgl.where(kc_mask[None, :], dots, float("-inf"))

        with pl.scope("Compute mask"):
            keep = dots > threshold[:, None]  # [0, 1, 0, 1]

        with pl.scope("Compute positions to write in for values above threshold"):
            smem_pos = ttgl.associative_scan(
                keep.to(ttgl.int32), 1, _combine_add
            ) - 1  # [(-1), 0, (0), 1]
            smem_pos = ttgl.where(keep, smem_pos, KC_BLOCK_SIZE)  # [(4), 0, (4), 1]
            kept_count = ttgl.sum(keep.to(ttgl.int32), axis=1)

        with pl.scope("Compute deltas"):
            masked_kc_idx = ttgl.where(keep, kc_offs[None, :], last_idx[:, None])
            running_max = ttgl.associative_scan(
                masked_kc_idx, 1, _combine_max
            )  # [(2), 5, (5), 7]
            shifted = ttgl.gather(running_max, shift_idx, 1)
            shifted = ttgl.where(
                block_range[None, :]==0, last_idx[:, None], shifted
            )  # [(2), 2, (5), 5]
            deltas = running_max - shifted  # [(0), 3, (0), 2]

        with pl.scope("Pack deltas into smem, then write this contiguously"):
            smem.scatter(  # scatter into smem, faster than into l2
                deltas, smem_pos, axis=1
            )  # writes into smem, making it [3, 2, ?, ? | 0, ?, ?, ?]
            ttgl.barrier()  # ensure smem write is done before loading to write into l2

            packed = smem.slice(
                0, KC_BLOCK_SIZE, dim=1
            ).load(scan_layout)  # [3, 2, ?, ?]
            out_mask = block_range[None, :] < kept_count[:, None]  # [1, 1, 0, 0]
            ttgl.store(
                lut_base + lut_cursor[:, None] + block_range[None, :],
                packed,
                mask=out_mask,
            )  # store into l2 contiguously
            ttgl.barrier()  # next iteration modifies smem, so ensure reading is done

        # Update for next block
        lut_cursor += kept_count
        last_idx = ttgl.max(
            ttgl.where(keep, kc_offs[None, :], last_idx[:, None]), axis=1
        )

    ###################
    ## Store results ##
    ###################

    ttgl.store(count_ptr + pid_bh*n_qc + q_offs, lut_cursor, mask=q_mask)
