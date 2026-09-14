# edge case if we have fewer than count_threshold clusters in total! beware

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
import math
import triton.profiler.language as pl

def histogram_threshold(
    # global
    qc, kc, last_kc_frac,
    mass_threshold, count_threshold,

    # per query-centroid
    maxes, 
    lower_edges, upper_edges,
    mass_offsets, count_offsets, 
    denominators,  # needs to be float32

    # out
    exp_histogram, counts_histogram,

    # config
    #QC_BLOCK_SIZE=64, KC_BLOCK_SIZE=128, N_BINS=128
    QC_BLOCK_SIZE=64, KC_BLOCK_SIZE=64, N_BINS=128,
    warps_per_cta=(4, 2)
):
    B, H, n_qc, D = qc.shape
    BH = B * H
    n_kc = kc.shape[2]
    n_q_blocks = triton.cdiv(n_qc, QC_BLOCK_SIZE)
    grid = (n_q_blocks * BH,)
    info = accumulate_kernel[grid](
        # global
        qc, kc, 

        # per query-centroid
        maxes, 
        lower_edges, upper_edges, 
        denominators,

        # special
        exp_histogram, counts_histogram,

        # other
        last_kc_frac, n_qc, n_kc, n_q_blocks, D=D,

        # config
        QC_BLOCK_SIZE=QC_BLOCK_SIZE, KC_BLOCK_SIZE=KC_BLOCK_SIZE, N_BINS=N_BINS,
        WARPS_M=warps_per_cta[0], WARPS_N=warps_per_cta[1],
        num_warps=math.prod(warps_per_cta),
    )#; print(info.n_regs, info.n_spills)

    threshold_grid = (BH*n_qc,)
    # timing on 1 million samples, per call: 0.3 ms
    threshold_kernel[threshold_grid](
        mass_threshold, float(count_threshold),  
        # allegedly triton is specializing on whether ints are powers of two so we cast to float
        exp_histogram, counts_histogram,
        mass_offsets, count_offsets,
        denominators,
        lower_edges, upper_edges,
        n_qc,
        N_BINS=N_BINS,
    )
    return lower_edges, upper_edges, mass_offsets, count_offsets

@gluon.jit(do_not_specialize=["last_kc_frac", "n_qc", "n_kc", "n_q_blocks"])
def accumulate_kernel(
    # global
    qc_ptr, kc_ptr,

    # per query-centroid
    maxes_ptr,
    lower_edges_ptr, upper_edges_ptr,
    denominators_ptr,

    # special
    exp_histogram_ptr, counts_histogram_ptr,

    # other
    last_kc_frac, n_qc, n_kc, n_q_blocks,
    D: ttgl.constexpr,

    # config
    QC_BLOCK_SIZE: ttgl.constexpr, KC_BLOCK_SIZE: ttgl.constexpr, N_BINS: ttgl.constexpr,
    WARPS_M: ttgl.constexpr, WARPS_N: ttgl.constexpr,
):
    log_last_kc_frac = tl.log(last_kc_frac)

    #############
    ## Layouts ##
    #############

    # mma accumulator, (QC, KC) fp32
    warps_per_cta: ttgl.constexpr = [WARPS_M, WARPS_N]
    acc_layout: ttgl.constexpr = ttgl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=warps_per_cta, instr_shape=[16, 8],
    )
    lhs_layout: ttgl.constexpr = ttgl.DotOperandLayout(parent=acc_layout, operand_index=0, k_width=8)
    rhs_layout: ttgl.constexpr = ttgl.DotOperandLayout(parent=acc_layout, operand_index=1, k_width=8)

    # load layouts, coalesced along the memory-contiguous dim (D)
    THREADS_D: ttgl.constexpr = D // 8
    qc_load_layout: ttgl.constexpr = ttgl.BlockedLayout(
        [1, 8], [32 // THREADS_D, THREADS_D], [WARPS_M, WARPS_N], [1, 0]
    )   # (QC, D), threads along D
    kct_load_layout: ttgl.constexpr = ttgl.BlockedLayout(
        [8, 1], [THREADS_D, 32 // THREADS_D], [WARPS_N, WARPS_M], [0, 1]
    )  # transposed load to (D, KC), threads along D

    # flush/zero layout for the [QC, N_BINS] histograms, coalesced along bins (?)
    hist_layout: ttgl.constexpr = ttgl.BlockedLayout(
        [1, 1], [1, 32], [WARPS_M, WARPS_N], [1, 0]
    )

    # per-row vectors ([QC]) live in the row-slice of the acc layout so [:, None]
    # broadcasts against dots without conversions
    row_layout: ttgl.constexpr = ttgl.SliceLayout(1, acc_layout)
    smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
        swizzle_byte_width=128, element_bitwidth=32, rank=2
    )
    #smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(
    #    vec=1, per_phase=1, max_phase=1, order=[1, 0]
    #)  # this almost doubles time

    #################
    ## IDs/offsets ##
    #################

    linear_id = ttgl.program_id(0)
    pid_q = linear_id % n_q_blocks
    pid_bh = linear_id // n_q_blocks
    rsqrt_d: ttgl.constexpr = 1.0 / (D ** 0.5)
 
    q_offs = pid_q * QC_BLOCK_SIZE + ttgl.arange(0, QC_BLOCK_SIZE, layout=row_layout)
    q_mask = q_offs < n_qc
    q_base = pid_bh * n_qc + q_offs
 
    kc_range = ttgl.arange(0, KC_BLOCK_SIZE, layout=ttgl.SliceLayout(0, acc_layout))
    kct_d = ttgl.arange(0, D, layout=ttgl.SliceLayout(1, kct_load_layout))
    kct_k = ttgl.arange(0, KC_BLOCK_SIZE, layout=ttgl.SliceLayout(0, kct_load_layout))
 
    ####################
    ## One-time loads ##
    ####################

    q_offs_l = pid_q * QC_BLOCK_SIZE + ttgl.arange(0, QC_BLOCK_SIZE, layout=ttgl.SliceLayout(1, qc_load_layout))
    d_offs_l = ttgl.arange(0, D, layout=ttgl.SliceLayout(0, qc_load_layout))
    qc_block = ttgl.load(
        qc_ptr + pid_bh * n_qc * D + q_offs_l[:, None] * D + d_offs_l[None, :],
        mask=(q_offs_l < n_qc)[:, None],
        other=0.0,
    )  # (QC, D)
    qc_block = ttgl.convert_layout(qc_block, lhs_layout)
 
    lower_edges = ttgl.load(lower_edges_ptr + q_base, mask=q_mask, other=0.0)
    upper_edges = ttgl.load(upper_edges_ptr + q_base, mask=q_mask, other=1.0)
    row_maxes = ttgl.load(maxes_ptr + q_base, mask=q_mask, other=0.0)
    inv_bin_widths = N_BINS / (upper_edges - lower_edges)
 
    ###########################
    ## Allocations and inits ##
    ###########################

    exp_hist = ttgl.allocate_shared_memory(ttgl.float32, [QC_BLOCK_SIZE, N_BINS], smem_layout)
    cnt_hist = ttgl.allocate_shared_memory(ttgl.int32, [QC_BLOCK_SIZE, N_BINS], smem_layout)
    exp_hist.store(ttgl.zeros([QC_BLOCK_SIZE, N_BINS], ttgl.float32, hist_layout))
    cnt_hist.store(ttgl.zeros([QC_BLOCK_SIZE, N_BINS], ttgl.int32, hist_layout))
    ttgl.barrier()

    #########################
    ## Loop over kc-blocks ##
    #########################

    for kc_start in range(0, n_kc, KC_BLOCK_SIZE):
        kc_offs = kc_start + kc_range  # used for masks
        kc_mask = kc_offs < n_kc
 
        with pl.scope("kc_load"):  # ~1/10 of time
            # load kc transposed so it is ready for the dot, coalesced along D (D, KC)
            kct_offs = kc_start + kct_k
            kc_block_t = ttgl.load(
                kc_ptr + pid_bh * n_kc * D + kct_offs[None, :] * D + kct_d[:, None],
                mask=(kct_offs < n_kc)[None, :],
                other=0.0,
            )
            kc_block_t = ttgl.convert_layout(kc_block_t, rhs_layout)
    
        with pl.scope("mma"):  # ~4% of time
            # dot product between blocks of qc and kc, gives (QC, KC) in fp32
            acc = ttgl.zeros([QC_BLOCK_SIZE, KC_BLOCK_SIZE], ttgl.float32, acc_layout)
            dots = ttgl.nvidia.ampere.mma_v2(qc_block, kc_block_t, acc) * rsqrt_d
            dots = ttgl.where(kc_offs[None, :] == n_kc - 1, dots + log_last_kc_frac, dots)
    
        
        with pl.scope("binning"):  # ~1/10 of time
            # find bins before applying exponential
            bin_idx = ((dots - lower_edges[:, None]) * inv_bin_widths[:, None]).to(ttgl.int32)
            hist_mask = q_mask[:, None] & kc_mask[None, :] & (bin_idx >= 0) & (bin_idx < N_BINS)  # mask for values outside range
            bin_idx = ttgl.minimum(ttgl.maximum(bin_idx, 0), N_BINS - 1)  # clamp to avoid out-of-bounds smem addresses
    
            # apply stabilized exponential, in place to save memory
            dots = ttgl.exp(dots - row_maxes[:, None])
            dots = ttgl.where(kc_mask[None, :], dots, 0.0)

        with pl.scope("hist_scatter"):  # ~3/4 of time
            # update smem histograms using scatter adds
            exp_hist.atomic_scatter_add(dots, bin_idx, axis=1, mask=hist_mask)
            cnt_hist.atomic_scatter_add(
                ttgl.full([QC_BLOCK_SIZE, KC_BLOCK_SIZE], 1, ttgl.int32, acc_layout),
                bin_idx, axis=1, mask=hist_mask
            )
    
    ###################
    ## Store results ##
    ###################

    # each (bh, q) histogram row is owned by exactly one program, so plain
    # stores replace the global atomics entirely (buffers need not be pre-zeroed)
    ttgl.barrier()
 
    fl_q = pid_q * QC_BLOCK_SIZE + ttgl.arange(0, QC_BLOCK_SIZE, layout=ttgl.SliceLayout(1, hist_layout))
    fl_b = ttgl.arange(0, N_BINS, layout=ttgl.SliceLayout(0, hist_layout))
    fl_offs = (pid_bh * n_qc + fl_q[:, None]) * N_BINS + fl_b[None, :]
    fl_mask = (fl_q < n_qc)[:, None]
 
    ttgl.store(exp_histogram_ptr + fl_offs, exp_hist.load(hist_layout), mask=fl_mask)
    ttgl.store(counts_histogram_ptr + fl_offs, cnt_hist.load(hist_layout), mask=fl_mask)
 
@triton.jit(do_not_specialize=["mass_threshold", "count_threshold", "n_qc"])
def threshold_kernel(
    # global
    mass_threshold, count_threshold,

    # per query-centroid
    exp_histogram_ptr, counts_histogram_ptr,
    mass_offsets_ptr, count_offsets_ptr, 
    denominators_ptr,
    lower_edges_ptr, upper_edges_ptr,

    # other
    n_qc, 
    N_BINS: tl.constexpr,
):
    linear_id = tl.program_id(0)

    pid_q = linear_id % n_qc
    pid_bh = linear_id // n_qc
    q_idx = pid_bh * n_qc + pid_q

    denominator = tl.load(denominators_ptr + q_idx)
    bin_indices = tl.arange(0, N_BINS)
    lower_edge = tl.load(lower_edges_ptr + q_idx)
    upper_edge = tl.load(upper_edges_ptr + q_idx)
    bin_width = (upper_edge - lower_edge) / N_BINS

    # Load offsets, we init them with 0s so we can always load them
    mass_offset = tl.load(mass_offsets_ptr + q_idx)
    count_offset = tl.load(count_offsets_ptr + q_idx)

    hist_base = q_idx * N_BINS
    reverse_indices = N_BINS -1 -bin_indices

    # Find edge-bin (within which the exp threshold lies) and the exp/count in each bin
    # Exp
    exp_threshold = denominator*mass_threshold -mass_offset
    reversed_exp_bins = tl.load(exp_histogram_ptr +hist_base +reverse_indices)
    exp_cumsum = tl.cumsum(reversed_exp_bins, axis=0)  # cumulative sum from the top (to avoid accumulating numerical errors)
    exp_exceeds = exp_cumsum >= exp_threshold  # first bin that exceeds threshold
    exp_first_exceeding = tl.min(tl.where(exp_exceeds, bin_indices, N_BINS), axis=0)  # mask exceeding, take first unmasked
    exp_edge_bin = N_BINS -1 -exp_first_exceeding  # convert to unreversed bin index

    # Count
    row_count_threshold = count_threshold -count_offset
    reversed_count_bins = tl.load(counts_histogram_ptr +hist_base +reverse_indices)
    count_cumsum = tl.cumsum(reversed_count_bins, axis=0)
    count_exceeds = count_cumsum >= row_count_threshold
    count_first_exceeding = tl.min(tl.where(count_exceeds, bin_indices, N_BINS), axis=0)
    count_edge_bin = N_BINS -1 -count_first_exceeding

    # Actual edge bin and summaries for potential refinement round
    edge_bin = tl.minimum(exp_edge_bin, count_edge_bin)
    edge_reverse_idx = N_BINS -1 -edge_bin

    # Compute updated values    
    new_lower_edge = lower_edge + edge_bin * bin_width
    new_upper_edge = new_lower_edge + bin_width
    exp_above = tl.sum(tl.where(bin_indices < edge_reverse_idx, reversed_exp_bins, 0.0), axis=0)  # the total mass in bins above the threshold bin
    count_above = tl.sum(tl.where(bin_indices < edge_reverse_idx, reversed_count_bins, 0), axis=0)

    # Save new bin edges, offsets, the threshold, and also denominators if first run
    tl.store(lower_edges_ptr + q_idx, new_lower_edge)
    tl.store(upper_edges_ptr + q_idx, new_upper_edge)
    tl.store(mass_offsets_ptr + q_idx, mass_offset + exp_above)
    tl.store(count_offsets_ptr + q_idx, count_offset + count_above)
