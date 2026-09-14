import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

def launch_minmax_kernel(
    qc,
    kc,
    last_kc_frac,
    QC_BLOCK_SIZE=64,
    KC_BLOCK_SIZE=128,
    WARPS_M=4,
    WARPS_N=1,
):
    B, H, n_qc, D = qc.shape
    BH = B * H
    n_kc = kc.shape[2]

    mins = torch.empty((BH, n_qc), dtype=torch.float32, device=qc.device)
    maxes = torch.empty_like(mins)
    denom = torch.empty_like(mins)

    n_q_blocks = triton.cdiv(n_qc, QC_BLOCK_SIZE)
    grid = (BH * n_q_blocks,)

    minmax_kernel[grid](
        qc, kc,
        mins, maxes, denom,
        last_kc_frac, n_qc, n_kc, n_q_blocks,
        D=D,
        QC_BLOCK_SIZE=QC_BLOCK_SIZE,
        KC_BLOCK_SIZE=KC_BLOCK_SIZE,
        WARPS_M=WARPS_M,
        WARPS_N=WARPS_N,
        num_warps=WARPS_M * WARPS_N,
    )
    return mins, maxes, denom

@gluon.jit(do_not_specialize=["last_kc_frac", "n_qc", "n_kc", "n_q_blocks"])
def minmax_kernel(
    # global
    qc_ptr, kc_ptr,

    # per query-centroid
    min_ptr, max_ptr, denom_ptr,

    # other
    last_kc_frac, n_qc, n_kc, n_q_blocks,
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
    THREADS_D: ttgl.constexpr = D // 8  # 8 fp16 per thread = 128-bit loads
    qc_load_layout: ttgl.constexpr = ttgl.BlockedLayout(
        [1, 8], [32 // THREADS_D, THREADS_D], [WARPS_M, WARPS_N], [1, 0]
    )
    kct_load_layout: ttgl.constexpr = ttgl.BlockedLayout(
        [8, 1], [THREADS_D, 32 // THREADS_D], [WARPS_N, WARPS_M], [0, 1]
    )
    row_layout: ttgl.constexpr = ttgl.SliceLayout(1, acc_layout)

    #################
    ## IDs/offsets ##
    #################

    linear_id = ttgl.program_id(0)
    pid_q = linear_id % n_q_blocks
    pid_bh = linear_id // n_q_blocks
    rsqrt_d: ttgl.constexpr = 1.0 / (D ** 0.5)

    q_offs = pid_q * QC_BLOCK_SIZE + ttgl.arange(0, QC_BLOCK_SIZE, layout=row_layout)
    q_mask = q_offs < n_qc

    kc_range = ttgl.arange(0, KC_BLOCK_SIZE, layout=ttgl.SliceLayout(0, acc_layout))
    kct_d = ttgl.arange(0, D, layout=ttgl.SliceLayout(1, kct_load_layout))
    kct_k = ttgl.arange(0, KC_BLOCK_SIZE, layout=ttgl.SliceLayout(0, kct_load_layout))

    ####################
    ## One-time loads ##
    ####################

    q_offs_l = pid_q*QC_BLOCK_SIZE + ttgl.arange(0, QC_BLOCK_SIZE, layout=ttgl.SliceLayout(1, qc_load_layout))
    d_offs_l = ttgl.arange(0, D, layout=ttgl.SliceLayout(0, qc_load_layout))
    qc_block = ttgl.load(
        qc_ptr + pid_bh*n_qc*D + q_offs_l[:, None]*D + d_offs_l[None, :],
        mask=(q_offs_l < n_qc)[:, None], other=0.0,
    )
    qc_block = ttgl.convert_layout(qc_block, lhs_layout)

    ###########################
    ## Allocations and inits ##
    ###########################

    row_min = ttgl.full([QC_BLOCK_SIZE], float("inf"), ttgl.float32, row_layout)
    row_max = ttgl.full([QC_BLOCK_SIZE], float("-inf"), ttgl.float32, row_layout)
    row_denom = ttgl.zeros([QC_BLOCK_SIZE], ttgl.float32, row_layout)

    #########################
    ## Loop over kc-blocks ##
    #########################

    for kc_start in range(0, n_kc, KC_BLOCK_SIZE):
        # Dots!
        kc_offs = kc_start + kc_range
        kc_mask = kc_offs < n_kc

        kct_offs = kc_start + kct_k
        kc_block_t = ttgl.load(
            kc_ptr + pid_bh*n_kc*D + kct_offs[None, :]*D + kct_d[:, None],
            mask=(kct_offs < n_kc)[None, :], other=0.0,
        )
        kc_block_t = ttgl.convert_layout(kc_block_t, rhs_layout)

        acc = ttgl.zeros([QC_BLOCK_SIZE, KC_BLOCK_SIZE], ttgl.float32, acc_layout)
        dots = (ttgl.nvidia.ampere.mma_v2(qc_block, kc_block_t, acc) * rsqrt_d)
        dots = ttgl.where(kc_offs[None, :]==n_kc - 1, dots + log_last_kc_frac, dots)

        # Now update the min, max, and denominator 
        block_min = ttgl.min(ttgl.where(kc_mask[None, :], dots, float("inf")), axis=1)
        block_max = ttgl.max(ttgl.where(kc_mask[None, :], dots, float("-inf")), axis=1)

        row_min = ttgl.minimum(row_min, block_min)
        new_max = ttgl.maximum(row_max, block_max)
        dots_masked = ttgl.where(kc_mask[None, :], dots, float("-inf"))
        row_denom = (
            row_denom*ttgl.exp(row_max-new_max)
            + ttgl.sum(ttgl.exp(dots_masked-new_max[:, None]), axis=1)
        )
        row_max = new_max

    #####################################################
    ## Add some margin to avoid numerical issues later ##
    #####################################################

    eps: ttgl.constexpr = 1e-5
    row_min -= eps
    row_max += eps
    row_denom *= ttgl.exp(-eps)

    ###################
    ## Store results ##
    ###################

    base = pid_bh * n_qc + q_offs
    ttgl.store(min_ptr + base, row_min, mask=q_mask)
    ttgl.store(max_ptr + base, row_max, mask=q_mask)    
    ttgl.store(denom_ptr + base, row_denom, mask=q_mask)