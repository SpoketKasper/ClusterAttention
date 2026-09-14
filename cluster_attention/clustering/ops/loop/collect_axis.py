import triton
import triton.language as tl

# API
def collect_axis(vectors, perm, proj, offs, szs, axis, max_size, target_size, block_m=128, num_warps=4):
    batch_size, n = perm.shape
    d = vectors.shape[2]
    S = offs.shape[0]
    _collect_axis_kernel[(batch_size * S, triton.cdiv(max_size, block_m))](
        vectors,
        perm,
        offs,
        szs,
        axis,
        proj,
        n,
        S,
        d,
        target_size,
        BLOCK_M=block_m,
        num_warps=num_warps,
    )

# Kernel
@triton.jit(do_not_specialize=["N", "S", "T"])
def _collect_axis_kernel(
    vectors,
    perm,
    offs,
    szs,
    axis,
    proj,
    N,
    S,
    D,
    T,
    BLOCK_M: tl.constexpr,
):
    pid_bs = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = pid_bs // S
    pid_s = pid_bs - pid_b * S

    size = tl.load(szs + pid_s)
    if size <= T:
        return
    off = tl.load(offs + pid_s)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < size

    idx = tl.load(perm + pid_b * N + off + offs_m, mask=mask_m, other=0)
    a = tl.load(axis + pid_b * S + pid_s)
    val = tl.load(vectors + (pid_b * N + idx) * D + a, mask=mask_m, other=0.0)
    #tl.store(proj + pid_b * N + off + offs_m, val.to(tl.float32), mask=mask_m)
    tl.store(proj + pid_b * N + off + offs_m, val, mask=mask_m)
