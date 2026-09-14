import torch
import triton
import triton.language as tl

# API
def find_best_axis(vectors, perm, offs, szs, max_size, sample, target_size, num_warps=4):
    batch_size, n = perm.shape
    d = vectors.shape[2]
    S = offs.shape[0]
    axis = torch.empty((batch_size, S), device=vectors.device, dtype=torch.int32)
    _find_best_axis_kernel[(batch_size*S,)](
        vectors,
        perm,
        offs,
        szs,
        axis,
        n,
        S,
        min(sample, max_size),
        d,
        target_size,
        BLOCK_N=triton.next_power_of_2(sample),
        BLOCK_D=triton.next_power_of_2(d),
        num_warps=num_warps,
    )
    return axis

# Kernel
@triton.jit(do_not_specialize=["N", "S", "SAMPLE", "T"])
def _find_best_axis_kernel(
    vectors,
    perm,
    offs,
    szs,
    axis_out,
    N,
    S,
    SAMPLE,
    D,
    T,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = pid // S
    pid_s = pid - pid_b * S

    size = tl.load(szs + pid_s)
    if size <= T:
        return
    off = tl.load(offs + pid_s)

    ns = tl.minimum(SAMPLE, size)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < ns
    mask_d = offs_d < D

    rand_pos = tl.minimum((tl.rand(pid, offs_n)*size).to(tl.int32), size-1)
    sample_pos = tl.where(size > ns, rand_pos, offs_n)

    idx = tl.load(perm + pid_b * N + off + sample_pos, mask=mask_n, other=0)
    x = tl.load(
        vectors + (pid_b * N + idx[:, None]) * D + offs_d[None, :],
        mask=mask_n[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)

    mean = tl.sum(x, axis=0) / ns
    centered = tl.where(mask_n[:, None] & mask_d[None, :], x-mean[None, :], 0.0)
    var = tl.sum(centered * centered, axis=0)
    var = tl.where(mask_d, var, -float("inf"))
    tl.store(axis_out + pid_b*S + pid_s, tl.argmax(var, axis=0).to(tl.int32))

