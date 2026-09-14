import torch
import triton
import triton.language as tl

# API
def center(keys, mu, out=None, block_n=256):
    B, N, D = keys.shape
    if out is None:
        out = torch.empty_like(keys)
    grid = (B, triton.cdiv(N, block_n))
    _center_kernel[grid](keys, mu, out, N, D, BLOCK_N=block_n)
    return out

# Kernel
@triton.jit(do_not_specialize=["N"])
def _center_kernel(
    keys_ptr, mu_ptr, out_ptr,
    N, D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    mask = offs_n[:, None] < N

    k = tl.load(keys_ptr + pid_b * N * D + offs_n[:, None] * D + offs_d[None, :], mask=mask).to(tl.float32)
    m = tl.load(mu_ptr + pid_b * D + offs_d).to(tl.float32)
    result = (k - m[None, :]).to(tl.bfloat16)

    tl.store(out_ptr + pid_b * N * D + offs_n[:, None] * D + offs_d[None, :], result, mask=mask)
