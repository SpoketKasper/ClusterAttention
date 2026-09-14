import torch
import triton
import triton.language as tl

# API
def normalize(X, inplace=True):
    X_flat = X.contiguous().view(-1, X.shape[-1])  # flattening to (something, D), just a view
    N_rows, D = X_flat.shape
    X_normalized_flat = X_flat if inplace else torch.empty_like(X_flat)
    BLOCK_M = 16
    grid = (triton.cdiv(N_rows, BLOCK_M),)
    _normalize_kernel[grid](
        X_flat, X_normalized_flat,
        N_rows, D,
        BLOCK_M=BLOCK_M,
    )
    return X_normalized_flat.view(X.shape)  # putting back into original shape

# Kernel
@triton.jit(do_not_specialize=["N_rows"])
def _normalize_kernel(
    x_ptr, x_normalized_ptr,
    N_rows, D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid*BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    mask = rows[:, None] < N_rows
    x = tl.load(
        x_ptr + rows[:, None]*D + cols[None, :], mask=mask, other=0.0
    )
    original_dtype = x.dtype
    x = x.to(tl.float32)
    eps = 1e-8
    inv_norm = 1.0 / tl.sqrt(tl.sum(x*x, axis=1)+eps)
    out = x*inv_norm[:, None]
    tl.store(
        x_normalized_ptr + rows[:, None]*D + cols[None, :], 
        out.to(original_dtype), mask=mask
    )
