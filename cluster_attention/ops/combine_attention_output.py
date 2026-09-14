import torch
import triton
import triton.language as tl

#########
## API ##
#########

def combine_attention(out1, lse1, out2, lse2, out=None, lse_out=None):
    D = out1.shape[-1]
    BHN = lse1.numel()  # grid, one program per row
    if out is None: out = torch.empty_like(out1)
    if lse_out is None: lse_out = torch.empty_like(lse1)
    _combine_attention_kernel[(BHN,)](out1, lse1, out2, lse2, out, lse_out, D)
    return out, lse_out

############
## Kernel ##
############

# use numerically stable logsumexp, like in torch.logsumexp
@triton.jit
def _combine_attention_kernel(
    out1_ptr, lse1_ptr,
    out2_ptr, lse2_ptr,
    out_ptr, lse_out_ptr,
    D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, D)

    lse1 = tl.load(lse1_ptr + row)
    lse2 = tl.load(lse2_ptr + row)
    out1 = tl.load(out1_ptr + row*D + offs_d).to(tl.float32)
    out2 = tl.load(out2_ptr + row*D + offs_d).to(tl.float32)

    m = tl.maximum(lse1, lse2)
    a1 = tl.exp(lse1 - m)
    a2 = tl.exp(lse2 - m)
    z = a1 + a2

    out = (a1*out1 + a2*out2) / z
    lse = m + tl.log(z)

    tl.store(out_ptr + row*D + offs_d, out.to(tl.bfloat16))
    tl.store(lse_out_ptr + row, lse)
