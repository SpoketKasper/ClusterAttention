# sage_with_custom_mask is adapted from the patterns in spas_sage_attn.core, 
# see SpargeAttn in this repo which comes from https://github.com/thu-ml/SpargeAttn
# Licenced under Apache 2.0

import torch
import math
from spas_sage_attn.utils import get_vanilla_qk_quant
import spas_sage_attn._qattn as qattn
import spas_sage_attn._fused as fused

from ...utils import print_cuda_tensors

def lut_to_idx(lut, valid_block_num, B, H, n_qc, n_kc, device):  # boolean output
    BH = B * H
    max_nv = max(valid_block_num.max().item(), 1)
    arange_nv = torch.arange(max_nv, device=device)
    lut_flat = lut.reshape(BH, n_qc, -1)
    vbn_flat = valid_block_num.reshape(BH, n_qc)
    nv_mask = arange_nv[None, None, :] < vbn_flat[:, :, None]
    deltas = lut_flat[:, :, :max_nv].long() * nv_mask
    abs_indices = deltas.cumsum(-1)
    selected = torch.zeros(BH, n_qc, n_kc, dtype=torch.bool, device=device)
    selected.scatter_(2, abs_indices.clamp(max=n_kc - 1), nv_mask)
    selected = selected.view(B, H, n_qc, n_kc)
    return selected

def idx_to_lut(selected):
    B, H, n_qc, n_kc = selected.shape
    valid_block_num = selected.sum(-1).int()
    max_sel = max(valid_block_num.max().item(), 1)

    indices = torch.arange(n_kc, device=selected.device).expand_as(selected)
    big = indices.masked_fill(~selected, n_kc).sort(-1).values[:, :, :, :max_sel]

    deltas = big.diff(dim=-1, prepend=big.new_zeros(B, H, n_qc, 1))

    # Pad back to full width with zeros
    out = torch.zeros(B, H, n_qc, n_kc, device=selected.device, dtype=torch.int32)
    out[:, :, :, :max_sel] = deltas.int()
    return out, valid_block_num.int()

def sage_with_custom_mask(q_s, k_s, v_s, lut, valid_block_num, return_lse=False):
    B, H, N, D = q_s.shape

    assert q_s.dtype == torch.bfloat16 and q_s.is_contiguous()
    assert k_s.dtype == torch.bfloat16 and k_s.is_contiguous()
    assert v_s.is_contiguous()
    v = v_s.to(torch.float16)

    km = k_s.mean(dim=-2, keepdim=True)
    q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q_s, k_s, km, 64, 128)
    
    scale = 1.0 / (D ** 0.5)

    padded_len = (N + 127) // 128 * 128
    v_tp = torch.empty((B, H, D, padded_len), dtype=v.dtype, device=v.device)
    fused.transpose_pad_permute_cuda(v, v_tp, 1)
    v_fp8 = torch.empty(v_tp.shape, dtype=torch.float8_e4m3fn, device=v.device)
    v_scale = torch.empty((B, H, D), dtype=torch.float32, device=v.device)
    fused.scale_fuse_quant_cuda(v_tp, v_fp8, v_scale, N, 2.25, 1)

    #print_cuda_tensors()

    o = torch.empty_like(q_s)
    # quantization granularity can be increased here
    writeback = qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_sm90(
        q_int8, k_int8, v_fp8, o, lut, valid_block_num,
        q_scale, k_scale, v_scale, 1, False, 1, scale, return_lse
    )
    if return_lse:
        # correcting the offset from sage

        # better? using the quantized values so maybe slightly more correct.
        #qs = q_scale.repeat_interleave(64, dim=-1)[..., :N]
        #q_hat = q_int8.float()*qs[..., None]
        #correction = (q_hat.float()*km.float()).sum(-1)*scale*math.log2(math.e)

        correction = (q_s.float()*km.float()).sum(-1)*scale*math.log2(math.e)
        return o, writeback+correction
    else:
        return o