import torch
import math
import triton
import triton.language as tl
from ..utils import Timer

####################
## Kernel wrapper ##
####################

# on medium size, everything before: 0.35 ms, kernel: 1.73 ms
def compensate_query_to_mean(
    q, k, v, lut, valid_block_num, cs_q, cs_k, out_sel, lse_sel,
    log2_lse=True, BLOCK_KC=128
):        
    B, H, N, D = q.shape
    BH = B*H
    n_qc, n_kc = (N +cs_q -1)//cs_q, (N +cs_k -1)//cs_k
    scale = D**-0.5

    #######################
    ## Compute centroids ##
    #######################

    n_full_k, rem_k = N//cs_k, N%cs_k
 
    kc = k[:, :, :n_full_k*cs_k].view(B, H, n_full_k, cs_k, D).mean(3)
    #kc = k[:, :, :n_full_k * cs_k].view(B, H, n_full_k, cs_k, D).sum(3, dtype=torch.float32) / cs_k
    vc = v[:, :, :n_full_k*cs_k].view(B, H, n_full_k, cs_k, D).mean(3)
    if rem_k:
        kc = torch.cat([kc, k[:, :, n_full_k*cs_k:N].mean(2, keepdim=True)], dim=2)
        vc = torch.cat([vc, v[:, :, n_full_k*cs_k:N].mean(2, keepdim=True)], dim=2)
 
    counts = torch.full((n_kc,), cs_k, device=q.device, dtype=torch.float32)
    if rem_k:
        counts[-1] = rem_k
    log_counts = counts.log()

    ################
    ## Decode LUT ##
    ################

    lut = lut.view(BH, n_qc, -1)
    max_lut_len = lut.shape[-1]
    mask = torch.arange(max_lut_len, device=lut.device) < valid_block_num.view(BH, n_qc, 1)
    lut.masked_fill_(~mask, 0)
    lut.cumsum_(-1)
 
    ################
    ## Run kernel ##
    ################
    
    if log2_lse:
        lse_sel = lse_sel * math.log(2)  # convert log2 to ln
    out = torch.empty_like(out_sel)
    lse = torch.empty_like(lse_sel)
    _centroid_compensation_kernel[(BH, n_qc)](
        q, kc, vc,
        log_counts, lut, valid_block_num,
        out_sel, lse_sel, out, lse,
        scale, N, n_kc, n_qc, max_lut_len,
        BLOCK_Q=cs_q, D=D, BLOCK_KC=BLOCK_KC,
    )
    lse = lse / math.log(2)  # convert back for consistency
    return out, lse.view(B, H, N)

############
## Kernel ##
############

#@triton.jit
@triton.jit(do_not_specialize=["N", "n_kc", "n_qc", "max_lut_len"])
def _centroid_compensation_kernel(
    Q, kc_ptr, vc_ptr, 
    log_counts_ptr, lut_ptr, valid_count_ptr, out_sel_ptr, lse_sel_ptr, 
    out_ptr, lse_ptr,
    sm_scale, N, n_kc, n_qc, max_lut_len,
    BLOCK_Q: tl.constexpr, D: tl.constexpr, BLOCK_KC: tl.constexpr,
):
    bh = tl.program_id(0)
    qb = tl.program_id(1)

    q_range = qb*BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_range = tl.arange(0, D)
    kc_range = tl.arange(0, BLOCK_KC)

    base_bhnd = (bh*N).to(tl.int64) * D
    base_bhkd = (bh*n_kc).to(tl.int64) * D
    base_bhn = bh * N
    lut_base = (bh*n_qc + qb).to(tl.int64) * max_lut_len

    q = tl.load(Q + base_bhnd + q_range[:, None]*D + d_range[None, :],
                mask=q_range[:, None] < N, other=0.0)

    m_i = tl.full([BLOCK_Q], value=float('-inf'), dtype=tl.float32)
    d_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, D], dtype=tl.float32)

    # cursor into sorted selected indices
    n_sel = tl.load(valid_count_ptr + bh*n_qc + qb)
    sel_cursor = 0
    sel_offs = tl.arange(0, BLOCK_KC)

    for kc_start in range(0, n_kc, BLOCK_KC):
        kc_offs = kc_start + kc_range
        mask_kc = kc_offs < n_kc

        # load window of selected indices starting at cursor
        window = tl.load(
            lut_ptr + lut_base + sel_cursor + sel_offs,
            mask=(sel_cursor+sel_offs)<n_sel, other=n_kc+1
        )

        # broadcast compare: is each centroid in the selected set?
        is_selected = tl.sum((kc_offs[:, None]==window[None, :]).to(tl.int32), axis=1) > 0

        # advance cursor past indices that fall in this tile
        sel_cursor += tl.sum((window<kc_start + BLOCK_KC).to(tl.int32))

        k = tl.load(
            kc_ptr + base_bhkd + kc_offs[:, None]*D + d_range[None, :],
            mask=mask_kc[:, None], other=0.0
        )
        v = tl.load(
            vc_ptr + base_bhkd + kc_offs[:, None]*D + d_range[None, :],
            mask=mask_kc[:, None], other=0.0
        )
        log_counts = tl.load(log_counts_ptr + kc_offs, mask=mask_kc, other=0.0)

        logits = tl.dot(q, tl.trans(k))*sm_scale + log_counts[None, :]
        logits = tl.where(is_selected[None, :] | ~mask_kc[None, :], float('-inf'), logits)

        m_ij = tl.max(logits, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(logits - m_new[:, None])

        # guard against fully-masked tiles where m_ij = -inf
        safe = m_ij > float('-inf')
        d_i = tl.where(safe, d_i*alpha + tl.sum(p, axis=1), d_i)
        acc = tl.where(safe[:, None], acc*alpha[:, None] + tl.dot(p.to(v.dtype), v), acc)
        m_i = tl.where(safe, m_new, m_i)
        #d_i = d_i * alpha + tl.sum(p, axis=1)
        #acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        #m_i = m_new

    lse_unsel = m_i + tl.log(d_i)
    #out_unsel = acc / d_i[:, None]

    lse_s = tl.load(lse_sel_ptr + base_bhn + q_range, mask=(q_range<N), other=0.0).to(tl.float32)
    out_s = tl.load(
        out_sel_ptr + base_bhnd + q_range[:, None]*D + d_range[None, :],
        mask=(q_range[:, None]<N), other=0.0
    ).to(tl.float32)

    max_lse = tl.maximum(lse_s, lse_unsel)
    w_s = tl.exp(lse_s - max_lse)
    w_u = tl.exp(lse_unsel - max_lse)

    # w_s * out_s + exp(m_i - max_lse) * acc, divided by (w_s + w_u)
    contrib_unsel = tl.exp(m_i - max_lse)[:, None] * acc
    out_combined = (w_s[:, None] * out_s + contrib_unsel) / (w_s + w_u)[:, None]
    tl.store(
        out_ptr + base_bhnd + q_range[:, None]*D + d_range[None, :],
        out_combined.to(q.dtype), mask=q_range[:, None] < N
    )

    lse_combined = max_lse + tl.log(w_s + w_u)
    tl.store(lse_ptr + base_bhn + q_range, lse_combined, mask=q_range<N)

##############
## Fallback ##
##############

# slow pytorch implementation for experiments
from .block_sparse_attention import (
    lut_to_idx, idx_to_lut, flashinfer_block_sparse_nonvariable_attn
)
def compensated_block_sparse_attention(q, k, v, lut, valid_block_num, cs_q, cs_k, mean_to_mean=True):
    B, H, N, D = q.shape
    n_qc = (N + cs_q - 1) // cs_q
    n_kc = (N + cs_k - 1) // cs_k
    scale = D ** -0.5

    # Part 0: Decode LUT for the selected mask
    selected = lut_to_idx(lut, valid_block_num, B, H, n_qc, n_kc, q.device)

    # Part 1: Sparse attention on selected blocks via flashinfer
    out_sel, lse_sel = flashinfer_block_sparse_nonvariable_attn(
        q, k, v, lut, valid_block_num, cs_q, cs_k, 
        return_lse=True, return_fp16=True
    )
    lse_sel = lse_sel * math.log(2)  # convert log2 → ln

    # Part 2: Centroid compensation for unselected blocks
    k_pad = torch.zeros(B, H, n_kc * cs_k, D, device=q.device, dtype=q.dtype)
    v_pad = torch.zeros_like(k_pad)
    k_pad[:, :, :N] = k
    v_pad[:, :, :N] = v
    counts = torch.full((n_kc,), cs_k, device=q.device, dtype=q.dtype)
    counts[-1] = N - (n_kc - 1) * cs_k

    k_cent = k_pad.view(B, H, n_kc, cs_k, D).sum(3) / counts[None, None, :, None]
    v_cent = v_pad.view(B, H, n_kc, cs_k, D).sum(3) / counts[None, None, :, None]

    log_counts = counts.log()

    if mean_to_mean:
        q_pad2 = torch.zeros(B, H, n_qc * cs_q, D, device=q.device, dtype=q.dtype)
        q_pad2[:, :, :N] = q
        q_counts = torch.full((n_qc,), cs_q, device=q.device, dtype=q.dtype)
        q_counts[-1] = N - (n_qc - 1) * cs_q
        q_cent = q_pad2.view(B, H, n_qc, cs_q, D).sum(3) / q_counts[None, None, :, None]

        cent_logits = torch.einsum('bhqd,bhkd->bhqk', q_cent.float(), k_cent.float()) * scale
        cent_logits = cent_logits + log_counts[None, None, None, :]
        cent_logits.masked_fill_(selected, float('-inf'))

        lse_unsel = torch.logsumexp(cent_logits, dim=-1)
        attn_unsel = torch.softmax(cent_logits, dim=-1)
        out_unsel = torch.einsum('bhqk,bhkd->bhqd', attn_unsel, v_cent.float())

        lse_unsel = lse_unsel.repeat_interleave(cs_q, dim=2)[:, :, :N]
        out_unsel = out_unsel.repeat_interleave(cs_q, dim=2)[:, :, :N]
    else:
        sel_tok = selected.repeat_interleave(cs_q, dim=2)[:, :, :N]
        cent_logits = torch.einsum('bhnd,bhkd->bhnk', q.float(), k_cent.float()) * scale
        cent_logits = cent_logits + log_counts[None, None, None, :]
        cent_logits.masked_fill_(sel_tok, float('-inf'))

        lse_unsel = torch.logsumexp(cent_logits, dim=-1)
        attn_unsel = torch.softmax(cent_logits, dim=-1)
        out_unsel = torch.einsum('bhnk,bhkd->bhnd', attn_unsel, v_cent.float())

    # Part 3: Combine
    lse_sel = lse_sel.float()
    lse_unsel = lse_unsel.float()
    out_sel = out_sel.float()
    out_unsel = out_unsel.float()

    max_lse = torch.maximum(lse_sel, lse_unsel)
    w_sel = torch.exp(lse_sel - max_lse)
    w_unsel = torch.exp(lse_unsel - max_lse)
    w_total = w_sel + w_unsel

    out = (w_sel[..., None] * out_sel + w_unsel[..., None] * out_unsel.nan_to_num()) / w_total[..., None]
    return out.to(q.dtype)
