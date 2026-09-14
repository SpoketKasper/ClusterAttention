import torch
import math

# the function in sage,
# qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90,
# uses a lut that is starting point and deltas
def top_idx_to_lut(top_idx, kv_counts, n_blocks_q, n_blocks_k, n_assign, device):
    valid_block_num = kv_counts.to(torch.int32).contiguous()
    lut = torch.zeros(*top_idx.shape[:-1], n_blocks_k, device=device, dtype=torch.int32)
    lut[..., :top_idx.shape[-1]] = top_idx
    lut[..., 1:top_idx.shape[-1]] -= top_idx[..., :-1]
    return lut.contiguous(), valid_block_num

def topk_assignment(q_s, k_s, B, H, n_assign, n_blocks_q, n_blocks_k, cs_q, cs_k, D, N):
    n_full_q = N // cs_q
    rem_q = N % cs_q
    q_c = q_s[:, :, :n_full_q * cs_q, :].view(B, H, n_full_q, cs_q, D).mean(3)
    if rem_q:
        q_c = torch.cat([q_c, q_s[:, :, n_full_q * cs_q:N, :].mean(2, keepdim=True)], dim=2)

    n_full_k = N // cs_k
    rem_k = N % cs_k
    k_c = k_s[:, :, :n_full_k * cs_k, :].view(B, H, n_full_k, cs_k, D).mean(3)
    if rem_k:
        k_c = torch.cat([k_c, k_s[:, :, n_full_k * cs_k:N, :].mean(2, keepdim=True)], dim=2)

    sim = torch.einsum("bhqd,bhkd->bhqk", q_c, k_c) / math.sqrt(D)
    if rem_k:
        sim[:, :, :, -1] += math.log(rem_k / cs_k)
    _, top_idx = sim.topk(n_assign, dim=-1)

    assignment = torch.zeros(B, H, n_blocks_q, n_blocks_k, device=q_s.device, dtype=torch.int32)
    assignment[..., :n_assign] = top_idx.sort(dim=-1).values
    counts = torch.full((B, H, n_blocks_q), n_assign, device=q_s.device, dtype=torch.int32)
    return assignment, counts

def adaptive_assignment(q_s, k_s, B, H, n_blocks_q, n_blocks_k, cs_q, cs_k, D, N, target_recall=0.9, min_k=16):
    # last block might be partial
    n_full_q = N // cs_q
    rem_q = N % cs_q
    q_c = q_s[:, :, :n_full_q * cs_q, :].view(B, H, n_full_q, cs_q, D).mean(3)
    if rem_q:
        q_c = torch.cat([q_c, q_s[:, :, n_full_q * cs_q:N, :].mean(2, keepdim=True)], dim=2)

    n_full_k = N // cs_k
    rem_k = N % cs_k
    k_c = k_s[:, :, :n_full_k * cs_k, :].view(B, H, n_full_k, cs_k, D).mean(3)
    if rem_k:
        k_c = torch.cat([k_c, k_s[:, :, n_full_k * cs_k:N, :].mean(2, keepdim=True)], dim=2)

    sim = torch.einsum("bhqd,bhkd->bhqk", q_c.float(), k_c.float()) / math.sqrt(D)
    if rem_k:
        sim[:, :, :, -1] += math.log(rem_k / cs_k)

    weights = torch.softmax(sim.float(), dim=-1)
    sorted_weights, sorted_idx = weights.sort(dim=-1, descending=False)
    cumsum_small = sorted_weights.cumsum(dim=-1)
    n_exclude = (cumsum_small <= (1.0 - target_recall)).sum(dim=-1)
    n_needed = (n_blocks_k - n_exclude).clamp(min=min_k, max=n_blocks_k)
    max_n = n_needed.max().item()
    top_idx = sorted_idx.flip(-1)[:, :, :, :max_n]
    return top_idx, n_needed

def assignment(q_s, k_s, cs_q, cs_k, target_recall, target_count, adaptive_k):
    B, H, N, D = q_s.shape
    n_blocks_q, n_blocks_k = (N +cs_q -1) //cs_q, (N +cs_k -1) //cs_k
    if not adaptive_k:
        top_idx, n_needed = topk_assignment(q_s, k_s, B, H, target_count, n_blocks_q, n_blocks_k, cs_q, cs_k, D, N)
    else:
        top_idx, n_needed = adaptive_assignment(q_s, k_s, B, H, n_blocks_q, n_blocks_k, cs_q, cs_k, D, N, target_recall=target_recall, min_k=target_count)
    lut, valid_block_num = top_idx_to_lut(top_idx, kv_counts, n_blocks_q, n_blocks_k, n_assign, q_s.device)
    return lut, valid_block_num