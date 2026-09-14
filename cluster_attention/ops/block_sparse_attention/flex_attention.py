import torch
import torch.nn.functional as F
import math

from torch.nn.attention.flex_attention import flex_attention, BlockMask
compiled_flex_attention = torch.compile(flex_attention)

def flex_attention(q_s, k_s, v_s, top_idx, kv_counts, B, H, N, n_blocks_q, n_assign, cs_q, cs_k, device):
        pad_size = math.lcm(cs_k, cs_q)
        pad = (pad_size - N % pad_size) % pad_size
        N_pad = N + pad
        if pad > 0:
            q_s = F.pad(q_s, (0, 0, 0, pad))
            k_s = F.pad(k_s, (0, 0, 0, pad))
            v_s = F.pad(v_s, (0, 0, 0, pad))
        n_blocks_q_pad = N_pad // cs_q
        if n_blocks_q_pad > n_blocks_q:
            extra = n_blocks_q_pad - n_blocks_q
            top_idx = F.pad(top_idx, (0, 0, 0, extra), value=0)
            if kv_counts is not None:
                kv_counts = F.pad(kv_counts, (0, extra), value=1)
        block_mask = flex_build_block_mask(top_idx, kv_counts, B, H, n_blocks_q_pad, n_assign, N, N_pad, cs_q, cs_k, device)
        out_s = compiled_flex_attention(q_s, k_s, v_s, block_mask=block_mask, kernel_options={"BLOCK_M": cs_q, "BLOCK_N": cs_k})
        return out_s[:, :, :N, :]

def flex_build_block_mask(top_idx, kv_counts, B, H, n_blocks_q, n_assign, N, N_pad, cs_q, cs_k, device):
    if kv_counts is not None:
        kv_num_blocks = kv_counts.to(torch.int32)
    else:
        kv_num_blocks = torch.full((B, H, n_blocks_q), n_assign, device=device, dtype=torch.int32)
    kv_indices = top_idx.to(torch.int32)
    def mask_mod(b, h, q_idx, kv_idx):
        return kv_idx < N
    block_mask = BlockMask(
        seq_lengths=(N_pad, N_pad),
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=None,
        full_kv_indices=None,
        q_num_blocks=None,
        q_indices=None,
        full_q_num_blocks=None,
        full_q_indices=None,
        BLOCK_SIZE=(cs_q, cs_k),
        #mask_mod=None,
        mask_mod=mask_mod
    )
    return block_mask
