import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from .ops import assignment, sage_with_custom_mask, compensate_query_to_mean
from .core import AttentionBackend
from .clustering import (
  linear_transforms_diagonalized_clustering, 
  attention_scores_diagonalized_clustering
)

##################
## DenseBackend ##
##################

try:
    #from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
    from sageattention import sageattn
except Exception as e:
    print(e)
class DenseBackend(AttentionBackend):
    is_dense = True

    def __init__(self, use_sage=False):
        super().__init__(c=1)
        self.use_sage = use_sage
        self.use_model_fallback = not use_sage  # the basic dense one can fall back

    def forward(self, q, k, v, layer_idx, return_lse=False):
        if self.use_sage:
            return sageattn(q, k, v, tensor_layout="HND", is_causal=False), None
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):  # forcing FA2
            return F.scaled_dot_product_attention(q, k, v, enable_gqa=True), None
            
#####################################
## Other AttentionBackend versions ##
#####################################

class RandomClusterBackend(AttentionBackend):
    def __init__(
            self, k=16, k_frac=None, cluster_size=(64, 64), 
            random_assignment=False, nonrandom_queries=False, nonrandom_keys=False, device="cuda",
            compensate=0
        ):
        super().__init__()
        self.k = k
        self.cluster_size = cluster_size
        self.random_assignment = random_assignment
        self.nonrandom_queries = nonrandom_queries
        self.nonrandom_keys = nonrandom_keys
        if k_frac:
            assert (k_frac <= 1) and (k_frac > 0), "need k_frac between 0 (exclusive) and 1 (inclusive)"
        self.k_frac = k_frac  # if defined overrides "k"
        self.compensate = compensate

    def forward(self, q, k, v, layer_idx, return_lse=False):
        B, H, N, D = q.shape
        cs_k, cs_q = self.cluster_size
        n_blocks_k = (N + cs_k - 1) // cs_k

        # Random permutations instead of learned clustering
        q_perm_small = torch.randperm(N, device=q.device).expand(B, H, -1)
        k_perm_small = torch.randperm(N, device=q.device).expand(B, H, -1)

        if self.nonrandom_queries:
            k_flat, q_flat = k.reshape(B * H, N, D), q.reshape(B * H, N, D)  # just contiguous views, no copies
            v_flat = v.reshape(B * H, N, D)
            _, _, _, q_perm = linear_transforms_diagonalized_clustering(
                k_flat, q_flat, v_flat,
                use_ranking_objective_keys=True,
                key_target_size=cs_k, query_target_size=cs_q
            )
            q_perm_small = q_perm.view(B, H, N)
        if self.nonrandom_keys:
            k_flat, q_flat = k.reshape(B * H, N, D), q.reshape(B * H, N, D)  # just contiguous views, no copies
            v_flat = v.reshape(B * H, N, D)
            _, k_perm, _, _ = linear_transforms_diagonalized_clustering(
                k_flat, q_flat, v_flat,
                use_ranking_objective_keys=True,
                key_target_size=cs_k, query_target_size=cs_q
            )
            k_perm_small = k_perm.view(B, H, N)

        q_perm = q_perm_small.unsqueeze(-1).expand(-1, -1, -1, D)
        k_perm = k_perm_small.unsqueeze(-1).expand(-1, -1, -1, D)

        q = q.gather(2, q_perm)
        k = k.gather(2, k_perm)
        v = v.gather(2, k_perm)

        #n_assign = min(self.k, n_blocks_k)
        n_assign = int(self.k_frac*n_blocks_k) if self.k_frac else min(self.k, n_blocks_k)
        if self.random_assignment:
            n_blocks_q = (N + cs_q - 1) // cs_q
            indices = torch.stack([
                torch.randperm(n_blocks_k, device=q.device)[:n_assign]
                for _ in range(n_blocks_q)
            ])
            indices, _ = indices.sort(dim=-1)
            deltas = indices.clone()
            deltas[..., 1:] = indices[..., 1:] - indices[..., :-1]

            lut = torch.zeros(B, H, n_blocks_q, n_blocks_k, device=q.device, dtype=torch.int32)
            lut[:, :, :, :n_assign] = deltas.to(torch.int32)
            valid_block_num = torch.full((B, H, n_blocks_q), n_assign, device=q.device, dtype=torch.int32)
        else:
            lut, valid_block_num = assignment(q, k, v, cs_q, cs_k, 0, n_assign, False)

        out = sage_with_custom_mask(
            q, k, v, lut, valid_block_num,
            return_lse=(return_lse or self.compensate > 0)
        )
        if self.compensate > 0:
            out_sel, lse_sel = out
            out, lse = compensate_query_to_mean(
                q, k, v, lut, valid_block_num,
                cs_q, cs_k, out_sel, lse_sel
            )
        elif return_lse:
            out, lse = out
        else:
            out, lse = out, None
        result = torch.empty_like(out)
        result.scatter_(2, q_perm, out)
        if lse is not None:
            lse_out = torch.empty_like(lse)
            lse_out.scatter_(2, q_perm_small, lse)
            lse = lse_out
        return result, lse

#####################################
## For extracting QKV from a layer ##
#####################################

class QKVProbe(AttentionBackend):
    def __init__(self, target_layer=0):
        super().__init__(c=1)
        self.target_layer = target_layer
        self.captured = None

    def forward(self, q, k, v, layer_idx):
        if layer_idx==self.target_layer:
            self.captured = (k.detach(), q.detach(), v.detach())
        return F.scaled_dot_product_attention(q, k, v), None
