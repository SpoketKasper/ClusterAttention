import torch
#torch._logging.set_logs(recompiles=True)

from .ops import (
    assignment, group_by_cluster, ungroup_back, compensate_query_to_mean,
    HAS_SPARGE, HAS_FLASHINFER,
)
if HAS_FLASHINFER: from .ops import flashinfer_block_sparse_attn
if HAS_SPARGE: from .ops import sage_with_custom_mask
from .core import AttentionBackend
from .clustering import (
  linear_transforms_diagonalized_clustering, 
  attention_scores_diagonalized_clustering
)

class ClusterAttentionBackend(AttentionBackend):
    def __init__(
        self, device="cuda", c=0, 

        # selection
        k=16, k_frac=None,  # k and k_frac are used in top-k. if k_frac is not None it overrides k.
        target_recall=0.9, min_k=16,  # min_k is used in adaptive
        adaptive_k=False,
        # other
        use_flashinfer=False, 
        compensate=0,
        hooks=[],

        # clustering params (should make a config class)
        cluster_size=(128, 64), intermediate_sizes=(None, None), 
        transform=True, 
        ridge_epsilon=None, eigenvalue_floor=None,  # these are not used
        nonlinear=False, 
    ):  
        if ridge_epsilon is None:
            ridge_epsilon = torch.tensor(0.0, device=device, dtype=torch.float32)        
        if eigenvalue_floor is None:
            eigenvalue_floor = torch.tensor(0.0, device=device, dtype=torch.float32)

        assert c == 0, "c must be 0 for this backend"
        super().__init__(c)
        self.k = k  # used in top-k
        if k_frac:
            assert (k_frac <= 1) and (k_frac > 0), "need k_frac between 0 (exclusive) and 1 (inclusive)"
        self.k_frac = k_frac
        if isinstance(cluster_size, int): cluster_size = (cluster_size, cluster_size)
        self.cluster_size = cluster_size
        self.adaptive_k = adaptive_k
        self.target_recall = target_recall
        self.min_k = min_k  # used in adaptive
        self.transform = transform
        self.ridge_epsilon = ridge_epsilon
        self.eigenvalue_floor = eigenvalue_floor
        self.nonlinear = nonlinear
        self.compensate = compensate
        self._hooks = hooks
        self.intermediate_sizes = intermediate_sizes

        architecture = torch.cuda.get_device_capability()
        if architecture[0] == 9:  # sm90, H100 and H200
            sage_sizes = (128, 64)  # (cs_k, cs_q)
        else:  # sm80, but rest of the code is not built for this
            sage_sizes = (64, 128)
        if not use_flashinfer:
            if not HAS_SPARGE:
                raise ImportError("Sparge is not installed, set use_flashinfer=True to use FlashInfer")
            if not (cluster_size == sage_sizes):
                raise ValueError("Sparge requires specific cluster sizes")
            self.block_sparse_attention = sage_with_custom_mask
        else:
            if not HAS_FLASHINFER:
                raise ValueError("FlashInfer is not installed")
            self.block_sparse_attention = flashinfer_block_sparse_attn
        
    def forward(self, q, k, v, layer_idx, return_lse=False):
        B, H, N, D = q.shape
        device = q.device

        cs_k, cs_q = self.cluster_size
        # counts clusters actually, n_qc and n_kc.
        n_blocks_q, n_blocks_k = (N +cs_q -1) //cs_q, (N +cs_k -1) //cs_k

        # 1a. Cluster unpadded keys and queries directly
        with self._record("clustering"):
            k_flat, q_flat = k.reshape(B * H, N, D), q.reshape(B * H, N, D)  # just contiguous views, no copies
            v_flat = v.reshape(B * H, N, D)
            if not self.nonlinear:
                _, k_perm, _, q_perm = linear_transforms_diagonalized_clustering(
                    k_flat, q_flat, 
                    v_flat,
                    attention_backend=self, 
                    use_ranking_objective_keys=(not self.adaptive_k),
                    skip_transforms=(not self.transform),
                    ridge_epsilon=self.ridge_epsilon,
                    eigenvalue_floor=self.eigenvalue_floor,
                    key_target_size=cs_k, query_target_size=cs_q,
                    key_intermediate_size=self.intermediate_sizes[0], 
                    query_intermediate_size=self.intermediate_sizes[1], 
                )
            else:
                _, k_perm, _, q_perm = attention_scores_diagonalized_clustering(
                    k_flat, q_flat, key_target_size=cs_k, query_target_size=cs_q
                )
            # all views, no memory cost
            k_perm_small, q_perm_small = k_perm.view(B, H, N), q_perm.view(B, H, N)
            k_perm, q_perm = k_perm_small.unsqueeze(-1).expand(-1, -1, -1, D), q_perm_small.unsqueeze(-1).expand(-1, -1, -1, D)

        # 1b. Sort so cluster members are contiguously placed
        with self._record("group_by_cluster"):
            # when perm is not None, labels is not used. so labels is actually not used here (a consequence of same size clusters)
            q, q_perm = group_by_cluster(None, q, D, perm=q_perm)  # q_s
            k, k_perm = group_by_cluster(None, k, D, perm=k_perm)  # k_s
            v, _ = group_by_cluster(None, v, D, perm=k_perm)  # v_s

        # 2. Assignment
        with self._record("assignment"):
            # n_assign needs a better name
            if self.adaptive_k:
                n_assign = min(self.min_k, n_blocks_k)
                target_recall = self.target_recall
            else:
                n_assign = self.k_frac*n_blocks_k if self.k_frac else min(self.k, n_blocks_k)
                target_recall = 0
            lut, valid_block_num = assignment(
                q, k, v, cs_q, cs_k, target_recall, n_assign, self.adaptive_k,  # the fallback branches on adaptive_k
            )

        # 3a. Compute sparse attention
        if self.compensate==1: raise NotImplementedError("no mean-to-mean yet :(")
        with self._record("block_sparse_attention"):
            out = self.block_sparse_attention(q, k, v, lut, valid_block_num, (self.compensate>0 or return_lse))
            # we now produce lut right away, should move this op into the fallbacks
        if (self.compensate==0 and not return_lse):
            out, lse = out, None
        elif self.compensate>0:  
            # compensate=1 (mean-mean) behaves like compensate=2 (query-mean) right now
            out_sel, lse_sel = out
            with self._record("compensate_query_to_mean"):
                out, lse = compensate_query_to_mean(
                    q, k, v, lut, valid_block_num, cs_q, cs_k, out_sel, lse_sel
                )
        else:
            out, lse = out
              
        # 3b. Unsort attention output back
        with self._record("ungroup_back"):
            out = ungroup_back(out, q_perm)
            if lse is not None:
                lse = ungroup_back(lse, q_perm_small)

        # Do hooks
        if self._hooks:
            ctx = locals()
            for hook in self._hooks:
                hook(ctx)
            
        return out, lse
