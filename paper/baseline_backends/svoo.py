# Code adapted from SVOO https://github.com/Mutual-Luo/SVOO
# Licensed under Apache 2.0

import torch

from cluster_attention import AttentionBackend

from svoo.co_clustering import *
from svoo.co_clustering import (
    _require_flashinfer_sparse,
    _env_int,
)
from svoo.kernels.triton.permute import (
    permute_tensor_by_labels_triton,
    apply_inverse_permutation_triton,
)

# copied from https://github.com/Mutual-Luo/SVOO/blob/main/svoo/co_clustering.py
# to add return_lse parameter
def dynamic_block_sparse_fwd_flashinfer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask_map: torch.Tensor,
    block_row_sz: torch.Tensor,
    block_col_sz: torch.Tensor,
    is_cpu: bool = True,
    return_lse = False
):
    """
    Launcher for the Flashinfer dynamic block sparse attention kernel.

    Args:
        q (torch.Tensor): Query tensor, shape [B, H, S, D].
        k (torch.Tensor): Key tensor, shape [B, H, S, D].
        v (torch.Tensor): Value tensor, shape [B, H, S, D].
        block_mask_map (torch.Tensor): Boolean mask, shape [B, H, qc_num, kc_num]. Currently must on CPU.
        block_row_sz (torch.Tensor): Query block sizes, shape [B, H, qc_num]. Currently must on CPU.
        block_col_sz (torch.Tensor): Key block sizes, shape [B, H, kc_num]. Currently must on CPU.
        is_cpu (bool): Whether to run on CPU. Flashinfer default is to run on CPU. We switch to GPU for faster planning. Default is True.
    """
    flashinfer_sparse = _require_flashinfer_sparse()
    # Input shape check
    B, H, S, D = q.shape
    qc_num = block_row_sz.shape[-1]
    kc_num = block_col_sz.shape[-1]
    assert block_mask_map.shape == (B, H, qc_num, kc_num)

    assert (
        all(t.device == torch.device("cpu") for t in [block_mask_map, block_row_sz, block_col_sz]) if is_cpu else True
    )

    # Check if block_col_sz and block_row_sz are the same for each head
    assert torch.all(block_col_sz.sum(dim=2) == block_col_sz.sum(dim=2)[0, 0])
    assert torch.all(block_row_sz.sum(dim=2) == block_row_sz.sum(dim=2)[0, 0])


    workspace_bytes = _env_int("SVOO_FLASHINFER_SPARSE_WORKSPACE_BYTES", 128 * 1024 * 1024)
    f_buffer = torch.empty((workspace_bytes,), dtype=torch.uint8, device=q.device)
    flashinfer_backend = os.environ.get("SVOO_FLASHINFER_SPARSE_BACKEND", "auto").lower()
    if flashinfer_backend not in ("auto", "fa2", "fa3"):
        raise ValueError(
            "SVOO_FLASHINFER_SPARSE_BACKEND must be one of: auto, fa2, fa3; "
            f"got {flashinfer_backend!r}"
        )
    wrapper = make_variable_block_sparse_attention_wrapper(
        flashinfer_sparse, f_buffer, backend=flashinfer_backend
    )
    int_workspace_bytes = _env_int("SVOO_FLASHINFER_SPARSE_INT_WORKSPACE_BYTES", 0)
    if int_workspace_bytes > 0:
        i_buffer = torch.empty((int_workspace_bytes,), dtype=torch.uint8, device=q.device)
        wrapper.reset_workspace_buffer(
            float_workspace_buffer=f_buffer,
            int_workspace_buffer=i_buffer,
        )

    # Reshape inputs to (B * H, ...)
    q = q.reshape(B * H, S, D)
    k = k.reshape(B * H, S, D)
    v = v.reshape(B * H, S, D)
    block_mask_map = block_mask_map.reshape(B * H, qc_num, kc_num)
    block_row_sz = block_row_sz.reshape(B * H, qc_num)
    block_col_sz = block_col_sz.reshape(B * H, kc_num)

    wrapper.plan(
        block_mask_map=block_mask_map,
        block_row_sz=block_row_sz,
        block_col_sz=block_col_sz,
        num_qo_heads=B * H,
        num_kv_heads=B * H,
        head_dim=D,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
    )

    out = wrapper.run(q, k, v, return_lse=return_lse)  # [num_qo_heads, qo_len, head_dim]
    if return_lse:
        o, lse = out
    else:
        o, lse = out, None
    del wrapper, f_buffer
    if int_workspace_bytes > 0:
        del i_buffer
    o = o.reshape(B, H, S, D)
    return o, lse

# modified from WanAttn_SAPAttn_Processor
# the source function does not distinguish between the two cfg passes, 
# either both cluster or both reuse clusters
# it also does not use centroids_inits in wan, but maybe in some other model
class SVOOBackend(AttentionBackend):
    def __init__(self, n_layers, 

        # defaults for all Wan versions in the paper
        num_q_centroids=256, num_k_centroids=1024,
        top_p=0.9, min_kc_ratio=0.1,
        kmeans_iters_init=2,

        # specific to video generation, defaults for Wan 2.1 14B
        reset_every=True,  # False for video generation
        # below have not been wired in, as they require state management
        kmeans_iters_step=2, 
        start_reuse_step=11,  # when skipping for 10, it starts at 11 anyway
        reuse_interval=20 
    ):
        super().__init__()
        self.num_q_centroids = num_q_centroids
        self.num_k_centroids = num_k_centroids
        self.top_p = top_p
        self.min_kc_ratio = min_kc_ratio
        self.kmeans_iters_init = kmeans_iters_init

        self.reset_every = reset_every
        self.kmeans_iters_step = kmeans_iters_step
        self.start_reuse_step = start_reuse_step
        self.reuse_interval = reuse_interval

        self.centroids_init = [False] * n_layers
        self.cache = [None] * n_layers

    def reset(self):
        self.centroids_init = [False] * len(self.centroids_init)
        self.cache = [None] * len(self.centroids_init)

    def should_recompute(self, layer_idx):
        if (
            self.reset_every  # triggers on not-Wan so guards
            or self.diffusion_step+1<self.start_reuse_step
            or self.cache[layer_idx] is None
        ):
            return True
        else:
            return (self.diffusion_step+1-self.start_reuse_step) % self.reuse_interval == 0
    
    def forward(self, q, k, v, layer_idx, return_lse=True):
        B, H, S, D = q.shape

        if self.should_recompute(layer_idx):
            q_flat = q.reshape(B * H, S, D)
            k_flat = k.reshape(B * H, S, D)

            iters = self.kmeans_iters_step if self.centroids_init[layer_idx] else self.kmeans_iters_init
            self.centroids_init[layer_idx] = True

            qlabels, qcentroids, qcluster_sizes, klabels, kcentroids, kcluster_sizes = co_cluster_tokens(
                q_flat, k_flat, self.num_q_centroids, self.num_k_centroids, max_iters=iters
            )

            q_cluster_sizes = qcluster_sizes.view(B, H, self.num_q_centroids)
            k_cluster_sizes = kcluster_sizes.view(B, H, self.num_k_centroids)

            dynamic_map = identify_dynamic_map(
                qcentroids.view(B, H, self.num_q_centroids, D),
                kcentroids.view(B, H, self.num_k_centroids, D),
                q_cluster_sizes,
                k_cluster_sizes,
                self.top_p,
                self.min_kc_ratio,
            )

            q_perm, q_sorted_indices = permute_tensor_by_labels_triton(q, qlabels, dim=2)
            k_perm, k_sorted_indices = permute_tensor_by_labels_triton(k, klabels, dim=2)
            
            self.cache[layer_idx] = {
                'qlabels': qlabels,
                'klabels': klabels,
                'q_sorted_indices': q_sorted_indices,
                'k_sorted_indices': k_sorted_indices,
                'dynamic_map': dynamic_map,
                'q_cluster_sizes': q_cluster_sizes,
                'k_cluster_sizes': k_cluster_sizes,
            }
        else:
            cached = self.cache[layer_idx]
            qlabels = cached['qlabels']
            klabels = cached['klabels']
            q_sorted_indices = cached['q_sorted_indices']
            k_sorted_indices = cached['k_sorted_indices']
            dynamic_map = cached['dynamic_map']
            q_cluster_sizes = cached['q_cluster_sizes']
            k_cluster_sizes = cached['k_cluster_sizes']
            
            q_perm, _ = permute_tensor_by_labels_triton(q, qlabels, dim=2, sorted_indices=q_sorted_indices)
            k_perm, _ = permute_tensor_by_labels_triton(k, klabels, dim=2, sorted_indices=k_sorted_indices)

        v_perm, _ = permute_tensor_by_labels_triton(v, klabels, dim=2, sorted_indices=k_sorted_indices)

        #out_perm = dynamic_block_sparse_fwd_triton(
        #    q_perm, k_perm, v_perm, dynamic_map, q_cluster_sizes, k_cluster_sizes
        #)
        out_perm, lse_perm = dynamic_block_sparse_fwd_flashinfer(
            q_perm, k_perm, v_perm, dynamic_map, q_cluster_sizes, k_cluster_sizes, is_cpu=False,
            return_lse=return_lse
        )

        out = apply_inverse_permutation_triton(out_perm, q_sorted_indices, dim=2)
        if lse_perm is not None:
            lse_perm = lse_perm.view(B, H, S)
            idx = q_sorted_indices.view(B, H, S)
            lse = torch.empty_like(lse_perm)
            lse.scatter_(2, idx, lse_perm)
        else:
            lse = None
        if self.reset_every:
            self.reset()
        return out, lse