from cluster_attention import AttentionBackend

from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda, spas_sage2_attn_meansim_cuda
class SpargeAttnBackend(AttentionBackend):
    def __init__(self, k=16, k_frac=None, cluster_size=(64, None), adaptive=False, cdfthreshd=0.98, simthreshd1=-0.1):
        if isinstance(cluster_size, int):
            cluster_size = (cluster_size, cluster_size)
        super().__init__()
        self.k = k
        if k_frac:
            assert (k_frac <= 1) and (k_frac > 0), "need k_frac between 0 (exclusive) and 1 (inclusive)"
        self.k_frac = k_frac  # if defined overrides "k"
        self.cluster_size = cluster_size
        self.adaptive = adaptive
        self.cdfthreshd = cdfthreshd
        self.simthreshd1 = simthreshd1

    def forward(self, q, k, v, layer_idx, return_lse=False):
        if self.adaptive:
            #out, sparsity = spas_sage2_attn_meansim_cuda(
            out = spas_sage2_attn_meansim_cuda(
                q, k, v,
                simthreshd1=self.simthreshd1,
                cdfthreshd=self.cdfthreshd,  # 0.6 default for sparge
                is_causal=False,
                #return_sparsity=True,
                return_lse=return_lse
            )
            #print(f"sparge sparsity: {sparsity:.3f} (blocks attended: {(1-sparsity)*100:.1f}%)")
            if return_lse:
                return out[0], out[1]
            else:
                return out, None          
        if self.k_frac:
            topk = self.k_frac
        else:
            N = q.shape[2]
            key_cluster_size = self.cluster_size[0]
            n_blocks = (N + key_cluster_size - 1) // key_cluster_size
            topk = self.k / n_blocks
        out = spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=topk, is_causal=False, return_lse=return_lse)
        if return_lse:
            return out[0], out[1]
        else:
            return out, None