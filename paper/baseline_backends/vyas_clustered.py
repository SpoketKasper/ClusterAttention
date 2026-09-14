import torch

from cluster_attention import AttentionBackend

##############################
## Vyas Clustered Attention ##
##############################

from fast_transformers.attention import ImprovedClusteredAttention
from fast_transformers.masking import FullMask, LengthMask

class VyasClusteredBackend(AttentionBackend):
    attn_params = ["clusters", "iterations", "bits", "hash_bias", "topk"]

    def __init__(
        self,
        attention_dropout=0.0,
        # paper defaults
        clusters=100, vyas_k=32,
        iterations=10, bits=63, 
        # True is the class default, but paper seems to indicate False
        hash_bias=True,
    ):
        super().__init__()
        self.clusters = clusters
        self.iterations = iterations
        self.bits = bits
        self.hash_bias = hash_bias
        self.topk = vyas_k  # last one, builds attn on this

    def build_attn(self):
        kwargs = {}
        for param in self.attn_params:
            if getattr(self, param) is not None:
                kwargs[param] = getattr(self, param)
        self.attn = ImprovedClusteredAttention(**kwargs, attention_dropout=0.0)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name not in self.attn_params: return  # dont rebuild on other things, or recurse-bug
        if not all(hasattr(self, p) for p in self.attn_params): return  # only build after all have been set
        self.build_attn()

    def forward(self, q, k, v, layer_idx, return_lse=False):
        B, H, S, D = q.shape
        input_dtype = q.dtype
        out = self.attn(
            q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(),
            FullMask(N=S, M=S, device=q.device),
            LengthMask(torch.full((B,), S, dtype=torch.long, device=q.device)),
            LengthMask(torch.full((B,), S, dtype=torch.long, device=q.device)),
        )
        return out.transpose(1, 2).to(input_dtype), None