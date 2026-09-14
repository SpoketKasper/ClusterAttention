import torch
import torch.nn.functional as F
import math

from cluster_attention import AttentionBackend

##################
## Scatterbrain ##
##################

from fly_src.models.attention.sblocal_attention import SBLocalAttention

class ScatterbrainBackend(AttentionBackend):
    attn_params = [
        "dim_heads", "softmax_eps", "local_context", "nb_features", "ortho_scaling",
    ]

    def __init__(
        self,
        dim_heads=128,
        # defaults
        softmax_eps=0.0, local_context=256, nb_features=None, ortho_scaling=0,
    ):
        super().__init__()
        self.dim_heads = dim_heads
        self.softmax_eps = softmax_eps
        self.local_context = local_context
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

    def build_attn(self):
        kwargs = {}
        for param in self.attn_params:
            if getattr(self, param) is not None:
                kwargs[param] = getattr(self, param)
        self.attn = SBLocalAttention(**kwargs, softmax_temp=None, attention_dropout=0.0)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name not in self.attn_params: return  # dont rebuild on other things, or recurse-bug
        if not all(hasattr(self, p) for p in self.attn_params): return  # only build after all have been set
        self.build_attn()

    def forward(self, q, k, v, layer_idx, return_lse=False):
        dtype_in = q.dtype
        q_in = q.transpose(1, 2).to(torch.float32)
        k_in = k.transpose(1, 2).to(torch.float32)
        v_in = v.transpose(1, 2).to(torch.float32)
        out, _ = self.attn(q_in, k_in, v_in)
        # out is (B, S, H, D)
        out = out.transpose(1, 2).to(dtype_in)
        lse = None if not return_lse else self.attn._last_log_normalization
        return out, lse/math.log(2)