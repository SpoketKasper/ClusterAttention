# The modified WanSelfAttention is adapted from Wan 2.1 https://github.com/Wan-Video/Wan2.1
# Licensed under Apache 2.0

import torch
from wan.modules.model import WanSelfAttention, rope_apply

from cluster_attention import ModifiedAttention

class ModifiedWanSelfAttention(ModifiedAttention, WanSelfAttention):
    locations = {"blocks": ["self_attn"]}
    calls_per_step = 2  # property of the pinned implementation of Wan 2.1
    layer_warmup = 1  # same as in the SVOO paper

    #diffusion_step_warmup = 10  # same as in the SVOO paper (set it from the model loader instead)
    #call_count = 0  # class attribute, an instance attribute is created on first increment

    def forward(self, x, seq_lens, grid_sizes, freqs):
        diffusion_step = self.call_count // self.calls_per_step
        self.call_count += 1  # can safely increment after finding diffusion_step
        if (
            self.attention_backend.use_model_fallback 
            or self.layer_idx<self.layer_warmup
            or diffusion_step < self.diffusion_step_warmup
        ):  
            # for fairness towards the model, but could also be interesting with a DenseBackend that also 
            # does the transposes, as a attention_backend could just as well run [B, S, H/n, D]
            return WanSelfAttention.forward(self, x, seq_lens, grid_sizes, freqs)

        # hacking in to communicate with the attention_backend
        self.attention_backend.diffusion_step = diffusion_step
        # not using seq_lens as that is only used when B>1
        q, k, v, state = self.decompose(x, grid_sizes, freqs)
        with torch.amp.autocast('cuda', enabled=False):  # just making sure Wan doesnt mess with our dtypes
            out, lse = self.attention_backend(q, k, v, self.layer_idx, return_lse=False)
        return self.recompose(out, lse, state)
    
    def decompose(self, x, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v
        
        q, k, v = qkv_fn(x)

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        # below is specific for attention_backends, but WanSelfAttention also casts to bfloat16
        input_dtype = q.dtype
        q = q.permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()  # [B, H, S, D]
        k = k.permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()
        v = v.permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()

        state = {"B": b, "S": s, "C": n * d, "input_dtype": input_dtype}
        return q, k, v, state

    def recompose(self, out, lse, state):
        out = out.transpose(1, 2).reshape(state["B"], state["S"], state["C"])
        out = out.to(state["input_dtype"])
        return self.o(out)
