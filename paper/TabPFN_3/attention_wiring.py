# The modified ICLAttention is adapted from TabPFN-3 
# https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/architectures/tabpfn_v3.py
# Licensed under Prior Labs License, see LICENSE.txt

import torch
import torch.nn.functional as F
from tabpfn.architectures.tabpfn_v3 import KVCacheEntry, ICLAttention

from cluster_attention import ModifiedAttention

class ModifiedICLAttention(ModifiedAttention, ICLAttention):
    locations = {"icl_blocks": ["icl_attention"]}

    def forward(self, x_BRE, single_eval_pos, *, cached_kv=None, return_kv=False):
        if (cached_kv is not None) or self.attention_backend.use_model_fallback:
            return ICLAttention.forward(self, x_BRE, single_eval_pos, cached_kv=cached_kv, return_kv=return_kv)
        with torch.autocast("cuda", enabled=False):  # so autocast doesnt interfere with deliberate precision choices
            q, k, v, state = self.decompose(x_BRE, single_eval_pos, return_kv=return_kv)
            out, lse = self.attention_backend(q, k, v, self.layer_idx, return_lse=state.get("needs_lse", False))
            return self.recompose(out, lse, state)

    def decompose(self, x_BRE, single_eval_pos, *, return_kv=False):
        B, R, _ = x_BRE.shape
        N = R if single_eval_pos is None else single_eval_pos
        x_train = x_BRE[:, :N]

        # note! here runs in float32 (slow) as it is inside disabled autocast. 
        # likely also applied to softmax_scaling_layer and out_projection.
        # fixing should give larger speedup than what is in the preprint.
        q = self.q_projection(x_BRE).view(B, R, self.num_heads, self.head_dim)
        k = self.k_projection(x_train).view(B, N, self.num_kv_heads, self.head_dim)
        v = self.v_projection(x_train).view(B, N, self.num_kv_heads, self.head_dim)

        sl = getattr(self, 'softmax_scaling_layer', None)
        if sl is not None:
            #if self.attention_backend.c != 1:
            #    cluster_size = getattr(self.attention_backend, 'cluster_size', 1)
            #    cs = cluster_size[1] if isinstance(cluster_size, tuple) else cluster_size
            #    actual_k = min(self.attention_backend.k * cs, N)
            #    q = sl(q, actual_k)
            #else:
            #    q = sl(q, N)
            q = sl(q, N)  # using full N for softmax_scaling_layer

        # autocast will go to float16, so slightly different from the ModifiedAttention calls
        input_dtype = q.dtype
        q = q.permute(0, 2, 1, 3).to(torch.bfloat16)
        k = k.permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()
        v = v.permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()

        q_train = q[:, :, :N, :]
        q_test = q[:, :, N:, :]

        state = {
            "q_test": q_test, "k": k, "v": v,
            "B": B, "R": R, "N": N, "return_kv": return_kv,
            "input_dtype": input_dtype,
        }
        return q_train.contiguous(), k, v, state

    def recompose(self, out_train, lse, state):
        q_test, k, v = state["q_test"], state["k"], state["v"]
        B, R = state["B"], state["R"]

        if self.num_kv_heads_test is not None:
            nh = self.num_kv_heads_test
            k_test = k[:, :nh]
            v_test = v[:, :nh]
        else:
            k_test = k
            v_test = v

        # this should never run in our testing, as our test->train calls have a cached_kv 
        # and fall back to ICLAttention.
        out_test = F.scaled_dot_product_attention(q_test, k_test, v_test)

        out = torch.cat([out_train, out_test], dim=2)
        out = out.permute(0, 2, 1, 3).reshape(B, R, self.head_dim * self.num_heads)
        out = out.to(state["input_dtype"])  # another cast
        result = self.out_projection(out)

        kv_entry = None
        if state["return_kv"]:
            k_cache = k.permute(0, 2, 1, 3)
            v_cache = v.permute(0, 2, 1, 3)
            if self.num_kv_heads_test is not None:
                nh = self.num_kv_heads_test
                k_cache = k_cache[:, :, :nh].contiguous()
                v_cache = v_cache[:, :, :nh].contiguous()
            kv_entry = KVCacheEntry(key=k_cache.detach(), value=v_cache.detach())

        return result, kv_entry
