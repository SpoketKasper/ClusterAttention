# The modified ViT Attention is adapted from timm https://github.com/huggingface/pytorch-image-models
# Licensed under Apache 2.0

import torch
import torch.nn.functional as F
import math
from timm.models.vision_transformer import Attention

from cluster_attention import ModifiedAttention
from cluster_attention.ops import combine_attention
from cluster_attention.clustering.ops import matmul  # special matmul that takes in bf16 and returns fp32

# no use_model_fallback here as timm does the same ops (we have removed those unused by DINOv2)
class ModifiedViTAttention(ModifiedAttention, Attention):
    locations = {"blocks": ["attn"]}

    def decompose(self, x, **kwargs):
        B, N, C = x.shape  # C for channel, aka hidden dimension
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        #q, k, v = q.contiguous(), k.contiguous(), v.contiguous()  # not worth it when we reassign right after

        state = {"B": B, "N": N, "C": C, "needs_lse": True}
        if not self.attention_backend.is_dense:
            # picking out cls token
            q_cls = q[:, :, :1]  # [B, H, 1, D]
            k_cls, v_cls = k[:, :, :1], v[:, :, :1]

            # attention from cls token onto content (and itself)
            state["out_cls"] = F.scaled_dot_product_attention(q_cls, k, v)

            # removing cls token
            q = q[:, :, 1:].contiguous()
            k = k[:, :, 1:].contiguous()
            v = v[:, :, 1:].contiguous()

            # attention from content tokens onto cls
            scale = q.shape[-1] ** -0.5
            #state["lse_cls_kv"] = (q * k_cls).sum(-1) * scale * math.log2(math.e)  # [B, H, N-1]
            B, H, Nm1, D = q.shape
            state["lse_cls_kv"] = matmul(
                q.reshape(B*H, Nm1, D), k_cls.reshape(B*H, 1, D), transB=True
            ).reshape(B, H, Nm1) * scale * math.log2(math.e)  # [B*H, N-1, 1] in fp32
            state["out_cls_kv"] = v_cls.expand_as(q).contiguous()  # [B, H, N-1, D]

        return q, k, v, state

    def recompose(self, out_modified, lse_modified, state):
        if "out_cls" in state:  # meaning, we are not running a DenseBackend
            out_rest, _ = combine_attention(
                out_modified, lse_modified, state["out_cls_kv"], state["lse_cls_kv"]
            )
            out = torch.cat([state["out_cls"], out_rest], dim=2)
        else:
            out = out_modified
        out = out.transpose(1, 2).reshape(state["B"], state["N"], state["C"])
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
