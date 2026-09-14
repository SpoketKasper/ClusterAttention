# Adapted from NystromAttention on https://github.com/lucidrains/nystrom-attention
# MIT license

import torch
import torch.nn.functional as F

from cluster_attention import AttentionBackend

###################
## Nyströmformer ##
###################

from nystrom_attention.nystrom_attention import moore_penrose_iter_pinv
from einops import reduce
from math import ceil

class NystromBackend(AttentionBackend):
    def __init__(
        self,
        num_landmarks=256, pinv_iterations=10,
        smooth=False  # smooth is original but requires local similarity
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.smooth = smooth

    def forward(self, q, k, v, layer_idx, return_lse=False):
        B, H, S, D = q.shape
        m = min(self.num_landmarks, S)
        scale = D ** -0.5

        if self.smooth:
            remainder = S % m
            if remainder > 0:
                padding = m - remainder
                q = F.pad(q, (0, 0, 0, padding))
                k = F.pad(k, (0, 0, 0, padding))
                v = F.pad(v, (0, 0, 0, padding))
                S_padded = S + padding
            else:
                S_padded = S

            q = q * scale

            l = ceil(S_padded / m)
            q_landmarks = reduce(q, 'b h (n l) d -> b h n d', 'sum', l=l) / l
            k_landmarks = reduce(k, 'b h (n l) d -> b h n d', 'sum', l=l) / l

            sim1 = torch.einsum('b h i d, b h j d -> b h i j', q, k_landmarks)
            sim2 = torch.einsum('b h i d, b h j d -> b h i j', q_landmarks, k_landmarks)
            sim3 = torch.einsum('b h i d, b h j d -> b h i j', q_landmarks, k)

            attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
            attn2_inv = moore_penrose_iter_pinv(attn2.to(torch.float32), self.pinv_iterations)

            out = (attn1 @ attn2_inv.to(v.dtype)) @ (attn3 @ v)

            if remainder > 0:
                out = out[:, :, :S]
        else:
            indices = torch.randperm(S, device=q.device)[:m]
            q_landmarks = q[:, :, indices]
            k_landmarks = k[:, :, indices]

            q_scaled = q * scale

            sim1 = torch.einsum('bhid, bhjd -> bhij', q_scaled, k_landmarks)
            sim2 = torch.einsum('bhid, bhjd -> bhij', q_landmarks * scale, k_landmarks)
            sim3 = torch.einsum('bhid, bhjd -> bhij', q_landmarks * scale, k)

            attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
            attn2_inv = moore_penrose_iter_pinv(attn2.float(), self.pinv_iterations).to(q.dtype)

            out = (attn1 @ attn2_inv) @ (attn3 @ v)
        return out, None