# prints out the error-decomposition terms 
import torch

from ..ops import lut_to_idx

def diagnose_sparse_error_hook(ctx):
    if ctx['self'].compensate>0:
        raise NotImplementedError("Did not write code from indices to bools although that would be simple")
    else:
        B, H, N, D = ctx["q"].shape
        cs_q = ctx["cs_q"]
        cs_k = ctx["cs_k"] 
        n_kc = (N + cs_k - 1) // cs_k
        selected = lut_to_idx(ctx['lut'], ctx['valid_block_num'], B, H, (N + cs_q - 1) // cs_q, n_kc, ctx["q"].device)

    diagnose_sparse_error(
        ctx["q"], 
        ctx["k"],
        ctx["v"],
        selected,
        ctx["cs_q"],
        ctx["cs_k"], 
        qs_per_head=8,
    )

def diagnose_sparse_error(q, k, v, selected, cs_q, cs_k, qs_per_head=8):
    B, H, N, D = q.shape
    scale = D**-0.5  # softmax temperature

    if B>1: print("Note: only sampling from the first batch")

    for h in range(H):
        print(f"Head: {h}")
        # convert to float (as accumulation happens in float inside the kernels)
        q_h, k_h, v_h = map(lambda x: x[0, h].float(), (q, k, v))

        # sampling some queries
        ids = torch.randint(0, N, (qs_per_head,), device=q.device)  # sampling ids
        qc_ids = ids//cs_q  # {0, 63, 64} -> {0, 0, 1} (for each sampled query)
        qs = q_h[ids]  # the actual queries

        # softmax computations
        logits = qs @ k_h.T * scale
        lse = torch.logsumexp(logits, dim=-1, keepdim=True)  # denominator
        w = torch.exp(logits-lse)  # weights

        # getting the selected key-clusters for all samples queries
        sel_mask = selected[0, h, qc_ids].repeat_interleave(cs_k, dim=-1)[:, :N]  # repeat to key-level

        for i in range(qs_per_head):
            wi = w[i]
            sel_mask_i = sel_mask[i]

            # expressions from the paper
            w_S = wi[sel_mask_i].sum()
            o_S = (wi[sel_mask_i].unsqueeze(-1) * v_h[sel_mask_i]).sum(0) / w_S
            o_Sbar = (wi[~sel_mask_i, None] * v_h[~sel_mask_i]).sum(0) / (1-w_S)
            
            # actual error
            o_dense = wi @ v_h
            actual_error = o_dense-o_S

            # prints
            print(f"  Query: {ids[i].item()}")
            print(f"    1-w_S = {(1-w_S).item():.3g}")
            print(f"    |o_Sbar - o_S| = {(o_Sbar-o_S).norm().item():.3g}")
            print(f"    |(1-w_S)(o_Sbar-o_S)| = {((1-w_S)*(o_Sbar-o_S)).norm().item():.3g}")
            print(f"    |actual_error| = {actual_error.norm().item():.3g}")