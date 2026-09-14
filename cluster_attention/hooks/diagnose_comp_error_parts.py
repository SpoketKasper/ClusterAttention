# removes jensen and cov separately to see how output error changes, ablation on terms. uses the exact expression from the appendix.
# in testing now, cov seems to dominate (a few to 10x larger). the preprint currently claims the opposite.

import torch
from ..ops import lut_to_idx

def error_ablation_hook(ctx):
    if ctx['self'].compensate>0:
        raise NotImplementedError("Did not write code from indices to bools although that would be simple")
    else:
        B, H, N, D = ctx["q"].shape
        cs_q = ctx["cs_q"]
        cs_k = ctx["cs_k"] 
        n_kc = (N + cs_k - 1) // cs_k
        selected = lut_to_idx(ctx['lut'], ctx['valid_block_num'], B, H, (N + cs_q - 1) // cs_q, n_kc, ctx["q"].device)

    error_ablation(
        ctx["q"], 
        ctx["k"],
        ctx["v"],
        selected,
        ctx["cs_q"],
        ctx["cs_k"], 
        ctx['layer_idx'],
    )

def error_ablation(q, k, v, selected, cs_q, cs_k, layer_idx, qs_per_head=2):
    B, H, N, D = q.shape
    scale = D ** -0.5
    n_kc = (N + cs_k - 1) // cs_k

    # Cluster centroids
    n_full = N//cs_k  # number of full clusters
    rem = N%cs_k
    kc = k[:, :, :n_full*cs_k].view(B, H, n_full, cs_k, D).mean(3)
    vc = v[:, :, :n_full*cs_k].view(B, H, n_full, cs_k, D).mean(3)
    if rem:
        kc = torch.cat([kc, k[:, :, n_full*cs_k:].mean(2, keepdim=True)], 2)
        vc = torch.cat([vc, v[:, :, n_full*cs_k:].mean(2, keepdim=True)], 2)
    counts = torch.full((n_kc,), cs_k, device=q.device, dtype=torch.float32)
    if rem: counts[-1] = rem

    #errs = {name: [] for name in ['sparse', 'comp', 'comp_no_jensen', 'comp_no_cov']}

    for h in range(H):
        print(f"Head: {h}")
        # convert to float (as accumulation happens in float inside the kernels)
        q_h, k_h, v_h, kc_h, vc_h = map(lambda x: x[0, h].float(), (q, k, v, kc, vc))

        # sampling some queries
        ids = torch.randint(0, N, (qs_per_head,), device=q.device)
        qc_ids = ids // cs_q
        qs = q_h[ids]

        # dense computation for the sampled queries
        logits = qs @ k_h.T * scale
        lse = logits.logsumexp(-1, keepdim=True)
        w = torch.exp(logits - lse)
        o_dense = w @ v_h

        # getting the selected key-clusters for all samples queries
        print(selected.dtype)
        sel_cluster = selected[0, h, qc_ids].bool()
        sel_mask = sel_cluster.repeat_interleave(cs_k, dim=-1)[:, :N]  # repeat to key-level

        for i in range(qs_per_head):
            wi = w[i]
            unsel_cluster = ~sel_cluster[i]
            if not unsel_cluster.any(): continue  # skip fully assigned

            sel_mask_i = sel_mask[i]

            # sparse output
            w_S = wi[sel_mask_i].sum()
            o_S = (wi[sel_mask_i, None]*v_h[sel_mask_i]).sum(0) / w_S

            # query->centroid comp on unselected clusters
            log_wc = qs[i] @ kc_h.T * scale
            centroid_logits = log_wc + counts.log()
            centroid_logits[~unsel_cluster] = float('-inf')  # masking out selected
            lse_unsel = centroid_logits.logsumexp(-1)
            o_unsel = torch.softmax(centroid_logits, dim=-1) @ vc_h

            # lse merge (numerically stable) giving ohat_S=o_comp
            lse_sel = lse[i, 0] + torch.log(w_S)
            max_lse = torch.max(lse_sel, lse_unsel)
            w_sel_n = torch.exp(lse_sel-max_lse)
            w_uns_n = torch.exp(lse_unsel-max_lse)
            o_comp = (w_sel_n*o_S + w_uns_n*o_unsel) / (w_sel_n + w_uns_n)

            # wbarc
            wi_c = wi[:n_full*cs_k].view(n_full, cs_k)  # cutting to even clusters
            wbarc = wi_c.sum(1) / counts[:n_full]
            if rem: wbarc = torch.cat([wbarc, wi[n_full*cs_k:].sum(dim=0, keepdim=True)/counts[-1]])

            # Jensen term: |c|(w_bar_c-centroid_weight_c)(vc-o_comp)
            wc = torch.exp(log_wc - lse[i, 0])
            delta_c = wbarc-wc
            jensen_term = (
                counts[unsel_cluster, None]*delta_c[unsel_cluster, None] * (vc_h[unsel_cluster]-o_comp)
            ).sum(0)

            # covariance term: sum(wv)-sum(w)v_bar identity
            vi_c = v_h[:n_full*cs_k].view(n_full, cs_k, D)  # cutting to even clusters
            wv_c_sum = (wi_c.unsqueeze(-1) * vi_c).sum(1)
            if rem: wv_c_sum = torch.cat([wv_c_sum, (wi[n_full*cs_k:, None] * v_h[n_full*cs_k:]).sum(dim=0, keepdim=True)])
            covariance_term = (
                wv_c_sum[unsel_cluster] - counts[unsel_cluster, None]*wbarc[unsel_cluster, None]*vc_h[unsel_cluster]
            ).sum(0)

            # store errors
            #errs['sparse'].append((o_dense[i]-o_S).norm().item())
            #errs['comp'].append((o_dense[i]-o_comp).norm().item())
            #errs['comp_no_jensen'].append((o_dense[i]-o_comp-jensen_term).norm().item())
            #errs['comp_no_cov'].append((o_dense[i]-o_comp-covariance_term).norm().item())
            print(f"  Query: {ids[i].item()}")
            print("  sparse:", (o_dense[i]-o_S).norm().item())
            print("  compensated:", (o_dense[i]-o_comp).norm().item())
            print("  comp minus Jensen:", (o_dense[i]-o_comp-jensen_term).norm().item())
            print("  comp minus covariance:", (o_dense[i]-o_comp-covariance_term).norm().item())
            print("  comp minus both:", (o_dense[i]-o_comp-jensen_term-covariance_term).norm().item())

            # precomputed cov using query centroid
            qc_vec = q_h[qc_ids[i]*cs_q : min((qc_ids[i]+1)*cs_q, N)].mean(0)
            cov_precomp = torch.zeros(D, device=q.device)
            for c in range(n_kc):
                if not unsel_cluster[c]:
                    continue
                if c < n_full:
                    sl = slice(c*cs_k, (c+1)*cs_k)
                else:
                    sl = slice(n_full*cs_k, N)
                wj = torch.exp(qc_vec @ k_h[sl].T * scale)
                cov_precomp += (wj[:, None] * v_h[sl]).sum(0) - wj.sum() * vc_h[c]
            print(f"  comp minus cov_precomp: {(o_dense[i]-o_comp-cov_precomp).norm().item():.8f}")
            
    #print(f"layer {layer_idx}:")
    #for name in errs:
    #    print(f"  {name}: {sum(errs[name]) / len(errs[name]):.6f}")