import torch
torch.backends.cuda.matmul.allow_tf32 = False  # default, but important here!
import math

from .ops import _second_moment, cov, _eigh_clamp, normalize, center
from .recursive_splitting import recursive_split

############
## Linear ##
############

@torch.no_grad()
def linear_transforms_diagonalized_clustering(
    keys, queries, values=None, verbose=False, 
    key_target_size=None, query_target_size=None, use_ranking_objective_keys=True, 
    key_intermediate_size=None, query_intermediate_size=None, 
    skip_transforms=False,  # for benchmarking
    ridge_epsilon=torch.tensor(0.0, device="cuda", dtype=torch.float32),
    eigenvalue_floor=torch.tensor(0.0, device="cuda", dtype=torch.float32),
    **split_kwargs
):  
    device = keys.device
    BH = keys.shape[0]

    key_split_kwargs = {
        **split_kwargs, 
        **({"target_size": key_target_size} if key_target_size is not None else {}),
        **({"intermediate_size": key_intermediate_size} if key_intermediate_size is not None else {})
    }
    query_split_kwargs = {
        **split_kwargs, 
        **({"target_size": query_target_size} if query_target_size is not None else {}),
        **({"intermediate_size": query_intermediate_size} if query_intermediate_size is not None else {})
    }

    mu_k = keys.mean(dim=1, dtype=torch.float32)  # fp32 accumulation
    transformed_keys = center(keys, mu_k)
    # now transformed_keys are centered
    M_Kc = _second_moment(transformed_keys).float()  # casting if not using custom contraction

    if not skip_transforms:
        M_Q = _second_moment(queries).float()  # casting if not using custom contraction

        eigvals_all, eigvecs_all = _eigh_clamp(torch.cat([M_Q, M_Kc], dim=0), ridge_epsilon, eigenvalue_floor)
        R = eigvecs_all[:BH] * eigvals_all[:BH].sqrt().unsqueeze(-2)
        R_k = (eigvecs_all[BH:] * eigvals_all[BH:].sqrt().unsqueeze(-2)).to(queries.dtype)

        projected_queries = torch.bmm(queries, R_k)
        if use_ranking_objective_keys:
            projected_queries = normalize(projected_queries)
        mu_qp = projected_queries.mean(dim=1, dtype=torch.float32)  # fp32 accumulation
        projected_queries = center(projected_queries, mu_qp, out=projected_queries)
        M_norm = _second_moment(projected_queries)

        cov_keys = R.transpose(-1, -2) @ M_Kc @ R  # fp32 matmul
        _, eigvecs2_all = _eigh_clamp(torch.cat([cov_keys, M_norm], dim=0))
        transformed_keys = torch.bmm(transformed_keys, (R @ eigvecs2_all[:BH]).to(keys.dtype))
        transformed_queries = torch.bmm(projected_queries, eigvecs2_all[BH:].to(queries.dtype))
    else:
        transformed_queries = queries.clone()
        if use_ranking_objective_keys:
            transformed_queries = normalize(transformed_queries)
        mu_q = transformed_queries.mean(dim=1, dtype=torch.float32)  # fp32 accumulation
        transformed_queries = center(transformed_queries, mu_q, out=transformed_queries)
        # now transformed_queries are centered
        M_Qc = _second_moment(transformed_queries)

        _, eigvecs2_all = _eigh_clamp(torch.cat([M_Kc, M_Qc], dim=0))
        transformed_keys = torch.bmm(transformed_keys, eigvecs2_all[:BH].to(keys.dtype))
        # now transformed_keys are in the pc-base
        transformed_queries = torch.bmm(transformed_queries, eigvecs2_all[BH:].to(queries.dtype))
        # now transformed_queries are in the pc-base

    # interesting to check remaining dimensions here
    key_clusters, perm_k = recursive_split(transformed_keys, verbose=verbose, **key_split_kwargs)[:2]
    query_clusters, perm_q = recursive_split(transformed_queries, verbose=verbose, **query_split_kwargs)[:2]
    return (key_clusters, perm_k, query_clusters, perm_q)

###############
## Nonlinear ##
###############
# this one is not so carefully worked through, consider it experimental

def kmeans(x, k, chunk_size=4096, min_steps=10):
    H, N, D = x.shape
    idx = torch.randperm(N, device=x.device)[:k]
    centers = x[:, idx].clone()
    n_steps = max(min_steps, (N + chunk_size - 1) // chunk_size)
    for step in range(n_steps):
        start = (step * chunk_size) % N
        end = min(start + chunk_size, N)
        x_chunk = x[:, start:end]
        n_chunk = x_chunk.shape[1]
        x_sq = (x_chunk * x_chunk).sum(-1, keepdim=True)
        c_sq = (centers * centers).sum(-1, keepdim=True)
        dists = x_sq - 2 * torch.bmm(x_chunk, centers.mT) + c_sq.mT
        assigns = dists.argmin(dim=-1)
        new_centers = torch.zeros(H, k, D, device=x.device, dtype=torch.float32)
        counts = torch.zeros(H, k, device=x.device, dtype=torch.float32)
        new_centers.scatter_add_(1, assigns.unsqueeze(-1).expand_as(x_chunk), x_chunk.float())
        counts.scatter_add_(1, assigns, torch.ones(H, n_chunk, device=x.device, dtype=torch.float32))
        mask = counts > 0
        centers = torch.where(mask.unsqueeze(-1), (new_centers / counts.unsqueeze(-1).clamp(min=1)).to(x.dtype), centers)
    return centers
def attention_scores_diagonalized_clustering(
        keys, 
        queries, 
        m=None, 
        tau=5.0,
        typical_token_count=1024,
        verbose=False, 
        key_target_size=None,
        query_target_size=None,
        **split_kwargs
    ):
    key_split_kwargs = {**split_kwargs, **({"target_size": key_target_size} if key_target_size is not None else {})}
    query_split_kwargs = {**split_kwargs, **({"target_size": query_target_size} if query_target_size is not None else {})}

    H, N, D = keys.shape
    if m is None:
        m = D

    ref_queries = kmeans(queries, m)
    ref_keys = kmeans(keys, m)

    scale_keys = (math.log(N) / math.log(typical_token_count)) / (tau * math.sqrt(D))
    scale_queries = (math.log(m) / math.log(typical_token_count)) / (tau * math.sqrt(D))

    keys = torch.softmax(torch.bmm(keys, ref_queries.mT).float() * scale_keys, dim=-2).to(keys.dtype)  # overwriting for memory
    queries = torch.softmax(torch.bmm(queries, ref_keys.mT).float() * scale_queries, dim=-1).to(keys.dtype)  # overwriting for memory
    # "keys" and "queries" now hold softmax activations for the keys and the queries onto the references

    keys = keys - keys.mean(dim=1, dtype=torch.float32).unsqueeze(1).to(keys.dtype)  # overwriting again
    queries = queries - queries.mean(dim=1, dtype=torch.float32).unsqueeze(1).to(keys.dtype)  # overwriting again
    # "keys" and "queries" now hold centered softmax activations

    M_Kc = _second_moment(keys)
    M_Qc = _second_moment(queries)
    _, eigvecs2_all = _eigh_clamp(torch.cat([M_Kc, M_Qc], dim=0))
    keys = torch.bmm(keys, eigvecs2_all[:H].to(keys.dtype))
    queries = torch.bmm(queries, eigvecs2_all[H:].to(keys.dtype))
    # "keys" and "queries" now hold softmax activations projected onto their pcs

    key_clusters, perm_k = recursive_split(keys, verbose=verbose, **key_split_kwargs)[:2]
    query_clusters, perm_q = recursive_split(queries, verbose=verbose, **query_split_kwargs)[:2]
    return (key_clusters, perm_k, query_clusters, perm_q)
