import torch

def group_by_cluster(labels, x, D, perm=None):
    if perm is None:
        perm = labels.argsort(dim=-1).unsqueeze(-1).expand(-1, -1, -1, D)
    x_s = x.gather(2, perm)
    return x_s, perm

def ungroup_back(out_s, perm):
    out = torch.empty_like(out_s)
    out.scatter_(2, perm, out_s)
    return out