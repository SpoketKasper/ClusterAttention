# AI-written file

import torch

def find_best_axis(vectors, groups, sample):
    batch_size, G, size = groups.shape
    d = vectors.shape[2]
    ns = min(sample, size)

    if size > ns:
        rand_idx = torch.randint(0, size, (batch_size, G, ns), device=groups.device)
        sampled = groups.gather(2, rand_idx)
    else:
        sampled = groups[:, :, :ns]

    # sampled: [batch, G, ns] -> gather vectors
    idx = sampled.long()
    # vectors: [batch, n, d]
    vecs = vectors[:, None].expand(-1, G, -1, -1)  # [batch, G, n, d]
    x = vecs.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, d))  # [batch, G, ns, d]
    x = x.float()

    mean = x.mean(dim=2, keepdim=True)
    centered = x - mean
    var = (centered ** 2).sum(dim=2)  # [batch, G, d]
    return var.argmax(dim=-1).int()


def project_coord(vectors, groups, axis):
    batch_size, G, size = groups.shape
    idx = groups.long()  # [batch, G, size]
    # axis: [batch, G] — which coordinate to pick
    # For each element, load vectors[b, idx[b,g,m], axis[b,g]]
    flat_idx = idx.unsqueeze(-1).expand(-1, -1, -1, vectors.shape[2])
    vecs = vectors[:, None].expand(-1, G, -1, -1).gather(2, flat_idx)  # [batch, G, size, d]
    a = axis.long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, size, 1)
    proj = vecs.gather(3, a).squeeze(-1).float()  # [batch, G, size]
    return proj


def partition(proj, groups, left_size):
    batch_size, G, size = groups.shape
    right_size = size - left_size
    left = torch.empty(batch_size, G, left_size, device=groups.device, dtype=groups.dtype)
    right = torch.empty(batch_size, G, right_size, device=groups.device, dtype=groups.dtype)

    for b in range(batch_size):
        for g in range(G):
            vals = proj[b, g]  # [size]
            idx = vals.argsort()
            sorted_groups = groups[b, g][idx]
            left[b, g] = sorted_groups[:left_size]
            right[b, g] = sorted_groups[left_size:]
    return left, right


def _second_moment(X):
    X = X.flatten(1, 2) if X.dim() == 4 else X
    return torch.bmm(X.transpose(1, 2), X) / X.shape[1]

def _eigh_clamp(M, ridge_epsilon=0, eigenvalue_floor=0):  # gpu eigh
    assert M.dtype == torch.float32 or M.dtype == torch.float64
    eigvals, eigvecs = torch.linalg.eigh(M)
    mean_eigval = eigvals.mean(dim=-1, keepdim=True)
    eigvals = (1 - ridge_epsilon) * eigvals + ridge_epsilon * mean_eigval
    return eigvals.clamp(min=mean_eigval*eigenvalue_floor), eigvecs

def normalize(vectors, inplace=None):
    return torch.nn.functional.normalize(vectors, dim=-1)
