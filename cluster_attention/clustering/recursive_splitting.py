import torch
import numpy as np
from functools import lru_cache

from .ops import find_split_direction, project, partition

@lru_cache(maxsize=None)
def plan_splits_device(n, target_size, device, intermediate_size=None):
    levels = []
    for offs, szs, lefts in plan_splits(n, target_size, intermediate_size):
        max_size = int(szs.max())
        levels.append((
            max_size,
            *(torch.from_numpy(x).to(device) for x in (offs, szs, lefts)),
        ))
    return levels
def plan_splits(n, target_size, intermediate_size=None):
    if intermediate_size is not None:
        return plan_splits_staged(n, target_size, intermediate_size)
    t = target_size
    szs = np.array([n], dtype=np.int64)
    levels = []
    while szs.max() > t:
        lefts = np.where(szs > t, (szs + t) // (2 * t) * t, szs)
        offs = np.concatenate(([0], np.cumsum(szs)[:-1]))
        levels.append((offs, szs, lefts))
        szs = np.stack([lefts, szs - lefts], 1).ravel()
    return levels

# for example useful for splitting keys and queries together, 
# or having a clean way to combine clusters into larger same-sized
def plan_splits_staged(n, t_fine, t_coarse):
    szs = np.array([n], dtype=np.int32)
    levels = []
    for t in (t_coarse, t_fine):
        while szs.max() > t:
            lefts = np.where(szs>t, (szs+t)//(2*t) * t, szs)
            offs = np.concatenate(([0], np.cumsum(szs)[:-1])).astype(np.int32)
            levels.append((offs, szs, lefts))
            szs = np.stack([lefts, szs-lefts], 1).ravel()
    return levels

# general for both diagonalized and iterative pc1 based on what 
# find_split_direction, project, and partition are
@torch.no_grad()
def recursive_split(vectors, target_size=64, intermediate_size=None, sample=128, verbose=False, attention_backend=None, allow_sort=False):
    batch_size, n, d = vectors.shape
    device = vectors.device
    levels = plan_splits_device(n, target_size, device, intermediate_size)
    perm = torch.arange(n, device=device, dtype=torch.int32).repeat(batch_size, 1)
    buf = torch.empty_like(perm)  # our double (writing) buffer!
    #proj = torch.empty(batch_size, n, device=device, dtype=torch.float32)  
    # might want float32 if not using diagonalized 
    proj = torch.empty(batch_size, n, device=device, dtype=torch.bfloat16)

    for max_size, offs, szs, lefts in levels:
        direction = find_split_direction(vectors, perm, offs, szs, max_size, sample, target_size)
        project(vectors, perm, proj, offs, szs, direction, max_size, target_size)
        partition(proj, perm, buf, offs, szs, lefts, max_size, target_size, allow_sort=allow_sort)
        perm, buf = buf, perm
    
    return None, perm, None
