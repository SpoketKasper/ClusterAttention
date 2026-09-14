import torch

############
## Config ##
############

use_custom_contraction = True
use_custom_eigh = True  # necessary to use cuda graph

#############
## Imports ##
#############

try:  # try to get triton kernels
    from .normalize import normalize
    from .center import center
    from .loop import find_split_direction, project, partition
except Exception as e:
    print(e)
    raise NotImplementedError("Fallbacks are old and probably not correct")
    from .fallbacks import (
        normalize,
        find_best_axis as find_split_direction, 
        project_coord as project, 
        partition
    )

if use_custom_contraction:  # takes in bf16, accumulates in fp32, returns fp32. fast and accurate.
    try:
        from .contraction import Contraction
        matmul = Contraction()
        def _second_moment(X):
            X = X.flatten(1, 2) if X.dim() == 4 else X
            return matmul(X, X, transA=True) / X.shape[1]
        def cov(A, B):
            return matmul(A, B, transA=True) / A.shape[1]

    except Exception as e:
        print(e)
        from .fallbacks import _second_moment
        def cov(A, B):
            return A.T * B / A.shape[1]

else:
    from .fallbacks import _second_moment

if use_custom_eigh:
    try:
        # cuda graph for transforms (requires batchable eigh)
        from .batched_eigh import BatchedEigh
        #eigh_16 = BatchedEigh(16, 64)
        #eigh_32 = BatchedEigh(32, 64)
        def _eigh_clamp(M, ridge_epsilon=0, eigenvalue_floor=0):
            assert M.dtype == torch.float32
            #e = eigh_32 if M.shape[0] == 32 else eigh_16
            #eigvals, eigvecs = e(M)
            eigvals, eigvecs = BatchedEigh.eigh(M)
            mean_eigval = eigvals.mean(dim=-1, keepdim=True)
            eigvals = (1 - ridge_epsilon) * eigvals + ridge_epsilon * mean_eigval
            return eigvals.clamp(min=mean_eigval*eigenvalue_floor), eigvecs
        print("Using graphable eigh wrapper")
    except Exception as e:
        print(e)
        from .fallbacks import _eigh_clamp
else:
    from .fallbacks import _eigh_clamp
