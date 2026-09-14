from ..cuda_graph_cached import CUDAGraphCached

############
## Config ##
############

# this affects latency for assignment and block_sparse_attention, which are outside the graph
use_cuda_graph = True
# maybe turn into a bool in the AttentionBackend? and use parameter 'enabled' in CUDAGraphCached calls

#############
## Imports ##
#############

from .transformed_clustering import (
  linear_transforms_diagonalized_clustering, 
  attention_scores_diagonalized_clustering
)

if use_cuda_graph:
    try:
        linear_transforms_diagonalized_clustering = CUDAGraphCached(linear_transforms_diagonalized_clustering)  
        attention_scores_diagonalized_clustering = CUDAGraphCached(attention_scores_diagonalized_clustering)  
        print("Using CUDA-graph speedups")
    except:
        print("Speedups not available")

def clear_graph_cache():  # should live with the CUDAGraphCached definition probably
    if isinstance(linear_transforms_diagonalized_clustering, CUDAGraphCached):
        linear_transforms_diagonalized_clustering.clear()
    if isinstance(attention_scores_diagonalized_clustering, CUDAGraphCached):
        attention_scores_diagonalized_clustering.clear()
