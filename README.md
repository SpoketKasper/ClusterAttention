# ClusterAttention: A training-free routing method for sparse bidirectional attention

Kasper Nordenram* Amelie Dittmann*

*Independent researcher

July 18, 2026

## Abstract

This paper introduces ClusterAttention, a training-free wrapper for bidirectional
attention layers. Through recursive splitting in transformed spaces informed by the
queries for the keys and vice-versa, it separately groups the keys and queries of each
attention-head into localized clusters of a fixed size. Each query-cluster is then as-
signed a set of key-clusters based on cluster-level scoring, and attention is performed
between the tokens in the clusters. The clustering is done in sizes that are powers
of 2, allowing for optimal tiling on a GPU. ClusterAttention operates on any set of
key-value-pairs and queries, without requiring the input to have structure, such as
a meaningful ordering. It targets large token-counts and models with meaningfully
sparse attention patterns, where the overhead of accurate clustering is outweighed
by the savings from highly sparse attention.

Preliminary evaluation has been carried out on the vision transformer DINOv2
with SpargeAttn as baseline [1]. We evaluate the Pareto front of latency on a
Nvidia H100 GPU versus distortion of image embeddings compared to dense at-
tention. We observe that while ClusterAttention does have meaningful routing
overhead, the impact of this diminishes as the number of tokens grows. At 143,641
tokens, the more accurate routing of ClusterAttention makes it Pareto-better than
SpargeAttn for much of the front, and specifically, there is indication that it reaches
low representation-distortions at lower latency than SpargeAttn. Since ClusterAt-
tention targets high token-count use cases, this is a promising indication, although
more extensive testing is required before drawing conclusions.¹ ²

---

¹ This is a preprint, and method development has not concluded. More extensive evaluation is planned. This report will be updated as work progresses, and the code will become available on https://github.com/SpoketKasper/ClusterAttention.

² Changes from the June 16 version: The adaptive method and its derivation were added. Numerics in forming the metric matrices were fixed, leading to better convergence to dense attention. This improved performance overall, giving a lower crossover with SpargeAttn, so we now only report results on images up to native resolution. A histogram-based assignment kernel was also written, enabling faster assignment.

[1] J. Zhang, C. Xiang, H. Huang, J. Wei, H. Xi, J. Zhu, and J. Chen, "SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference," 2025.