# ClusterAttention: A training-free routing method for sparse bidirectional attention

Kasper Nordenram* Amelie Dittmann*

*Independent researcher

June 16, 2026

## Abstract

This paper introduces ClusterAttention, a training-free wrapper for bidirectional attention layers. Through recursive splitting in transformed spaces informed by the queries for the keys and vice-versa, it separately groups the keys and queries of each attention-head into localized clusters of a fixed size. Each query-cluster is then assigned a set of key-clusters based on cluster-level scoring, and attention is performed between the tokens in the clusters. The clustering is done in sizes that are powers of 2, allowing for optimal tiling on a GPU. ClusterAttention operates on any set of key-value-pairs and queries, without requiring the input to have any structure, such as a meaningful ordering. It targets large token-counts and models with meaningfully sparse attention patterns, where the overhead of accurate clustering is outweighed by the savings from highly sparse attention.

Preliminary evaluation has been carried out on DINOv2 with SpargeAttn as baseline [1]. We evaluate the Pareto front of latency on a Nvidia H100 GPU vs. distortion of image embeddings compared to dense attention. We observe that while ClusterAttention does have meaningful routing overhead, the impact of this diminishes as the number of tokens grows. At 360,000 tokens, the more accurate routing of ClusterAttention makes it Pareto-better than SpargeAttn for much of the front. Since ClusterAttention targets token counts on the scale of one million and more, this is a promising indication, although more extensive testing is required before drawing conclusions.¹

---

¹ This is a preprint, and method development has not concluded. More extensive evaluation is planned. This report will be updated as work progresses, and the code will become available on https://github.com/SpoketKasper/ClusterAttention.

[1] J. Zhang, C. Xiang, H. Huang, J. Wei, H. Xi, J. Zhu, and J. Chen, "SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference," 2025.
