# ClusterAttention: A training-free speedup of bidirectional attention

Kasper Nordenram* Amelie Dittmann*

*Independent researcher

August 27, 2026

## Abstract

This paper introduces ClusterAttention, a general training-free speedup of bidirectional attention layers. Existing sparse attention methods either rely on structure in the input, such as order in language or spatial proximity in images, or use slow clustering processes amortized over several forward passes. ClusterAttention instead uses a fast recursive clustering method that adapts to the geometry of the keys and queries in each attention head to produce useful clusters. This method allows setting the size of the clusters arbitrarily. We utilize this by setting all clusters to be a fixed size that is a power of two, allowing the block-sparse attention to run at the same latency per query-key interaction as dense attention on GPUs. We also derive an expression for the output error in sparse attention, that explains the counterintuitive experimental finding that tight clusters can lead to larger errors than random clusters. We then derive the error when excluded clusters are compensated through their centroids, and show that this error shrinks with tighter clusters. We integrate this compensation into the method.

On large-scale tabular data ClusterAttention speeds up TabPFN-3 [1] by two to six times, while retaining at least 99% of the dense accuracy. To our knowledge, it is the first training-free method that can be successfully applied in the setting of unstructured input and a single forward pass. For video generation with Wan 2.1-14B T2V [2], ClusterAttention achieves output closer to dense attention and a larger speedup (1.8x versus 1.4x) compared to SVOO [3], a leading method developed specifically for this domain, both run without offline calibration. ¹

---

¹ Evaluation is currently limited due to time and budget constraints. The code will be available on
https://github.com/SpoketKasper/ClusterAttention, where the full preprint history can be found.

[1] Léo Grinsztajn et al. *TabPFN-3: Technical Report.* 2026. arXiv: 2605.13986 [cs.LG]
[2] Team Wan et al. *Wan: Open and Advanced Large-Scale Video Generative Models.* 2025. arXiv: 2503.20314 [cs.CV].
[3] Jiayi Luo et al. *Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering.* 2026. arXiv: 2603.18636 [cs.CV].