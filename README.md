Code for project ClusterAttention (https://arxiv.org/abs/2608.26965), a training-free speedup of bidirectional attention.

For use in a project, the only dependencies are the three libraries in `requirements_part1.txt` and `requirements_part2.txt`. Note that the code may run with other packages, but using these ensures reproducing the performance seen in the paper. `requirements_optional.txt` is optional. The `attention_wiring.py` files in the `paper` folder show how to wire the modified attention backend into a model. Essentially, what you need is `cluster_attention.ModifiedAttention`, `cluster_attention.ModifiedModel`, and `cluster_attention.ClusterAttentionBackend`. You then define a modified attention class, that mixes the original attention or your model and `ModifiedAttention`, and also defines `locations`, where this modification should be injected. You then simply call
```python
your_modified_model = ModifiedModel(your_model, ClusterAttentionBackend(...), [YourModifiedAttention])
```
and you are ready to use your model with ClusterAttention.

The experiments in the preprint are run on Modal. To run this and the evaluations, pip-install the requirements in `paper/paper_requirements.txt` into a local virtual environment. Reproducing the results on DINOv2 and TabPFN-3 requires downloading data, how to do this is described in each folder. Each folder also contains the results from the preprint.

The code is currently written to run on SM90 architectures such as Nvidia H100 and H200. The modifications to run on other architectures are likely small, but performance could be suboptimal if no tuning is carried out.

Citation:
```
@article{nordenram2026clusterattention,
  title={ClusterAttention: A training-free speedup of bidirectional attention},
  author={Nordenram, Kasper and Dittmann, Amelie},
  journal={arXiv preprint arXiv:2608.26965},
  year={2026}
}
```