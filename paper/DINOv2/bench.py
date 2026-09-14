import modal

from paper.modal_setup import make_image
from paper.bench_utils import (
  Run, save, 
  FlattenedCosineSimilarity, PerPatchCosineSimilarity, 
  perform_runs, ProfileTask
)

############
## Config ##
############

version = "v1000"
run_type = "pareto"

if run_type=="pareto":
    image = 12  # "hamngatan"  # deer, hamngatan, or number of images from div8k
    outer_sweep_param = ("image", [image] if isinstance(image, str) else range(image), False, True, False)  # pareto
    # for pareto: small: 2072, medium: 250*14=3500, large: 5306, xlarge: 600*14
    #res = 2300
    res = 2072
    n_runs = 1  # per image
elif run_type=="profile":
    image = 0  # (index here, set to 0)
    # for latency sweep/profiling: [14*25, 14*50, 14*100, 14*250, 14*500, 14*750, 14*1000]
    res = [14*25, 14*50, 14*100, 14*250, 14*500, 14*750, 14*1000]
    outer_sweep_param = ("res", res, True, False, True)  # profile
    n_runs = 3

topk_fracs = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
target_recalls = [0.6, 0.8, 0.9, 0.95, 0.99, 1.0]

compute_reference = True

#############
## Pre-run ##
#############

# data and model setup (not changed)
model_name, n_layers, patch_size = "vit_large_patch14_dinov2.lvd142m", 24, 14
# other (not changed)
cluster_size = (128, 64)
# modal config (not changed)
gpu = "H100!"
timeout = 1800
if run_type=="pareto":
    setup = {"model": model_name, "gpu": gpu, "resolution": res}  # pareto
elif run_type=="profile":
    setup = {"model": model_name, "gpu": gpu}  # profile

def bench_pareto():
    import torch
    from cluster_attention import (
        ClusterAttentionBackend, RandomClusterBackend, DenseBackend
    )
    from cluster_attention.hooks.diagnose_comp_error_parts import error_ablation_hook
    from cluster_attention.hooks.diagnose_sparse_error import diagnose_sparse_error_hook
    from paper.baseline_backends import (
        SpargeAttnBackend, SVOOBackend, ScatterbrainBackend
    )
    from paper.DINOv2.bench_wiring import ViTModelLoader, ViTImageLoader
    model_loader = ViTModelLoader(res, model_name)
    data_loader = ViTImageLoader(res, image)

    runs = [
        # dense
        Run(name="dense", attention_backend=DenseBackend(), n_runs=n_runs, warmup=3),
        #Run(name="dense_sage", attention_backend=DenseBackend(use_sage=True), n_runs=n_runs, warmup=3),

        # topk
        #Run(
        #    name="clusterattention_topk", 
        #    attention_backend=ClusterAttentionBackend(
        #       cluster_size=cluster_size, hooks=[
        #            error_ablation_hook, 
        #            diagnose_sparse_error_hook
        #        ]),
        #    sweep_param=("k_frac", topk_fracs), n_runs=n_runs, warmup=("k", [17, 33, 65])
        #),
        Run(
            name="clusterattention_topk_finecomp", 
            attention_backend=ClusterAttentionBackend(cluster_size=cluster_size, compensate=2),
            sweep_param=("k_frac", topk_fracs), n_runs=n_runs, warmup=("k", [17, 33, 65])
        ),
        #Run(
        #    name="clusterattention_topk_notransforms", 
        #    attention_backend=ClusterAttentionBackend(cluster_size=cluster_size, transform=False),
        #    sweep_param=("k_frac", topk_fracs), n_runs=n_runs, warmup=("k", [17, 33, 65])
        #),
        #Run(
        #    name="clusterattention_topk_notransforms_finecomp", 
        #    attention_backend=ClusterAttentionBackend(cluster_size=cluster_size, transform=False, compensate=2),
        #    sweep_param=("k_frac", topk_fracs), n_runs=n_runs, warmup=("k", [17, 33, 65])
        #),
        #Run(
        #    name="random_topk", 
        #    attention_backend=RandomClusterBackend(cluster_size=cluster_size),
        #    sweep_param=("k_frac", topk_fracs), n_runs=n_runs, warmup=("k", [17, 33, 65])
        #),
        #Run(
        #    name="random_topk_finecomp", 
        #    attention_backend=RandomClusterBackend(cluster_size=cluster_size, compensate=2),
        #    sweep_param=("k_frac", topk_fracs), n_runs=n_runs, warmup=("k", [17, 33, 65])
        #),
        #Run(
        #    name="randomkeys_topk", 
        #    attention_backend=RandomClusterBackend(cluster_size=cluster_size, nonrandom_queries=True),
        #    sweep_param=("k_frac", topk_fracs), n_runs=n_runs, warmup=("k", [17, 33, 65])
        #),
        #Run(
        #    name="randomqueries_topk", 
        #    attention_backend=RandomClusterBackend(cluster_size=cluster_size, nonrandom_keys=True),
        #    sweep_param=("k_frac", topk_fracs), n_runs=n_runs, warmup=("k", [17, 33, 65])
        #),
        #Run(
        #    name="randomrandom_topk", 
        #    attention_backend=RandomClusterBackend(cluster_size=cluster_size, random_assignment=True),
        #    sweep_param=("k_frac", topk_fracs), n_runs=n_runs, warmup=("k", [17, 33, 65])
        #),
        #Run(
        #    name="spargeattn_topk", 
        #    attention_backend=SpargeAttnBackend(cluster_size=cluster_size),
        #    sweep_param=("k_frac", topk_fracs), n_runs=n_runs, warmup=("k", [17, 33, 65])
        #),

        # adaptive
        #Run(
        #    name="clusterattention_adaptive", 
        #    attention_backend=ClusterAttentionBackend(cluster_size=cluster_size, adaptive_k=True),
        #    sweep_param=("target_recall", target_recalls), n_runs=n_runs, warmup=("target_recall", [0.63, 0.73, 0.83])
        #),
        #Run(
        #    name="clusterattention_adaptive_notransforms", 
        #    attention_backend=ClusterAttentionBackend(cluster_size=cluster_size, adaptive_k=True, transform=False),
        #    sweep_param=("target_recall", target_recalls), n_runs=n_runs, warmup=("target_recall", [0.63, 0.73, 0.83])
        #),
        #Run(
        #    name="clusterattention_adaptive_finecomp", 
        #    attention_backend=ClusterAttentionBackend(cluster_size=cluster_size, adaptive_k=True, compensate=2),
        #    sweep_param=("target_recall", target_recalls), n_runs=n_runs, warmup=("target_recall", [0.63, 0.73, 0.83])
        #),
        #Run(
        #    name="clusterattention_adaptive_notransforms_finecomp", 
        #    attention_backend=ClusterAttentionBackend(cluster_size=cluster_size, adaptive_k=True, transform=False, compensate=2),
        #    sweep_param=("target_recall", target_recalls), n_runs=n_runs, warmup=("target_recall", [0.63, 0.73, 0.83])
        #),
        #Run(
        #    name="spargeattn_adaptive_m01", 
        #    attention_backend=SpargeAttnBackend(cluster_size=cluster_size, adaptive=True, simthreshd1=-0.1),
        #    sweep_param=("cdfthreshd", target_recalls), n_runs=n_runs, warmup=("cdfthreshd", [0.63, 0.73, 0.83])
        #),
        #Run(
        #    name="spargeattn_adaptive_06", 
        #    attention_backend=SpargeAttnBackend(cluster_size=cluster_size, adaptive=True, simthreshd1=0.6),
        #    sweep_param=("cdfthreshd", target_recalls), n_runs=n_runs, warmup=("cdfthreshd", [0.63, 0.73, 0.83])
        #),
        #Run(
        #    name="svoo", 
        #    attention_backend=SVOOBackend(n_layers=n_layers),
        #    sweep_param=("top_p", target_recalls), n_runs=n_runs, warmup=("top_p", [0.63, 0.73, 0.83])    
        #),

        # other
        #Run(
        #    name="scatterbrain", 
        #    attention_backend=ScatterbrainBackend(dim_heads=64),
        #    sweep_param=("local_context", [32, 64, 128, 4096]), n_runs=n_runs, warmup=("local_context", [32, 64, 128])    
        #),
    ]
 
    evals = [FlattenedCosineSimilarity, PerPatchCosineSimilarity]
    return perform_runs(runs, setup, model_loader, data_loader, evals, outer_sweep_param, compute_reference=compute_reference)

def bench_profile():
    from cluster_attention import ClusterAttentionBackend
    from paper.vit.vit_wiring import ViTModelLoader, ViTImageLoader
    model_loader = ViTModelLoader(res, model_name)
    data_loader = ViTImageLoader(res, image)
    runs = [Run(name="clusterattention_topk_finecomp", 
        attention_backend=ClusterAttentionBackend(cluster_size=cluster_size, k=64, compensate=2),
        #attention_backend=ClusterAttentionBackend(cluster_size=cluster_size, k_frac=0.1, compensate=2),
        task=ProfileTask(),
        warmup=3,#3/0
        n_runs=n_runs
    )]
    evals = []
    return perform_runs(runs, setup, model_loader, data_loader, evals, outer_sweep_param, compute_reference=False)

#########
## Run ##
#########

name = "whatever"
volume, volume_path = modal.Volume.from_name("vit-bench-data"), "/images"
app = modal.App(name)
@app.function(
    gpu=gpu, 
    image=make_image(
        customize=lambda img: 
            img
            .run_commands("pip install timm==1.0.28 torchvision==0.27.1 --no-deps")
            .pip_install("huggingface-hub==1.27.0")
    ),
    timeout=timeout, 
    volumes={volume_path: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def bench_remote():
    # to make tests, just put random stuff here followed by a raise
    if run_type=="pareto":
        return bench_pareto()
    elif run_type=="profile":
        return bench_profile()

@app.local_entrypoint()
def main():
    save(bench_remote.remote(), version, run_type, path_base="paper/DINOv2/results")