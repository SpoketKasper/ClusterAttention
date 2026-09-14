import modal

from paper.modal_setup import make_image
from paper.bench_utils import Run, save, perform_runs

############
## Config ##
############

# dataset configs
version = "Rain_in_Australia"

# attention_backend sweep configs
#topk_fracs = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
#target_recalls = [0.6, 0.8, 0.9, 0.95, 0.99, 1.0]
topk_fracs = [0.1, 0.4, 0.8, 1.0]
topk_fracs_w_small = [0.01, 0.05] + topk_fracs  # running extra for finecomp
target_recalls = [0.8, 0.9, 1.0]
target_recalls_w_low = [0.1, 0.4] + target_recalls  # running extra for svoo on the smoking and drinking dataset

# warmup config
same_size_warmup = True  # set True for SVOO
# now running True for all. ClusterAttention has no noticable 
# first-run-on-new-size overhead, but the other tabpfn parts do.
big_run_features = 25  # only matters if above is False
# 200k for datasets_100k and dataset_150k, 500k for dataset_400k, 700k for dataset_600k
# set slightly larger than dataset to avoid first-run memory allocation, 
# but not spend unnecessary compute

#############
## Pre-run ##
#############

DATASETS = [
    "Smoking_and_Drinking_Dataset_with_body_signal",  # 634460, 198270
    "Data_Science_for_Good_Kiva_Crowdfunding",  # 429571, 134241
    "CDC_Diabetes_Health_Indicators",  # 162355, 50736, (21)
    "accelerometer",  # 97922, 30601
    "walking-activity",  # 95572, 29867
    "Rain_in_Australia",  # 93094, 29092
    "customer_satisfaction_in_airline",  # 83123, 25976
    "dabetes_130-us_hospitals",  # 65129, 20354
]
# skipping the smaller customer_satisfaction_in_airline and diabetes_130-us_hospitals
DATASET_150k = ["CDC_Diabetes_Health_Indicators"]  # 162355, 50736, (21)
DATASET_400k = ["Data_Science_for_Good_Kiva_Crowdfunding"]  # 429571, 134241, (11)
DATASET_600k = ["Smoking_and_Drinking_Dataset_with_body_signal"]  # 634460, 198270 (23)
# modal config (not changed)
gpu = "H100!"
timeout = 1800
# config (not changed)
n_layers = 24  # model info
n_estimators = 1  # default is 8, but we use 1
cluster_size = (128, 64)
setup = {"model": "tabpfn_v3", "gpu": gpu, "n_estimators": n_estimators}
match version:  # hardcoding bench 
    case "accelerometer":
        n_runs = 3
        outer_sweep_param = ("dataset", ["accelerometer"], False, True, False)
        big_run_length = 200_000
        max_test_samples = None
    case "walking-activity":
        n_runs = 3
        outer_sweep_param = ("dataset", ["walking-activity"], False, True, False)
        big_run_length = 200_000
        max_test_samples = None
    case "Rain_in_Australia":
        n_runs = 3
        outer_sweep_param = ("dataset", ["Rain_in_Australia"], False, True, False)
        big_run_length = 200_000
        max_test_samples = None
    case "dataset_150k":
        n_runs = 3
        outer_sweep_param = ("dataset", DATASET_150k, False, True, False)
        big_run_length = 200_000
        max_test_samples = None
    case "dataset_400k":
        n_runs = 1
        outer_sweep_param = ("dataset", DATASET_400k, False, True, False)
        big_run_length = 500_000
        max_test_samples = None
    case "dataset_600k":
        n_runs = 1
        outer_sweep_param = ("dataset", DATASET_600k, False, True, False)
        big_run_length = 700_000
        #max_test_samples = 10_000
        max_test_samples = None

def bench_pareto():
    from cluster_attention import (
        ClusterAttentionBackend, RandomClusterBackend, DenseBackend
    )
    from cluster_attention.hooks.diagnose_comp_error_parts import error_ablation_hook
    from cluster_attention.hooks.diagnose_sparse_error import diagnose_sparse_error_hook
    from paper.baseline_backends import (
        SVOOBackend, NystromBackend, VyasClusteredBackend,
    )
        
    from paper.TabPFN_3.bench_wiring import (
        TabPFN3BackboneLoader, TALENTLoaderWithPreprocessing, Accuracy
    )
    model_loader = TabPFN3BackboneLoader()
    data_loader = TALENTLoaderWithPreprocessing(
        same_size_warmup=same_size_warmup, 
        big_run_length=big_run_length, 
        big_run_features=big_run_features,
        max_test_samples=max_test_samples
    )
    runs = [
        # dense
        #Run(name="dense", attention_backend=DenseBackend(), n_runs=n_runs),
        #Run(name="dense_sage", attention_backend=DenseBackend(use_sage=True), n_runs=n_runs),

        # topk
        Run(
            name="clusterattention_topk", 
            attention_backend=ClusterAttentionBackend(
               cluster_size=cluster_size, hooks=[
                    #error_ablation_hook, 
                    #diagnose_sparse_error_hook,
                ]),
            sweep_param=("k_frac", topk_fracs), n_runs=n_runs, warmup=("k", [17, 33, 65])
        ),
        #Run(
        #    name="clusterattention_topk_finecomp", 
        #    attention_backend=ClusterAttentionBackend(cluster_size=cluster_size, compensate=2),
        #    sweep_param=("k_frac", topk_fracs_w_small), n_runs=n_runs, warmup=("k", [17, 33, 65])
        #),
        #Run(
        #    name="random_topk", 
        #    attention_backend=RandomClusterBackend(cluster_size=cluster_size),
        #    sweep_param=("k_frac", topk_fracs), n_runs=n_runs, warmup=("k", [17, 33, 65])
        #),

        # adaptive
        #Run(
        #    name="clusterattention_adaptive", 
        #    attention_backend=ClusterAttentionBackend(
        #        cluster_size=cluster_size, adaptive_k=True, hooks=[
        #            error_ablation_hook, 
        #            diagnose_sparse_error_hook
        #        ]
        #    ),
        #    sweep_param=("target_recall", target_recalls), n_runs=n_runs, warmup=("target_recall", [0.63, 0.73, 0.83])
        #),
        #Run(
        #    name="svoo", 
        #    attention_backend=SVOOBackend(n_layers=n_layers),
        #    sweep_param=("top_p", target_recalls), 
        #    #sweep_param=("top_p", target_recalls_w_low),
        #    n_runs=n_runs, warmup=("top_p", [0.63, 0.73, 0.83])  
        #),

        # other
        #Run(
        #    name="nyström",
        #    attention_backend=NystromBackend(pinv_iterations=10),
        #    sweep_param=("num_landmarks", [512, 1024, 2048]),
        #    n_runs=n_runs, warmup=("num_landmarks", [512, 1024, 2048])  
        #)
        #Run(
        #    name="vyas-cluster",
        #    attention_backend=VyasClusteredBackend(),
        #    sweep_param=("clusters", [100, 250, 500, 1000]),
        #    n_runs=n_runs, warmup=("clusters", [100, 250, 500, 1000])  
        #)
    ]
    evals = [Accuracy]
    return perform_runs(runs, setup, model_loader, data_loader, evals, outer_sweep_param, compute_reference=False)


#########
## Run ##
#########

volume, volume_path = modal.Volume.from_name("talent-large-datasets"), "/root/talent-large-datasets"
app = modal.App("tabpfn-bench")

tabpfn_image = make_image(customize=lambda img: img.run_commands(
    "pip install tabpfn==8.3.0 --no-deps")  # dont want to wait to rerun everything
    .pip_install(
        "pydantic==2.13.0", "pydantic-settings==2.15.0", "safetensors==0.8.0", "huggingface-hub==1.27.0",
        "scipy==1.18.0",
        "scikit-learn==1.8.0"  # probably not performance critical, at least in the measured paths
    )
    .run_commands(  # triggering model download into the image
        "python -c 'from tabpfn import TabPFNClassifier; TabPFNClassifier().fit([[0, 1],[1, 0]], [0, 1])'",
        secrets=[modal.Secret.from_dotenv()]
    )
    #.pip_install("tilelang==0.1.13")
)
@app.function(
    gpu=gpu,
    image=tabpfn_image,
    timeout=timeout,
    volumes={volume_path: volume},
    secrets=[modal.Secret.from_dotenv()],
)
def bench_remote():
    # small patch to remove specialized error handling
    from contextlib import contextmanager
    import tabpfn.classifier
    @contextmanager
    def _passthrough(*args, **kwargs):
        yield
    tabpfn.classifier.handle_oom_errors = _passthrough

    # if there are compiles etc. 
    import torch
    torch._logging.set_logs(recompiles=True)

    ### temporary debug
    #import tabpfn.architectures.tabpfn_v3 as v3_mod
    #_orig = v3_mod.scaled_dot_product_attention
    #def _debug_sdpa(q, k=None, v=None, _backends_override=None, **kwargs):
    #    print(f"SDPA dtype: q={q.dtype}")
    #    return _orig(q, k, v, _backends_override, **kwargs)
    #v3_mod.scaled_dot_product_attention = _debug_sdpa
    ###
    # swap in the one you want to run here

    #from tabpfn.architectures.shared.fa3_backend import FA3_BACKEND
    #print(FA3_BACKEND.is_available())  # False! so we are not using FA3 for dense.
    return bench_pareto()

@app.local_entrypoint()
def main():
    save(bench_remote.remote(), version, "pareto", path_base="paper/TabPFN_3/results")
