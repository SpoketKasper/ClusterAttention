# Modifying wan.utils.utils.cache_video from https://github.com/Wan-Video/Wan2.1
# Licensed under Apache 2.0

# note! needs --detach in the modal call, so "modal run --detach paper/Wan_2d1_14B_T2V/bench.py"
import modal

from paper.modal_setup import make_image
from paper.bench_utils import Run, save, perform_runs

# bench config
version = "v1"
n_runs = 1
topk_fracs = [0.2]  # reasonable based on svg-ear numbers (0.3 for sparge and 0.2-0.26 for them)
# open-sora prompt set of 12 prompts (from SpargeAttn repo)
outer_sweep_param = ("prompt_idx", [0], False, False, False)  # set to 0, 1,...

def bench_pareto():
    from cluster_attention import ClusterAttentionBackend, DenseBackend
    from paper.baseline_backends import SVOOBackend
    from paper.Wan_2d1_14B_T2V.bench_wiring import WanModelLoader, WanPromptLoader

    model_loader = WanModelLoader(task=task, size=size, output_dir=volume_path+"/wan")
    data_loader = WanPromptLoader()

    runs = [
        #Run(
        #    name="dense", 
        #    attention_backend=DenseBackend(),
        #    n_runs=n_runs
        #),
        #Run(
        #    name="clusterattention_topk_finecomp",
        #    attention_backend=ClusterAttentionBackend(cluster_size=cluster_size, compensate=2),
        #    sweep_param=("k_frac", topk_fracs), n_runs=n_runs
        #),
        Run(
            name="svoo",
            attention_backend=SVOOBackend(n_layers=n_layers, reset_every=False),
            sweep_param=("top_p", [0.9]), n_runs=n_runs
        )
    ]
    return perform_runs(
        runs, setup, model_loader, data_loader, [], outer_sweep_param, compute_reference=False
    )

#############
## Pre-run ##
#############

# modal config (not changed)
gpu = "H200"
timeout = 3600
# model config (not changed)
task = "t2v-14B"
size = "1280*720"
n_layers = 40  # Wan 14B
# ca config (not changed)
cluster_size = (128, 64)
setup = {"model": task, "gpu": gpu, "size": size}

#########
## Run ##
#########

name = "wan-bench"
volume, volume_path = modal.Volume.from_name("wan-results", create_if_missing=True), "/root/wan-results"
app = modal.App(name)

wan_image = make_image(
    customize=lambda img: 
        img
        .pip_install("torchvision==0.27.1", extra_options="--no-deps")
        .pip_install("huggingface-hub==1.27.0")
        .run_commands(
            "hf download Wan-AI/Wan2.1-T2V-14B --local-dir /root/Wan2.1-T2V-14B",
        secrets=[modal.Secret.from_dotenv()]
        ).run_commands(
            "git clone https://github.com/Wan-Video/Wan2.1.git /opt/wan2.1"
            " && cd /opt/wan2.1 && git checkout 9737cba9c1c3c4d04b33fcad41c111989865d315"
        ).env({"PYTHONPATH": "/opt/wan2.1:/opt/svoo"}
        ).pip_install(
            "easydict==1.13", 
            "diffusers==0.40.0", 
            "ftfy==6.3.1", 
            "imageio==2.37.4", 
            "imageio-ffmpeg==0.6.0",
            "transformers==5.14.0",
            "tokenizers==0.22.2",
        ).pip_install(
            "accelerate==1.14.0", extra_options="--no-deps"
        ).pip_install(
            "flash-attn==2.8.3.post1+cu.13.0.torch.2.12",
            extra_index_url="https://wheels.astral.sh/simple/cu130/",
            extra_options="--no-deps"
        )
)
@app.function(
    gpu=gpu,
    image=wan_image,
    timeout=timeout,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={volume_path: volume},
)
def bench_remote():
    # patching the cache_video function #
    import wan.utils.utils as wu
    import torch, torchvision

    _orig_cache_video = wu.cache_video
    def patched_cache_video(tensor, save_file=None, fps=30, suffix='.mp4', nrow=8, normalize=True, value_range=(-1, 1), retry=5):
        import os.path as osp
        cache_file = osp.join('/tmp', wu.rand_name(suffix=suffix)) if save_file is None else save_file
        tensor = tensor.clamp(min(value_range), max(value_range))
        tensor = torch.stack([
            torchvision.utils.make_grid(u, nrow=nrow, normalize=normalize, value_range=value_range)
            for u in tensor.unbind(2)
        ], dim=1).permute(1, 2, 3, 0)
        tensor = (tensor * 255).to(torch.uint8).cpu()
        import imageio
        writer = imageio.get_writer(cache_file, fps=fps, codec='libx264', quality=8)
        for frame in tensor.numpy():
            writer.append_data(frame)
        writer.close()
        return cache_file
    wu.cache_video = patched_cache_video
    #####################################

    result = bench_pareto()
    save(result, version, "pareto", path_base=volume_path+"/wan/bench")
    volume.commit()

# not saving stuff locally
@app.local_entrypoint()
def main():
    #bench_remote.spawn()
    bench_remote.remote()