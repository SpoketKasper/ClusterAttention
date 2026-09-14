# Modifying Fast Transformers ('vyas clustered') https://github.com/idiap/fast-transformers/tree/master
# MIT license

# Modifying Scatterbrain https://github.com/HazyResearch/fly
# Licensed under Apache 2.0

import modal

def make_image(customize=None):
    image = (
        # base image
        modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.12")

        # build infra, necessary for both cluster_attention (sparge, triton) and baselines
        .apt_install("gcc", "g++", "git")
        .env({"TORCH_CUDA_ARCH_LIST": "9.0"})  # setting only h100/h200 architecture for faster compilation
        .env({"CXX": "g++", "CC": "gcc"})  # so compilation uses g++ instead of clang++, required by pytorch
        .pip_install("ninja==1.13.0")  # for faster compilation
        .pip_install("torch==2.12.1")  # ensuring compilation against correct pytorch version

        # baselines
        # svoo
        .run_commands(
            "git clone https://github.com/Mutual-Luo/SVOO /opt/svoo "
            "&& cd /opt/svoo "
            "&& git checkout e4ae67b579766bcbe820bda7d34e104ff4c82d5f"
        )
        # adacluster, was not successfully integrated
        #.run_commands(
        #    "git clone https://github.com/USTC-MLSys/Adacluster /opt/adacluster"
        #    " && cd /opt/adacluster && git checkout e7bed1c475a596ca6057fa7da2e5b3c37909b536"
        # nyström
        .run_commands("pip install nystrom-attention==0.0.14 --no-deps")
        # vyas clustered
        .run_commands("pip install wheel")
        .run_commands(
            "git clone https://github.com/idiap/fast-transformers.git "
            "&& cd fast-transformers "
            "&& git checkout 2ad36b97e64cb93862937bd21fcc9568d989561f "
            # modding to run on h100, is hardcoded to older architectures
            "&& sed -i 's/compute_[0-9]*/compute_90/g' setup.py "
            "&& pip install . --no-build-isolation --no-deps"
        )        
        # scatterbrain
        .pip_install("einops==0.8.2")
        .run_commands(
            "git clone https://github.com/HazyResearch/fly /opt/fly "
            "&& cd /opt/fly "
            "&& git checkout 6b73449a6b3e228af9e4afe4f153a384e9b537b9 "
            "&& mv /opt/fly/src /opt/fly/fly_src "
            # modding file to store already computed lse
            "&& find /opt/fly/fly_src -name '*.py' -exec sed -i 's/from src\\./from fly_src./g' {} \\; "
            "&& find /opt/fly/fly_src -name '*.py' -exec sed -i 's/import src\\./import fly_src./g' {} \\; "
            "&& sed -i \"/return rearrange(out, 'b h t d/i\\        self._last_log_normalization = log_normalization.squeeze(-1)\" /opt/fly/fly_src/models/attention/sblocal_attention.py"
        )
        # putting stuff that cant be pip-installed onto path
        .env({"PYTHONPATH": "/opt/fly:/opt/svoo"})

        # dense with sageattention, baseline but currently inside cluster_attention folder
        .run_commands(
            "git clone https://github.com/thu-ml/SageAttention "
            "&& cd SageAttention "
            "&& git checkout d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5 "
            "&& python setup.py install"
        )
        
        # packages for cluster_attention
        # locally modified SpargeAttn
        .add_local_dir("SpargeAttn", remote_path="/opt/SpargeAttn", copy=True, ignore=[".git", "__pycache__", "build", "*.egg-info"])
        .run_commands("cd /opt/SpargeAttn && python setup.py install")
        # optional deps
        .pip_install_from_requirements("requirements_optional.txt")
        # performance-critical packages last (important to use --no-deps in 'customize' where there can be conflict)
        .pip_install_from_requirements("requirements_part1.txt")  # the pytorch pins triton 3.7.1 so we need two parts
        .pip_install_from_requirements("requirements_part2.txt")
        # note! below was used in the paper results, 
        # now switched to stable 'triton==3.8.0' released on August 28 
        #.pip_install("git+https://github.com/triton-lang/triton.git@8d749c1660d399b71d8a115bd0aacfa7f955d65b", env={"MAX_JOBS": "8"})
    
        # specific data, not really used anymore
        .add_local_file(local_path="paper/DINOv2/data/deer.jpg", remote_path="/data/deer.jpg", copy=True)
        .add_local_file(local_path="paper/DINOv2/data/hamngatan.jpg", remote_path="/data/hamngatan.jpg", copy=True)
    )
    if customize:
        image = customize(image)
    image = image.add_local_dir(".", remote_path="/root", ignore=[".git", "__pycache__", "*.pyc", ".venv", "outputs", "**/data", "div8k_raw", ".modal", "SpargeAttn"])
    return image
