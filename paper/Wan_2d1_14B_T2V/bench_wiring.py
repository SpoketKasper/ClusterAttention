from cluster_attention import ModifiedModel
from paper.bench_utils import ModelLoader, BenchDataLoader
from paper.Wan_2d1_14B_T2V.attention_wiring import ModifiedWanSelfAttention

####################################
## Convenience class for benching ##
####################################

class WanPromptLoader(BenchDataLoader):
    def __init__(self, prompts_path="/opt/SpargeAttn/evaluate/testing_prompts.txt"):
        self.prompt_idx = 0
        self._prompts = None
        self.prompts_path = prompts_path

    def _load_prompts(self):
        if self._prompts is None:
            with open(self.prompts_path) as f:
                self._prompts = [line.strip() for line in f if line.strip()]

    def __call__(self):
        self._load_prompts()
        return self._prompts[self.prompt_idx]
    
class WanModelLoader(ModelLoader):
    forward_warmup = True
    wrapper_classes = [ModifiedWanSelfAttention]
    task = None
    size = None
    ckpt_dir = None
    output_dir = None
    save_prefix = None
    
    def __init__(
        self, task="t2v-14B", size="1280*720", ckpt_dir="./Wan2.1-T2V-14B",
        output_dir=None, save_prefix="wan"
    ):
        self.task = task
        self.size = size
        self.ckpt_dir = ckpt_dir
        self.output_dir = output_dir
        self.save_prefix = save_prefix

    def __call__(self):
        import wan
        from wan.configs import WAN_CONFIGS
        cfg = WAN_CONFIGS[self.task]
        return wan.WanT2V(
            config=cfg,
            checkpoint_dir=self.ckpt_dir,
            # all below the only reasonable setting for single-gpu
            device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False,
        )

    def wrap(self, pipeline, attention_backend):  # overriding as we need to call pipeline.model
        return ModifiedModel(pipeline.model, attention_backend, self.wrapper_classes)

    # same as forward but just three diffusion steps
    def warmup_forward(self, pipeline, prompt):
        for layer in pipeline.model.blocks:
            layer.self_attn.call_count = 0
            layer.self_attn.diffusion_step_warmup = 1  # so we get one dense and two modified in warmup

        from wan.configs import SIZE_CONFIGS
        # defaults from https://github.com/Wan-Video/Wan2.1/blob/main/generate.py
        video = pipeline.generate(
            prompt,
            size=SIZE_CONFIGS[self.size],
            frame_num=81,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=3,
            guide_scale=5.0,
            seed=0,
            offload_model=False,  # not default but faster
        )

        # reset call_count
        for layer in pipeline.model.blocks:  # hardcoded :/ could use the class property?
            layer.self_attn.call_count = 0
            layer.self_attn.diffusion_step_warmup = 10  # same as SVOO
        return video
    
    def forward(self, pipeline, prompt):
        from wan.configs import SIZE_CONFIGS
        # defaults from https://github.com/Wan-Video/Wan2.1/blob/main/generate.py
        video = pipeline.generate(
            prompt,
            size=SIZE_CONFIGS[self.size],
            frame_num=81,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=50,
            guide_scale=5.0,
            seed=0,
            offload_model=False,  # not default but faster
        )

        # reset call_count
        for layer in pipeline.model.blocks:  # hardcoded :/ could use the class property?
            layer.self_attn.call_count = 0
        return video

    def post_forward(self, pipeline, video, prompt):
        # export video
        from wan.utils.utils import cache_video
        from wan.configs import WAN_CONFIGS
        cfg = WAN_CONFIGS[self.task]
        from datetime import datetime
        import os
        # values from https://github.com/Wan-Video/Wan2.1/blob/main/generate.py
        run_name = getattr(self, 'run_name', 'unknown')
        # adding timestamp to avoid an expensive overwrite
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            self.output_dir,
            f"{self.save_prefix}_{run_name}_{self.prompt_idx}_{ts}.mp4"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cache_video(
            tensor=video[None], 
            save_file=path, 
            fps=cfg.sample_fps,
            nrow=1, 
            normalize=True, 
            value_range=(-1, 1)
        )
        return video