import torch
import torch.nn.functional as F
import gc
from dataclasses import dataclass
import json
import os
import glob

# sweep classes?

###################################
## Generic classes and functions ##
###################################

@dataclass
class Sample:
    setup: dict  # model name, for example pixel count,...
    #token_count: int
    name: str  # (includes params)
    sweep_param: str  # the name in setup of the param that was swept over, (any  # (name, value),)
    latency: float
    evals: dict  # {cosine_sim: 0.992, accuracy: 0.84,...}

@dataclass
class Run:
    name: str  # method name, e.g. "clusterattention_topk"
    attention_backend: any
    sweep_param: any = None  # (name, value list) or None
    #outer_sweep_param: any = None  # sweep_param where the model needs to be reloaded/rewarmed/new ref
    task: any = None  # Task or None
    n_runs: int = 3
    #n_warmup: int = 3
    warmup: any = 3  # number for default attention_backend config, or same format as sweep_param to modify params between runs
    compute_reference: bool = True  # False when dense is infeasible at this size

@torch.no_grad()
def collect_sample(model_loader, model, data_loader, x, run, reference, setup, evals):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = model_loader.forward(model, x)
    end.record()
    torch.cuda.synchronize()
    out = model_loader.post_forward(model, out, x)  # things we dont want to time
    return Sample(
        setup=setup,
        name=run.name,
        #sweep_param=run.sweep_param[0],
        sweep_param=run.sweep_param[0] if run.sweep_param else None,
        latency=start.elapsed_time(end),
        evals={e.name: e.compute(out, reference, data_loader) for e in evals},# if reference is not None else {},
    )

# model_loader owns forward, as it defines the return format (such as embeddings or whatever)
@torch.no_grad()
def perform_runs(
    runs, setup, model_loader, data_loader, evals=[], 
    outer_sweep_param=None,  # (name, vals, reload_model, recalculate_reference, rewarm_on_outer)
    compute_reference=True, 
):
    samples = []
    
    do_outer_sweep = outer_sweep_param is not None
    outer_sweep_iterator = outer_sweep_param[1] if do_outer_sweep else range(1)
    model = None
    reference = None
    for outer_i, outer_sweep_val in enumerate(outer_sweep_iterator):
        print(f"Outer loop iteration: {outer_i} with {outer_sweep_param[0]}={outer_sweep_val}")
        if do_outer_sweep:
            setattr(model_loader, outer_sweep_param[0], outer_sweep_val)
            setattr(data_loader, outer_sweep_param[0], outer_sweep_val)
            setup = {**setup, outer_sweep_param[0]: outer_sweep_val}
        
        if model is not None and (do_outer_sweep and outer_sweep_param[2]): 
            teardown(model)
            model = None
        if model is None: model = model_loader()

        x = data_loader()  # data_loader can here also stash things like correct labels etc.

        if do_outer_sweep and outer_sweep_param[3]:
            reference = None
        if reference is None and compute_reference:
            model_loader.run_name = "reference"
            reference = model_loader.forward(model, x)  # before wrapping (so it is dense)
            reference = model_loader.post_forward(model, reference, x)

        for run in runs:  # each is a new attention_backend
            print(f"  Run: {run.name}")
            wrapped = model_loader.wrap(model, run.attention_backend)  # returns a handle to unwrap

            if outer_i==0 or (do_outer_sweep and outer_sweep_param[4]):
                print("  Warming up")
                warmup(run, model_loader, model, data_loader, x)

            do_sweep = run.sweep_param is not None
            sweep_iterator = run.sweep_param[1] if do_sweep else range(1)
            for sweep_val in sweep_iterator:
                run_setup = {**setup}
                if do_sweep:
                    print(f"    Setting {run.sweep_param[0]}={sweep_val}")
                    setattr(run.attention_backend, run.sweep_param[0], sweep_val)
                    run_setup[run.sweep_param[0]] = sweep_val
                    #wrapped = model_loader.wrap(model, run.attention_backend)  # shouldnot be doing anything

                run_samples = []
                for i in range(run.n_runs):
                    print(f"    Run {i}")
                    model_loader.run_name = run.name
                    if run.task:
                        run.task.prepare(run.attention_backend)
                    s = collect_sample(model_loader, model, data_loader, x, run, reference, run_setup, evals)
                    if run.task:
                        extracted = run.task.extract(run.attention_backend)
                        s.evals.update(extracted)
                    run_samples.append(s)

                #wrapped.unwrap()
                samples.extend(run_samples)

            # unwrap to leave fresh (although next wrap would otherwise just override)
            wrapped.unwrap()
    return samples

def warmup(run, model_loader, model, data_loader, x):  # model comes in wrapped
    if model_loader.forward_warmup and data_loader.data_warmup:
        raise ValueError("Not currently possible to do both forward_warmup and data_warmup")
    if model_loader.forward_warmup:
        model_loader.warmup_forward(model, x)
    elif data_loader.data_warmup:
        for x_warmup in data_loader.warmup_data():
            model_loader.forward(model, x_warmup)
    else:
        sweep_warmup = isinstance(run.warmup, tuple)
        warmup_iterator = run.warmup[1] if sweep_warmup else range(run.warmup)
        for warmup_val in warmup_iterator:
            if sweep_warmup:
                setattr(run.attention_backend, run.warmup[0], warmup_val)
            torch.compiler.cudagraph_mark_step_begin()  # most likely not doing anything here
            model_loader.forward(model, x)

def teardown(model):
    from cluster_attention.clustering import clear_graph_cache
    del model
    gc.collect()
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    clear_graph_cache()

def save(samples, version, tag, path_base):
    out_dir = f"{path_base}/{version}"
    os.makedirs(out_dir, exist_ok=True)
    existing = glob.glob(f"{out_dir}/{tag}_*.json")
    if existing:
        idx = max(int(p.split("_")[-1].split(".")[0]) for p in existing) + 1
    else:
        idx = 0
    out = f"{out_dir}/{tag}_{idx}.json"
    with open(out, "w") as f:
        json.dump([s.__dict__ for s in samples], f, indent=2)
    print(f"Wrote {out}")

def load(version, path_base):
    samples = []
    for path in sorted(glob.glob(f"{path_base}/{version}/*.json")):
        with open(path) as f:
            samples += [Sample(**d) for d in json.load(f)]
    return samples

##################
## Base classes ##
##################

class Eval:
    name: str
    def compute(*args):
        raise NotImplementedError("Subclass this class")

class ModelLoader:
    forward_warmup = False  # if there is a special forward to use in warmup 
    wrapper_classes = None  # the modified attention versions

    def wrap(self, model, attention_backend):
        from cluster_attention import ModifiedModel  # importing in here it happens on modal
        return ModifiedModel(model, attention_backend, self.wrapper_classes)

    def __call__(self):  # fetches and returns the model
        raise NotImplementedError("Subclass this class")
    def forward(self, model, x):  # calls the forward on the model (no nn.Module so not invoked by __call__)
        raise NotImplementedError("Subclass this class")
    def post_forward(self, model, out, x):  # does nothing unless overridden, used for things we do not want to time
        return out
    
class BenchDataLoader:
    data_warmup = False

    def __call__(self):
        raise NotImplementedError("Subclass this class")

    def warmup_data(self):
        raise NotImplementedError("Optionally implement")

class Task:
    def prepare(self, attention_backend):
        raise NotImplementedError("Subclass this class")
    def extract(self, attention_backend):
        raise NotImplementedError("Subclass this class")

########################
## Example subclasses ##
########################

class FlattenedCosineSimilarity(Eval):  # like in SageAttention, flattened over all samples
    name = "flattened_cosine_similarity"
    def compute(embeddings, reference_embeddings, *args):  # (B, N, D) flattened to (B, N*D)
        return F.cosine_similarity(
            embeddings.flatten(1).float(), reference_embeddings.flatten(1).float(), dim=1
        ).mean().item()

class PerPatchCosineSimilarity(Eval):
    name = "per_patch_cosine_similarity"
    def compute(embeddings, reference, *args):
        sims = F.cosine_similarity(
            embeddings.reshape(-1, embeddings.shape[-1]).float(),
            reference.reshape(-1, reference.shape[-1]).float(),
            dim=1
        )
        return {"mean": sims.mean().item(), "p5": sims.quantile(0.05).item()}

class RecallTask(Task):
    def prepare(self, attention_backend):
        attention_backend.start_recall_logging()
    def extract(self, attention_backend):
        return {"recall_by_layer": attention_backend.get_recall_stats()}

class ProfileTask(Task):
    def __init__(self, print=False):
        super().__init__()
        self.print = print
    def prepare(self, attention_backend):
        attention_backend.start_profiling()
    def extract(self, attention_backend):
        timings = list(attention_backend.get_timings())
        if self.print:
            print(timings)
        return {"timings": timings}
