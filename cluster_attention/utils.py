import time
import torch
import numpy as np
import gc
import sys

class Timer:
    def __init__(self, key, printer=print, accumulator=None):
        self.key = key
        self.printer = printer
        self.accumulator = accumulator
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start_time
        if self.accumulator is not None:
            self.accumulator[self.key].append(self.elapsed)
        else:
            self.printer(f"{self.key}: {self.elapsed:.5f}s")
        return False

    @staticmethod
    def summary(accumulator):
        for key, vals in accumulator.items():
            mean, std = np.mean(vals), np.std(vals)
            print(f"{key:30s}  {mean*1000:8.2f} ± {std*1000:6.2f} ms  (n={len(vals)})")

def time100withwarmup(fn, name):
    for i in range(5):
        fn()
    with Timer(f"100 runs of {name}"):
        for i in range(100):
            fn()

try:
    import triton.profiler as proton
    import subprocess
    import torch

    class proton_profiler:
        def __init__(self, name="kernel", context="python"):
            self.name = name
            self.context = context

        def __enter__(self):
            self.session = proton.start(self.name, context=self.context)
            return self

        def __exit__(self, *args):
            torch.cuda.synchronize()
            proton.finalize(self.session)
            r = subprocess.run(
                ["proton-viewer", "-m", "time/us", self.name + ".hatchet"],
                capture_output=True, text=True,
            )
            print(r.stdout)
            if r.stderr:
                print(r.stderr)

        def scope(self, label):
            return proton.scope(label)                    
except Exception as e:
    print(e)

def print_cuda_tensors(top=10):
    tensors = [o for o in gc.get_objects() if torch.is_tensor(o) and o.is_cuda]
    seen = set()
    rows = []
    for t in tensors:
        ptr = t.untyped_storage().data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        rows.append(t)
    rows.sort(key=lambda t: -t.untyped_storage().nbytes())

    def owners(t):
        found = []
        for ref in gc.get_referrers(t):
            if ref is rows or ref is tensors:
                continue
            if isinstance(ref, type(sys._getframe())):
                found.append(f"frame:{ref.f_code.co_name}:{ref.f_lineno}")
                continue
            if isinstance(ref, dict):
                ks = [k for k, v in ref.items() if v is t and isinstance(k, str)]
                for owner in gc.get_referrers(ref):
                    if hasattr(owner, "__dict__") and owner.__dict__ is ref:
                        found.append(f"{type(owner).__name__}.{ks[0] if ks else '?'}")
                        break
                else:
                    if ks:
                        found.append(f"dict[{ks[0]}]")
            elif isinstance(ref, (list, tuple)):
                for owner in gc.get_referrers(ref):
                    if hasattr(owner, "__dict__"):
                        attr = [k for k, v in owner.__dict__.items() if v is ref]
                        found.append(f"{type(owner).__name__}.{attr[0] if attr else '?'}[]")
                        break
                    if isinstance(owner, dict):
                        ks = [k for k, v in owner.items() if v is ref and isinstance(k, str)]
                        if ks:
                            found.append(f"dict[{ks[0]}][]")
                            break
        return found

    for t in rows[:top]:
        print(f"{t.untyped_storage().nbytes()/1e6:10.1f} MB  {tuple(t.shape)}  {t.dtype}  {owners(t)}")

def print_leak_stacks(min_mb=20):
    snap = torch.cuda.memory._snapshot()
    from collections import defaultdict
    groups = defaultdict(lambda: [0, 0])
    for seg in snap["segments"]:
        for blk in seg["blocks"]:
            if blk["state"] != "active_allocated" or blk["size"] < min_mb * 1e6:
                continue
            frames = blk.get("frames") or []
            key = tuple(f"{f['filename']}:{f['line']}:{f['name']}" for f in frames[:8])
            groups[key][0] += blk["size"]
            groups[key][1] += 1
    for key, (size, n) in sorted(groups.items(), key=lambda x: -x[1][0]):
        print(f"\n{size/1e6:.0f} MB in {n} blocks:")
        for line in key:
            print(f"    {line}")
