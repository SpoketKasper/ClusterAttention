import torch
import torch.nn as nn
from contextlib import contextmanager, nullcontext

# class that holds all the logic and potentially states, 
# weights (like in the dsa indexer), etc. needed for modified/sparse attention
class AttentionBackend(nn.Module):
    is_dense = False  # used for different handling of special tokens
    use_model_fallback = False  # for falling back to model-attentions when they exist

    def __init__(self, c=0):
        super().__init__()
        self.c = c  # could be used for sparsity annealing, as a scaling factor on exp weights
        self._profile = False
        self._events = []

    def forward(self, q, k, v, layer_idx, return_lse=False):
        # should return out and lse (can be None)
        raise NotImplementedError

    # Timing stuff! For internal timing
    def start_profiling(self):
        self._profile = True
        self._events.clear()

    def stop_profiling(self):
        self._profile = False

    def get_timings(self):
        torch.cuda.synchronize()
        results = []
        for name, start, end in self._events:
            results.append((name, start.elapsed_time(end)))
        self._events.clear()
        return results

    def print_timings(self):
        for name, ms in self.get_timings():
            print(f"{name}: {ms:.2f} ms")

    def _record(self, name):
        if not self._profile:
            return nullcontext()
        return self._timed_section(name)

    @contextmanager
    def _timed_section(self, name):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        yield
        e.record()
        self._events.append((name, s, e))
