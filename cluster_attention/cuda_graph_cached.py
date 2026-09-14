# mostly AI-written file
import torch

# registrer all instances here, clear all with class method?
class CUDAGraphCached:
    def __init__(self, fn):
        self.fn = fn
        self.cache = {}
        self.pool = torch.cuda.graph_pool_handle()

    def _make_key(self, args, kwargs):
        shape_key = tuple(a.shape for a in args if isinstance(a, torch.Tensor))
        kw_key = []
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kw_key.append((k, "__tensor__", v.shape))
            else:
                try:
                    hash(v)
                    kw_key.append((k, v))
                except TypeError:
                    continue
        return shape_key + tuple(sorted(kw_key))

    def _split_kwargs(self, kwargs):
        tensor_kw = {}
        scalar_kw = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                tensor_kw[k] = v
            else:
                try:
                    hash(v)
                    scalar_kw[k] = v
                except TypeError:
                    continue  # drop unhashable non-tensors (attention_backend, etc.)
        return tensor_kw, scalar_kw

    def __call__(self, *args, enabled=True, **kwargs):
        if not enabled:
            return self.fn(*args, **kwargs)

        tensor_kw, scalar_kw = self._split_kwargs(kwargs)
        key = self._make_key(args, {**tensor_kw, **scalar_kw})

        if key not in self.cache:
            placeholders = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
            kw_placeholders = {k: v.clone() for k, v in tensor_kw.items()}
            self.fn(*placeholders, **kw_placeholders, **scalar_kw)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                result = self.fn(*placeholders, **kw_placeholders, **scalar_kw)

            tensor_placeholders = [p for p in placeholders if isinstance(p, torch.Tensor)]
            self.cache[key] = (g, tensor_placeholders, kw_placeholders, result)

        g, tensor_placeholders, kw_placeholders, result = self.cache[key]
        j = 0
        for a in args:
            if isinstance(a, torch.Tensor):
                tensor_placeholders[j].copy_(a)
                j += 1
        for k, v in tensor_kw.items():
            kw_placeholders[k].copy_(v)
        g.replay()

        if isinstance(result, tuple):
            return tuple(r.clone() if isinstance(r, torch.Tensor) else r for r in result)
        elif isinstance(result, torch.Tensor):
            return result.clone()
        return result

    def clear(self):
        self.cache.clear()
 