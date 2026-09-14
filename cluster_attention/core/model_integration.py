import torch.nn as nn

# a mixin for the attention that we modify. 
# by default uses decompose, forward, recompose,
# but the forward can be overridden
class ModifiedAttention:
    locations = {
        "name of transformer block": [
            "name of attention in that transformer block"
        ]
    }  # used by ModifiedModel to find where to inject the ModifiedAttention

    def decompose(self, x, **kwargs):
        raise NotImplementedError

    def recompose(self, out, lse, state):
        raise NotImplementedError

    # we may want an explicit bypass here, for cases like in TabPFNv3,
    # so overriding the forward is necessary less often

    def forward(self, *args, **kwargs):
        with self.attention_backend._record("decompose"):
            q, k, v, state = self.decompose(*args, **kwargs)
        with self.attention_backend._record("forward"):
            out, lse = self.attention_backend(q, k, v, self.layer_idx, return_lse=state.get("needs_lse", False))
        with self.attention_backend._record("recompose"):
            return self.recompose(out, lse, state)

# handle for wrapping and unwrapping a model.
class ModifiedModel(nn.Module):
    def __init__(self, root_module, attention_backend, wrapper_classes):
        super().__init__()
        self.model = root_module
        self.attention_backend = attention_backend
        self._originals = []

        merged = {}
        for cls in wrapper_classes:
            for root_attr, names in cls.locations.items():
                mapping = merged.setdefault(root_attr, {})
                for name in names:
                    mapping[name] = cls

        idx = 0
        for root_attr, mapping in merged.items():
            idx = self._wrap_recursively(getattr(root_module, root_attr), mapping, idx)

    def _wrap_recursively(self, module, mapping, idx):
        for name, child in module.named_children():
            if name in mapping:
                cls = mapping[name]
                if not hasattr(child, '_original_cls'):
                    child._original_cls = child.__class__
                self._originals.append((child, child._original_cls))
                child.__class__ = cls
                child.attention_backend = self.attention_backend
                child.layer_idx = idx
                idx += 1
            else:
                idx = self._wrap_recursively(child, mapping, idx)
        return idx

    def unwrap(self):
        for module, original_cls in self._originals:
            module.__class__ = original_cls
            for attr in ('attention_backend', 'layer_idx', '_original_cls'):
                if hasattr(module, attr):
                    delattr(module, attr)
        return self.model

    @property
    def wrappers(self):
        return [module for module, _ in self._originals]
