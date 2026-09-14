# recent methods
try:
    from .sparge_attn import SpargeAttnBackend
except Exception as e:
    print(f"Import of SpargeAttn failed with: {e}")
try:
    from .svoo import SVOOBackend
except Exception as e:
    print(f"Import of SVOO failed with: {e}")

# older methods
try:
    from .nystrom import NystromBackend
except Exception as e:
    print(f"Import of Nyström attention failed with: {e}")
try:
    from .scatterbrain import ScatterbrainBackend
except Exception as e:
    print(f"Import of Scatterbrain failed with: {e}")
try:
    from .vyas_clustered import VyasClusteredBackend
except Exception as e:
    print(f"Import of Vyas et al. clustered failed with: {e}")