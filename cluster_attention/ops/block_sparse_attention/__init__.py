try:
    from .flashinfer import flashinfer_block_sparse_attn, flashinfer_block_sparse_nonvariable_attn
    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False
try:
    from .sage_attention import sage_with_custom_mask, lut_to_idx, idx_to_lut
    HAS_SPARGE = True
except ImportError:
    HAS_SPARGE = False