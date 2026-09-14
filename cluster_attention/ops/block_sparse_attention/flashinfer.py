import torch
import flashinfer

_wrappers = {}

def _get_wrapper(device):
    if device not in _wrappers:
        workspace = torch.empty(128*1024*1024, dtype=torch.uint8, device=device)
        _wrappers[device] = flashinfer.VariableBlockSparseAttentionWrapper(workspace)
    return _wrappers[device]

def lut_to_block_mask(lut, valid_block_num, n_kb):  # BSR is much nicer than LUT
    lut = lut.reshape(-1, lut.shape[-2], lut.shape[-1])
    valid_block_num = valid_block_num.reshape(-1, valid_block_num.shape[-1])
    BH, n_qb, n_kc = lut.shape

    col_range = torch.arange(n_kc, device=lut.device)
    valid = col_range[None, None, :] < valid_block_num[:, :, None]

    deltas = lut.clone()
    deltas[~valid] = 0
    abs_idx = torch.cumsum(deltas, dim=2)

    block_mask = torch.zeros(BH, n_qb, n_kb, dtype=torch.bool, device=lut.device)
    bh = torch.arange(BH, device=lut.device)[:, None, None].expand_as(abs_idx)
    qb = torch.arange(n_qb, device=lut.device)[None, :, None].expand_as(abs_idx)
    block_mask[bh[valid], qb[valid], abs_idx[valid].long()] = True
    return block_mask

# timings at 3500 tokens
def flashinfer_block_sparse_attn(
        q_s, k_s, v_s, lut, valid_block_num, QC_BLOCK_SIZE=64, KC_BLOCK_SIZE=128,
        return_lse=False, #return_fp32=False
):
    B, H, N, D = q_s.shape
    BH = B * H
    n_qb = (N + QC_BLOCK_SIZE - 1) // QC_BLOCK_SIZE
    n_kb = (N + KC_BLOCK_SIZE - 1) // KC_BLOCK_SIZE

    block_mask = lut_to_block_mask(lut, valid_block_num, n_kb)  # 1 ms

    row_sz = torch.full((n_qb,), QC_BLOCK_SIZE, dtype=torch.int32, device=q_s.device)
    row_sz[-1] = N - (n_qb-1)*QC_BLOCK_SIZE
    col_sz = torch.full((n_kb,), KC_BLOCK_SIZE, dtype=torch.int32, device=q_s.device)
    col_sz[-1] = N - (n_kb-1)*KC_BLOCK_SIZE

    # this block: 
    wrapper = _get_wrapper(q_s.device)
    wrapper.plan(
        block_mask,
        row_sz.expand(BH, n_qb).contiguous(),
        col_sz.expand(BH, n_kb).contiguous(),
        BH, BH, D,
        sm_scale=1.0 / (D ** 0.5),
        q_data_type=q_s.dtype, kv_data_type=k_s.dtype,
    )  # 0.1 ms

    # run wants (num_heads, seq_len, head_dim), which is just a view of (B, H, N, D)
    out = wrapper.run(  # 100 ms!
        q_s.reshape(BH, N, D), k_s.reshape(BH, N, D), v_s.reshape(BH, N, D),
        return_lse=return_lse, #o_data_type=(torch.float32 if return_fp32 else q_s.dtype)
    )
    if return_lse:
        o, lse = out
        return o.reshape(B, H, N, D), lse.reshape(B, H, N)
    else:
        o = out
        return o.reshape(B, H, N, D)

_bsr_wrappers = {}

def _get_bsr_wrapper(device):
    if device not in _bsr_wrappers:
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        _bsr_wrappers[device] = flashinfer.BlockSparseAttentionWrapper(workspace)
    return _bsr_wrappers[device]

def flashinfer_block_sparse_nonvariable_attn(
        q_s, k_s, v_s, lut, valid_block_num, QC_BLOCK_SIZE=64, KC_BLOCK_SIZE=128,
        return_lse=False, return_fp16=False
):
    B, H, N, D = q_s.shape
    BH = B * H
    n_qb = (N + QC_BLOCK_SIZE - 1) // QC_BLOCK_SIZE
    n_kb = (N + KC_BLOCK_SIZE - 1) // KC_BLOCK_SIZE
    N_padded = n_kb * KC_BLOCK_SIZE

    block_mask = lut_to_block_mask(lut, valid_block_num, n_kb)  # (BH, n_qb, n_kb)

    pad = N_padded - N
    if pad > 0:
        k_s = torch.nn.functional.pad(k_s, (0, 0, 0, pad))
        v_s = torch.nn.functional.pad(v_s, (0, 0, 0, pad))

    q_flat = q_s.reshape(BH, N, D)
    k_flat = k_s.reshape(BH, N_padded, D)
    v_flat = v_s.reshape(BH, N_padded, D)

    o_dtype = torch.float16 if return_fp16 else q_s.dtype
    out_all = torch.empty(BH, N, D, dtype=o_dtype, device=q_s.device)
    lse_all = torch.empty(BH, N, device=q_s.device) if return_lse else None

    wrapper = _get_bsr_wrapper(q_s.device)

    for bh in range(BH):
        head_mask = block_mask[bh]

        max_nnz = (torch.iinfo(torch.int32).max //
                (QC_BLOCK_SIZE * KC_BLOCK_SIZE)) - 1

        # Conservative: even a fully dense row fits.
        rows_per_call = max(1, max_nnz // n_kb)

        for qb0 in range(0, n_qb, rows_per_call):
            qb1 = min(qb0 + rows_per_call, n_qb)
            hm = head_mask[qb0:qb1]

            nnz_per_row = hm.sum(dim=1)
            indptr = torch.zeros(
                len(nnz_per_row) + 1, dtype=torch.int32, device=q_s.device
            )
            indptr[1:] = nnz_per_row.cumsum(0)
            indices = hm.nonzero(as_tuple=True)[1].to(torch.int32)

            mask = None
            if pad > 0:
                nnz = indices.numel()
                mask = torch.ones(
                    nnz, QC_BLOCK_SIZE, KC_BLOCK_SIZE,
                    dtype=torch.bool, device=q_s.device
                )
                mask[indices == n_kb - 1, :, KC_BLOCK_SIZE - pad:] = False

            q0 = qb0 * QC_BLOCK_SIZE
            q1 = min(qb1 * QC_BLOCK_SIZE, N)
            M_chunk = q1 - q0

            wrapper.plan(
                indptr, indices,
                M=M_chunk, N=N_padded,
                R=QC_BLOCK_SIZE, C=KC_BLOCK_SIZE,
                num_qo_heads=1, num_kv_heads=1,
                head_dim=D,
                mask=mask,
                sm_scale=1.0 / (D ** 0.5),
                q_data_type=q_s.dtype,
                kv_data_type=k_s.dtype,
                o_data_type=o_dtype,
            )

            result = wrapper.run(
                q_flat[bh, q0:q1].unsqueeze(1),
                k_flat[bh].unsqueeze(1),
                v_flat[bh].unsqueeze(1),
                return_lse=return_lse,
            )

            if return_lse:
                out_all[bh, q0:q1] = result[0].squeeze(1)
                lse_all[bh, q0:q1] = result[1].squeeze(1)
            else:
                out_all[bh, q0:q1] = result.squeeze(1)
                
    if return_lse:
        return out_all.reshape(B, H, N, D), lse_all.reshape(B, H, N)
    return out_all.reshape(B, H, N, D)