import torch

from .minmax_kernel import launch_minmax_kernel
from .histogram_kernel import histogram_threshold
from .create_lut_gluon import launch_lut_kernel

# buffers to avoid reallocations (no big difference but feels like good practice)
class AssignmentBuffers:
    def __init__(self, device="cuda"):
        self._device = device
        self._allocated = False

    def ensure(self, B, H, BH, max_n_qc, max_n_kc, D, N_BINS=128):
        needed = BH * max_n_qc
        if self._allocated and self._capacity >= needed and self._max_n_kc >= max_n_kc:
            return
        print("(allocating more memory for assignment)")
        self._capacity = needed
        self._max_n_kc = max_n_kc

        self.qc = torch.empty(B, H, max_n_qc, D, device=self._device, dtype=torch.bfloat16)
        self.kc = torch.empty(B, H, max_n_kc, D, device=self._device, dtype=torch.bfloat16)
        self.cluster_assignment = torch.empty(BH * max_n_qc * max_n_kc, device=self._device, dtype=torch.int32)
        self.valid_block_num = torch.empty(needed, device=self._device, dtype=torch.int32)
        self.mins = torch.empty(needed, device=self._device, dtype=torch.float32)
        self.maxes = torch.empty(needed, device=self._device, dtype=torch.float32)
        self.denominators = torch.empty(needed, device=self._device, dtype=torch.float32)
        self.lower_edges = torch.empty(needed, device=self._device, dtype=torch.float32)
        self.upper_edges = torch.empty(needed, device=self._device, dtype=torch.float32)
        self.mass_offsets = torch.empty(needed, device=self._device, dtype=torch.float32)
        self.count_offsets = torch.empty(needed, device=self._device, dtype=torch.int32)
        self.exp_histogram = torch.empty(needed * N_BINS, device=self._device, dtype=torch.float32)
        self.counts_histogram = torch.empty(needed * N_BINS, device=self._device, dtype=torch.int32)
        self._allocated = True

_shared_bufs = AssignmentBuffers(device="cuda")

def assignment(
    q_s, k_s, v_s,
    cs_q, cs_k,
    target_recall, min_k,
    adaptive_k,
    BLOCK_SIZE_KC=128, BLOCK_SIZE_QC=64,
    N_BINS=128,
):
    B, H, N, D = q_s.shape
    n_qc = (N + cs_q - 1) // cs_q
    n_kc = (N + cs_k - 1) // cs_k
    BH = B * H

    rem_k, rem_q = N % cs_k, N % cs_q
    n_full_k, n_full_q = N // cs_k, N // cs_q
    last_kc_frac = (rem_k or cs_k) / cs_k

    _shared_bufs.ensure(B, H, BH, n_qc, n_kc, D)
    bufs = _shared_bufs

    # Compute centroids into pre-allocated buffers
    qc = bufs.qc[:B, :H, :n_qc]
    kc = bufs.kc[:B, :H, :n_kc]

    qc[:, :, :n_full_q] = q_s[:, :, :n_full_q * cs_q].view(B, H, n_full_q, cs_q, D).mean(3)
    if rem_q:
        qc[:, :, n_full_q:n_qc] = q_s[:, :, n_full_q * cs_q:N].mean(2, keepdim=True)
    kc[:, :, :n_full_k] = k_s[:, :, :n_full_k * cs_k].view(B, H, n_full_k, cs_k, D).mean(3)
    if rem_k:
        kc[:, :, n_full_k:n_kc] = k_s[:, :, n_full_k * cs_k:N].mean(2, keepdim=True)

    # Slice all buffers to actual size
    mins = bufs.mins[:BH * n_qc]
    maxes = bufs.maxes[:BH * n_qc]
    denominators = bufs.denominators[:BH * n_qc]
    lower_edges = bufs.lower_edges[:BH * n_qc]
    upper_edges = bufs.upper_edges[:BH * n_qc]
    mass_offsets = bufs.mass_offsets[:BH * n_qc]
    count_offsets = bufs.count_offsets[:BH * n_qc]
    valid_block_num = bufs.valid_block_num[:BH * n_qc]
    cluster_assignment = bufs.cluster_assignment[:BH * n_qc * n_kc]
    exp_histogram = bufs.exp_histogram[:BH * n_qc * N_BINS]
    counts_histogram = bufs.counts_histogram[:BH * n_qc * N_BINS]

    mass_offsets.zero_()
    count_offsets.zero_()
    cluster_assignment.zero_()
    exp_histogram.zero_()
    counts_histogram.zero_()

    launch_minmax_kernel(qc, kc, last_kc_frac, mins, maxes, denominators)

    lower_edges.copy_(mins)
    upper_edges.copy_(maxes)

    for i in range(2):
        lower_edges, upper_edges, mass_offsets, count_offsets = histogram_threshold(
            qc, kc, last_kc_frac, target_recall, min_k,
            maxes, lower_edges, upper_edges,
            mass_offsets, count_offsets, denominators,
            exp_histogram, counts_histogram,
            N_BINS=N_BINS,
        )

    cluster_assignment, valid_block_num = launch_lut_kernel(
        qc, kc, lower_edges, last_kc_frac,
        cluster_assignment, valid_block_num,
    )
    return (
        cluster_assignment[:BH * n_qc * n_kc].view(B, H, n_qc, n_kc),
        valid_block_num[:BH * n_qc].view(B, H, n_qc)
    )
