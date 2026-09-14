import torch
import triton
import triton.language as tl

def partition(
    proj, perm, buf, offs, szs, lefts, max_size, target_size, 
    sm_factor=32,  # how many programs we want per sm, 32: 9966, 
    block_m_max=512, num_warps_simple=2, num_warps_sliced=4, allow_sort=False
):
    batch_size, n = perm.shape
    S = offs.shape[0]
    num_rows = batch_size * S
    device = perm.device

    num_slices = sm_factor * _sm_count(device) // num_rows
    #if allow_sort and (max_size <= block_m_max):  # some us slower, but may be useful later
    #    block_m = triton.next_power_of_2(max_size)
    #    _partition_sort_kernel[(num_rows,)](
    #        proj, perm, buf, offs, szs, lefts,
    #        n, S, target_size,
    #        BLOCK_M=block_m, num_warps=num_warps_simple,  # have not benched warps
    #    )
    #    return
    # why is 2 here faster? 
    # that means we go to simple when there are less than 2 blocks per slice
    if num_slices <= 2:
        block_m = min(block_m_max, triton.next_power_of_2(max_size))
        _partition_simple_kernel[(num_rows,)](
            proj, perm, buf, offs, szs, lefts,
            n, S, target_size,
            BLOCK_M=block_m, num_warps=num_warps_simple,
        )
    else:
        _partition_sliced(
            proj, perm, buf, offs, szs, lefts, max_size, target_size,
            num_rows, num_slices, device, n, S,
            block_m=block_m_max, num_warps=num_warps_sliced,
        )

def deterministic_debug_partition(  # uses only simple as it is deterministic (has not been verified)
    proj, perm, buf, offs, szs, lefts, max_size, target_size,
    block_m_max=512, num_warps=2,
):
    batch_size, n = perm.shape
    S = offs.shape[0]
    num_rows = batch_size * S
    block_m = min(block_m_max, triton.next_power_of_2(max_size))
    _partition_simple_kernel[(num_rows,)](
        proj, perm, buf, offs, szs, lefts,
        n, S, target_size,
        BLOCK_M=block_m, num_warps=num_warps,
    )

###########
## Utils ##
###########

def _sm_count(device):
    return torch.cuda.get_device_properties(device).multi_processor_count

@triton.jit
def bf16_to_sortable_ints(vals):
    bits = vals.to(tl.uint16, bitcast=True)
    sign = bits >> 15
    keys = tl.where(sign != 0, bits ^ 0xFFFF, bits ^ 0x8000)
    return keys.to(tl.int32)

##########
## Sort ##
##########

# not so fast but also not so slow, might be useful fused with collect or even full level
@triton.jit(do_not_specialize=["N", "S", "T"])
def _partition_sort_kernel(
    proj_ptr, perm_ptr, buf, offs, szs, lefts,
    N, S, T,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = pid // S
    pid_s = pid - pid_b * S

    size = tl.load(szs + pid_s)
    off = tl.load(offs + pid_s)
    row = pid_b * N + off
    offs_m = tl.arange(0, BLOCK_M)
    mask = offs_m < size

    if size <= T:
        tl.store(buf + row + offs_m, tl.load(perm_ptr + row + offs_m, mask=mask), mask=mask)
        return

    proj = tl.load(proj_ptr + row + offs_m, mask=mask, other=0.0)
    keys = bf16_to_sortable_ints(proj)
    # pad with max so they sort to the end
    keys = tl.where(mask, keys, 0x7FFF)
    perm = tl.load(perm_ptr + row + offs_m, mask=mask, other=0)
    # combine and sort on the keys (so perm gets sorted)
    combined = (keys.to(tl.int64) << 32) | (perm.to(tl.int64) & 0xFFFFFFFF)
    combined = tl.sort(combined, dim=0)
    sorted_perm = (combined & 0xFFFFFFFF).to(tl.int32)
    tl.store(buf + row + offs_m, sorted_perm, mask=mask)

############
## Simple ##
############

@triton.jit(do_not_specialize=["N", "S", "T"])
def _partition_simple_kernel(
    proj, perm, buf, offs, szs, lefts,
    N, S, T,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = pid // S
    pid_s = pid - pid_b * S

    size = tl.load(szs + pid_s)
    off = tl.load(offs + pid_s)
    row = pid_b * N + off

    if size <= T:
        # finished segment: pass through so buf stays complete after the swap
        for start in range(0, size, BLOCK_M):
            offs_m = start + tl.arange(0, BLOCK_M)
            mask = offs_m < size
            tl.store(buf + row + offs_m, tl.load(perm + row + offs_m, mask=mask), mask=mask)
        return

    left_size = tl.load(lefts + pid_s)
    bins = tl.arange(0, 256)

    # pass 1: hi-byte histogram
    hist = tl.zeros([256], tl.int32)
    for start in range(0, size, BLOCK_M):
        offs_m = start + tl.arange(0, BLOCK_M)
        mask = offs_m < size
        vals = tl.load(proj + row + offs_m, mask=mask, other=0.0)
        hi = bf16_to_sortable_ints(vals) >> 8
        hist += tl.histogram(tl.where(mask, hi, 0), 256)
    pad = (size + BLOCK_M - 1) // BLOCK_M * BLOCK_M - size
    hist = tl.where(bins == 0, hist - pad, hist)

    # select hi bucket
    cum = tl.cumsum(hist, axis=0)
    before = cum - hist
    kth = left_size - 1
    in_bucket = (before <= kth) & (cum > kth)
    hi_bucket = tl.min(tl.where(in_bucket, bins, 255), axis=0)
    before_hi = tl.max(tl.where(in_bucket, before, 0), axis=0)

    # pass 2: lo-byte histogram within the hi bucket
    hist = tl.zeros([256], tl.int32)
    forced = 0
    for start in range(0, size, BLOCK_M):
        offs_m = start + tl.arange(0, BLOCK_M)
        mask = offs_m < size
        vals = tl.load(proj + row + offs_m, mask=mask, other=0.0)
        keys = bf16_to_sortable_ints(vals)
        sel = mask & ((keys >> 8) == hi_bucket)
        hist += tl.histogram(tl.where(sel, keys & 0xFF, 0), 256)
        forced += tl.sum((~sel).to(tl.int32), axis=0)
    hist = tl.where(bins == 0, hist - forced, hist)

    # select lo bucket -> threshold and eq budget
    cum = tl.cumsum(hist, axis=0)
    before = cum - hist
    kth = left_size - before_hi - 1
    in_bucket = (before <= kth) & (cum > kth)
    lo_bucket = tl.min(tl.where(in_bucket, bins, 255), axis=0)
    before_lo = tl.max(tl.where(in_bucket, before, 0), axis=0)
    threshold = (hi_bucket << 8) | lo_bucket
    eq_budget = left_size - before_hi - before_lo

    # pass 3: scatter with sequential counters (deterministic, no atomics)
    left_pos = 0
    right_pos = 0
    eq_taken = 0
    for start in range(0, size, BLOCK_M):
        offs_m = start + tl.arange(0, BLOCK_M)
        mask = offs_m < size
        vals = tl.load(proj + row + offs_m, mask=mask, other=0.0)
        keys = bf16_to_sortable_ints(vals)
        lt = mask & (keys < threshold)
        eq = mask & (keys == threshold)
        eq_rank = tl.cumsum(eq.to(tl.int32), axis=0) - 1
        take_left = lt | (eq & ((eq_taken + eq_rank) < eq_budget))
        take_right = mask & ~take_left

        p = tl.load(perm + row + offs_m, mask=mask, other=0)
        lpos = left_pos + tl.cumsum(take_left.to(tl.int32), axis=0) - 1
        rpos = right_pos + tl.cumsum(take_right.to(tl.int32), axis=0) - 1
        tl.store(buf + row + lpos, p, mask=take_left)
        tl.store(buf + row + left_size + rpos, p, mask=take_right)

        left_pos += tl.sum(take_left.to(tl.int32), axis=0)
        right_pos += tl.sum(take_right.to(tl.int32), axis=0)
        eq_taken += tl.sum(eq.to(tl.int32), axis=0)

############
## Sliced ##
############

def _partition_sliced(
    proj, perm, buf, offs, szs, lefts, max_size, target_size,
    num_rows, num_slices, device, n, S,
    block_m=512, num_warps=4,
):
    hists = torch.zeros(2, num_rows, 256, device=device, dtype=torch.int32)
    hi_bucket = torch.empty(num_rows, device=device, dtype=torch.int32)
    before_hi = torch.empty(num_rows, device=device, dtype=torch.int32)
    threshold = torch.empty(num_rows, device=device, dtype=torch.int32)
    eq_budget = torch.empty(num_rows, device=device, dtype=torch.int32)
    counters = torch.zeros(3, num_rows, device=device, dtype=torch.int32)

    grid_slices = (num_rows, num_slices)
    grid_blocks = (num_rows, triton.cdiv(max_size, block_m))
    _sliced_hist_kernel[grid_slices](
        proj, offs, szs, hi_bucket, hists[0],
        n, S, target_size, num_slices,
        LO_PASS=False,
        BLOCK_M=block_m, num_warps=num_warps,
    )
    _sliced_select_kernel[(num_rows,)](
        hists[0], szs, lefts, hi_bucket, before_hi, 
        threshold, eq_budget, S, target_size, LO_PASS=False
    )
    _sliced_hist_kernel[grid_slices](
        proj, offs, szs, hi_bucket, hists[1],
        n, S, target_size, num_slices,
        LO_PASS=True,
        BLOCK_M=block_m, num_warps=num_warps,
    )
    _sliced_select_kernel[(num_rows,)](
        hists[1], szs, lefts, hi_bucket, before_hi,
        threshold, eq_budget, S, target_size, LO_PASS=True
    )
    _sliced_scatter_kernel[grid_blocks](
        proj, perm, buf, offs, szs, lefts,
        threshold, eq_budget, counters,
        n, S, target_size,
        BLOCK_M=block_m, num_warps=num_warps,
    )


@triton.jit(do_not_specialize=["N", "S", "T", "NUM_SLICES"])
def _sliced_hist_kernel(
    proj, offs, szs, hi_bucket, hist,
    N, S, T, NUM_SLICES,
    LO_PASS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_slice = tl.program_id(1)
    pid_b = pid_row // S
    pid_s = pid_row - pid_b * S

    size = tl.load(szs + pid_s)
    if size <= T:
        return
    slice_len = (size + NUM_SLICES - 1) // NUM_SLICES
    start0 = pid_slice * slice_len
    if start0 >= size:
        return
    end = tl.minimum(start0 + slice_len, size)
    off = tl.load(offs + pid_s)
    row = pid_b * N + off

    if LO_PASS:
        selected_hi = tl.load(hi_bucket + pid_row)

    h = tl.zeros([256], tl.int32)
    forced = 0
    for start in range(start0, end, BLOCK_M):
        offs_m = start + tl.arange(0, BLOCK_M)
        mask = offs_m < end
        vals = tl.load(proj + row + offs_m, mask=mask, other=0.0)
        keys = bf16_to_sortable_ints(vals)
        if LO_PASS:
            sel = mask & ((keys >> 8) == selected_hi)
            b = keys & 0xFF
        else:
            sel = mask
            b = keys >> 8
        h += tl.histogram(tl.where(sel, b, 0), 256)
        forced += tl.sum((~sel).to(tl.int32), axis=0)
    bins = tl.arange(0, 256)
    h = tl.where(bins == 0, h - forced, h)
    tl.atomic_add(hist + pid_row * 256 + bins, h, sem="relaxed")


@triton.jit(do_not_specialize=["S", "T"])
def _sliced_select_kernel(
    hist, szs, lefts, hi_bucket, before_hi,
    threshold, eq_budget, S, T,
    LO_PASS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_s = pid_row % S
    size = tl.load(szs + pid_s)
    if size <= T:
        return
    left_size = tl.load(lefts + pid_s)

    kth = left_size - 1
    if LO_PASS:
        count_before_hi = tl.load(before_hi + pid_row)
        kth -= count_before_hi

    bins = tl.arange(0, 256)
    counts = tl.load(hist + pid_row * 256 + bins)
    cum = tl.cumsum(counts, axis=0)
    before = cum - counts
    in_bucket = (before <= kth) & (cum > kth)
    bucket = tl.min(tl.where(in_bucket, bins, 255), axis=0)
    count_before = tl.max(tl.where(in_bucket, before, 0), axis=0)

    if LO_PASS:
        hi = tl.load(hi_bucket + pid_row)
        tl.store(threshold + pid_row, (hi << 8) | bucket)
        tl.store(eq_budget + pid_row, kth + 1 - count_before)
    else:
        tl.store(hi_bucket + pid_row, bucket)
        tl.store(before_hi + pid_row, count_before)


@triton.jit(do_not_specialize=["N", "S", "T"])
def _sliced_scatter_kernel(
    proj, perm, buf, offs, szs, lefts,
    threshold, eq_budget, counters,
    N, S, T,
    BLOCK_M: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_block = tl.program_id(1)
    pid_b = pid_row // S
    pid_s = pid_row - pid_b * S

    size = tl.load(szs + pid_s)
    off = tl.load(offs + pid_s)
    row = pid_b * N + off
    offs_m = pid_block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs_m < size

    if size <= T:
        # finished segment, pass through
        tl.store(buf + row + offs_m, tl.load(perm + row + offs_m, mask=mask), mask=mask)
        return

    left_size = tl.load(lefts + pid_s)
    thresh = tl.load(threshold + pid_row)
    budget = tl.load(eq_budget + pid_row)

    vals = tl.load(proj + row + offs_m, mask=mask, other=0.0)
    keys = bf16_to_sortable_ints(vals)
    lt = mask & (keys < thresh)
    eq = mask & (keys == thresh)

    num_rows = tl.num_programs(0)
    eq_count = tl.sum(eq.to(tl.int32), axis=0)
    eq_start = tl.atomic_add(counters + pid_row, eq_count, sem="relaxed")
    eq_rank = tl.cumsum(eq.to(tl.int32), axis=0) - 1
    take_left = lt | (eq & ((eq_start + eq_rank) < budget))
    take_right = mask & ~take_left

    left_count = tl.sum(take_left.to(tl.int32), axis=0)
    right_count = tl.sum(take_right.to(tl.int32), axis=0)
    left_start = tl.atomic_add(counters + num_rows + pid_row, left_count, sem="relaxed")
    right_start = tl.atomic_add(counters + 2 * num_rows + pid_row, right_count, sem="relaxed")

    p = tl.load(perm + row + offs_m, mask=mask, other=0)
    lpos = left_start + tl.cumsum(take_left.to(tl.int32), axis=0) - 1
    rpos = right_start + tl.cumsum(take_right.to(tl.int32), axis=0) - 1
    tl.store(buf + row + lpos, p, mask=take_left)
    tl.store(buf + row + left_size + rpos, p, mask=take_right)