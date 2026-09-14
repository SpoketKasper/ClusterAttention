# AI-written file

import cupy as cp
import torch

def _make_kernel(n, threads_per_block=0):
    assert n & (n - 1) == 0 and n >= 4, "N must be a power of 2"
    assert threads_per_block >= n // 2, "need at least one thread per pair"
    N = n
    STRIDE = N + 1  # pad rows to avoid shared memory bank conflicts
    pairs = N // 2
    tpp = threads_per_block // pairs  # threads per pair
    elems = N // tpp  # elements per thread
    source = rf'''
extern "C" __global__
void __launch_bounds__({threads_per_block})
jacobi_eigh(
    const float* __restrict__ A_in,   // (B, 128, 128)
    float* __restrict__ W_out,        // (B, 128)
    float* __restrict__ V_out,        // (B, 128, 128)
    int num_sweeps
) {{
    #define N {N}
    #define STRIDE {STRIDE}
    #define PAIRS {pairs}
    #define TPP {tpp}
    #define ELEMS {elems}

    extern __shared__ float smem[];
    float* A = smem;                        // N * STRIDE
    float* V = A + N * STRIDE;              // N * STRIDE
    float* cs = V + N * STRIDE;             // PAIRS * 2

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int pair_id = tid / TPP;
    const int local_id = tid % TPP;
    const int elem_start = local_id * ELEMS;

    // ---- load A, init V = I ------------------------------------------------
    const float* A_g = A_in + bid * N * N;
    for (int i = tid; i < N * N; i += {threads_per_block}) {{
        int r = i / N, c = i % N;
        A[r * STRIDE + c] = A_g[i];
    }}
    for (int i = tid; i < N * N; i += {threads_per_block}) {{
        int r = i / N, c = i % N;
        V[r * STRIDE + c] = (r == c) ? 1.0f : 0.0f;
    }}
    __syncthreads();

    // ---- sweeps -------------------------------------------------------------
    for (int sweep = 0; sweep < num_sweeps; sweep++) {{
        for (int round = 0; round < N - 1; round++) {{

            // round-robin tournament: 64 independent pairs per round
            int p, q;
            if (pair_id == 0) {{
                p = N - 1;
                q = round;
            }} else {{
                p = ((round - pair_id) % (N - 1) + (N - 1)) % (N - 1);
                q = (round + pair_id) % (N - 1);
            }}
            if (p > q) {{ int tmp = p; p = q; q = tmp; }}

            // one thread per pair computes rotation params
            if (local_id == 0) {{
                float app = A[p * STRIDE + p];
                float aqq = A[q * STRIDE + q];
                float apq = A[p * STRIDE + q];

                float cc, ss;
                if (fabsf(apq) < 1e-30f) {{
                    cc = 1.0f; ss = 0.0f;
                }} else {{
                    float tau = (aqq - app) / (2.0f * apq);
                    float t;
                    if (tau >= 0.0f)
                        t =  1.0f / ( tau + sqrtf(1.0f + tau * tau));
                    else
                        t = -1.0f / (-tau + sqrtf(1.0f + tau * tau));
                    cc = rsqrtf(1.0f + t * t);
                    ss = t * cc;
                }}
                cs[pair_id * 2    ] = cc;
                cs[pair_id * 2 + 1] = ss;
            }}
            __syncthreads();

            const float c = cs[pair_id * 2];
            const float s = cs[pair_id * 2 + 1];

            // right multiply  A ← A · J   (update columns p, q)
            #pragma unroll
            for (int i = 0; i < ELEMS; i++) {{
                int k = elem_start + i;
                int kp = k * STRIDE + p;
                int kq = k * STRIDE + q;
                float akp = A[kp];
                float akq = A[kq];
                A[kp] = c * akp - s * akq;
                A[kq] = s * akp + c * akq;
            }}
            __syncthreads();

            // left multiply  A ← Jᵀ · A   (update rows p, q)
            #pragma unroll
            for (int i = 0; i < ELEMS; i++) {{
                int k = elem_start + i;
                int pk = p * STRIDE + k;
                int qk = q * STRIDE + k;
                float apk = A[pk];
                float aqk = A[qk];
                A[pk] = c * apk - s * aqk;
                A[qk] = s * apk + c * aqk;
            }}
            __syncthreads();

            // accumulate eigenvectors  V ← V · J
            #pragma unroll
            for (int i = 0; i < ELEMS; i++) {{
                int k = elem_start + i;
                int kp = k * STRIDE + p;
                int kq = k * STRIDE + q;
                float vkp = V[kp];
                float vkq = V[kq];
                V[kp] = c * vkp - s * vkq;
                V[kq] = s * vkp + c * vkq;
            }}
            __syncthreads();
        }}
    }}

    // ---- store results ------------------------------------------------------
    for (int i = tid; i < N; i += {threads_per_block}) {{
        W_out[bid * N + i] = A[i * STRIDE + i];
    }}

    float* V_g = V_out + bid * N * N;
    for (int i = tid; i < N * N; i += {threads_per_block}) {{
        int r = i / N, c = i % N;
        V_g[i] = V[r * STRIDE + c];
    }}
}}
'''
    kern = cp.RawKernel(source, 'jacobi_eigh')
    smem = N * STRIDE * 4 * 2 + pairs * 2 * 4  # bytes
    kern.max_dynamic_shared_size_bytes = smem
    return kern, smem


_kernels = {}

import numpy as np

# 16,64,64
# threads_per_block=128: 1107, 
# threads_per_block=256: 816, 
# threads_per_block=512: 786,
# threads_per_block=1024: 878,
def jacobi_eigh(A, num_sweeps=10, threads_per_block=128):
    B, N, _ = A.shape
    assert A.dtype == torch.float32 and A.is_cuda

    key = (N, threads_per_block)
    if key not in _kernels:
        _kernels[key] = _make_kernel(N, threads_per_block)
    kern, smem = _kernels[key]

    W = torch.empty(B, N, device=A.device, dtype=torch.float32)
    V = torch.empty(B, N, N, device=A.device, dtype=torch.float32)

    A = A.contiguous()
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        kern((B,), (threads_per_block,),
             (np.uint64(A.data_ptr()), np.uint64(W.data_ptr()),
              np.uint64(V.data_ptr()), np.int32(num_sweeps)),
             shared_mem=smem)
    return W, V

def compare_eigh(W1, V1, W2, V2, A=None):
    print("eigenvalue max diff:      ", (W1 - W2).abs().max().item())

    # eigenvectors: fix sign per column before comparing element-wise
    sign = torch.sign((V1 * V2).sum(dim=-2, keepdim=True))
    print("eigenvector max diff:     ", (V1 - sign * V2).abs().max().item())

    # subspace agreement, immune to sign and degenerate-cluster mixing
    M = V1.transpose(-1, -2) @ V2
    eye = torch.eye(V1.shape[-1], device=V1.device).expand_as(M)
    print("|V1^T V2| vs I max diff:  ", (M.abs() - eye).abs().max().item())

    if A is not None:
        for name, W, V in (("1", W1, V1), ("2", W2, V2)):
            R = V @ torch.diag_embed(W) @ V.transpose(-1, -2) - A
            print(f"reconstruction error {name}:   ", R.abs().max().item())

def test_and_time():
    import torch
    from cluster_attention.clustering.ops.custom_jacobi import jacobi_eigh, compare_eigh
    from cluster_attention.clustering.ops.batched_eigh import BatchedEigh
    from cluster_attention.utils import time100withwarmup
    from cluster_attention.cuda_graph_cached import CUDAGraphCached

    torch.manual_seed(42)
    B, N = 16, 64

    eigh = BatchedEigh(B, N)

    jacobi_eigh = CUDAGraphCached(jacobi_eigh)
    eigh = CUDAGraphCached(eigh)

    # random PSD matrices (like covariance matrices)
    X = torch.randn(B, N, N, device="cuda")
    A = X @ X.transpose(-1, -2) / N

    # our kernel
    W, V = jacobi_eigh(A, num_sweeps=10)

    # cusolver wrapper
    W_wr, V_wr = eigh(A)

    # reference
    W_ref, V_ref = torch.linalg.eigh(A)

    compare_eigh(W, V, W_ref, V_ref)
    compare_eigh(W_wr, V_wr, W_ref, V_ref)

    time100withwarmup(lambda: jacobi_eigh(A, num_sweeps=10), "jacobi_eigh")
    time100withwarmup(lambda: eigh(A), "BatchedEigh")
    time100withwarmup(lambda: torch.linalg.eigh(A), "torch.linalg.eigh")

    key = next(iter(jacobi_eigh.cache))
    g = jacobi_eigh.cache[key][0]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(1000):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end), "us/call")  # ms over 1000 iters = us/call
