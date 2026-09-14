# AI-written file

import ctypes
import torch

_cublas = ctypes.cdll.LoadLibrary("libcublas.so")

CUDA_R_16BF = 14
CUDA_R_32F = 0
CUBLAS_COMPUTE_32F = 68


class Contraction:
    def __init__(self, device="cuda"):
        self._handle = ctypes.c_void_p()
        _cublas.cublasCreate_v2(ctypes.byref(self._handle))
        stream = torch.cuda.current_stream(device)
        _cublas.cublasSetStream_v2(self._handle, ctypes.c_void_p(stream.cuda_stream))

    def __call__(self, A, B, transA=False, transB=False):
        assert A.is_contiguous() and B.is_contiguous()

        stream = torch.cuda.current_stream(A.device)
        _cublas.cublasSetStream_v2(self._handle, ctypes.c_void_p(stream.cuda_stream))

        #A = A.contiguous()
        #B = B.contiguous()

        if A.dim() == 3:
            return self._batched(A, B, transA, transB)
        return self._single(A, B, transA, transB)
    
    def _single(self, A, B):
        m, k = A.shape
        _, n = B.shape
        C = torch.empty(m, n, dtype=torch.float32, device=A.device)

        _cublas.cublasGemmEx(
            self._handle,
            ctypes.c_int(0), ctypes.c_int(0),
            ctypes.c_int(n), ctypes.c_int(m), ctypes.c_int(k),
            ctypes.byref(ctypes.c_float(1.0)),
            ctypes.c_void_p(B.data_ptr()), ctypes.c_int(CUDA_R_16BF), ctypes.c_int(n),
            ctypes.c_void_p(A.data_ptr()), ctypes.c_int(CUDA_R_16BF), ctypes.c_int(k),
            ctypes.byref(ctypes.c_float(0.0)),
            ctypes.c_void_p(C.data_ptr()), ctypes.c_int(CUDA_R_32F), ctypes.c_int(n),
            ctypes.c_int(CUBLAS_COMPUTE_32F),
            ctypes.c_int(-1),
        )
        return C

    def _batched(self, A, B, transA, transB):
        a0, a1, a2 = A.shape
        b0, b1, b2 = B.shape
        m = a2 if transA else a1
        k = a1 if transA else a2
        n = b1 if transB else b2

        C = torch.empty(a0, m, n, dtype=torch.float32, device=A.device)

        # row-major → col-major: swap A↔B and flip trans flags
        opB = 1 if transA else 0   # becomes transB in col-major
        opA = 1 if transB else 0   # becomes transA in col-major

        ldB = b2  # leading dim of physical B layout
        ldA = a2  # leading dim of physical A layout

        _cublas.cublasGemmStridedBatchedEx(
            self._handle,
            ctypes.c_int(opA), ctypes.c_int(opB),
            ctypes.c_int(n), ctypes.c_int(m), ctypes.c_int(k),
            ctypes.byref(ctypes.c_float(1.0)),
            ctypes.c_void_p(B.data_ptr()), ctypes.c_int(CUDA_R_16BF), ctypes.c_int(ldB), ctypes.c_longlong(b1 * b2),
            ctypes.c_void_p(A.data_ptr()), ctypes.c_int(CUDA_R_16BF), ctypes.c_int(ldA), ctypes.c_longlong(a1 * a2),
            ctypes.byref(ctypes.c_float(0.0)),
            ctypes.c_void_p(C.data_ptr()), ctypes.c_int(CUDA_R_32F), ctypes.c_int(n), ctypes.c_longlong(m * n),
            ctypes.c_int(a0),
            ctypes.c_int(CUBLAS_COMPUTE_32F),
            ctypes.c_int(-1),
        )
        return C