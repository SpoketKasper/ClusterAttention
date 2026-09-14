# Note! Single workspace, so two consecutive calls with nothing in between
# will overwrite the eigenvalues of the first call
# AI-written file

import ctypes
import torch

_cusolver = ctypes.cdll.LoadLibrary("libcusolver.so")
CUDA_R_32F = 0


class BatchedEigh:
    _cache = {}

    def __init__(self, batch_size, n, device="cuda"):
        self.batch_size = batch_size
        self.n = n

        self._handle = ctypes.c_void_p()
        _cusolver.cusolverDnCreate(ctypes.byref(self._handle))

        stream = torch.cuda.current_stream(device)
        _cusolver.cusolverDnSetStream(self._handle, ctypes.c_void_p(stream.cuda_stream))

        self._params = ctypes.c_void_p()
        _cusolver.cusolverDnCreateParams(ctypes.byref(self._params))

        self._eigenvalues = torch.empty(batch_size, n, dtype=torch.float32, device=device)
        self._info = torch.empty(batch_size, dtype=torch.int32, device=device)

        dev_bytes = ctypes.c_size_t()
        host_bytes = ctypes.c_size_t()
        _cusolver.cusolverDnXsyevBatched_bufferSize(
            self._handle, self._params,
            ctypes.c_int(1),
            ctypes.c_int(1),
            ctypes.c_int64(n),
            ctypes.c_int(CUDA_R_32F),
            ctypes.c_void_p(0),
            ctypes.c_int64(n),
            ctypes.c_int(CUDA_R_32F),
            ctypes.c_void_p(0),
            ctypes.c_int(CUDA_R_32F),
            ctypes.byref(dev_bytes),
            ctypes.byref(host_bytes),
            ctypes.c_int64(batch_size),
        )

        self._dev_workspace = torch.empty(dev_bytes.value, dtype=torch.uint8, device=device)
        self._dev_bytes = dev_bytes.value

        self._host_bytes = host_bytes.value
        if self._host_bytes > 0:
            print(f"WARNING: cusolverDnXsyevBatched requires {self._host_bytes} bytes of host workspace, "
                  "CUDA graph capture may fail")
            self._host_workspace = (ctypes.c_byte * self._host_bytes)()
            self._host_ptr = ctypes.cast(self._host_workspace, ctypes.c_void_p)
        else:
            self._host_ptr = ctypes.c_void_p(0)

    def __call__(self, A):
        stream = torch.cuda.current_stream(A.device)
        _cusolver.cusolverDnSetStream(self._handle, ctypes.c_void_p(stream.cuda_stream))

        _cusolver.cusolverDnXsyevBatched(
            self._handle, self._params,
            ctypes.c_int(1),
            ctypes.c_int(1),
            ctypes.c_int64(self.n),
            ctypes.c_int(CUDA_R_32F),
            ctypes.c_void_p(A.data_ptr()),
            ctypes.c_int64(self.n),
            ctypes.c_int(CUDA_R_32F),
            ctypes.c_void_p(self._eigenvalues.data_ptr()),
            ctypes.c_int(CUDA_R_32F),
            ctypes.c_void_p(self._dev_workspace.data_ptr()),
            ctypes.c_size_t(self._dev_bytes),
            self._host_ptr,
            ctypes.c_size_t(self._host_bytes),
            ctypes.c_void_p(self._info.data_ptr()),
            ctypes.c_int64(self.batch_size),
        )
        return self._eigenvalues, A.transpose(-1, -2)

    @classmethod
    def eigh(cls, A):
        batch_size, n, _ = A.shape
        key = (batch_size, n, A.device)
        if key not in cls._cache:
            cls._cache[key] = cls(batch_size, n, A.device)
        return cls._cache[key](A)