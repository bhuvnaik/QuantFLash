"""
turbo_fused.py
Python wrapper for the fused TurboQuant CUDA kernel.

Compiles turbo_kernel.cu at import time via torch.utils.cpp_extension,
then provides FusedTurboQuant class that replaces the PyTorch baseline.

Usage:
    python turbo_fused.py          # runs benchmark comparing fused vs baseline
"""

import torch
import torch.nn.functional as F
import numpy as np
import ctypes
import os
import subprocess
import time
from scipy.stats import norm as scipy_norm


# ── compile the kernel ───────────────────────────────────────────────────────

def compile_kernel(cuda_file: str = "turbo_kernel.cu") -> ctypes.CDLL:
    """
    Compile turbo_kernel.cu to a shared library and load it.
    Uses nvcc directly — no torch extension overhead.
    """
    so_file = cuda_file.replace(".cu", ".so")

    # only recompile if .cu is newer than .so
    if (os.path.exists(so_file) and
            os.path.getmtime(so_file) > os.path.getmtime(cuda_file)):
        print(f"  Using cached {so_file}")
    else:
        print(f"  Compiling {cuda_file} -> {so_file} ...")
        cmd = [
            "nvcc", "-O3",
            "-arch=sm_86",          # RTX A5000 = Ampere sm_86
            "--compiler-options", "-fPIC",
            "-shared",
            cuda_file, "-o", so_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"nvcc failed:\n{result.stderr}")
        print(f"  Compiled successfully.")

    lib = ctypes.CDLL(os.path.abspath(so_file))

    # set argument types for turbo_encode_kernel
    lib.turbo_encode_kernel.argtypes = [
        ctypes.c_void_p,   # y
        ctypes.c_void_p,   # x
        ctypes.c_void_p,   # centroids
        ctypes.c_void_p,   # idx
        ctypes.c_void_p,   # r_unit
        ctypes.c_void_p,   # gamma
        ctypes.c_int,      # d
        ctypes.c_int,      # k
    ]
    lib.turbo_encode_kernel.restype = None

    # set argument types for turbo_encode_tiled_kernel
    lib.turbo_encode_tiled_kernel.argtypes = [
        ctypes.c_void_p,   # y
        ctypes.c_void_p,   # x
        ctypes.c_void_p,   # centroids
        ctypes.c_void_p,   # idx
        ctypes.c_void_p,   # r_unit
        ctypes.c_void_p,   # gamma
        ctypes.c_int,      # d
        ctypes.c_int,      # k
    ]
    lib.turbo_encode_tiled_kernel.restype = None

     
    lib.launch_turbo_encode.argtypes = [
         ctypes.c_void_p,  # y
         ctypes.c_void_p,  # x
         ctypes.c_void_p,  # centroids
         ctypes.c_void_p,  # idx
         ctypes.c_void_p,  # r_unit
         ctypes.c_void_p,  # gamma
         ctypes.c_int,     # n
         ctypes.c_int,     # d
         ctypes.c_int,     # k
    ]
    lib.launch_turbo_encode.restype = None
    return lib
# ── codebook construction ────────────────────────────────────────────────────

def solve_lloyd_max(d: int, b: int, n_iter: int = 2000) -> np.ndarray:
    k = 2 ** b
    std = 1.0 / np.sqrt(d)
    pdf = scipy_norm(0, std)
    centroids = pdf.ppf(np.linspace(0.5 / k, 1 - 0.5 / k, k))
    for _ in range(n_iter):
        bds = np.concatenate([[-np.inf], (centroids[:-1] + centroids[1:]) / 2, [np.inf]])
        nc = np.zeros(k)
        for i in range(k):
            lo, hi = bds[i], bds[i + 1]
            p = pdf.cdf(hi) - pdf.cdf(lo)
            nc[i] = centroids[i] if p < 1e-12 \
                else (std ** 2) * (pdf.pdf(lo) - pdf.pdf(hi)) / p
        if np.max(np.abs(nc - centroids)) < 1e-12:
            break
        centroids = nc
    return centroids.astype(np.float32)


# ── next power of 2 ──────────────────────────────────────────────────────────

def next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


# ── fused TurboQuant ─────────────────────────────────────────────────────────

class FusedTurboQuant:
    """
    TurboQuantProd with fused S2+S4+S5 kernel.

    Pipeline:
      [cuBLAS] S1: y = x @ Pi.T
      [FUSED]  S2+S4+S5: idx, r_unit_rot, gamma = fused_encode(y, x, centroids)
      [cuBLAS] S3: x_tilde_mse = centroids[idx] @ Pi   (via embedding + matmul)
      [cuBLAS] S6: z = sign(r_unit_rot @ S_eff.T)       S_eff = S @ Pi.T
      [cuBLAS] S7: x_tilde_qjl = gamma * (scale * z @ S_eff)
      [elem]   final: x_tilde = x_tilde_mse + x_tilde_qjl

    Key design: r_unit is returned in the ROTATED basis (y-space).
    We precompute S_eff = S @ Pi.T so QJL matmul works directly on
    the rotated residual without needing to rotate back.
    """

    def __init__(self, d: int, b: int, device: torch.device, lib: ctypes.CDLL):
        self.d      = d
        self.b      = b
        self.k      = 2 ** b
        self.device = device
        self.lib    = lib
        self.scale  = float(np.sqrt(np.pi / 2) / d)

        # rotation matrix Pi (orthogonal)
        G       = torch.randn(d, d, device=device)
        Q, _    = torch.linalg.qr(G)
        self.Pi = Q                                     # (d, d)

        # QJL matrix S; precompute S_eff = S @ Pi.T
        # so we can apply QJL directly to rotated residuals
        self.S     = torch.randn(d, d, device=device)  # (d, d)
        self.S_eff = self.S @ self.Pi.T                # (d, d)

        # codebook
        self.centroids = torch.tensor(
            solve_lloyd_max(d, b),
            dtype=torch.float32,
            device=device
        )                                               # (k,)

        # choose kernel variant
        self._use_tiled = (d > 1024)
        self._block_dim = min(next_pow2(d), 1024)

        # shared memory size: k floats (centroids) + block_dim floats (reduce)
        self._shmem_bytes = (self.k + self._block_dim) * 4

    def _launch_fused(self, y, x, idx, r_unit, gamma):
        n = x.shape[0]
        self.lib.launch_turbo_encode(
            ctypes.c_void_p(y.data_ptr()),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(self.centroids.data_ptr()),
            ctypes.c_void_p(idx.data_ptr()),
            ctypes.c_void_p(r_unit.data_ptr()),
            ctypes.c_void_p(gamma.data_ptr()),
            ctypes.c_int(n),
            ctypes.c_int(self.d),
            ctypes.c_int(self.k),
        )
    def encode(self, x: torch.Tensor):
        """
        Full TurboQuantProd encoding.
        x: (n, d) unit-norm fp32
        Returns: (idx, z, gamma, x_tilde_mse) for decoding
        """
        n = x.shape[0]

        # allocate outputs
        idx    = torch.zeros(n, self.d, dtype=torch.int16, device=self.device)
        r_unit = torch.zeros(n, self.d, device=self.device)
        gamma  = torch.zeros(n, device=self.device)

        # S1: rotate  [cuBLAS]
        y = x @ self.Pi.T                              # (n, d)

        # S2+S4+S5: fused codebook + residual + normalize
        self._launch_fused(y, x, idx, r_unit, gamma)
        torch.cuda.synchronize()

        # S3: dequantize MSE part  [embedding lookup + matmul]
        y_tilde     = self.centroids[idx.long()]       # (n, d)
        x_tilde_mse = y_tilde @ self.Pi               # (n, d)

        # S6: QJL on normalized rotated residual  [cuBLAS]
        # r_unit is in rotated basis, S_eff = S @ Pi.T
        z = torch.sign(r_unit @ self.S_eff.T)         # (n, d)
        z = torch.where(z == 0, torch.ones_like(z), z)

        # S7: QJL dequant  [cuBLAS]
        x_tilde_qjl = gamma.unsqueeze(-1) * (self.scale * (z @ self.S_eff))

        return x_tilde_mse + x_tilde_qjl

    def encode_baseline(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pure PyTorch baseline (no fused kernel) for comparison.
        Same mathematical operations, unfused.
        """
        # S1
        y = x @ self.Pi.T
        # S2 — materializes (n,d,k) distance tensor
        idx = (y.unsqueeze(-1) - self.centroids).abs().argmin(dim=-1)
        # S3
        x_tilde_mse = self.centroids[idx] @ self.Pi
        # S4
        r = x - x_tilde_mse
        # S5
        gamma  = torch.norm(r, dim=-1, keepdim=True)
        r_unit = F.normalize(r, dim=-1)
        # S6
        z = torch.sign(r_unit @ self.S.T)
        z = torch.where(z == 0, torch.ones_like(z), z)
        # S7
        x_tilde_qjl = gamma * (self.scale * (z @ self.S))
        return x_tilde_mse + x_tilde_qjl


# ── benchmark ────────────────────────────────────────────────────────────────

def benchmark(fn, n_warmup=20, n_repeat=100):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return np.mean(times), np.std(times)


def run_benchmark():
    device = torch.device("cuda")
    print("=" * 64)
    print("TurboQuant: Fused Kernel vs Baseline Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 64)

    print("\nCompiling kernel...")
    lib = compile_kernel("turbo_kernel.cu")

    configs = [
        (512,  4, 1),
        (512,  4, 16),
        (512,  4, 128),
        (512,  4, 1024),
        (512,  4, 4096),
        (512,  8, 1024),   # high bit-width — biggest win expected
        (1024, 4, 1024),
        (1536, 4, 1024),
    ]

    print(f"\n{'Config':<28} {'Baseline':>12} {'Fused':>12} {'Speedup':>10}")
    print("-" * 64)

    for d, b, n in configs:
        torch.manual_seed(42)
        x   = F.normalize(torch.randn(n, d, device=device), dim=-1)
        tq  = FusedTurboQuant(d, b, device, lib)

        t_base, _ = benchmark(lambda: tq.encode_baseline(x))
        t_fuse, _ = benchmark(lambda: tq.encode(x))
        speedup   = t_base / t_fuse

        print(f"d={d:4d} b={b} n={n:5d}  "
              f"{t_base:>10.4f}ms  "
              f"{t_fuse:>10.4f}ms  "
              f"{speedup:>9.2f}x")

    print("=" * 64)


if __name__ == "__main__":
    run_benchmark()
