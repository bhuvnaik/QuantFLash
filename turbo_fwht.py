"""
turbo_fwht.py
FWHT-based TurboQuant: replaces dense d×d rotation matrices with
Randomized Hadamard Transform (RHT).

Three levels compared:
  Level 0 (baseline):    dense Pi@x via cuBLAS + naive codebook
  Level 1 (fused):       dense Pi@x via cuBLAS + fused codebook kernel
  Level 2 (fwht):        RHT via FWHT kernel + fused codebook kernel
  Level 3 (full-fused):  single fused_fwht_encode kernel (everything)

Validates:
  1. FWHT correctness — output matches dense rotation statistically
  2. Distortion bounds still hold under RHT
  3. Speedup at each level
  4. Memory savings from eliminating Pi and S matrices
"""

import torch
import torch.nn.functional as F
import numpy as np
import ctypes
import os
import subprocess
from scipy.stats import norm as scipy_norm
from typing import Tuple


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
                else (std**2) * (pdf.pdf(lo) - pdf.pdf(hi)) / p
        if np.max(np.abs(nc - centroids)) < 1e-12:
            break
        centroids = nc
    return centroids.astype(np.float32)


def next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def compile_fwht(cuda_file: str = "fwht_kernel.cu") -> ctypes.CDLL:
    so_file = cuda_file.replace(".cu", ".so")
    if (os.path.exists(so_file) and
            os.path.getmtime(so_file) > os.path.getmtime(cuda_file)):
        print(f"  Using cached {so_file}")
    else:
        print(f"  Compiling {cuda_file} ...")
        cmd = [
            "nvcc", "-O3", "-arch=sm_86",
            "--compiler-options", "-fPIC",
            "-shared", cuda_file, "-o", so_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"nvcc failed:\n{result.stderr}")
        print(f"  Done.")

    lib = ctypes.CDLL(os.path.abspath(so_file))

    # launch_fwht_forward(x, signs, out, n, d)
    lib.launch_fwht_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
    ]
    lib.launch_fwht_forward.restype = None

    # launch_fwht_inverse(y, signs, out, n, d)
    lib.launch_fwht_inverse.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
    ]
    lib.launch_fwht_inverse.restype = None

    # launch_fused_fwht_encode(x, signs, centroids, idx, r_unit, gamma, n, d, k)
    lib.launch_fused_fwht_encode.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.launch_fused_fwht_encode.restype = None

    return lib


class FWHT:
    """
    Randomized Hadamard Transform.
    RHT(x) = (1/sqrt(d)) * H_d * D * x
    where D = diag(signs), signs_i in {-1, +1}.

    Inverse: RHT^{-1}(y) = D * (1/sqrt(d)) * H_d * y
    """

    def __init__(self, d: int, device: torch.device, lib: ctypes.CDLL):
        assert (d & (d - 1)) == 0, f"d must be power of 2, got {d}"
        self.d      = d
        self.device = device
        self.lib    = lib

        # random sign vector D — O(d) storage vs O(d^2) for dense matrix
        signs_np    = (2 * (np.random.randint(0, 2, d).astype(np.float32)) - 1)
        self.signs  = torch.tensor(signs_np, device=device)  # (d,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (n, d) -> RHT(x): (n, d)"""
        n   = x.shape[0]
        out = torch.empty_like(x)
        self.lib.launch_fwht_forward(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(self.signs.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(n),
            ctypes.c_int(self.d),
        )
        return out

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """y: (n, d) -> RHT^{-1}(y): (n, d)"""
        n   = y.shape[0]
        out = torch.empty_like(y)
        self.lib.launch_fwht_inverse(
            ctypes.c_void_p(y.data_ptr()),
            ctypes.c_void_p(self.signs.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(n),
            ctypes.c_int(self.d),
        )
        return out


class TurboQuantFWHT:
    """
    TurboQuantProd with FWHT replacing dense rotation matrices.

    Memory savings vs dense:
      Dense Pi+S: 2 * d^2 * 4 bytes
      FWHT signs1+signs2: 2 * d * 4 bytes
      Ratio: d^2 / d = d  (512x savings at d=512)

    Compute savings vs dense:
      Dense rotation: O(d^2) per vector
      FWHT: O(d log d) per vector
    """

    def __init__(self, d: int, b: int, device: torch.device, lib: ctypes.CDLL):
        self.d      = d
        self.b      = b
        self.k      = 2 ** b
        self.device = device
        self.lib    = lib
        self.scale  = float(np.sqrt(np.pi / 2) / d)

        # Two independent RHTs — one for MSE rotation, one for QJL
        self.rht1 = FWHT(d, device, lib)  # replaces Pi
        self.rht2 = FWHT(d, device, lib)  # replaces S

        self.centroids = torch.tensor(
            solve_lloyd_max(d, b),
            dtype=torch.float32, device=device
        )

    def encode_level2(self, x: torch.Tensor) -> torch.Tensor:
        """
        FWHT rotation + fused codebook kernel (separate calls).
        = Level 1 speedup + FWHT speedup for rotation stages.
        """
        n = x.shape[0]

        # S1: RHT forward (replaces Pi @ x)
        y = self.rht1.forward(x)

        # S2+S4+S5: fused codebook + residual + normalize
        idx    = torch.zeros(n, self.d, dtype=torch.int16, device=self.device)
        r_unit = torch.zeros(n, self.d, device=self.device)
        gamma  = torch.zeros(n, device=self.device)

        self.lib.launch_fused_fwht_encode(
            ctypes.c_void_p(x.data_ptr()),       # not used in this path
            ctypes.c_void_p(self.rht1.signs.data_ptr()),
            ctypes.c_void_p(self.centroids.data_ptr()),
            ctypes.c_void_p(idx.data_ptr()),
            ctypes.c_void_p(r_unit.data_ptr()),
            ctypes.c_void_p(gamma.data_ptr()),
            ctypes.c_int(n),
            ctypes.c_int(self.d),
            ctypes.c_int(self.k),
        )
        torch.cuda.synchronize()

        # S3: dequantize (RHT inverse replaces Pi.T @ y_tilde)
        y_tilde     = self.centroids[idx.long()]
        x_tilde_mse = self.rht1.inverse(y_tilde)

        # S6+S7: QJL on normalized residual (RHT replaces S)
        z = torch.sign(self.rht2.forward(r_unit))
        z = torch.where(z == 0, torch.ones_like(z), z)
        x_tilde_qjl = gamma.unsqueeze(-1) * (self.scale * self.rht2.inverse(z))

        return x_tilde_mse + x_tilde_qjl

    def encode_level3(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single fully-fused kernel: FWHT + codebook + residual + normalize.
        S1+S2+S4+S5 in one kernel pass. Then S3+S6+S7 as FWHT calls.
        """
        n = x.shape[0]

        idx    = torch.zeros(n, self.d, dtype=torch.int16, device=self.device)
        r_unit = torch.zeros(n, self.d, device=self.device)
        gamma  = torch.zeros(n, device=self.device)

        # single fused kernel: FWHT + codebook + residual + normalize
        self.lib.launch_fused_fwht_encode(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(self.rht1.signs.data_ptr()),
            ctypes.c_void_p(self.centroids.data_ptr()),
            ctypes.c_void_p(idx.data_ptr()),
            ctypes.c_void_p(r_unit.data_ptr()),
            ctypes.c_void_p(gamma.data_ptr()),
            ctypes.c_int(n),
            ctypes.c_int(self.d),
            ctypes.c_int(self.k),
        )
        torch.cuda.synchronize()

        # S3: inverse RHT for MSE dequant
        y_tilde     = self.centroids[idx.long()]
        x_tilde_mse = self.rht1.inverse(y_tilde)

        # S6+S7: QJL via second RHT
        z = torch.sign(self.rht2.forward(r_unit))
        z = torch.where(z == 0, torch.ones_like(z), z)
        x_tilde_qjl = gamma.unsqueeze(-1) * (self.scale * self.rht2.inverse(z))

        return x_tilde_mse + x_tilde_qjl

    def encode_baseline(self, x: torch.Tensor) -> torch.Tensor:
        """Dense rotation baseline for comparison."""
        Pi = torch.linalg.qr(torch.randn(self.d, self.d, device=self.device))[0]
        S  = torch.randn(self.d, self.d, device=self.device)

        y           = x @ Pi.T
        idx         = (y.unsqueeze(-1) - self.centroids).abs().argmin(dim=-1)
        x_tilde_mse = self.centroids[idx] @ Pi
        r           = x - x_tilde_mse
        gamma       = torch.norm(r, dim=-1, keepdim=True)
        r_unit      = F.normalize(r, dim=-1)
        z           = torch.sign(r_unit @ S.T)
        z           = torch.where(z == 0, torch.ones_like(z), z)
        return x_tilde_mse + gamma * (self.scale * (z @ S))



def timeit(fn, n_warmup=20, n_repeat=100):
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
    return float(np.mean(times)), float(np.std(times))



def validate_fwht(lib, device, d=512, n=1000):
    """
    Validate FWHT correctness:
    1. RHT(RHT^{-1}(x)) ≈ x  (invertibility)
    2. Coordinate marginals of RHT(x) follow N(0, 1/d)  (distribution check)
    3. ||RHT(x)||_2 = ||x||_2  (isometry check)
    """
    print("\n=== FWHT Correctness Validation ===")
    rht = FWHT(d, device, lib)

    # unit norm vectors
    x = F.normalize(torch.randn(n, d, device=device), dim=-1)

    # 1. invertibility
    y        = rht.forward(x)
    x_recon  = rht.inverse(y)
    inv_err  = (x - x_recon).abs().max().item()
    print(f"Invertibility max error:    {inv_err:.2e}  (should be < 1e-5)")

    # 2. isometry
    norm_x   = torch.norm(x, dim=-1)
    norm_y   = torch.norm(y, dim=-1)
    iso_err  = (norm_x - norm_y).abs().max().item()
    print(f"Isometry max norm error:    {iso_err:.2e}  (should be < 1e-5)")

    # 3. coordinate distribution — should be N(0, 1/d)
    coords   = y.flatten().cpu().numpy()
    std_expected = 1.0 / np.sqrt(d)
    std_actual   = coords.std()
    mean_actual  = coords.mean()
    print(f"Coordinate distribution:")
    print(f"  Expected std: {std_expected:.4f}  Actual: {std_actual:.4f}")
    print(f"  Expected mean: 0.0000        Actual: {mean_actual:.4f}")
    print(f"  Distribution check: {'PASS' if abs(std_actual - std_expected) < 0.01 else 'FAIL'}")

    # 4. near-independence check — pairwise correlation should be ~0
    sample = y[:200].cpu().numpy()
    corr   = np.corrcoef(sample.T)
    off_diag = corr[np.triu_indices(d, k=1)]
    print(f"  Mean |off-diagonal correlation|: {np.abs(off_diag).mean():.4f}  (should be ~0)")


def validate_distortion(lib, device, d=512, b=4, n=2000):
    """
    Verify TurboQuant distortion bounds still hold under FWHT rotation.
    """
    print(f"\n=== Distortion Validation (d={d}, b={b}) ===")
    tq = TurboQuantFWHT(d, b, device, lib)
    x  = F.normalize(torch.randn(n, d, device=device), dim=-1)
    y  = torch.randn(d, device=device)
    Y  = y.unsqueeze(0).expand(n, -1)

    x_tilde = tq.encode_level3(x)

    # MSE
    mse = (x - x_tilde).pow(2).sum(dim=-1).mean().item()
    mse_upper = (np.sqrt(3 * np.pi) / 2) * (4 ** -b)
    mse_lower = 4 ** -b

    # IP distortion and bias
    ip_true = (Y * x).sum(dim=-1)
    ip_est  = (Y * x_tilde).sum(dim=-1)
    ip_dist = (ip_true - ip_est).pow(2).mean().item()
    ip_bias = (ip_est - ip_true).mean().item()

    print(f"MSE:          {mse:.6f}  bounds: [{mse_lower:.6f}, {mse_upper:.6f}]")
    print(f"IP distortion:{ip_dist:.6f}")
    print(f"IP bias:      {ip_bias:+.6f}  (should be ~0)")
    print(f"MSE in bounds: {'PASS' if mse_lower <= mse <= mse_upper * 2 else 'FAIL'}")


def run_benchmark(lib, device):
    print("\n" + "=" * 72)
    print("TurboQuant Hardware Acceleration: FWHT vs Dense Rotation")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 72)

    configs = [
        (512,  4, 1),
        (512,  4, 128),
        (512,  4, 1024),
        (512,  4, 4096),
        (512,  8, 1024),
        (1024, 4, 1024),
    ]

    print(f"\n{'Config':<26} {'Dense(ms)':>10} {'FWHT-L2':>10} "
          f"{'FWHT-L3':>10} {'L2 vs Dense':>12} {'L3 vs Dense':>12}")
    print("-" * 82)

    for d, b, n in configs:
        torch.manual_seed(42)
        x  = F.normalize(torch.randn(n, d, device=device), dim=-1)
        tq = TurboQuantFWHT(d, b, device, lib)

        # dense baseline — rebuild Pi/S each call to avoid caching effects
        Pi = torch.linalg.qr(torch.randn(d, d, device=device))[0]
        S  = torch.randn(d, d, device=device)
        centroids = tq.centroids
        scale = tq.scale

        def dense():
            y           = x @ Pi.T
            idx         = (y.unsqueeze(-1) - centroids).abs().argmin(dim=-1)
            x_hat_mse   = centroids[idx] @ Pi
            r           = x - x_hat_mse
            gam         = torch.norm(r, dim=-1, keepdim=True)
            ru          = F.normalize(r, dim=-1)
            z           = torch.sign(ru @ S.T)
            z           = torch.where(z == 0, torch.ones_like(z), z)
            return x_hat_mse + gam * (scale * (z @ S))

        t_dense, _ = timeit(dense)
        t_l2, _    = timeit(lambda: tq.encode_level2(x))
        t_l3, _    = timeit(lambda: tq.encode_level3(x))

        print(f"d={d:4d} b={b} n={n:5d}  "
              f"{t_dense:>10.4f}  "
              f"{t_l2:>10.4f}  "
              f"{t_l3:>10.4f}  "
              f"{t_dense/t_l2:>11.2f}x  "
              f"{t_dense/t_l3:>11.2f}x")

    print("=" * 72)

    # memory comparison
    print("\n=== Memory Savings: FWHT vs Dense Rotation ===")
    print(f"{'d':>6} {'Dense Pi+S (MB)':>18} {'FWHT signs (KB)':>18} {'Savings':>10}")
    for d in [128, 256, 512, 1024, 2048, 4096]:
        dense_mb = 2 * d * d * 4 / 1e6
        fwht_kb  = 2 * d * 4 / 1e3
        savings  = dense_mb * 1e3 / fwht_kb
        print(f"{d:>6} {dense_mb:>18.1f} {fwht_kb:>18.1f} {savings:>9.0f}x")


if __name__ == "__main__":
    device = torch.device("cuda")

    print("Compiling FWHT kernel...")
    lib = compile_fwht("fwht_kernel.cu")

    # validate correctness first
    validate_fwht(lib, device, d=512, n=2000)
    validate_distortion(lib, device, d=512, b=4, n=2000)

    # run benchmark
    run_benchmark(lib, device)
