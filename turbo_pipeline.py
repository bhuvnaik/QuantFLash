"""
TurboQuant Full GPU Pipeline
Combines GPU SRHT (double sign FWHT) + Triton fused S2+S3+S5+S6

Stages:
  [GPU FWHT]   S0: norm extraction + normalize
  [GPU FWHT]   S1: SRHT forward  D2 @ H @ D1 @ x_unit
  [Triton]     S2+S3+S5+S6: codebook + norm_correction + residual + normalize
  [GPU FWHT]   S4: SRHT inverse  D1 @ H @ D2 @ y_tilde  (for MSE reconstruction)

Baseline comparison: CPU numpy SRHT + PyTorch S2
"""

import torch
import torch.nn.functional as F
import numpy as np
import triton
import triton.language as tl
from scipy.stats import norm as scipy_norm
import ctypes, os, sys

sys.path.insert(0, '/scratch/bhuvanc/turboquant_plus')
from turboquant.rotation import random_rotation_fast

os.chdir('/scratch/bhuvanc/turboquant')
from turbo_fwht import compile_fwht
from triton_turbo import turbo_fused_kernel, solve_lloyd_max


# ── Double-sign FWHT (SRHT matching turboquant_plus) ─────────────────────────

class SRHT_GPU:
    """
    GPU implementation of double-sign SRHT:
        forward:  y = D2 @ (1/sqrt(d)) @ H @ D1 @ x
        inverse:  x = D1 @ (1/sqrt(d)) @ H @ D2 @ y

    Uses our fwht_kernel.cu with two sign vectors.
    The kernel applies signs1 before butterfly, signs2 after.
    Forward: signs1=D1, signs2=D2
    Inverse: signs1=D2, signs2=D1  (reversed order)
    """

    def __init__(self, d: int, device: torch.device,
                 lib: ctypes.CDLL, seed: int = 42):
        assert (d & (d-1)) == 0, "d must be power of 2"
        self.d      = d
        self.device = device
        self.lib    = lib

        rng = np.random.default_rng(seed)
        signs1, signs2, _ = random_rotation_fast(d, rng)
        # truncate to d (random_rotation_fast pads to next pow2)
        self.D1 = torch.tensor(signs1[:d], dtype=torch.float32, device=device)
        self.D2 = torch.tensor(signs2[:d], dtype=torch.float32, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (n, d) → D2 @ H @ D1 @ x / sqrt(d)"""
        n   = x.shape[0]
        out = torch.empty_like(x)
        # apply D1 before butterfly, D2 after
        # our fwht_kernel: launch_fwht_forward(x, signs, out, n, d)
        # signs is applied before butterfly — so we need two calls
        # or we inline: apply D1 manually, then kernel with signs=D2

        # Step 1: apply D1 elementwise
        x_d1 = x * self.D1   # (n, d)

        # Step 2: FWHT butterfly + scale + apply D2
        # our kernel does: out = sign_after * WHT(sign_before * x) / sqrt(d)
        # set sign_before = ones (D1 already applied), sign_after = D2
        ones = torch.ones(self.d, device=self.device)
        self.lib.launch_fwht_forward(
            ctypes.c_void_p(x_d1.data_ptr()),
            ctypes.c_void_p(ones.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(n),
            ctypes.c_int(self.d),
        )
        # apply D2 after
        out = out * self.D2
        return out

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """y: (n, d) → D1 @ H @ D2 @ y / sqrt(d)"""
        n   = y.shape[0]
        out = torch.empty_like(y)

        # apply D2 before butterfly
        y_d2 = y * self.D2

        ones = torch.ones(self.d, device=self.device)
        self.lib.launch_fwht_inverse(
            ctypes.c_void_p(y_d2.data_ptr()),
            ctypes.c_void_p(ones.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(n),
            ctypes.c_int(self.d),
        )
        # apply D1 after
        out = out * self.D1
        return out


# ── Full GPU TurboQuant pipeline ──────────────────────────────────────────────

class TurboQuantGPU:
    """
    Complete TurboQuant encoding pipeline fully on GPU.

    S0: norm extraction            (GPU elementwise)
    S1: SRHT forward D2@H@D1      (GPU FWHT kernel)
    S2+S3+S5+S6: fused Triton     (Triton kernel)
    S4: SRHT inverse (optional)    (GPU FWHT kernel)
    """

    def __init__(self, d: int, b: int, device: torch.device,
                 fwht_lib: ctypes.CDLL, seed: int = 42):
        self.d      = d
        self.b      = b
        self.k      = 2**b
        self.device = device

        self.srht = SRHT_GPU(d, device, fwht_lib, seed)
        self.centroids = torch.tensor(
            solve_lloyd_max(d, b), dtype=torch.float32, device=device
        )

    def encode(self, x: torch.Tensor):
        """
        x: (n, d) fp32 real KV vectors (not unit norm)
        Returns: idx, r_unit, gamma, gamma_r
        """
        n = x.shape[0]

        # S0: norm extraction (GPU)
        norms   = torch.norm(x, dim=-1, keepdim=True)          # (n,1)
        gamma   = norms.squeeze(-1)                             # (n,)
        x_unit  = x / torch.clamp(norms, min=1e-8)             # (n,d)

        # S1: SRHT forward (GPU FWHT)
        y = self.srht.forward(x_unit)                          # (n,d)

        # S2+S3+S5+S6: Triton fused kernel
        idx     = torch.zeros(n, self.d, dtype=torch.int16, device=self.device)
        r_unit  = torch.zeros(n, self.d, dtype=torch.float32, device=self.device)
        gamma_r = torch.zeros(n, dtype=torch.float32, device=self.device)

        turbo_fused_kernel[(n,)](
            y, self.centroids, idx, r_unit, gamma_r,
            d=self.d, k=self.k, BLOCK_D=self.d,
        )
        torch.cuda.synchronize()

        return idx, r_unit, gamma, gamma_r

    def encode_baseline_cpu(self, x: torch.Tensor):
        """CPU SRHT + GPU codebook baseline (current state of the art)."""
        from turboquant.rotation import apply_fast_rotation_batch
        n    = x.shape[0]
        x_np = x.cpu().numpy()

        # S0: CPU norm extraction
        norms  = np.linalg.norm(x_np, axis=1, keepdims=True)
        gamma  = torch.tensor(norms.squeeze(), device=self.device, dtype=torch.float32)
        x_unit = (x_np / np.where(norms > 1e-8, norms, 1.0)).astype(np.float32)

        # S1: CPU SRHT
        s1 = self.srht.D1.cpu().numpy()
        s2 = self.srht.D2.cpu().numpy()
        y  = apply_fast_rotation_batch(x_unit, s1, s2, self.d)
        y_gpu = torch.tensor(y, device=self.device, dtype=torch.float32)

        # S2: GPU codebook (unfused baseline)
        idx = (y_gpu.unsqueeze(-1) - self.centroids).abs().argmin(dim=-1).short()

        # S3+S5+S6: PyTorch unfused
        y_tilde = self.centroids[idx.long()]
        yt_norm = torch.norm(y_tilde, dim=-1, keepdim=True)
        y_tilde = y_tilde / torch.clamp(yt_norm, min=1e-8)
        r_rot   = y_gpu - y_tilde
        gamma_r = torch.norm(r_rot, dim=-1)
        r_unit  = r_rot / torch.clamp(gamma_r.unsqueeze(-1), min=1e-8)

        return idx, r_unit, gamma, gamma_r


# ── correctness check ─────────────────────────────────────────────────────────

def validate(tq: TurboQuantGPU, x: torch.Tensor, tol: float = 1e-2):
    print(f"\n=== Correctness (n={x.shape[0]}, d={x.shape[1]}, b={tq.b}) ===")

    idx_g, ru_g, gam_g, gr_g = tq.encode(x)
    idx_b, ru_b, gam_b, gr_b = tq.encode_baseline_cpu(x)

    idx_ok = (idx_g == idx_b).float().mean().item()
    gam_ok = (gam_g - gam_b).abs().max().item()
    gr_ok  = (gr_g  - gr_b ).abs().max().item()
    ru_ok  = (ru_g  - ru_b ).abs().max().item()

    print(f"  idx agreement:   {idx_ok*100:.1f}%  (expect ~100%)")
    print(f"  gamma max diff:  {gam_ok:.6f}")
    print(f"  gamma_r max diff:{gr_ok:.6f}")
    print(f"  r_unit max diff: {ru_ok:.6f}")

    passed = idx_ok > 0.99 and gam_ok < tol and ru_ok < tol
    print(f"  PASS: {passed}")
    return passed


# ── benchmark ─────────────────────────────────────────────────────────────────

def timeit(fn, n_warmup=20, n_repeat=100):
    for _ in range(n_warmup): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.mean(times)), float(np.std(times))


def benchmark(tq: TurboQuantGPU, device, real_kv_path=None):
    print("\n" + "="*72)
    print("TurboQuant Full GPU Pipeline vs CPU SRHT Baseline")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"d={tq.d}, b={tq.b}, k={tq.k}")
    print("="*72)

    if real_kv_path and os.path.exists(real_kv_path):
        k_all = np.load(real_kv_path).astype(np.float32)
        print(f"Real KV vectors: {k_all.shape}")
    else:
        k_all = (np.random.randn(34272, tq.d) * 44).astype(np.float32)
        print(f"Synthetic vectors: {k_all.shape}")

    configs = [
        (1,     "decode  n=1",    "decode"),
        (8,     "decode  n=8",    "decode"),
        (32,    "decode  n=32",   "decode"),
        (128,   "decode  n=128",  "decode"),
        (512,   "prefill n=512",  "prefill"),
        (1024,  "prefill n=1K",   "prefill"),
        (4096,  "prefill n=4K",   "prefill"),
        (8192,  "prefill n=8K",   "prefill"),
        (34272, "prefill n=full", "prefill"),
    ]

    print(f"\n{'Config':<22} {'Phase':<8} {'CPU-SRHT(ms)':>13} "
          f"{'GPU-full(ms)':>13} {'Speedup':>9} {'Bottleneck':<15}")
    print("-"*85)

    for n, label, phase in configs:
        if n > len(k_all): continue
        x = torch.tensor(k_all[:n], device=device, dtype=torch.float32)

        t_cpu, _ = timeit(lambda: tq.encode_baseline_cpu(x))
        t_gpu, _ = timeit(lambda: tq.encode(x))

        speedup = t_cpu / t_gpu

        # identify bottleneck in GPU pipeline
        norms  = torch.norm(x, dim=-1, keepdim=True)
        x_unit = x / torch.clamp(norms, min=1e-8)

        t_srht,   _ = timeit(lambda: tq.srht.forward(x_unit))
        y = tq.srht.forward(x_unit)
        idx_t   = torch.zeros(n, tq.d, dtype=torch.int16, device=device)
        ru_t    = torch.zeros(n, tq.d, dtype=torch.float32, device=device)
        gr_t    = torch.zeros(n, dtype=torch.float32, device=device)
        t_triton, _ = timeit(lambda: turbo_fused_kernel[(n,)](
            y, tq.centroids, idx_t, ru_t, gr_t,
            d=tq.d, k=tq.k, BLOCK_D=tq.d,
        ))

        if t_srht > t_triton * 2:
            bottleneck = "SRHT"
        elif t_triton > t_srht * 2:
            bottleneck = "Triton"
        else:
            bottleneck = "launch overhead"

        print(f"{label:<22} {phase:<8} {t_cpu:>13.4f} "
              f"{t_gpu:>13.4f} {speedup:>8.2f}x  {bottleneck:<15}")

    # stage breakdown for key configs
    print(f"\n--- Stage breakdown (GPU pipeline) ---")
    print(f"{'Config':<22} {'S0 norm':>10} {'S1 SRHT':>10} "
          f"{'S2-S6 Triton':>14} {'Total':>10}")
    print("-"*68)

    for n, label, phase in [(1,"decode n=1","decode"),
                             (128,"decode n=128","decode"),
                             (1024,"prefill 1K","prefill"),
                             (4096,"prefill 4K","prefill")]:
        x = torch.tensor(k_all[:n], device=device, dtype=torch.float32)

        norms  = torch.norm(x, dim=-1, keepdim=True)
        x_unit = x / torch.clamp(norms, min=1e-8)
        y      = tq.srht.forward(x_unit)
        idx_t  = torch.zeros(n, tq.d, dtype=torch.int16, device=device)
        ru_t   = torch.zeros(n, tq.d, dtype=torch.float32, device=device)
        gr_t   = torch.zeros(n, dtype=torch.float32, device=device)

        t_norm,   _ = timeit(lambda: torch.norm(x, dim=-1, keepdim=True))
        t_srht,   _ = timeit(lambda: tq.srht.forward(x_unit))
        t_triton, _ = timeit(lambda: turbo_fused_kernel[(n,)](
            y, tq.centroids, idx_t, ru_t, gr_t,
            d=tq.d, k=tq.k, BLOCK_D=tq.d,
        ))
        t_total,  _ = timeit(lambda: tq.encode(x))

        print(f"{label:<22} {t_norm:>10.4f} {t_srht:>10.4f} "
              f"{t_triton:>14.4f} {t_total:>10.4f}")

    print("="*72)


if __name__ == "__main__":
    device = torch.device("cuda")

    print("Compiling FWHT kernel...")
    fwht_lib = compile_fwht("fwht_kernel.cu")

    tq = TurboQuantGPU(d=128, b=4, device=device, fwht_lib=fwht_lib)

    # validate on real KV vectors
    k_real = np.load('/scratch/bhuvanc/kv_vectors/k_all_vectors.npy').astype(np.float32)
    x_val  = torch.tensor(k_real[:64], device=device)
    ok = validate(tq, x_val)

    if ok:
        benchmark(tq, device, '/scratch/bhuvanc/kv_vectors/k_all_vectors.npy')
    else:
        print("Correctness failed — fix before benchmarking.")
