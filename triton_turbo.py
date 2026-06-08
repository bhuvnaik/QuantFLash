"""
Triton fused kernel for TurboQuant encoding pipeline.

Fuses S2+S3+S5+S6 in one Triton kernel (no butterfly inside kernel):
  S2: codebook lookup     idx = argmin_k |y_j - c_k|
  S3: norm correction     ỹ_unit = centroids[idx] / ||centroids[idx]||
  S5: residual in rotated space  r_rot = y - ỹ_unit
  S6: norm + normalize    gamma_r = ||r_rot||, r_unit = r_rot / gamma_r

S0 (norm extraction), S1 (SRHT forward), S4 (SRHT inverse) are done
outside the kernel in PyTorch — they are fast elementwise + FWHT ops.

One program per input vector. BLOCK_D threads per block = d.
No tl.gather across warp boundaries — avoids Triton 3.3.1 compiler bug.
"""

import torch
import torch.nn.functional as F
import numpy as np
import triton
import triton.language as tl
from scipy.stats import norm as scipy_norm
import sys, os

sys.path.insert(0, '/scratch/bhuvanc/turboquant_plus')
from turboquant.rotation import (
    random_rotation_fast,
    apply_fast_rotation_batch,
    apply_fast_rotation_transpose,
)


def solve_lloyd_max(d: int, b: int, n_iter: int = 2000) -> np.ndarray:
    k = 2**b; std = 1/np.sqrt(d)
    pdf = scipy_norm(0, std)
    c = pdf.ppf(np.linspace(0.5/k, 1-0.5/k, k))
    for _ in range(n_iter):
        bds = np.concatenate([[-np.inf], (c[:-1]+c[1:])/2, [np.inf]])
        nc = np.zeros(k)
        for i in range(k):
            lo, hi = bds[i], bds[i+1]
            p = pdf.cdf(hi) - pdf.cdf(lo)
            nc[i] = c[i] if p < 1e-12 else (std**2)*(pdf.pdf(lo)-pdf.pdf(hi))/p
        if np.max(np.abs(nc-c)) < 1e-12: break
        c = nc
    return c.astype(np.float32)


@triton.jit
def turbo_fused_kernel(
    Y_ptr,        # (n, d) fp32  rotated unit-norm vectors (SRHT output)
    C_ptr,        # (k,)   fp32  Lloyd-Max centroids
    IDX_ptr,      # (n, d) int16 output codebook indices
    RUNIT_ptr,    # (n, d) fp32  output normalized residual (rotated basis)
    GAMMA_R_ptr,  # (n,)   fp32  output residual norms
    d: tl.constexpr,
    k: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    One program per vector. BLOCK_D = d threads.
    Each thread owns one coordinate throughout.

    No cross-warp gather — all operations are elementwise per thread
    except tl.sum (reduction) which Triton handles natively.
    """
    vec_id = tl.program_id(0)
    tid    = tl.arange(0, BLOCK_D)
    base   = vec_id * d


    y = tl.load(Y_ptr + base + tid)   # (d,) — each thread owns y[tid]

    # No gather needed — we iterate over centroids as scalars
    best_dist = tl.full((BLOCK_D,), float('inf'), dtype=tl.float32)
    best_idx  = tl.zeros((BLOCK_D,), dtype=tl.int16)

    for ci in tl.static_range(k):
        c_val = tl.load(C_ptr + ci)                        # scalar
        dist  = tl.abs(y - c_val)                          # (d,) elementwise
        update = dist < best_dist
        best_dist = tl.where(update, dist, best_dist)
        best_idx  = tl.where(update,
                             tl.full((BLOCK_D,), ci, dtype=tl.int16),
                             best_idx)

    tl.store(IDX_ptr + base + tid, best_idx)


    # y_tilde[tid] = centroids[best_idx[tid]]
    # Use tl.static_range to avoid gather — thread tid checks all k centroids
    y_tilde = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for ci in tl.static_range(k):
        c_val   = tl.load(C_ptr + ci)
        is_best = best_idx == ci
        y_tilde = tl.where(is_best, tl.full((BLOCK_D,), c_val, dtype=tl.float32), y_tilde)

    # S3: norm correction — renormalize y_tilde to unit norm
    yt_norm_sq = tl.sum(y_tilde * y_tilde, axis=0)
    yt_norm    = tl.sqrt(yt_norm_sq)
    yt_inv     = tl.where(yt_norm > 1e-8, 1.0 / yt_norm, 0.0)
    y_tilde    = y_tilde * yt_inv
    # r_rot = y - y_tilde  (both unit-norm, so ||r_rot|| < 2)
    r_rot = y - y_tilde

    r_norm_sq = tl.sum(r_rot * r_rot, axis=0)
    gamma_r   = tl.sqrt(r_norm_sq)
    inv_gr    = tl.where(gamma_r > 1e-8, 1.0 / gamma_r, 0.0)
    r_unit    = r_rot * inv_gr

    tl.store(GAMMA_R_ptr + vec_id, gamma_r)
    tl.store(RUNIT_ptr + base + tid, r_unit)


class TritonTurboQuant:
    """
    TurboQuant encoder with Triton-fused S2+S3+S5+S6.

    Pipeline:
      [NumPy CPU]  S0: norm extraction + normalize
      [NumPy CPU]  S1: SRHT forward (D2 @ H @ D1)
      [Triton GPU] S2+S3+S5+S6: codebook + norm_correction + residual + normalize
      [NumPy CPU]  S4: SRHT inverse (for MSE reconstruction, separate)

    The Triton kernel handles the bottleneck stages on GPU.
    SRHT is done on CPU matching turboquant_plus reference implementation.
    """

    def __init__(self, d: int, b: int, device: torch.device, seed: int = 42):
        assert (d & (d-1)) == 0 and d <= 1024
        self.d = d
        self.b = b
        self.k = 2**b
        self.device = device

        rng = np.random.default_rng(seed)
        self.signs1, self.signs2, self.padded_d = random_rotation_fast(d, rng)

        self.centroids = torch.tensor(
            solve_lloyd_max(d, b), dtype=torch.float32, device=device
        )
        self.centroids_np = self.centroids.cpu().numpy()

    def _srht_forward_batch(self, x_unit: np.ndarray) -> np.ndarray:
        return apply_fast_rotation_batch(
            x_unit, self.signs1, self.signs2, self.padded_d
        ).astype(np.float32)

    def _srht_inverse_batch(self, y: np.ndarray) -> np.ndarray:
        return np.stack([
            apply_fast_rotation_transpose(y[i], self.signs1, self.signs2, self.padded_d)
            for i in range(len(y))
        ]).astype(np.float32)

    def encode(self, x: torch.Tensor):
        """
        Full encode pipeline.
        x: (n, d) fp32 real KV vectors (not unit norm)

        Returns:
            idx:     (n, d) int16
            r_unit:  (n, d) fp32  normalized residual in rotated basis
            gamma:   (n,)   fp32  original norms
            gamma_r: (n,)   fp32  residual norms
        """
        n = x.shape[0]
        x_np = x.cpu().numpy()

        # S0: norm extraction
        norms   = np.linalg.norm(x_np, axis=1, keepdims=True)
        gamma   = norms.squeeze().astype(np.float32)
        x_unit  = (x_np / np.where(norms > 1e-8, norms, 1.0)).astype(np.float32)

        # S1: SRHT forward (CPU)
        y = self._srht_forward_batch(x_unit)   # (n, d)

        # move y to GPU for Triton kernel
        y_gpu   = torch.tensor(y, device=self.device, dtype=torch.float32)
        idx     = torch.zeros(n, self.d, dtype=torch.int16, device=self.device)
        r_unit  = torch.zeros(n, self.d, dtype=torch.float32, device=self.device)
        gamma_r = torch.zeros(n, dtype=torch.float32, device=self.device)

        # S2+S3+S5+S6: Triton fused kernel
        grid = (n,)
        turbo_fused_kernel[grid](
            y_gpu, self.centroids,
            idx, r_unit, gamma_r,
            d=self.d, k=self.k, BLOCK_D=self.d,
        )
        torch.cuda.synchronize()

        return (idx,
                r_unit,
                torch.tensor(gamma, device=self.device),
                gamma_r)

    def encode_baseline(self, x: torch.Tensor):
        """Pure PyTorch/NumPy baseline for correctness comparison."""
        n = x.shape[0]
        x_np = x.cpu().numpy()

        # S0
        norms  = np.linalg.norm(x_np, axis=1, keepdims=True)
        gamma  = norms.squeeze().astype(np.float32)
        x_unit = (x_np / np.where(norms > 1e-8, norms, 1.0)).astype(np.float32)

        # S1
        y = self._srht_forward_batch(x_unit)
        y_t = torch.tensor(y, device=self.device)

        # S2
        idx = (y_t.unsqueeze(-1) - self.centroids).abs().argmin(dim=-1).short()

        # S3: norm correction
        y_tilde = self.centroids[idx.long()]
        yt_norm = torch.norm(y_tilde, dim=-1, keepdim=True)
        y_tilde = y_tilde / torch.clamp(yt_norm, min=1e-8)

        # S5: residual in rotated space
        r_rot   = y_t - y_tilde

        # S6
        gr      = torch.norm(r_rot, dim=-1)
        r_unit  = r_rot / torch.clamp(gr.unsqueeze(-1), min=1e-8)

        return (idx,
                r_unit,
                torch.tensor(gamma, device=self.device),
                gr)


def validate(tq, x, tol=1e-3):
    print(f"\n=== Correctness (n={x.shape[0]}, d={x.shape[1]}, b={tq.b}) ===")
    idx_f, ru_f, gam_f, gr_f = tq.encode(x)
    idx_b, ru_b, gam_b, gr_b = tq.encode_baseline(x)

    idx_ok = (idx_f == idx_b).all().item()
    gam_ok = (gam_f - gam_b).abs().max().item()
    gr_ok  = (gr_f  - gr_b ).abs().max().item()
    ru_ok  = (ru_f  - ru_b ).abs().max().item()

    print(f"  idx match:       {idx_ok}")
    print(f"  gamma max diff:  {gam_ok:.6f}")
    print(f"  gamma_r max diff:{gr_ok:.6f}")
    print(f"  r_unit max diff: {ru_ok:.6f}")

    passed = idx_ok and gam_ok < tol and gr_ok < tol and ru_ok < tol
    print(f"  PASS: {passed}")
    return passed



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


def benchmark(device, real_kv_path=None):
    print("\n" + "="*70)
    print("TurboQuant Triton Kernel Benchmark: Real KV Vectors")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*70)

    d, b = 128, 4
    tq   = TritonTurboQuant(d, b, device)
    C    = tq.centroids

    # load real KV vectors
    if real_kv_path and os.path.exists(real_kv_path):
        k_all = np.load(real_kv_path).astype(np.float32)
        print(f"Using real KV vectors: {k_all.shape}")
    else:
        k_all = (np.random.randn(34272, d) * 44).astype(np.float32)
        print(f"Using synthetic vectors: {k_all.shape}")

    # precompute SRHT-rotated vectors for GPU-only timing
    norms  = np.linalg.norm(k_all, axis=1, keepdims=True)
    x_unit = (k_all / np.where(norms > 1e-8, norms, 1.0)).astype(np.float32)
    y_all  = tq._srht_forward_batch(x_unit)

    configs = [
        # (n,      label,           phase)
        (1,        "decode n=1",    "decode"),
        (8,        "decode n=8",    "decode"),
        (32,       "decode n=32",   "decode"),
        (128,      "decode n=128",  "decode"),
        (512,      "prefill 512",   "prefill"),
        (1024,     "prefill 1K",    "prefill"),
        (4096,     "prefill 4K",    "prefill"),
        (8192,     "prefill 8K",    "prefill"),
        (34272,    "prefill full",  "prefill"),
    ]

    print(f"\n{'Config':<22} {'Phase':<8} "
          f"{'Baseline S2(ms)':>16} {'Triton(ms)':>12} {'Speedup':>9} {'BW saved':>10}")
    print("-"*80)

    for n, label, phase in configs:
        if n > len(y_all):
            continue

        y_gpu = torch.tensor(y_all[:n], device=device, dtype=torch.float32)
        x_gpu = torch.tensor(k_all[:n], device=device, dtype=torch.float32)

        # baseline: unfused S2 only (the bottleneck we're replacing)
        def baseline_s2():
            return (y_gpu.unsqueeze(-1) - C).abs().argmin(dim=-1)

        # triton: fused S2+S3+S5+S6 (GPU part only, excludes CPU SRHT)
        idx_out   = torch.zeros(n, d, dtype=torch.int16, device=device)
        ru_out    = torch.zeros(n, d, dtype=torch.float32, device=device)
        gr_out    = torch.zeros(n, dtype=torch.float32, device=device)

        def triton_fused():
            turbo_fused_kernel[(n,)](
                y_gpu, C, idx_out, ru_out, gr_out,
                d=d, k=tq.k, BLOCK_D=d,
            )

        t_base, _ = timeit(baseline_s2)
        t_trit, _ = timeit(triton_fused)

        # theoretical bytes: baseline reads (n,d,k) tensor; triton reads (n,d)
        bw_base = n * d * tq.k * 4   # distance tensor
        bw_trit = n * d * 4          # just y
        bw_ratio = bw_base / bw_trit

        print(f"{label:<22} {phase:<8} "
              f"{t_base:>15.4f}  {t_trit:>12.4f}  "
              f"{t_base/t_trit:>8.2f}x  {bw_ratio:>9.1f}x")

    # also show full encode (including CPU SRHT) for context
    print(f"\n--- Full encode pipeline (including CPU SRHT) ---")
    print(f"{'Config':<22} {'Triton full(ms)':>16} {'Baseline full(ms)':>18}")
    print("-"*58)

    for n, label, phase in [(1,"decode n=1","decode"),
                             (128,"decode n=128","decode"),
                             (4096,"prefill 4K","prefill")]:
        x_sub = torch.tensor(k_all[:n], device=device)

        t_full, _ = timeit(lambda: tq.encode(x_sub))
        t_base, _ = timeit(lambda: tq.encode_baseline(x_sub))

        print(f"{label:<22} {t_full:>15.4f}  {t_base:>18.4f}")

    print("="*70)


if __name__ == "__main__":
    device = torch.device("cuda")

    print("=== TurboQuant Triton Fused Kernel ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # correctness on real KV vectors
    k_real = np.load('/scratch/bhuvanc/kv_vectors/k_all_vectors.npy').astype(np.float32)
    x_small = torch.tensor(k_real[:32], device=device)

    tq = TritonTurboQuant(d=128, b=4, device=device)
    ok = validate(tq, x_small)

    if ok:
        print("\nCorrectness PASSED. Running benchmark...")
        benchmark(device, '/scratch/bhuvanc/kv_vectors/k_all_vectors.npy')
    else:
        print("\nCorrectness FAILED.")
