"""
Fused K+V TurboQuant kernel.

Instead of two separate encode() calls (2x launch overhead),
one kernel processes K and V vectors interleaved:
  thread block i encodes K[i] if i < n
  thread block i+n encodes V[i-n] if i >= n

One kernel launch, same arithmetic, half the overhead.
"""

import torch
import numpy as np
import triton
import triton.language as tl
import os, sys

os.chdir('/scratch/bhuvanc/turboquant')
sys.path.insert(0, '/scratch/bhuvanc/turboquant')

from turbo_fwht import compile_fwht
from turbo_pipeline import TurboQuantGPU
from triton_turbo import turbo_fused_kernel, solve_lloyd_max
from scipy.stats import norm as scipy_norm


@triton.jit
def turbo_kv_fused_kernel(
    K_ptr, V_ptr,
    C_ptr,
    K_idx_ptr, K_runit_ptr, K_gammar_ptr,
    V_idx_ptr, V_runit_ptr, V_gammar_ptr,
    n,
    d: tl.constexpr,
    k: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Grid: (2*n,)  — first n blocks handle K, next n blocks handle V.
    Each block encodes one vector: codebook lookup + norm correction
    + residual + normalize (S2+S3+S5+S6).

    Identical arithmetic to turbo_fused_kernel, just dispatches
    K vs V based on block index.
    """
    pid = tl.program_id(0)
    is_v = pid >= n
    row  = pid - n * is_v   # index within K or V

    offs = tl.arange(0, BLOCK_D)

    # load input vector (K or V)
    base_ptr = tl.where(is_v, V_ptr + row * d, K_ptr + row * d)
    y = tl.load(base_ptr + offs)

    # load codebook
    c = tl.load(C_ptr + tl.arange(0, k))  # (k,)

    # S2: find nearest centroid per coordinate
    # broadcast y (BLOCK_D,) vs c (k,) → distances (BLOCK_D, k)
    y_exp = tl.reshape(y,    [BLOCK_D, 1])
    c_exp = tl.reshape(c,    [1,       k])
    dist  = tl.abs(y_exp - c_exp)          # (BLOCK_D, k)
    idx   = tl.argmin(dist, axis=1)        # (BLOCK_D,)  int

    # S3: norm correction — build y_tilde from centroids, then unit-normalize
    y_tilde_raw = tl.load(C_ptr + idx)     # (BLOCK_D,) centroid values
    yt_sq_sum   = tl.sum(y_tilde_raw * y_tilde_raw, axis=0)
    yt_norm     = tl.sqrt(yt_sq_sum + 1e-16)
    y_tilde     = y_tilde_raw / yt_norm    # unit-normalized

    # S5: residual in rotated space
    r_rot  = y - y_tilde                   # (BLOCK_D,)

    # S6: residual norm and unit direction
    r_sq   = tl.sum(r_rot * r_rot, axis=0)
    gamma_r = tl.sqrt(r_sq + 1e-16)
    r_unit  = r_rot / gamma_r

    # write outputs to correct tensor (K or V)
    idx_ptr    = tl.where(is_v, V_idx_ptr    + row * d, K_idx_ptr    + row * d)
    runit_ptr  = tl.where(is_v, V_runit_ptr  + row * d, K_runit_ptr  + row * d)
    gammar_ptr = tl.where(is_v, V_gammar_ptr + row,     K_gammar_ptr + row)

    tl.store(idx_ptr   + offs, idx.to(tl.int16))
    tl.store(runit_ptr + offs, r_unit)
    tl.store(gammar_ptr,       gamma_r)


class TurboQuantKVFused(TurboQuantGPU):
    """
    Encodes K and V in a single fused Triton kernel launch.
    S0 (norm extraction) and S1 (SRHT) still run separately
    since they need to complete before the fused codebook step.
    """

    def encode_kv_fused(self, k: torch.Tensor, v: torch.Tensor):
        """
        k: (n, d), v: (n, d) — must be same shape
        Returns: (k_idx, k_runit, k_gamma, k_gammar,
                  v_idx, v_runit, v_gamma, v_gammar)
        """
        assert k.shape == v.shape
        n = k.shape[0]

        # S0: norm extraction (both)
        norms_k = torch.norm(k, dim=-1, keepdim=True)
        norms_v = torch.norm(v, dim=-1, keepdim=True)
        k_gamma = norms_k.squeeze(-1)
        v_gamma = norms_v.squeeze(-1)
        k_unit  = k / torch.clamp(norms_k, min=1e-8)
        v_unit  = v / torch.clamp(norms_v, min=1e-8)

        # S1: SRHT (both) — two calls but cheap
        y_k = self.srht.forward(k_unit)
        y_v = self.srht.forward(v_unit)

        # S2+S3+S5+S6: single fused kernel for both K and V
        k_idx    = torch.zeros(n, self.d, dtype=torch.int16,   device=self.device)
        k_runit  = torch.zeros(n, self.d, dtype=torch.float32, device=self.device)
        k_gammar = torch.zeros(n,         dtype=torch.float32, device=self.device)
        v_idx    = torch.zeros(n, self.d, dtype=torch.int16,   device=self.device)
        v_runit  = torch.zeros(n, self.d, dtype=torch.float32, device=self.device)
        v_gammar = torch.zeros(n,         dtype=torch.float32, device=self.device)

        # grid = 2*n: first n blocks = K, next n blocks = V
        turbo_kv_fused_kernel[(2 * n,)](
            y_k, y_v,
            self.centroids,
            k_idx, k_runit, k_gammar,
            v_idx, v_runit, v_gammar,
            n,
            d=self.d, k=self.k, BLOCK_D=self.d,
        )
        torch.cuda.synchronize()

        return (k_idx, k_runit, k_gamma, k_gammar,
                v_idx, v_runit, v_gamma, v_gammar)

    def encode_kv_sequential(self, k, v):
        k_out = self.encode(k)
        v_out = self.encode(v)
        return (*k_out, *v_out)


def validate(tq, k, v, tol=1e-4):
    print(f"\n=== Correctness (n={k.shape[0]}, d={k.shape[1]}) ===")

    (ki_f, kr_f, kg_f, kgr_f,
     vi_f, vr_f, vg_f, vgr_f) = tq.encode_kv_fused(k, v)

    (ki_s, kr_s, kg_s, kgr_s,
     vi_s, vr_s, vg_s, vgr_s) = tq.encode_kv_sequential(k, v)

    k_idx_ok = (ki_f == ki_s).all().item()
    v_idx_ok = (vi_f == vi_s).all().item()
    k_gr_err = (kgr_f - kgr_s).abs().max().item()
    v_gr_err = (vgr_f - vgr_s).abs().max().item()
    k_ru_err = (kr_f  - kr_s ).abs().max().item()
    v_ru_err = (vr_f  - vr_s ).abs().max().item()

    print(f"  K idx match:     {k_idx_ok}")
    print(f"  V idx match:     {v_idx_ok}")
    print(f"  K gamma_r err:   {k_gr_err:.2e}")
    print(f"  V gamma_r err:   {v_gr_err:.2e}")
    print(f"  K r_unit err:    {k_ru_err:.2e}")
    print(f"  V r_unit err:    {v_ru_err:.2e}")

    passed = k_idx_ok and v_idx_ok and k_gr_err < tol and v_gr_err < tol
    print(f"  PASS: {passed}")
    return passed


def timeit(fn, n_warmup=30, n_repeat=200):
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


def benchmark(tq, k_all, v_all):
    print("\n" + "="*74)
    print("Fused K+V vs Sequential K+V Encoding")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*74)

    configs = [
        (1,     "decode  n=1   "),
        (8,     "decode  n=8   "),
        (32,    "decode  n=32  "),
        (128,   "decode  n=128 "),
        (256,   "decode  n=256 "),
        (512,   "prefill n=512 "),
        (1024,  "prefill n=1K  "),
        (4096,  "prefill n=4K  "),
        (8192,  "prefill n=8K  "),
        (34272, "prefill n=full"),
    ]

    print(f"\n{'Config':<22} {'Sequential(ms)':>15} {'Fused(ms)':>11} "
          f"{'Speedup':>9} {'Saved(ms)':>10}")
    print("-"*70)

    for n, label in configs:
        if n > len(k_all): continue
        k = torch.tensor(k_all[:n], device=tq.device, dtype=torch.float32)
        v = torch.tensor(v_all[:n], device=tq.device, dtype=torch.float32)

        t_seq, _ = timeit(lambda: tq.encode_kv_sequential(k, v))
        t_fus, _ = timeit(lambda: tq.encode_kv_fused(k, v))

        speedup = t_seq / t_fus
        saved   = t_seq - t_fus

        print(f"{label:<22} {t_seq:>15.4f} {t_fus:>11.4f} "
              f"{speedup:>8.2f}x {saved:>10.4f}")

    print(f"\nTheoretical max speedup: ~1.5x")
    print(f"(S0+S1 still sequential, only S2-S6 fused into one launch)")

    # decode context
    n_dec = 28 * 8
    k_d   = torch.tensor(k_all[:n_dec], device=tq.device, dtype=torch.float32)
    v_d   = torch.tensor(v_all[:n_dec], device=tq.device, dtype=torch.float32)
    t_seq_d, _ = timeit(lambda: tq.encode_kv_sequential(k_d, v_d))
    t_fus_d, _ = timeit(lambda: tq.encode_kv_fused(k_d, v_d))

    print(f"\n--- Decode context (n={n_dec} = 28 layers × 8 KV heads) ---")
    print(f"Sequential:  {t_seq_d:.4f}ms  ({t_seq_d/22.7*100:.2f}% of 22.7ms decode)")
    print(f"Fused:       {t_fus_d:.4f}ms  ({t_fus_d/22.7*100:.2f}% of 22.7ms decode)")
    print(f"Saved:       {t_seq_d - t_fus_d:.4f}ms per token")
    print("="*74)


if __name__ == "__main__":
    device   = torch.device("cuda")
    fwht_lib = compile_fwht("fwht_kernel.cu")
    tq       = TurboQuantKVFused(d=128, b=4, device=device, fwht_lib=fwht_lib)

    k_all = np.load('/scratch/bhuvanc/kv_vectors/k_all_vectors.npy').astype(np.float32)
    v_all = np.load('/scratch/bhuvanc/kv_vectors/v_all_vectors.npy').astype(np.float32)

    k_val = torch.tensor(k_all[:64], device=device)
    v_val = torch.tensor(v_all[:64], device=device)
    ok = validate(tq, k_val, v_val)

    if ok:
        print("\nCorrectness PASSED. Running benchmark...")
        benchmark(tq, k_all, v_all)
    else:
        print("\nCorrectness FAILED.")


def encode_kv_fully_fused(tq, k: torch.Tensor, v: torch.Tensor):
    """
    Maximum fusion: single norm+SRHT pass over [K;V], single Triton launch.
    Optimal at decode and prefill n <= ~8K.
    At large prefill (n > 8K) sequential is preferred due to cat overhead.
    """
    from triton_turbo import turbo_fused_kernel
    n   = k.shape[0]
    kv  = torch.cat([k, v], dim=0)
    norms   = torch.norm(kv, dim=-1, keepdim=True)
    gamma   = norms.squeeze(-1)
    kv_unit = kv / torch.clamp(norms, min=1e-8)
    y_kv    = tq.srht.forward(kv_unit)

    idx    = torch.zeros(2*n, tq.d, dtype=torch.int16,   device=tq.device)
    runit  = torch.zeros(2*n, tq.d, dtype=torch.float32, device=tq.device)
    gammar = torch.zeros(2*n,       dtype=torch.float32, device=tq.device)

    turbo_fused_kernel[(2*n,)](
        y_kv, tq.centroids, idx, runit, gammar,
        d=tq.d, k=tq.k, BLOCK_D=tq.d,
    )
    torch.cuda.synchronize()

    return (idx[:n], runit[:n], gamma[:n], gammar[:n],
            idx[n:], runit[n:], gamma[n:], gammar[n:])
