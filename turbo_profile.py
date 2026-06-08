"""
TurboQuant Phase 1: Overhead Profiling
Measures per-stage latency, memory bandwidth, and arithmetic intensity
for TurboQuantMSE and TurboQuantProd on the RTX A5000.

Stages profiled:
  [MSE]  S1: rotation      Pi @ x
  [MSE]  S2: codebook      nearest centroid lookup
  [MSE]  S3: derotation    y_tilde @ Pi
  [PROD] S4: residual      x - x_tilde_mse
  [PROD] S5: normalize     r / ||r||
  [PROD] S6: QJL quant     sign(S @ r_unit)
  [PROD] S7: QJL dequant   scale * z @ S

Outputs:
  - results/profile_<timestamp>.json   raw numbers
  - results/profile_<timestamp>.txt    human-readable report
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import time
from scipy.stats import norm as scipy_norm
from typing import Dict, List


def solve_lloyd_max(d: int, b: int, n_iter: int = 2000) -> np.ndarray:
    k = 2 ** b
    std = 1.0 / np.sqrt(d)
    pdf = scipy_norm(0, std)
    centroids = pdf.ppf(np.linspace(0.5 / k, 1 - 0.5 / k, k))
    for _ in range(n_iter):
        bounds = np.concatenate([
            [-np.inf], (centroids[:-1] + centroids[1:]) / 2, [np.inf]
        ])
        new_c = np.zeros(k)
        for i in range(k):
            lo, hi = bounds[i], bounds[i + 1]
            p = pdf.cdf(hi) - pdf.cdf(lo)
            new_c[i] = centroids[i] if p < 1e-12 \
                else (std**2) * (pdf.pdf(lo) - pdf.pdf(hi)) / p
        if np.max(np.abs(new_c - centroids)) < 1e-12:
            break
        centroids = new_c
    return centroids.astype(np.float32)


class GPUTimer:
    """
    Accurate GPU kernel timing using CUDA events.
    Always synchronizes before/after to get true kernel time,
    not just launch time.
    """
    def __init__(self, n_warmup: int = 20, n_repeat: int = 100):
        self.n_warmup = n_warmup
        self.n_repeat = n_repeat

    def measure(self, fn, *args) -> Dict:
        """
        Returns dict with mean_ms, std_ms, min_ms, max_ms.
        """
        # warmup — important for CUDA JIT and caching effects
        for _ in range(self.n_warmup):
            fn(*args)
        torch.cuda.synchronize()

        times = []
        for _ in range(self.n_repeat):
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(*args)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # milliseconds

        times = np.array(times)
        return {
            "mean_ms": float(np.mean(times)),
            "std_ms":  float(np.std(times)),
            "min_ms":  float(np.min(times)),
            "max_ms":  float(np.max(times)),
        }



def ai_matmul(n: int, d: int) -> float:
    """
    Arithmetic intensity (FLOP/byte) for (n,d) @ (d,d) -> (n,d).
    FLOPs: 2*n*d*d  (multiply-add)
    Bytes: load A(n*d) + load B(d*d) + store C(n*d), all fp32 = *4
    """
    flops = 2 * n * d * d
    bytes_ = 4 * (n * d + d * d + n * d)
    return flops / bytes_


def ai_codebook(n: int, d: int, k: int) -> float:
    """
    Arithmetic intensity for nearest-centroid lookup.
    For each of n*d elements, compute distance to k centroids.
    FLOPs: n*d*k  (one subtraction + abs per centroid)
    Bytes: load x(n*d) + load centroids(k) + store idx(n*d int32)
    All fp32 except idx which is int32 (same size).
    """
    flops  = n * d * k
    bytes_ = 4 * (n * d + k + n * d)
    return flops / bytes_


def ai_elementwise(n: int, d: int, ops_per_element: int = 1) -> float:
    """
    Arithmetic intensity for elementwise ops (residual, normalize, sign).
    Bandwidth-bound: AI ~ ops_per_element / (2*4) bytes (read+write fp32).
    """
    flops  = ops_per_element * n * d
    bytes_ = 4 * 2 * n * d   # read input + write output
    return flops / bytes_



# RTX A5000 specs
A5000_TFLOPS_FP32  = 27.8    # TFLOPS
A5000_BW_GBs       = 768.0   # GB/s memory bandwidth
A5000_RIDGE_POINT  = (A5000_TFLOPS_FP32 * 1e12) / (A5000_BW_GBs * 1e9)  # FLOP/byte

def roofline_bound(ai: float, n_flops: float) -> float:
    """
    Returns roofline performance bound in TFLOPS given arithmetic intensity
    and the operation's FLOP count.
    """
    perf_compute  = A5000_TFLOPS_FP32
    perf_bw       = ai * A5000_BW_GBs / 1e3   # TFLOPS
    return min(perf_compute, perf_bw)


def actual_tflops(flops: float, time_ms: float) -> float:
    return flops / (time_ms * 1e-3) / 1e12


def bytes_moved(n: int, d: int, stage: str, k: int = 0) -> float:
    """Theoretical bytes moved for each stage (fp32 = 4 bytes)."""
    if stage == "matmul":        # (n,d) @ (d,d)
        return 4 * (n*d + d*d + n*d)
    elif stage == "codebook":    # (n,d) nearest neighbor in k centroids
        return 4 * (n*d + k + n*d)
    elif stage == "elementwise": # read + write
        return 4 * 2 * n * d
    return 0.0



def profile_all_stages(
    d: int,
    b: int,
    n: int,
    device: torch.device,
    timer: GPUTimer,
) -> Dict:
    """
    Profile each stage of TurboQuantMSE and TurboQuantProd independently.
    Returns dict of timing + roofline results.
    """
    k = 2 ** b

    torch.manual_seed(42)
    x  = F.normalize(torch.randn(n, d, device=device), dim=-1)
    Pi = torch.linalg.qr(torch.randn(d, d, device=device))[0]
    S  = torch.randn(d, d, device=device)
    centroids = torch.tensor(solve_lloyd_max(d, b), device=device)

    results = {"d": d, "b": b, "n": n, "stages": {}}

    # S1: rotation  y = x @ Pi.T 
    def s1(): return x @ Pi.T
    t = timer.measure(s1)
    flops = 2 * n * d * d
    ai    = ai_matmul(n, d)
    results["stages"]["S1_rotation"] = {
        **t,
        "flops": flops,
        "ai_flop_per_byte": ai,
        "actual_tflops": actual_tflops(flops, t["mean_ms"]),
        "roofline_tflops": roofline_bound(ai, flops),
        "bound": "compute" if ai >= A5000_RIDGE_POINT else "bandwidth",
        "bytes_theoretical": bytes_moved(n, d, "matmul"),
        "bandwidth_GBs": bytes_moved(n, d, "matmul") / (t["mean_ms"] * 1e-3) / 1e9,
    }

    # S2: codebook lookup
    y = x @ Pi.T
    def s2():
        dists = (y.unsqueeze(-1) - centroids).abs()
        return dists.argmin(dim=-1)
    t  = timer.measure(s2)
    fl = n * d * k
    ai = ai_codebook(n, d, k)
    results["stages"]["S2_codebook"] = {
        **t,
        "flops": fl,
        "ai_flop_per_byte": ai,
        "actual_tflops": actual_tflops(fl, t["mean_ms"]),
        "roofline_tflops": roofline_bound(ai, fl),
        "bound": "compute" if ai >= A5000_RIDGE_POINT else "bandwidth",
        "bytes_theoretical": bytes_moved(n, d, "codebook", k),
        "bandwidth_GBs": bytes_moved(n, d, "codebook", k) / (t["mean_ms"] * 1e-3) / 1e9,
    }

   
    idx     = (y.unsqueeze(-1) - centroids).abs().argmin(dim=-1)
    y_tilde = centroids[idx]
    def s3(): return y_tilde @ Pi
    t  = timer.measure(s3)
    fl = 2 * n * d * d
    ai = ai_matmul(n, d)
    results["stages"]["S3_derotation"] = {
        **t,
        "flops": fl,
        "ai_flop_per_byte": ai,
        "actual_tflops": actual_tflops(fl, t["mean_ms"]),
        "roofline_tflops": roofline_bound(ai, fl),
        "bound": "compute" if ai >= A5000_RIDGE_POINT else "bandwidth",
        "bytes_theoretical": bytes_moved(n, d, "matmul"),
        "bandwidth_GBs": bytes_moved(n, d, "matmul") / (t["mean_ms"] * 1e-3) / 1e9,
    }

    
    x_tilde_mse = y_tilde @ Pi
    def s4(): return x - x_tilde_mse
    t  = timer.measure(s4)
    fl = n * d
    ai = ai_elementwise(n, d, 1)
    results["stages"]["S4_residual"] = {
        **t,
        "flops": fl,
        "ai_flop_per_byte": ai,
        "actual_tflops": actual_tflops(fl, t["mean_ms"]),
        "roofline_tflops": roofline_bound(ai, fl),
        "bound": "bandwidth",
        "bytes_theoretical": bytes_moved(n, d, "elementwise"),
        "bandwidth_GBs": bytes_moved(n, d, "elementwise") / (t["mean_ms"] * 1e-3) / 1e9,
    }

   
    r = x - x_tilde_mse
    def s5(): return F.normalize(r, dim=-1)
    t  = timer.measure(s5)
    fl = 2 * n * d   # sqrt + divide
    results["stages"]["S5_normalize"] = {
        **t,
        "flops": fl,
        "ai_flop_per_byte": ai_elementwise(n, d, 2),
        "actual_tflops": actual_tflops(fl, t["mean_ms"]),
        "roofline_tflops": roofline_bound(ai_elementwise(n, d, 2), fl),
        "bound": "bandwidth",
        "bytes_theoretical": bytes_moved(n, d, "elementwise"),
        "bandwidth_GBs": bytes_moved(n, d, "elementwise") / (t["mean_ms"] * 1e-3) / 1e9,
    }

   
    r_unit = F.normalize(r, dim=-1)
    def s6(): return torch.sign(r_unit @ S.T)
    t  = timer.measure(s6)
    fl = 2 * n * d * d
    ai = ai_matmul(n, d)
    results["stages"]["S6_qjl_quant"] = {
        **t,
        "flops": fl,
        "ai_flop_per_byte": ai,
        "actual_tflops": actual_tflops(fl, t["mean_ms"]),
        "roofline_tflops": roofline_bound(ai, fl),
        "bound": "compute" if ai >= A5000_RIDGE_POINT else "bandwidth",
        "bytes_theoretical": bytes_moved(n, d, "matmul"),
        "bandwidth_GBs": bytes_moved(n, d, "matmul") / (t["mean_ms"] * 1e-3) / 1e9,
    }

    z     = torch.sign(r_unit @ S.T)
    scale = np.sqrt(np.pi / 2) / d
    def s7(): return scale * (z @ S)
    t  = timer.measure(s7)
    fl = 2 * n * d * d
    ai = ai_matmul(n, d)
    results["stages"]["S7_qjl_dequant"] = {
        **t,
        "flops": fl,
        "ai_flop_per_byte": ai,
        "actual_tflops": actual_tflops(fl, t["mean_ms"]),
        "roofline_tflops": roofline_bound(ai, fl),
        "bound": "compute" if ai >= A5000_RIDGE_POINT else "bandwidth",
        "bytes_theoretical": bytes_moved(n, d, "matmul"),
        "bandwidth_GBs": bytes_moved(n, d, "matmul") / (t["mean_ms"] * 1e-3) / 1e9,
    }

    def e2e_mse():
        y_   = x @ Pi.T
        idx_ = (y_.unsqueeze(-1) - centroids).abs().argmin(dim=-1)
        return centroids[idx_] @ Pi

    def e2e_prod():
        y_      = x @ Pi.T
        idx_    = (y_.unsqueeze(-1) - centroids).abs().argmin(dim=-1)
        xt_mse  = centroids[idx_] @ Pi
        r_      = x - xt_mse
        gamma_  = torch.norm(r_, dim=-1, keepdim=True)
        r_unit_ = F.normalize(r_, dim=-1)
        z_      = torch.sign(r_unit_ @ S.T)
        xt_qjl  = gamma_ * (scale * (z_ @ S))
        return xt_mse + xt_qjl

    t_mse  = timer.measure(e2e_mse)
    t_prod = timer.measure(e2e_prod)
    results["e2e_mse_ms"]  = t_mse["mean_ms"]
    results["e2e_prod_ms"] = t_prod["mean_ms"]

    # memory footprint
    bytes_per_float = 4
    results["memory"] = {
        "input_MB":      n * d * bytes_per_float / 1e6,
        "rotation_MB":   d * d * bytes_per_float / 1e6,  # Pi
        "qjl_matrix_MB": d * d * bytes_per_float / 1e6,  # S
        "codebook_MB":   k * bytes_per_float / 1e6,
        "output_mse_MB": n * d * bytes_per_float / 1e6,
        # TurboQuantProd output: idx (int32) + z (fp32) + gamma (fp32)
        "output_prod_MB": (n*d*4 + n*d*4 + n*4) / 1e6,
    }

    return results



def print_report(all_results: List[Dict]) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("TurboQuant Phase 1: Overhead Profiling Report")
    lines.append(f"GPU: NVIDIA RTX A5000  |  "
                 f"Peak FP32: {A5000_TFLOPS_FP32} TFLOPS  |  "
                 f"BW: {A5000_BW_GBs} GB/s")
    lines.append(f"Ridge point: {A5000_RIDGE_POINT:.1f} FLOP/byte")
    lines.append("=" * 72)

    for res in all_results:
        d, b, n = res["d"], res["b"], res["n"]
        lines.append(f"\n{'─'*72}")
        lines.append(f"d={d}  b={b} bits  n={n} vectors  "
                     f"(input: {res['memory']['input_MB']:.1f} MB)")
        lines.append(f"{'─'*72}")
        lines.append(f"{'Stage':<20} {'Time(ms)':>10} {'AI':>8} "
                     f"{'Bound':<12} {'Act.TFLOPS':>12} {'BW(GB/s)':>10}")
        lines.append(f"{'─'*72}")

        stage_labels = {
            "S1_rotation":   "S1 rotation",
            "S2_codebook":   "S2 codebook",
            "S3_derotation": "S3 derotation",
            "S4_residual":   "S4 residual",
            "S5_normalize":  "S5 normalize",
            "S6_qjl_quant":  "S6 QJL quant",
            "S7_qjl_dequant":"S7 QJL dequant",
        }

        for key, label in stage_labels.items():
            s = res["stages"][key]
            lines.append(
                f"{label:<20} {s['mean_ms']:>10.4f} "
                f"{s['ai_flop_per_byte']:>8.2f} "
                f"{s['bound']:<12} "
                f"{s['actual_tflops']:>12.4f} "
                f"{s['bandwidth_GBs']:>10.1f}"
            )

        lines.append(f"{'─'*72}")
        lines.append(f"  End-to-end MSE:  {res['e2e_mse_ms']:.4f} ms")
        lines.append(f"  End-to-end Prod: {res['e2e_prod_ms']:.4f} ms")
        lines.append(f"  Memory: Pi={res['memory']['rotation_MB']:.1f}MB  "
                     f"S={res['memory']['qjl_matrix_MB']:.1f}MB  "
                     f"codebook={res['memory']['codebook_MB']:.3f}MB")

    lines.append("\n" + "=" * 72)
    return "\n".join(lines)



def scaling_sweep(device: torch.device, timer: GPUTimer) -> List[Dict]:
    """
    Sweep over (d, b, n) configurations to characterize scaling behavior.
    Covers realistic KV cache dimensions from small to large models.
    """
    configs = [
        # (d,    b, n)      # representative use cases
        (128,   4, 1024),   # small model, short context
        (256,   4, 1024),
        (512,   4, 1024),   # ESM-2 650M attention dim
        (1024,  4, 1024),   # large model attention dim
        (512,   4, 4096),   # longer context
        (512,   4, 8192),
        (512,   2, 1024),   # varying bit-width
        (512,   3, 1024),
        (512,   4, 1024),
        (1536,  4, 1024),   # OpenAI embedding dim (paper's experiment)
    ]

    results = []
    for d, b, n in configs:
        print(f"  profiling d={d}, b={b}, n={n} ...", flush=True)
        try:
            r = profile_all_stages(d, b, n, device, timer)
            results.append(r)
        except RuntimeError as e:
            print(f"    SKIPPED (OOM or error): {e}")
    return results



if __name__ == "__main__":
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    print()

    timer = GPUTimer(n_warmup=20, n_repeat=100)

    print("Running scaling sweep...")
    results = scaling_sweep(device, timer)

    # save results
    os.makedirs("results", exist_ok=True)
    ts = int(time.time())

    json_path = f"results/profile_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results: {json_path}")

    report = print_report(results)
    print(report)

    txt_path = f"results/profile_{ts}.txt"
    with open(txt_path, "w") as f:
        f.write(report)
    print(f"Saved report: {txt_path}")
