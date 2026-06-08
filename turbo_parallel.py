"""
Parallel K and V TurboQuant encoding using CUDA streams.

Sequential (current):   encode(K) → encode(V)   cost = 2 * t_encode
Parallel  (this file):  stream1: encode(K)
                        stream2: encode(V)        cost = t_encode + overhead

Both streams run concurrently on the GPU — K and V are independent tensors,
no data dependency between them.
"""

import torch
import numpy as np
import ctypes, os, sys

os.chdir('/scratch/bhuvanc/turboquant')
sys.path.insert(0, '/scratch/bhuvanc/turboquant')

from turbo_fwht import compile_fwht
from turbo_pipeline import TurboQuantGPU
from triton_turbo import turbo_fused_kernel


class TurboQuantParallel(TurboQuantGPU):
    """
    Extends TurboQuantGPU with parallel K+V encoding on two CUDA streams.
    """

    def __init__(self, d, b, device, fwht_lib, seed=42):
        super().__init__(d, b, device, fwht_lib, seed)
        # two independent CUDA streams
        self.stream_k = torch.cuda.Stream(device=device)
        self.stream_v = torch.cuda.Stream(device=device)

    def encode_kv_parallel(self, k: torch.Tensor, v: torch.Tensor):
        """
        Encode K and V concurrently on separate CUDA streams.

        k: (n_k, d)  key vectors
        v: (n_v, d)  value vectors  (n_k == n_v in practice)

        Returns: (k_idx, k_r_unit, k_gamma, k_gamma_r,
                  v_idx, v_r_unit, v_gamma, v_gamma_r)
        """
        n_k, n_v = k.shape[0], v.shape[0]

        # pre-allocate outputs on GPU for both streams
        k_idx    = torch.zeros(n_k, self.d, dtype=torch.int16, device=self.device)
        k_runit  = torch.zeros(n_k, self.d, dtype=torch.float32, device=self.device)
        k_gamma  = torch.zeros(n_k, dtype=torch.float32, device=self.device)
        k_gammar = torch.zeros(n_k, dtype=torch.float32, device=self.device)

        v_idx    = torch.zeros(n_v, self.d, dtype=torch.int16, device=self.device)
        v_runit  = torch.zeros(n_v, self.d, dtype=torch.float32, device=self.device)
        v_gamma  = torch.zeros(n_v, dtype=torch.float32, device=self.device)
        v_gammar = torch.zeros(n_v, dtype=torch.float32, device=self.device)

       
        with torch.cuda.stream(self.stream_k):
            norms_k  = torch.norm(k, dim=-1, keepdim=True)
            k_gamma.copy_(norms_k.squeeze(-1))
            k_unit   = k / torch.clamp(norms_k, min=1e-8)
            y_k      = self.srht.forward(k_unit)
            turbo_fused_kernel[(n_k,)](
                y_k, self.centroids, k_idx, k_runit, k_gammar,
                d=self.d, k=self.k, BLOCK_D=self.d,
            )

        
        with torch.cuda.stream(self.stream_v):
            norms_v  = torch.norm(v, dim=-1, keepdim=True)
            v_gamma.copy_(norms_v.squeeze(-1))
            v_unit   = v / torch.clamp(norms_v, min=1e-8)
            y_v      = self.srht.forward(v_unit)
            turbo_fused_kernel[(n_v,)](
                y_v, self.centroids, v_idx, v_runit, v_gammar,
                d=self.d, k=self.k, BLOCK_D=self.d,
            )

    
        torch.cuda.current_stream().wait_stream(self.stream_k)
        torch.cuda.current_stream().wait_stream(self.stream_v)

        return (k_idx, k_runit, k_gamma, k_gammar,
                v_idx, v_runit, v_gamma, v_gammar)

    def encode_kv_sequential(self, k: torch.Tensor, v: torch.Tensor):
        """Sequential baseline: encode K then V on default stream."""
        k_out = self.encode(k)
        v_out = self.encode(v)
        return (*k_out, *v_out)



def validate_parallel(tqp, k, v, tol=1e-4):
    print(f"\n=== Correctness check (n={k.shape[0]}, d={k.shape[1]}) ===")

    # parallel
    (k_idx_p, k_ru_p, k_gam_p, k_gr_p,
     v_idx_p, v_ru_p, v_gam_p, v_gr_p) = tqp.encode_kv_parallel(k, v)
    torch.cuda.synchronize()

    # sequential reference
    (k_idx_s, k_ru_s, k_gam_s, k_gr_s,
     v_idx_s, v_ru_s, v_gam_s, v_gr_s) = tqp.encode_kv_sequential(k, v)
    torch.cuda.synchronize()

    k_idx_ok  = (k_idx_p == k_idx_s).all().item()
    v_idx_ok  = (v_idx_p == v_idx_s).all().item()
    k_gam_err = (k_gam_p - k_gam_s).abs().max().item()
    v_gam_err = (v_gam_p - v_gam_s).abs().max().item()
    k_ru_err  = (k_ru_p  - k_ru_s ).abs().max().item()
    v_ru_err  = (v_ru_p  - v_ru_s ).abs().max().item()

    print(f"  K idx match:     {k_idx_ok}")
    print(f"  V idx match:     {v_idx_ok}")
    print(f"  K gamma max err: {k_gam_err:.2e}")
    print(f"  V gamma max err: {v_gam_err:.2e}")
    print(f"  K r_unit maxerr: {k_ru_err:.2e}")
    print(f"  V r_unit maxerr: {v_ru_err:.2e}")

    passed = k_idx_ok and v_idx_ok and k_gam_err < tol and v_gam_err < tol
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


def benchmark(tqp, k_all):
    print("\n" + "="*72)
    print("Parallel K+V Encoding Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*72)

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

    print(f"\n{'Config':<22} {'Sequential(ms)':>15} {'Parallel(ms)':>13} "
          f"{'Speedup':>9} {'Saved(ms)':>10}")
    print("-"*72)

    for n, label in configs:
        if n > len(k_all): continue
        k = torch.tensor(k_all[:n], device=tqp.device, dtype=torch.float32)
        v = torch.tensor(k_all[:n], device=tqp.device, dtype=torch.float32)

        t_seq, _  = timeit(lambda: tqp.encode_kv_sequential(k, v))
        t_par, _  = timeit(lambda: tqp.encode_kv_parallel(k, v))

        speedup = t_seq / t_par
        saved   = t_seq - t_par

        print(f"{label:<22} {t_seq:>15.4f} {t_par:>13.4f} "
              f"{speedup:>8.2f}x {saved:>10.4f}")

    # theoretical max: perfect parallelism → speedup = 2x
    print(f"\nTheoretical max speedup: 2.00x (perfect overlap)")
    print(f"(Limited by: kernel launch overhead, SM availability, stream scheduling)")

    # show what this means for full inference
    print(f"\n--- Impact on full decode step ---")
    print(f"Decode step total:      ~22.7ms  (weight-bound)")
    n_dec = 28 * 8  # n_layers * n_kv_heads, one new token
    k_dec = torch.tensor(k_all[:n_dec], device=tqp.device, dtype=torch.float32)
    v_dec = torch.tensor(k_all[:n_dec], device=tqp.device, dtype=torch.float32)
    t_seq_dec, _ = timeit(lambda: tqp.encode_kv_sequential(k_dec, v_dec))
    t_par_dec, _ = timeit(lambda: tqp.encode_kv_parallel(k_dec, v_dec))
    print(f"Sequential K+V encode:  {t_seq_dec:.4f}ms  ({t_seq_dec/22.7*100:.2f}% of decode)")
    print(f"Parallel   K+V encode:  {t_par_dec:.4f}ms  ({t_par_dec/22.7*100:.2f}% of decode)")
    print(f"Time saved per token:   {t_seq_dec-t_par_dec:.4f}ms")

    print("="*72)


if __name__ == "__main__":
    device   = torch.device("cuda")
    fwht_lib = compile_fwht("fwht_kernel.cu")
    tqp      = TurboQuantParallel(d=128, b=4, device=device, fwht_lib=fwht_lib)

    k_all = np.load('/scratch/bhuvanc/kv_vectors/k_all_vectors.npy').astype(np.float32)
    v_all = np.load('/scratch/bhuvanc/kv_vectors/v_all_vectors.npy').astype(np.float32)

    # correctness
    k_val = torch.tensor(k_all[:64], device=device)
    v_val = torch.tensor(v_all[:64], device=device)
    ok = validate_parallel(tqp, k_val, v_val)

    if ok:
        print("\nCorrectness PASSED. Running benchmark...")
        benchmark(tqp, k_all)
    else:
        print("\nCorrectness FAILED — check stream isolation.")
