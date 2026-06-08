import torch
import torch.nn.functional as F
import numpy as np
import ctypes
import os
import subprocess
from scipy.stats import norm as scipy_norm


def solve_lloyd_max(d, b, n_iter=2000):
    k = 2**b; std = 1/np.sqrt(d); pdf = scipy_norm(0, std)
    c = pdf.ppf(np.linspace(0.5/k, 1-0.5/k, k))
    for _ in range(n_iter):
        bds = np.concatenate([[-np.inf], (c[:-1]+c[1:])/2, [np.inf]])
        nc = np.zeros(k)
        for i in range(k):
            lo, hi = bds[i], bds[i+1]; p = pdf.cdf(hi)-pdf.cdf(lo)
            nc[i] = c[i] if p < 1e-12 else (std**2)*(pdf.pdf(lo)-pdf.pdf(hi))/p
        if np.max(np.abs(nc-c)) < 1e-12: break
        c = nc
    return c.astype(np.float32)


def compile_lib(cuda_file):
    so_file = cuda_file.replace(".cu", ".so")
    if (os.path.exists(so_file) and
            os.path.getmtime(so_file) > os.path.getmtime(cuda_file)):
        print(f"  Using cached {so_file}")
    else:
        print(f"  Compiling {cuda_file} ...")
        cmd = ["nvcc", "-O3", "-arch=sm_86",
               "--compiler-options", "-fPIC",
               "-shared", cuda_file, "-o", so_file]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc failed:\n{r.stderr}")
        print(f"  Done.")
    lib = ctypes.CDLL(os.path.abspath(so_file))

    lib.launch_quant_attn.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float,
    ]
    lib.launch_quant_attn.restype = None

    lib.launch_pack_khat.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
    ]
    lib.launch_pack_khat.restype = None

    lib.launch_pack_z.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
    ]
    lib.launch_pack_z.restype = None

    return lib


class QuantAttn:
    """
    Fused quantized attention using TurboQuant compressed KV cache.

    Compressed key format per token:
      k_hat:   d/2 bytes  (4-bit MSE indices, 2 per byte)
      z_packed: d/8 bytes  (1-bit QJL signs, 8 per byte)
      gamma:   4 bytes    (fp32 residual norm)
    Total: d/2 + d/8 + 4 bytes vs d*2 bytes for fp16

    At d=128: 84 bytes vs 256 bytes = 3.0x compression
    """

    def __init__(self, d, b, device, attn_lib, fwht_lib):
        assert d in [64, 128, 256, 512], f"d must be 64/128/256/512, got {d}"
        assert (d & (d-1)) == 0, "d must be power of 2"
        self.d       = d
        self.b       = b
        self.k       = 2**b
        self.device  = device
        self.lib     = attn_lib
        self.scale   = float(np.sqrt(np.pi/2) / d)
        self.qk_scale = float(1.0 / np.sqrt(d))

    # FWHT sign vectors — one for MSE rotation, one for QJL
        signs_np       = (2*(np.random.randint(0,2,d).astype(np.float32))-1)
        self.signs     = torch.tensor(signs_np, device=device)
        signs_np2      = (2*(np.random.randint(0,2,d).astype(np.float32))-1)
        self.signs_qjl = torch.tensor(signs_np2, device=device)

        # Lloyd-Max codebook
        self.centroids = torch.tensor(
            solve_lloyd_max(d, b), dtype=torch.float32, device=device
        )
    def _fwht_torch(self, x):
        """Pure PyTorch FWHT for correctness validation."""
        d = x.shape[-1]
        h = x.clone()
        stride = 1
        while stride < d:
            for i in range(0, d, stride*2):
                a = h[..., i:i+stride].clone()
                b = h[..., i+stride:i+2*stride].clone()
                h[..., i:i+stride]          = a + b
                h[..., i+stride:i+2*stride] = a - b
            stride *= 2
        return h * (1.0 / np.sqrt(d))

    def encode_keys(self, keys):
        n = keys.shape[0]
        # rotate with FWHT (same rotation used in attend)
        y      = self._fwht_torch(keys * self.signs)
        idx    = (y.unsqueeze(-1) - self.centroids).abs().argmin(dim=-1)
        idx    = idx.to(torch.int16)
        x_hat_rot = self.centroids[idx.long()]
        r_rot  = y - x_hat_rot
        gamma  = torch.norm(r_rot, dim=-1)
        r_unit = F.normalize(r_rot, dim=-1)
        z      = torch.sign(self._fwht_torch(r_unit * self.signs_qjl))
        z      = torch.where(z==0, torch.ones_like(z), z)

        k_hat_packed = torch.zeros(n, self.d//2, dtype=torch.int8, device=self.device)
        self.lib.launch_pack_khat(
            ctypes.c_void_p(idx.data_ptr()),
            ctypes.c_void_p(k_hat_packed.data_ptr()),
            ctypes.c_int(n), ctypes.c_int(self.d),
        )
        z_packed = torch.zeros(n, self.d//8, dtype=torch.uint8, device=self.device)
        self.lib.launch_pack_z(
            ctypes.c_void_p(z.data_ptr()),
            ctypes.c_void_p(z_packed.data_ptr()),
            ctypes.c_int(n), ctypes.c_int(self.d),
        )
        torch.cuda.synchronize()
        return k_hat_packed, z_packed, gamma

    def attend(self, q, k_hat_packed, z_packed, gamma):
        n_q   = q.shape[0]
        n_ctx = k_hat_packed.shape[0]
        scores = torch.zeros(n_q, n_ctx, device=self.device)

        # pre-transform query with FWHT — done once, amortized over all keys
        q_fwht = self._fwht_torch(q * self.signs) * self.qk_scale

        self.lib.launch_quant_attn(
            ctypes.c_void_p(q_fwht.data_ptr()),
            ctypes.c_void_p(k_hat_packed.data_ptr()),
            ctypes.c_void_p(z_packed.data_ptr()),
            ctypes.c_void_p(gamma.data_ptr()),
            ctypes.c_void_p(self.centroids.data_ptr()),
            ctypes.c_void_p(self.signs.data_ptr()),
            ctypes.c_void_p(scores.data_ptr()),
            ctypes.c_int(n_q),
            ctypes.c_int(n_ctx),
            ctypes.c_int(self.d),
            ctypes.c_float(self.scale),
            ctypes.c_float(1.0),  # qk_scale already applied to q_fwht
        )
        torch.cuda.synchronize()
        return scores
    def attend_naive(self, q, keys):
        n_ctx  = keys.shape[0]
        y      = self._fwht_torch(keys * self.signs)
        idx    = (y.unsqueeze(-1) - self.centroids).abs().argmin(dim=-1)
        x_hat_rot = self.centroids[idx.long()]
        r_rot  = y - x_hat_rot
        gamma  = torch.norm(r_rot, dim=-1, keepdim=True)
        r_unit = F.normalize(r_rot, dim=-1)
        z      = torch.sign(self._fwht_torch(r_unit * self.signs_qjl))
        z      = torch.where(z==0, torch.ones_like(z), z)
        x_tilde_rot = x_hat_rot + gamma * self.scale * z
        q_fwht = self._fwht_torch(q * self.signs)
        return self.qk_scale * (q_fwht @ x_tilde_rot.T)

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


def validate(attn, device, d, n_ctx=64, n_q=8):
    """
    Correctness check: fused quantized attention vs naive reference.
    Use small n_ctx so we can inspect values.
    """
    print(f"\n=== Correctness Validation (d={d}, n_ctx={n_ctx}, n_q={n_q}) ===")
    torch.manual_seed(42)

    keys = F.normalize(torch.randn(n_ctx, d, device=device), dim=-1)
    q    = F.normalize(torch.randn(n_q,   d, device=device), dim=-1)

    # naive reference
    scores_ref = attn.attend_naive(q, keys)

    # fused quantized
    k_hat_packed, z_packed, gamma = attn.encode_keys(keys)
    scores_fused = attn.attend(q, k_hat_packed, z_packed, gamma)

    diff = (scores_ref - scores_fused).abs()
    print(f"Score max diff:  {diff.max().item():.6f}")
    print(f"Score mean diff: {diff.mean().item():.6f}")
    print(f"Score correlation: {torch.corrcoef(torch.stack([scores_ref.flatten(), scores_fused.flatten()]))[0,1].item():.6f}")

    # check ranking preservation (top-k accuracy)
    for k in [1, 4, 8]:
        if k > n_ctx: continue
        top_ref   = scores_ref[0].topk(k).indices.sort().values
        top_fused = scores_fused[0].topk(k).indices.sort().values
        match = (top_ref == top_fused).all().item()
        print(f"Top-{k} ranking preserved: {match}")


def benchmark(device, attn_lib, fwht_lib):
    print("\n" + "="*64)
    print("Fused Quantized Attention Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*64)

    configs = [
        # (d,   b, n_q, n_ctx)
        (128,  4,  1,   1024),    # single query, 1k context
        (128,  4,  1,   4096),    # single query, 4k context
        (128,  4,  1,   16384),   # single query, 16k context
        (128,  4,  8,   4096),    # small batch
        (128,  4,  32,  4096),    # larger batch
        (64,   4,  1,   4096),    # smaller head dim
        (256,  4,  1,   4096),    # larger head dim
    ]

    print(f"\n{'Config':<32} {'Naive(ms)':>10} {'Fused(ms)':>10} {'Speedup':>10} {'BW saved':>10}")
    print("-"*74)

    for d, b, n_q, n_ctx in configs:
        torch.manual_seed(42)
        attn = QuantAttn(d, b, device, attn_lib, fwht_lib)
        keys = F.normalize(torch.randn(n_ctx, d, device=device), dim=-1)
        q    = F.normalize(torch.randn(n_q,   d, device=device), dim=-1)

        k_hat_packed, z_packed, gamma = attn.encode_keys(keys)

        t_naive, _ = timeit(lambda: attn.attend_naive(q, keys))
        t_fused, _ = timeit(lambda: attn.attend(q, k_hat_packed, z_packed, gamma))

        # theoretical BW comparison
        bw_naive = n_q * n_ctx * d * 4      # fp32 keys
        bw_fused = n_q * n_ctx * (d//2 + d//8 + 4)  # compressed
        bw_ratio = bw_naive / bw_fused

        print(f"d={d:3d} b={b} n_q={n_q:2d} n_ctx={n_ctx:5d}  "
              f"{t_naive:>10.4f}  {t_fused:>10.4f}  "
              f"{t_naive/t_fused:>9.2f}x  {bw_ratio:>9.1f}x")

    print("="*64)

    # memory layout summary
    print("\n=== Compressed KV Cache Memory Layout ===")
    print(f"{'d':>6} {'fp16 (bytes)':>14} {'4+1bit (bytes)':>16} {'Compression':>12}")
    for d in [64, 128, 256, 512]:
        fp16_bytes   = d * 2
        quant_bytes  = d//2 + d//8 + 4
        ratio        = fp16_bytes / quant_bytes
        print(f"{d:>6} {fp16_bytes:>14} {quant_bytes:>16} {ratio:>11.1f}x")


if __name__ == "__main__":
    device = torch.device("cuda")
    print("Compiling kernels...")
    attn_lib = compile_lib("quant_attn_kernel.cu")
    fwht_lib = None
    b = 4

    attn128 = QuantAttn(128, b, device, attn_lib, fwht_lib)
    validate(attn128, device, d=128, n_ctx=64, n_q=8)

    attn64 = QuantAttn(64, b, device, attn_lib, fwht_lib)
    validate(attn64, device, d=64, n_ctx=64, n_q=8)

    benchmark(device, attn_lib, fwht_lib)
