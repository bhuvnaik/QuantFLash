import os, sys, ctypes, numpy as np, torch
import torch.nn.functional as F

os.chdir('/scratch/bhuvanc/turboquant')
sys.path.insert(0, '/scratch/bhuvanc/turboquant')
sys.path.insert(0, '/scratch/bhuvanc/turboquant_plus')

from turbo_fwht import compile_fwht
from turbo_pipeline import TurboQuantGPU

device   = torch.device("cuda")
fwht_lib = compile_fwht("fwht_kernel.cu")
tq4      = TurboQuantGPU(d=128, b=4, device=device, fwht_lib=fwht_lib)

# ── load QuantFlash kernel ────────────────────────────────────────────────────
qf_lib = ctypes.CDLL(os.path.abspath("quantflash_kernel.so"))
for fn, args in [
    ("launch_quantflash_v2", [
        ctypes.c_void_p]*7 + [ctypes.c_int]*4 + [ctypes.c_float]*2),
    ("launch_qf_pack_khat",
        [ctypes.c_void_p]*2 + [ctypes.c_int]*2),
    ("launch_qf_pack_z",
        [ctypes.c_void_p]*2 + [ctypes.c_int]*2),
]:
    getattr(qf_lib, fn).argtypes = args
    getattr(qf_lib, fn).restype  = None

# ── load old quant_attn kernel for comparison ─────────────────────────────────
old_lib = ctypes.CDLL(os.path.abspath("quant_attn_kernel.so"))
old_lib.launch_quant_attn.argtypes = [
    ctypes.c_void_p]*7 + [ctypes.c_int]*3 + [ctypes.c_float]*2
old_lib.launch_quant_attn.restype  = None

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"TurboQuant: d={tq4.d}, b={tq4.b}, k={tq4.k}")

# ── packing helpers ───────────────────────────────────────────────────────────

def pack_khat(idx, n, d):
    """idx: (n,d) int16 → packed (n, d/2) int8 with 4-bit nibbles."""
    out = torch.zeros(n, d//2, dtype=torch.int8, device=device)
    qf_lib.launch_qf_pack_khat(
        ctypes.c_void_p(idx.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(n), ctypes.c_int(d),
    )
    torch.cuda.synchronize()
    return out

def pack_z(r_unit, n, d):
    """r_unit: (n,d) fp32 signs → packed (n, d/8) uint8."""
    out = torch.zeros(n, d//8, dtype=torch.uint8, device=device)
    qf_lib.launch_qf_pack_z(
        ctypes.c_void_p(r_unit.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(n), ctypes.c_int(d),
    )
    torch.cuda.synchronize()
    return out

def encode_keys(keys):
    """Encode (n_ctx, d) fp32 keys → compressed QuantFlash format."""
    n = keys.shape[0]; d = tq4.d
    idx, r_unit, _, gamma_r = tq4.encode(keys)
    return pack_khat(idx, n, d), pack_z(r_unit, n, d), gamma_r

def encode_keys_old(keys):
    """Encode for old quant_attn kernel format (same packing, different lib)."""
    n = keys.shape[0]; d = tq4.d
    idx, r_unit, _, gamma_r = tq4.encode(keys)
    # old kernel uses same int8 packing for k_hat
    k_hat_old = torch.zeros(n, d//2, dtype=torch.int8, device=device)
    old_lib.launch_pack_khat = None  # old kernel has launch_pack_khat
    # use our packer since it's identical
    k_hat_old = pack_khat(idx, n, d)
    # old kernel uses float z (not packed uint8) — unpack from r_unit directly
    z_old = torch.zeros(n, d//8, dtype=torch.uint8, device=device)
    qf_lib.launch_qf_pack_z(
        ctypes.c_void_p(r_unit.data_ptr()),
        ctypes.c_void_p(z_old.data_ptr()),
        ctypes.c_int(n), ctypes.c_int(d),
    )
    torch.cuda.synchronize()
    return k_hat_old, z_old, gamma_r

def run_quantflash(q, k_hat, z_packed, gamma_r, n_q, n_ctx):
    d = tq4.d; K = tq4.k
    scores = torch.zeros(n_q, n_ctx, device=device)
    qf_lib.launch_quantflash_v2(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k_hat.data_ptr()),
        ctypes.c_void_p(z_packed.data_ptr()),
        ctypes.c_void_p(gamma_r.data_ptr()),
        ctypes.c_void_p(tq4.centroids.data_ptr()),
        ctypes.c_void_p(tq4.srht.D1.data_ptr()),
        ctypes.c_void_p(scores.data_ptr()),
        ctypes.c_int(n_q), ctypes.c_int(n_ctx),
        ctypes.c_int(d), ctypes.c_int(K),
        ctypes.c_float(float(1.0/d)),
        ctypes.c_float(float(1.0/np.sqrt(d))),
    )
    torch.cuda.synchronize()
    return scores

def run_old_qattn(q, k_hat, z_packed, gamma_r, n_q, n_ctx):
    d = tq4.d
    scores = torch.zeros(n_q, n_ctx, device=device)
    old_lib.launch_quant_attn(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k_hat.data_ptr()),
        ctypes.c_void_p(z_packed.data_ptr()),
        ctypes.c_void_p(gamma_r.data_ptr()),
        ctypes.c_void_p(tq4.centroids.data_ptr()),
        ctypes.c_void_p(tq4.srht.D1.data_ptr()),
        ctypes.c_void_p(scores.data_ptr()),
        ctypes.c_int(n_q), ctypes.c_int(n_ctx), ctypes.c_int(d),
        ctypes.c_float(float(1.0/d)),
        ctypes.c_float(float(1.0/np.sqrt(d))),
    )
    torch.cuda.synchronize()
    return scores

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
    return float(np.mean(times))

# ── correctness ───────────────────────────────────────────────────────────────
print("\n" + "="*68)
print("CORRECTNESS")
print("="*68)

torch.manual_seed(42)
d = tq4.d; n_ctx = 512; n_q = 4
keys = F.normalize(torch.randn(n_ctx, d, device=device), dim=-1)
q    = F.normalize(torch.randn(n_q,   d, device=device), dim=-1)

k_hat, z_packed, gamma_r = encode_keys(keys)
qf_scores    = run_quantflash(q, k_hat, z_packed, gamma_r, n_q, n_ctx)
exact_scores = float(1.0/np.sqrt(d)) * (q @ keys.T)

diff = (qf_scores - exact_scores).abs()
print(f"  Score max diff:   {diff.max().item():.4f}")
print(f"  Score mean diff:  {diff.mean().item():.4f}")
print(f"  Cosine sim:       "
      f"{F.cosine_similarity(qf_scores.reshape(1,-1), exact_scores.reshape(1,-1)).item():.6f}")

for topk in [1, 4, 16]:
    qf_top = qf_scores.topk(topk, dim=-1).indices
    ex_top = exact_scores.topk(topk, dim=-1).indices
    recall = sum(len(set(qf_top[i].tolist()) & set(ex_top[i].tolist()))
                 for i in range(n_q)) / (n_q * topk)
    print(f"  Top-{topk:2d} recall:    {recall*100:.1f}%")

# ── memory layout ─────────────────────────────────────────────────────────────
bytes_fp16 = d * 2
bytes_qf   = d//2 + d//8 + 4
lut_bytes  = d * tq4.k * 4

print(f"\n  fp16 key:       {bytes_fp16} bytes/key")
print(f"  QuantFlash key: {bytes_qf} bytes/key  "
      f"(k_hat={d//2} z={d//8} gamma_r=4)")
print(f"  Bandwidth reduction: {bytes_fp16/bytes_qf:.2f}x")
print(f"  LUT: {lut_bytes} bytes = {lut_bytes/1024:.1f}KB "
      f"(built once per query, amortized over all keys)")

# ── main speed benchmark ──────────────────────────────────────────────────────
print("\n" + "="*72)
print("SPEED BENCHMARK: QuantFlash vs fp16 cuBLAS vs old quant_attn")
print("="*72)

print(f"\n{'n_q':>4} {'n_ctx':>7}  "
      f"{'fp16(ms)':>10} {'old_qa(ms)':>11} {'QF(ms)':>8}  "
      f"{'QF/fp16':>9} {'QF/old':>8} {'bw_saved':>10}")
print("-"*80)

configs = [
    (1,   1024), (1,  4096), (1,  16384), (1,  65536),
    (8,   4096), (8,  16384), (8,  65536),
    (32,  4096), (32, 16384),
]

for n_q, n_ctx in configs:
    torch.manual_seed(n_ctx)
    keys     = F.normalize(torch.randn(n_ctx, d, device=device), dim=-1)
    q        = F.normalize(torch.randn(n_q,   d, device=device), dim=-1)
    q_f16    = q.half()
    keys_f16 = keys.half()

    k_hat, z_packed, gamma_r = encode_keys(keys)

    ms_fp16 = timeit(lambda: q_f16 @ keys_f16.T)
    ms_old  = timeit(lambda: run_old_qattn(
        q, k_hat, z_packed, gamma_r, n_q, n_ctx))
    ms_qf   = timeit(lambda: run_quantflash(
        q, k_hat, z_packed, gamma_r, n_q, n_ctx))

    bw_ratio = (n_ctx * bytes_fp16) / (n_ctx * bytes_qf)

    print(f"{n_q:>4} {n_ctx:>7}  "
          f"{ms_fp16:>10.4f} {ms_old:>11.4f} {ms_qf:>8.4f}  "
          f"{ms_fp16/ms_qf:>8.2f}x {ms_old/ms_qf:>7.2f}x "
          f"{bw_ratio:>9.2f}x")

# ── bandwidth utilization ─────────────────────────────────────────────────────
print("\n" + "="*68)
print("BANDWIDTH UTILIZATION (n_q=1, decode regime)")
print("="*68)

bw_peak = 768.0
print(f"\n{'n_ctx':>8}  {'fp16(GB/s)':>12} {'QF(GB/s)':>10} "
      f"{'fp16_util':>11} {'QF_util':>9} {'speedup':>9}")
print("-"*64)

for n_ctx in [1024, 4096, 16384, 65536]:
    torch.manual_seed(n_ctx)
    keys     = F.normalize(torch.randn(n_ctx, d, device=device), dim=-1)
    q        = F.normalize(torch.randn(1,     d, device=device), dim=-1)
    q_f16    = q.half(); keys_f16 = keys.half()
    k_hat, z_packed, gamma_r = encode_keys(keys)

    ms_fp16 = timeit(lambda: q_f16 @ keys_f16.T)
    ms_qf   = timeit(lambda: run_quantflash(
        q, k_hat, z_packed, gamma_r, 1, n_ctx))

    bw_fp16 = (n_ctx*bytes_fp16) / (ms_fp16*1e-3) / 1e9
    bw_qf   = (n_ctx*bytes_qf)   / (ms_qf  *1e-3) / 1e9

    print(f"{n_ctx:>8}  {bw_fp16:>11.1f}GB/s {bw_qf:>9.1f}GB/s "
          f"{bw_fp16/bw_peak*100:>10.1f}% {bw_qf/bw_peak*100:>8.1f}% "
          f"{ms_fp16/ms_qf:>8.2f}x")

print(f"\nA5000 peak HBM bandwidth: {bw_peak:.0f} GB/s")
print("="*68)
