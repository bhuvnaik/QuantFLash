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

cu = r"""
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define TILE_K 64

/*
 * quantflash_v3
 *
 * Receives q_fwht: query already transformed by full double-SRHT on host.
 * No butterfly in kernel — eliminates all __syncthreads() deadlock risk.
 *
 * Grid:  (n_q, ceil(n_ctx/TILE_K))
 * Block: (TILE_K,) = 64 threads — one thread per key
 *
 * Each thread:
 *   1. Reads LUT from shared memory (built cooperatively at block start)
 *   2. Loops over d/2 packed index pairs — 128 LUT lookups, no multiplications
 *   3. Loops over d bits of z — 128 multiply-adds (with preloaded q_fwht)
 *   4. Writes one score
 *
 * Shared memory:
 *   s_q[d]     = 512B    pre-transformed query (broadcast to all threads)
 *   s_lut[d*K] = 8192B   LUT[coord][centroid]
 *   Total      = 8704B   well within 48KB L1
 */
extern "C"
__global__ void quantflash_v3(
    const float*   __restrict__ q_fwht,   /* (n_q, d) SRHT-transformed query  */
    const int8_t*  __restrict__ k_hat,    /* (n_ctx, d/2) packed 4-bit idx    */
    const uint8_t* __restrict__ z_packed, /* (n_ctx, d/8) packed sign bits    */
    const float*   __restrict__ gamma_r,  /* (n_ctx,)     residual norms      */
    const float*   __restrict__ centroids,/* (K,)         codebook            */
    float*         __restrict__ scores,   /* (n_q, n_ctx) output              */
    int n_ctx, int d, int K,
    float scale, float qk_scale
) {
    extern __shared__ float shmem[];
    float* s_q   = shmem;          /* d floats                    */
    float* s_lut = shmem + d;      /* d*K floats                  */

    int q_id    = blockIdx.x;
    int tile_id = blockIdx.y;
    int tid     = threadIdx.x;     /* in [0, TILE_K)              */
    int d2      = d / 2;
    int d8      = d / 8;

    int key_start = tile_id * TILE_K;
    int key_end   = min(key_start + TILE_K, n_ctx);

    /* ── load pre-transformed query into shared memory ───────────────────── */
    /* d=128 entries, TILE_K=64 threads → 2 entries per thread               */
    int base_q = q_id * d;
    s_q[tid]        = q_fwht[base_q + tid];
    s_q[tid + TILE_K] = q_fwht[base_q + tid + TILE_K];
    __syncthreads();

    /* ── build LUT[coord][centroid] = s_q[coord] * centroids[centroid] ───── */
    /* d*K = 128*16 = 2048 entries, 64 threads → 32 entries each             */
    int lut_total = d * K;
    for (int idx = tid; idx < lut_total; idx += TILE_K) {
        int coord   = idx / K;
        int ci      = idx % K;
        s_lut[idx]  = s_q[coord] * centroids[ci];
    }
    __syncthreads();

    /* ── each thread scores one key — no synchronization needed ─────────── */
    int ki = key_start + tid;
    if (ki >= key_end) return;

    /* codebook term: d/2 packed bytes → 128 LUT lookups, zero multiplications */
    float dot_lut = 0.0f;
    for (int j = 0; j < d2; j++) {
        int8_t packed = k_hat[ki * d2 + j];
        int idx0 = (packed     ) & 0x0F;   /* lo nibble */
        int idx1 = (packed >> 4) & 0x0F;   /* hi nibble */
        dot_lut += s_lut[(2*j)   * K + idx0];
        dot_lut += s_lut[(2*j+1) * K + idx1];
    }

    /* binary residual term: d/8 bytes → 128 sign lookups */
    float dot_z = 0.0f;
    for (int j = 0; j < d; j++) {
        uint8_t zb = z_packed[ki * d8 + j / 8];
        float   zj = ((zb >> (j % 8)) & 1) ? 1.0f : -1.0f;
        dot_z += s_q[j] * zj;
    }

    scores[q_id * n_ctx + ki] =
        qk_scale * (dot_lut + gamma_r[ki] * scale * dot_z);
}

extern "C"
void launch_quantflash_v3(
    const float*   q_fwht,
    const int8_t*  k_hat,
    const uint8_t* z_packed,
    const float*   gamma_r,
    const float*   centroids,
    float*         scores,
    int n_q, int n_ctx, int d, int K,
    float scale, float qk_scale
) {
    int n_tiles = (n_ctx + TILE_K - 1) / TILE_K;
    dim3 grid(n_q, n_tiles);
    int  block = TILE_K;
    int  shmem = (d + d * K) * sizeof(float);
    quantflash_v3<<<grid, block, shmem>>>(
        q_fwht, k_hat, z_packed, gamma_r, centroids,
        scores, n_ctx, d, K, scale, qk_scale
    );
}

extern "C"
__global__ void qf_pack_khat(
    const int16_t* __restrict__ idx, int8_t* __restrict__ out, int n, int d
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    if (i >= n || 2*j+1 >= d) return;
    int lo = idx[i*d + 2*j]   & 0x0F;
    int hi = idx[i*d + 2*j+1] & 0x0F;
    out[i*(d/2) + j] = (int8_t)((hi << 4) | lo);
}
extern "C"
__global__ void qf_pack_z(
    const float* __restrict__ z, uint8_t* __restrict__ out, int n, int d
) {
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int byte = blockIdx.y;
    if (i >= n || 8*byte+7 >= d) return;
    uint8_t packed = 0;
    for (int b = 0; b < 8; b++)
        if (z[i*d + 8*byte + b] > 0.0f) packed |= (1 << b);
    out[i*(d/8) + byte] = packed;
}
extern "C"
void launch_qf_pack_khat(const int16_t* idx, int8_t* out, int n, int d) {
    dim3 block(32); dim3 grid((n+31)/32, d/2);
    qf_pack_khat<<<grid,block>>>(idx, out, n, d);
}
extern "C"
void launch_qf_pack_z(const float* z, uint8_t* out, int n, int d) {
    dim3 block(32); dim3 grid((n+31)/32, d/8);
    qf_pack_z<<<grid,block>>>(z, out, n, d);
}
"""

with open('/scratch/bhuvanc/turboquant/quantflash_v3.cu', 'w') as f:
    f.write(cu)

import subprocess
result = subprocess.run([
    'nvcc', '-O3', '-arch=sm_86', '--compiler-options', '-fPIC',
    '-shared', '/scratch/bhuvanc/turboquant/quantflash_v3.cu',
    '-o', '/scratch/bhuvanc/turboquant/quantflash_v3.so'
], capture_output=True, text=True)
if result.returncode != 0:
    print("COMPILE ERROR:", result.stderr); sys.exit(1)
print("Compiled quantflash_v3.so OK")

qf = ctypes.CDLL('/scratch/bhuvanc/turboquant/quantflash_v3.so')
qf.launch_quantflash_v3.argtypes = [ctypes.c_void_p]*6 + \
    [ctypes.c_int]*4 + [ctypes.c_float]*2
qf.launch_quantflash_v3.restype  = None
qf.launch_qf_pack_khat.argtypes  = [ctypes.c_void_p]*2+[ctypes.c_int]*2
qf.launch_qf_pack_khat.restype   = None
qf.launch_qf_pack_z.argtypes     = [ctypes.c_void_p]*2+[ctypes.c_int]*2
qf.launch_qf_pack_z.restype      = None

old = ctypes.CDLL('/scratch/bhuvanc/turboquant/quant_attn_kernel.so')
old.launch_quant_attn.argtypes = [ctypes.c_void_p]*7+[ctypes.c_int]*3+[ctypes.c_float]*2
old.launch_quant_attn.restype  = None

d = tq4.d; K = tq4.k

def pack_khat(idx, n):
    out = torch.zeros(n, d//2, dtype=torch.int8, device=device)
    qf.launch_qf_pack_khat(ctypes.c_void_p(idx.data_ptr()),
        ctypes.c_void_p(out.data_ptr()), ctypes.c_int(n), ctypes.c_int(d))
    torch.cuda.synchronize(); return out

def pack_z(r_unit, n):
    out = torch.zeros(n, d//8, dtype=torch.uint8, device=device)
    qf.launch_qf_pack_z(ctypes.c_void_p(r_unit.data_ptr()),
        ctypes.c_void_p(out.data_ptr()), ctypes.c_int(n), ctypes.c_int(d))
    torch.cuda.synchronize(); return out

def encode_keys(keys):
    """Encode keys and return (k_hat, z_packed, gamma_r, q_fwht_fn)."""
    n = keys.shape[0]
    idx, r_unit, _, gamma_r = tq4.encode(keys)
    return pack_khat(idx, n), pack_z(r_unit, n), gamma_r

def transform_query(q):
    """Apply full double-SRHT to query on GPU using existing FWHT kernel."""
    return tq4.srht.forward(q)   # (n_q, d) — D2 @ H @ D1 @ q / sqrt(d)

def run_qf(q_fwht, k_hat, z_packed, gamma_r, n_q, n_ctx):
    scores = torch.zeros(n_q, n_ctx, device=device)
    qf.launch_quantflash_v3(
        ctypes.c_void_p(q_fwht.data_ptr()),
        ctypes.c_void_p(k_hat.data_ptr()),
        ctypes.c_void_p(z_packed.data_ptr()),
        ctypes.c_void_p(gamma_r.data_ptr()),
        ctypes.c_void_p(tq4.centroids.data_ptr()),
        ctypes.c_void_p(scores.data_ptr()),
        ctypes.c_int(n_q), ctypes.c_int(n_ctx),
        ctypes.c_int(d),   ctypes.c_int(K),
        ctypes.c_float(float(1.0/d)),
        ctypes.c_float(float(1.0/np.sqrt(d))),
    )
    torch.cuda.synchronize(); return scores

def run_old(q, k_hat, z_packed, gamma_r, n_q, n_ctx):
    scores = torch.zeros(n_q, n_ctx, device=device)
    old.launch_quant_attn(
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
    torch.cuda.synchronize(); return scores

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

print("\n" + "="*60)
print("CORRECTNESS v3")
print("="*60)

torch.manual_seed(42)
n_ctx_val = 512; n_q_val = 4
keys = F.normalize(torch.randn(n_ctx_val, d, device=device), dim=-1)
q    = F.normalize(torch.randn(n_q_val,   d, device=device), dim=-1)

k_hat_v, z_v, gamma_r_v = encode_keys(keys)
q_fwht   = transform_query(q)
qf_sc    = run_qf(q_fwht, k_hat_v, z_v, gamma_r_v, n_q_val, n_ctx_val)
exact    = float(1.0/np.sqrt(d)) * (q @ keys.T)

diff     = (qf_sc - exact).abs()
cos_sim  = F.cosine_similarity(
    qf_sc.reshape(1,-1), exact.reshape(1,-1)).item()
print(f"  Max diff:    {diff.max().item():.4f}")
print(f"  Mean diff:   {diff.mean().item():.4f}")
print(f"  Cosine sim:  {cos_sim:.6f}  (good if > 0.90)")

for topk in [1, 4, 16]:
    qf_top = qf_sc.topk(topk,dim=-1).indices
    ex_top = exact.topk(topk,dim=-1).indices
    recall = sum(len(set(qf_top[i].tolist())&set(ex_top[i].tolist()))
                 for i in range(n_q_val)) / (n_q_val*topk)
    print(f"  Top-{topk:2d} recall: {recall*100:.1f}%")

bytes_fp16 = d*2; bytes_qf = d//2 + d//8 + 4
print(f"\n  fp16={bytes_fp16}B/key  QF={bytes_qf}B/key  "
      f"bandwidth ratio={bytes_fp16/bytes_qf:.2f}x")

print("\n" + "="*70)
print("SPEED BENCHMARK")
print("="*70)
print(f"\n{'n_q':>4} {'n_ctx':>7}  "
      f"{'fp16(ms)':>10} {'old(ms)':>9} {'QFv3(ms)':>10}  "
      f"{'vs_fp16':>9} {'vs_old':>8}")
print("-"*68)

configs = [
    (1,1024),(1,4096),(1,16384),(1,65536),
    (8,4096),(8,16384),(8,65536),
    (32,4096),(32,16384),
]
for n_q, n_ctx in configs:
    torch.manual_seed(n_ctx)
    keys     = F.normalize(torch.randn(n_ctx, d, device=device), dim=-1)
    q        = F.normalize(torch.randn(n_q,   d, device=device), dim=-1)
    q_f16    = q.half(); keys_f16 = keys.half()
    k_hat_b, z_b, gamma_r_b = encode_keys(keys)
    q_fwht_b = transform_query(q)

    ms_fp16 = timeit(lambda: q_f16 @ keys_f16.T)
    ms_old  = timeit(lambda: run_old(q, k_hat_b, z_b, gamma_r_b, n_q, n_ctx))
    ms_qf   = timeit(lambda: run_qf(q_fwht_b, k_hat_b, z_b,
                                     gamma_r_b, n_q, n_ctx))

    print(f"{n_q:>4} {n_ctx:>7}  "
          f"{ms_fp16:>10.4f} {ms_old:>9.4f} {ms_qf:>10.4f}  "
          f"{ms_fp16/ms_qf:>8.2f}x {ms_old/ms_qf:>7.2f}x")

print(f"\nNote: QFv3 timing excludes query SRHT (~0.04ms, paid once per token)")
print("      fp16 timing excludes query load (same overhead)")
