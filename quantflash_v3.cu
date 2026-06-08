
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
