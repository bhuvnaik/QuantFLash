/*
 * quantflash_kernel.cu  v2
 *
 * Key design change from v1:
 *   v1: block=(d/2,), threads reduce over coordinates, loop over keys  → BAD
 *   v2: block=(TILE_K,), each thread handles ONE key entirely           → GOOD
 *
 * Each thread computes one complete attention score:
 *   score_i = Σ_j LUT[j][idx_j] + gamma_r_i * scale * dot_z_i
 *
 * LUT[d][K] = 8KB built once per block, reused across all TILE_K keys.
 * No reductions needed per key — each thread is independent.
 *
 * Query SRHT: full double-sign  y = D2 * WHT(D1 * q) / sqrt(d)
 * (v1 was missing D2 — causing correctness failure)
 *
 * Grid:  (n_q, ceil(n_ctx/TILE_K))
 * Block: (TILE_K,) = 64 threads, one per key in tile
 *
 * Shared memory:
 *   s_q[d]       transformed query fp32        512B
 *   s_lut[d*K]   LUT[coord][centroid] fp32     8KB
 *   s_tmp[d]     FWHT butterfly scratch         512B
 *   Total: ~9.5KB  (L1 = 48KB, easily fits)
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define TILE_K 64

/* ── FWHT butterfly over shared memory ──────────────────────────────────────
 * Called by first d threads (tid < d) to transform s_q in place.
 * Uses s_tmp as scratch to avoid race conditions.
 */
__device__ void fwht_shared(float* s_q, float* s_tmp, int tid, int d) {
    // only first d threads participate
    if (tid >= d) return;
    for (int stride = 1; stride < d; stride <<= 1) {
        // each thread reads one element, computes butterfly
        int group = tid / stride;
        int pos   = tid % stride;
        int i     = group * (stride * 2) + pos;
        int j     = i + stride;
        float a = s_q[i], b = s_q[j];
        __syncthreads();
        s_q[i] = a + b;
        s_q[j] = a - b;
        __syncthreads();
    }
    (void)s_tmp;
}

extern "C"
__global__ void quantflash_kernel_v2(
    const float*   __restrict__ q,          /* (n_q, d)       fp32               */
    const int8_t*  __restrict__ k_hat,      /* (n_ctx, d/2)   packed 4-bit       */
    const uint8_t* __restrict__ z_packed,   /* (n_ctx, d/8)   packed signs       */
    const float*   __restrict__ gamma_r,    /* (n_ctx,)       residual norms     */
    const float*   __restrict__ centroids,  /* (K,)           codebook           */
    const float*   __restrict__ D1,         /* (d,)           SRHT sign vector 1 */
    const float*   __restrict__ D2,         /* (d,)           SRHT sign vector 2 */
    float*         __restrict__ scores,     /* (n_q, n_ctx)   output             */
    int n_ctx, int d, int K,
    float scale, float qk_scale
) {
    // shmem layout: s_q[d] | s_lut[d*K] | s_tmp[d]
    extern __shared__ float shmem[];
    float* s_q   = shmem;           // d floats = 512B
    float* s_lut = shmem + d;       // d*K floats = 8KB
    float* s_tmp = shmem + d + d*K; // d floats scratch

    int q_id    = blockIdx.x;       // which query
    int tile_id = blockIdx.y;       // which key tile
    int tid     = threadIdx.x;      // in [0, TILE_K)

    int key_start = tile_id * TILE_K;
    int key_end   = min(key_start + TILE_K, n_ctx);
    int base_q    = q_id * d;

    /* ── Step 1: load query and apply D1 ─────────────────────────────────── */
    // threads 0..d-1 each load one coordinate
    // TILE_K=64, d=128 → need two passes if TILE_K < d
    if (tid < d) {
        s_q[tid]   = q[base_q + tid] * D1[tid];
    }
    __syncthreads();

    /* ── Step 2: FWHT butterfly ───────────────────────────────────────────── */
    // only first d=128 threads participate; TILE_K=64 so we need 2 passes
    // since TILE_K(64) < d(128), each thread handles 2 coordinates
    for (int stride = 1; stride < d; stride <<= 1) {
        // thread tid handles coordinates (2*tid) and (2*tid+1) -- NO
        // actually with block=(TILE_K=64) and d=128, each thread handles
        // two butterfly operations
        int i0 = 2 * tid;
        int j0 = i0 + stride;
        int group0 = i0 / (stride * 2);
        int pos0   = i0 % (stride * 2);

        // only do the butterfly if i and j are a valid pair
        // valid pair: pos0 < stride
        if (pos0 < stride && j0 < d) {
            float a = s_q[i0], b = s_q[j0];
            __syncthreads();
            s_q[i0] = a + b;
            s_q[j0] = a - b;
        } else {
            __syncthreads();
        }
        __syncthreads();
    }
    // scale by 1/sqrt(d) and apply D2
    if (tid < d/2) {
        s_q[2*tid]   = s_q[2*tid]   * rsqrtf((float)d) * D2[2*tid];
        s_q[2*tid+1] = s_q[2*tid+1] * rsqrtf((float)d) * D2[2*tid+1];
    }
    __syncthreads();

    /* ── Step 3: build LUT[j][i] = s_q[j] * centroids[i] ────────────────── */
    // d*K = 128*16 = 2048 entries
    // TILE_K=64 threads → each handles 2048/64 = 32 entries
    int lut_total = d * K;
    for (int idx = tid; idx < lut_total; idx += TILE_K) {
        int coord    = idx / K;
        int cent_idx = idx % K;
        s_lut[idx]   = s_q[coord] * centroids[cent_idx];
    }
    __syncthreads();

    /* ── Step 4: each thread scores ONE key ──────────────────────────────── */
    int ki = key_start + tid;
    if (ki < key_end) {
        int d2 = d / 2;
        int d8 = d / 8;

        // --- codebook term: 128 LUT lookups, no multiplications ----------
        float dot_lut = 0.0f;
        for (int j = 0; j < d2; j++) {
            int8_t packed = k_hat[ki * d2 + j];
            int idx0 = (packed >> 0) & 0x0F;  // lo nibble
            int idx1 = (packed >> 4) & 0x0F;  // hi nibble
            dot_lut += s_lut[(2*j)   * K + idx0];
            dot_lut += s_lut[(2*j+1) * K + idx1];
        }

        // --- binary residual term: 128 sign lookups ----------------------
        float dot_z = 0.0f;
        for (int j = 0; j < d; j++) {
            uint8_t zb = z_packed[ki * d8 + j / 8];
            float   zj = ((zb >> (j % 8)) & 1) ? 1.0f : -1.0f;
            dot_z += s_q[j] * zj;
        }

        // --- final score -------------------------------------------------
        scores[q_id * n_ctx + ki] =
            qk_scale * (dot_lut + gamma_r[ki] * scale * dot_z);
    }
}

extern "C"
void launch_quantflash_v2(
    const float*   q,
    const int8_t*  k_hat,
    const uint8_t* z_packed,
    const float*   gamma_r,
    const float*   centroids,
    const float*   D1,
    const float*   D2,
    float*         scores,
    int n_q, int n_ctx, int d, int K,
    float scale, float qk_scale
) {
    int n_tiles = (n_ctx + TILE_K - 1) / TILE_K;
    dim3 grid(n_q, n_tiles);
    int  block = TILE_K;
    int  shmem = (d + d * K + d) * sizeof(float);
    quantflash_kernel_v2<<<grid, block, shmem>>>(
        q, k_hat, z_packed, gamma_r, centroids, D1, D2,
        scores, n_ctx, d, K, scale, qk_scale
    );
}

/* packing kernels — unchanged */
extern "C"
__global__ void qf_pack_khat(
    const int16_t* __restrict__ idx,
    int8_t*        __restrict__ out,
    int n, int d
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
    const float*  __restrict__ z,
    uint8_t*      __restrict__ out,
    int n, int d
) {
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int byte = blockIdx.y;
    if (i >= n || 8*byte+7 >= d) return;
    uint8_t packed = 0;
    for (int b = 0; b < 8; b++) {
        if (z[i*d + 8*byte + b] > 0.0f) packed |= (1 << b);
    }
    out[i*(d/8) + byte] = packed;
}

extern "C"
void launch_qf_pack_khat(const int16_t* idx, int8_t* out, int n, int d) {
    dim3 block(32); dim3 grid((n+31)/32, d/2);
    qf_pack_khat<<<grid, block>>>(idx, out, n, d);
}
extern "C"
void launch_qf_pack_z(const float* z, uint8_t* out, int n, int d) {
    dim3 block(32); dim3 grid((n+31)/32, d/8);
    qf_pack_z<<<grid, block>>>(z, out, n, d);
}
