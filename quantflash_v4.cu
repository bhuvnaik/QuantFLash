
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define TILE_K 64

extern "C"
__global__ void quantflash_v4(
    const float*   __restrict__ q_fwht,
    const int8_t*  __restrict__ k_hat,
    const uint8_t* __restrict__ z_packed,
    const float*   __restrict__ gamma,    /* (n_ctx,) original key norms  */
    const float*   __restrict__ gamma_r,  /* (n_ctx,) residual norms      */
    const float*   __restrict__ centroids,
    float*         __restrict__ scores,
    int n_ctx, int d, int K,
    float scale, float qk_scale
) {
    extern __shared__ float shmem[];
    float* s_q   = shmem;
    float* s_lut = shmem + d;

    int q_id    = blockIdx.x;
    int tile_id = blockIdx.y;
    int tid     = threadIdx.x;
    int d2      = d / 2;
    int d8      = d / 8;
    int key_start = tile_id * TILE_K;
    int key_end   = min(key_start + TILE_K, n_ctx);
    int base_q    = q_id * d;

    /* load query */
    s_q[tid]          = q_fwht[base_q + tid];
    s_q[tid + TILE_K] = q_fwht[base_q + tid + TILE_K];
    __syncthreads();

    /* build LUT */
    int lut_total = d * K;
    for (int idx = tid; idx < lut_total; idx += TILE_K) {
        s_lut[idx] = s_q[idx / K] * centroids[idx % K];
    }
    __syncthreads();

    int ki = key_start + tid;
    if (ki >= key_end) return;

    /* codebook dot product via LUT */
    float dot_lut = 0.0f;
    for (int j = 0; j < d2; j++) {
        int8_t packed = k_hat[ki * d2 + j];
        dot_lut += s_lut[(2*j)   * K + ((packed     ) & 0x0F)];
        dot_lut += s_lut[(2*j+1) * K + ((packed >> 4) & 0x0F)];
    }

    /* binary residual dot product */
    float dot_z = 0.0f;
    for (int j = 0; j < d; j++) {
        uint8_t zb = z_packed[ki * d8 + j / 8];
        dot_z += s_q[j] * (((zb >> (j % 8)) & 1) ? 1.0f : -1.0f);
    }

    /* apply gamma: original key norm scaling */
    float gk = gamma[ki];
    scores[q_id * n_ctx + ki] =
        qk_scale * gk * (dot_lut + gamma_r[ki] * scale * dot_z);
}

extern "C"
void launch_quantflash_v4(
    const float*   q_fwht,
    const int8_t*  k_hat,
    const uint8_t* z_packed,
    const float*   gamma,
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
    quantflash_v4<<<grid, block, shmem>>>(
        q_fwht, k_hat, z_packed, gamma, gamma_r, centroids,
        scores, n_ctx, d, K, scale, qk_scale
    );
}

extern "C"
__global__ void qf_pack_khat(
    const int16_t* __restrict__ idx, int8_t* __restrict__ out, int n, int d
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; int j = blockIdx.y;
    if (i >= n || 2*j+1 >= d) return;
    int lo = idx[i*d + 2*j] & 0x0F; int hi = idx[i*d + 2*j+1] & 0x0F;
    out[i*(d/2) + j] = (int8_t)((hi << 4) | lo);
}
extern "C"
__global__ void qf_pack_z(
    const float* __restrict__ z, uint8_t* __restrict__ out, int n, int d
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; int byte = blockIdx.y;
    if (i >= n || 8*byte+7 >= d) return;
    uint8_t packed = 0;
    for (int b = 0; b < 8; b++)
        if (z[i*d + 8*byte + b] > 0.0f) packed |= (1 << b);
    out[i*(d/8) + byte] = packed;
}
extern "C"
void launch_qf_pack_khat(const int16_t* idx, int8_t* out, int n, int d) {
    dim3 block(32); dim3 grid((n+31)/32, d/2);
    qf_pack_khat<<<grid,block>>>(idx, out, n, d); }
extern "C"
void launch_qf_pack_z(const float* z, uint8_t* out, int n, int d) {
    dim3 block(32); dim3 grid((n+31)/32, d/8);
    qf_pack_z<<<grid,block>>>(z, out, n, d); }
