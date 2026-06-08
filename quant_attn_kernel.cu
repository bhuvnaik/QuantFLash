#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define KEYS_PER_BLOCK 4

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

/*
 * quant_attn_warp_kernel
 *
 * Accepts pre-transformed query q_fwht = FWHT(q * signs) / sqrt(d).
 * No butterfly needed inside kernel — pure dot product computation.
 *
 * Grid:  (n_q, ceil(n_ctx/KEYS_PER_BLOCK))
 * Block: (32, KEYS_PER_BLOCK)
 *
 * Each warp handles one key.
 * Each lane handles d/32 coordinates.
 * Warp shuffle reduction — zero __syncthreads() in hot path.
 */
extern "C"
__global__ void quant_attn_warp_kernel(
    const float*   __restrict__ q_fwht,
    const int8_t*  __restrict__ k_hat,
    const uint8_t* __restrict__ z_packed,
    const float*   __restrict__ gamma,
    const float*   __restrict__ centroids,
    float*         __restrict__ scores,
    int n_ctx,
    int d,
    float scale,
    float qk_scale
) {
    // shared memory: q_fwht row + centroids
    extern __shared__ float shmem[];
    float* s_q    = shmem;        // d floats
    float* s_cent = shmem + d;    // 16 floats

    int q_id    = blockIdx.x;
    int tile_id = blockIdx.y;
    int warp_id = threadIdx.y;
    int lane    = threadIdx.x;
    int base_q  = q_id * d;

    int coords_per_thread = d / 32;
    int flat_tid  = threadIdx.y * blockDim.x + threadIdx.x;
    int n_threads = blockDim.x * blockDim.y;

    // load pre-transformed query into shared memory
    for (int i = flat_tid; i < d; i += n_threads)
        s_q[i] = q_fwht[base_q + i];

    // load centroids
    if (flat_tid < 16) s_cent[flat_tid] = centroids[flat_tid];
    __syncthreads();

    // each warp processes one key
    int ki = tile_id * KEYS_PER_BLOCK + warp_id;
    if (ki >= n_ctx) return;

    float gamma_ki = gamma[ki];
    int base_z = ki * (d / 8);

    float dot_mse = 0.0f;
    float dot_z   = 0.0f;

    for (int c = 0; c < coords_per_thread; c++) {
        int coord = lane * coords_per_thread + c;

        // unpack 4-bit index
        int8_t byte_val = k_hat[ki * (d/2) + coord/2];
        int idx = (coord % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
        dot_mse += s_q[coord] * s_cent[idx];

        // unpack 1-bit z
        uint8_t z_byte = z_packed[base_z + coord/8];
        float zv = ((z_byte >> (coord % 8)) & 1) ? 1.0f : -1.0f;
        dot_z += s_q[coord] * zv;
    }

    // warp reduce — no __syncthreads()
    dot_mse = warp_reduce_sum(dot_mse);
    dot_z   = warp_reduce_sum(dot_z);

    if (lane == 0) {
        scores[q_id * n_ctx + ki] =
            qk_scale * (dot_mse + gamma_ki * scale * dot_z);
    }
}

extern "C"
void launch_quant_attn(
    const float*   q_fwht,
    const int8_t*  k_hat,
    const uint8_t* z_packed,
    const float*   gamma,
    const float*   centroids,
    const float*   signs,   // unused — kept for API compatibility
    float*         scores,
    int n_q, int n_ctx, int d,
    float scale, float qk_scale
) {
    int n_tiles = (n_ctx + KEYS_PER_BLOCK - 1) / KEYS_PER_BLOCK;
    dim3 grid(n_q, n_tiles);
    dim3 block(32, KEYS_PER_BLOCK);
    int  shmem = (d + 16) * sizeof(float);
    quant_attn_warp_kernel<<<grid, block, shmem>>>(
        q_fwht, k_hat, z_packed, gamma, centroids,
        scores, n_ctx, d, scale, qk_scale
    );
}

extern "C"
__global__ void pack_khat_kernel(
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
__global__ void pack_z_kernel(
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
void launch_pack_khat(
    const int16_t* idx, int8_t* out, int n, int d
) {
    dim3 block(32);
    dim3 grid((n+31)/32, d/2);
    pack_khat_kernel<<<grid, block>>>(idx, out, n, d);
}

extern "C"
void launch_pack_z(
    const float* z, uint8_t* out, int n, int d
) {
    dim3 block(32);
    dim3 grid((n+31)/32, d/8);
    pack_z_kernel<<<grid, block>>>(z, out, n, d);
}
