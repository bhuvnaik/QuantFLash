/*
 * turbo_kernel.cu
 * Fused kernel for TurboQuant encoding pipeline.
 * Fuses stages S2+S4+S5:
 *   S2: codebook lookup  argmin_k |y_j - c_k|
 *   S4: residual         r = x - dequant(idx)
 *   S5: normalize        r_unit = r / ||r||
 *
 * gridDim.x = n, blockDim.x = next_pow2(d) <= 1024
 *
 * Compile:
 *   nvcc -O3 -arch=sm_86 -shared --compiler-options -fPIC \
 *        turbo_kernel.cu -o turbo_kernel.so
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

extern "C"
__global__ void turbo_encode_kernel(
    const float*  __restrict__ y,
    const float*  __restrict__ x,
    const float*  __restrict__ centroids,
    int16_t*      __restrict__ idx,
    float*        __restrict__ r_unit,
    float*        __restrict__ gamma,
    int d,
    int k
) {
    extern __shared__ float shmem[];
    float* s_centroids = shmem;
    float* s_reduce    = shmem + k;

    int vec_id = blockIdx.x;
    int coord  = threadIdx.x;
    int base   = vec_id * d;

    if (coord < k) {
        s_centroids[coord] = centroids[coord];
    }
    __syncthreads();

    float y_j = (coord < d) ? y[base + coord] : 0.0f;

    float best_dist = 1e30f;
    int   best_idx  = 0;
    for (int c = 0; c < k; c++) {
        float dist = fabsf(y_j - s_centroids[c]);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx  = c;
        }
    }

    if (coord < d) {
        idx[base + coord] = (int16_t) best_idx;
    }

    float r_j = (coord < d) ? (y_j - s_centroids[best_idx]) : 0.0f;

    s_reduce[coord] = r_j * r_j;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (coord < stride) {
            s_reduce[coord] += s_reduce[coord + stride];
        }
        __syncthreads();
    }

    float norm     = sqrtf(s_reduce[0]);
    float inv_norm = (norm > 1e-8f) ? (1.0f / norm) : 0.0f;

    if (coord == 0) {
        gamma[vec_id] = norm;
    }

    if (coord < d) {
        r_unit[base + coord] = r_j * inv_norm;
    }
}

extern "C"
__global__ void turbo_encode_tiled_kernel(
    const float*  __restrict__ y,
    const float*  __restrict__ x,
    const float*  __restrict__ centroids,
    int16_t*      __restrict__ idx,
    float*        __restrict__ r_unit,
    float*        __restrict__ gamma,
    int d,
    int k
) {
    extern __shared__ float shmem[];
    float* s_centroids = shmem;
    float* s_reduce    = shmem + k;

    int vec_id = blockIdx.x;
    int tid    = threadIdx.x;
    int base   = vec_id * d;

    for (int c = tid; c < k; c += blockDim.x) {
        s_centroids[c] = centroids[c];
    }
    __syncthreads();

    float partial = 0.0f;
    for (int coord = tid; coord < d; coord += blockDim.x) {
        float y_j = y[base + coord];
        float best_dist = 1e30f;
        int   best_c    = 0;
        for (int c = 0; c < k; c++) {
            float dist = fabsf(y_j - s_centroids[c]);
            if (dist < best_dist) { best_dist = dist; best_c = c; }
        }
        idx[base + coord]    = (int16_t) best_c;
        float r_j            = y_j - s_centroids[best_c];
        r_unit[base + coord] = r_j;
        partial             += r_j * r_j;
    }

    s_reduce[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_reduce[tid] += s_reduce[tid + stride];
        __syncthreads();
    }

    float norm     = sqrtf(s_reduce[0]);
    float inv_norm = (norm > 1e-8f) ? (1.0f / norm) : 0.0f;

    if (tid == 0) gamma[vec_id] = norm;
    __syncthreads();

    for (int coord = tid; coord < d; coord += blockDim.x) {
        r_unit[base + coord] *= inv_norm;
    }
}

extern "C"
void launch_turbo_encode(
    const float* y,
    const float* x,
    const float* centroids,
    int16_t*     idx,
    float*       r_unit,
    float*       gamma,
    int n, int d, int k
) {
    // next power of 2 >= d, capped at 1024
    int block = 1;
    while (block < d && block < 1024) block <<= 1;

    int shmem_bytes = (k + block) * sizeof(float);

    if (d <= 1024) {
        turbo_encode_kernel<<<n, block, shmem_bytes>>>(
            y, x, centroids, idx, r_unit, gamma, d, k
        );
    } else {
        int tile = 512;
        int shmem_tiled = (k + tile) * sizeof(float);
        turbo_encode_tiled_kernel<<<n, tile, shmem_tiled>>>(
            y, x, centroids, idx, r_unit, gamma, d, k
        );
    }
}
