#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

extern "C"
__global__ void fwht_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ signs,
    float*       __restrict__ out,
    int d, int log2d
) {
    extern __shared__ float s[];
    int vec_id = blockIdx.x;
    int tid    = threadIdx.x;   // tid in [0, d/2)
    int base   = vec_id * d;

    // each thread loads 2 elements
    s[2*tid]   = x[base + 2*tid]   * signs[2*tid];
    s[2*tid+1] = x[base + 2*tid+1] * signs[2*tid+1];
    __syncthreads();

    // butterfly: tid handles one pair per stage
    for (int stride = 1; stride < d; stride <<= 1) {
        int group  = tid / stride;
        int pos    = tid % stride;
        int i      = group * (stride << 1) + pos;
        int j      = i + stride;
        float a    = s[i];
        float b    = s[j];
        __syncthreads();
        s[i] = a + b;
        s[j] = a - b;
        __syncthreads();
    }

    float scale    = rsqrtf((float) d);
    out[base + 2*tid]   = s[2*tid]   * scale;
    out[base + 2*tid+1] = s[2*tid+1] * scale;
}

extern "C"
__global__ void fwht_inverse_kernel(
    const float* __restrict__ y,
    const float* __restrict__ signs,
    float*       __restrict__ out,
    int d, int log2d
) {
    extern __shared__ float s[];
    int vec_id = blockIdx.x;
    int tid    = threadIdx.x;   // tid in [0, d/2)
    int base   = vec_id * d;

    s[2*tid]   = y[base + 2*tid];
    s[2*tid+1] = y[base + 2*tid+1];
    __syncthreads();

    for (int stride = 1; stride < d; stride <<= 1) {
        int group  = tid / stride;
        int pos    = tid % stride;
        int i      = group * (stride << 1) + pos;
        int j      = i + stride;
        float a    = s[i];
        float b    = s[j];
        __syncthreads();
        s[i] = a + b;
        s[j] = a - b;
        __syncthreads();
    }

    float scale        = rsqrtf((float) d);
    out[base + 2*tid]   = s[2*tid]   * scale * signs[2*tid];
    out[base + 2*tid+1] = s[2*tid+1] * scale * signs[2*tid+1];
}

extern "C"
__global__ void fused_fwht_encode_kernel(
    const float*  __restrict__ x,
    const float*  __restrict__ signs,
    const float*  __restrict__ centroids,
    int16_t*      __restrict__ idx,
    float*        __restrict__ r_unit,
    float*        __restrict__ gamma,
    int d, int k
) {
    extern __shared__ float shmem[];
    float* s_work      = shmem;
    float* s_centroids = shmem + d;
    float* s_reduce    = shmem + d + k;

    int vec_id = blockIdx.x;
    int tid    = threadIdx.x;   // tid in [0, d/2)
    int base   = vec_id * d;

    // load centroids
    if (2*tid   < k) s_centroids[2*tid]   = centroids[2*tid];
    if (2*tid+1 < k) s_centroids[2*tid+1] = centroids[2*tid+1];

    // load x with sign flip
    s_work[2*tid]   = x[base + 2*tid]   * signs[2*tid];
    s_work[2*tid+1] = x[base + 2*tid+1] * signs[2*tid+1];
    __syncthreads();

    // butterfly
    for (int stride = 1; stride < d; stride <<= 1) {
        int group = tid / stride;
        int pos   = tid % stride;
        int i     = group * (stride << 1) + pos;
        int j     = i + stride;
        float a   = s_work[i];
        float b   = s_work[j];
        __syncthreads();
        s_work[i] = a + b;
        s_work[j] = a - b;
        __syncthreads();
    }

    float scale = rsqrtf((float) d);
    float y0    = s_work[2*tid]   * scale;
    float y1    = s_work[2*tid+1] * scale;

    // codebook lookup for both coordinates
    float bd0 = 1e30f, bd1 = 1e30f;
    int   bc0 = 0,     bc1 = 0;
    for (int c = 0; c < k; c++) {
        float cv = s_centroids[c];
        float d0 = fabsf(y0 - cv);
        float d1 = fabsf(y1 - cv);
        if (d0 < bd0) { bd0 = d0; bc0 = c; }
        if (d1 < bd1) { bd1 = d1; bc1 = c; }
    }
    idx[base + 2*tid]   = (int16_t) bc0;
    idx[base + 2*tid+1] = (int16_t) bc1;

    float r0 = y0 - s_centroids[bc0];
    float r1 = y1 - s_centroids[bc1];

    // reduction for ||r||^2 — each thread contributes 2 elements
    s_reduce[tid] = r0*r0 + r1*r1;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_reduce[tid] += s_reduce[tid + stride];
        __syncthreads();
    }

    float norm     = sqrtf(s_reduce[0]);
    float inv_norm = (norm > 1e-8f) ? (1.0f / norm) : 0.0f;

    if (tid == 0) gamma[vec_id] = norm;

    r_unit[base + 2*tid]   = r0 * inv_norm;
    r_unit[base + 2*tid+1] = r1 * inv_norm;
}

extern "C"
void launch_fwht_forward(
    const float* x, const float* signs, float* out, int n, int d
) {
    int log2d = 0, tmp = d;
    while (tmp > 1) { log2d++; tmp >>= 1; }
    // blockDim = d/2 (each thread handles 2 elements)
    fwht_forward_kernel<<<n, d/2, d * sizeof(float)>>>(x, signs, out, d, log2d);
}

extern "C"
void launch_fwht_inverse(
    const float* y, const float* signs, float* out, int n, int d
) {
    int log2d = 0, tmp = d;
    while (tmp > 1) { log2d++; tmp >>= 1; }
    fwht_inverse_kernel<<<n, d/2, d * sizeof(float)>>>(y, signs, out, d, log2d);
}

extern "C"
void launch_fused_fwht_encode(
    const float* x, const float* signs, const float* centroids,
    int16_t* idx, float* r_unit, float* gamma,
    int n, int d, int k
) {
    // blockDim = d/2, shmem = d + k + d/2 floats
    int shmem = (2*d + k) * sizeof(float);
    fused_fwht_encode_kernel<<<n, d/2, shmem>>>(
        x, signs, centroids, idx, r_unit, gamma, d, k
    );
}
