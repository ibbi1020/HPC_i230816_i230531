#include <cuda_runtime.h>
#include <stdio.h>
#include "convolve_gpu.h"

// ============================================================
// Optional constant memory
// ============================================================
__constant__ float c_Kernel[64];

__device__ __forceinline__ float k_at(int k, int R, const float* kernel)
{
#ifdef USE_CONST
    return c_Kernel[R - k];
#else
    return kernel[R - k];
#endif
}


// ============================================================
// Shared horizontal convolution (FIXED tile loads)
// ============================================================
template<int R>
__global__
void horiz_shared(const float* __restrict__ in,
                  float* __restrict__ out,
                  const float* __restrict__ kernel,
                  int w, int h, int pitch)
{
    const int BW = 32;
    const int BH = 8;

    __shared__ float tile[BH][BW + 2*R];

    const int gx0 = blockIdx.x * BW;     // block origin (x)
    const int gy0 = blockIdx.y * BH;     // block origin (y)

    const int tx  = threadIdx.x;         // 0..BW-1
    const int ty  = threadIdx.y;         // 0..BH-1

    const int x   = gx0 + tx;
    const int y   = gy0 + ty;

    // ---- LOAD TILE + HALO (from block-aligned base) ----
    const int start_x = gx0 - R;         // base column for this tile row

    // Each thread loads 1..2 elements of its row (covers BW+2R)
    for (int i = tx; i < BW + 2*R; i += BW) {
        int gx = start_x + i;
        float v = 0.0f;
        if (y < h && gx >= 0 && gx < w) {
            v = in[(size_t)y * pitch + gx];
        }
        tile[ty][i] = v;
    }
    __syncthreads();

    if (x >= w || y >= h) return;

    // Zero-pad edges like CPU reference
    if (x < R || x >= w - R) {
        out[(size_t)y * pitch + x] = 0.0f;
        return;
    }

    float acc = 0.0f;
    #pragma unroll
    for (int k = -R; k <= R; k++) {
        float v = tile[ty][tx + k + R];
        acc += v * k_at(k, R, kernel);
    }
    out[(size_t)y * pitch + x] = acc;
}


// ============================================================
// Shared vertical convolution (FIXED tile loads)
// ============================================================
template<int R>
__global__
void vert_shared(const float* __restrict__ in,
                 float* __restrict__ out,
                 const float* __restrict__ kernel,
                 int w, int h, int pitch)
{
    const int BW = 32;
    const int BH = 8;

    __shared__ float tile[BH + 2*R][BW];

    const int gx0 = blockIdx.x * BW;     // block origin (x)
    const int gy0 = blockIdx.y * BH;     // block origin (y)

    const int tx  = threadIdx.x;         // 0..BW-1
    const int ty  = threadIdx.y;         // 0..BH-1

    const int x   = gx0 + tx;
    const int y   = gy0 + ty;

    // ---- LOAD TILE + HALO (from block-aligned base) ----
    const int start_y = gy0 - R;         // base row for this tile column

    // Each thread loads 1..2 elements of its column (covers BH+2R)
    for (int i = ty; i < BH + 2*R; i += BH) {
        int gy = start_y + i;
        float v = 0.0f;
        if (x < w && gy >= 0 && gy < h) {
            v = in[gy * pitch + x];
        }
        tile[i][tx] = v;
    }
    __syncthreads();

    if (x >= w || y >= h) return;

    // Zero-pad edges like CPU reference
    if (y < R || y >= h - R) {
        out[(size_t)y * pitch + x] = 0.0f;
        return;
    }

    float acc = 0.0f;
    #pragma unroll
    for (int k = -R; k <= R; k++) {
        float v = tile[ty + k + R][tx];
        acc += v * k_at(k, R, kernel);
    }
    out[(size_t)y * pitch + x] = acc;
}



// ============================================================
// Constant-memory upload
// ============================================================
extern "C"
void uploadKernelToConst(const float* hKernel, int K)
{
    cudaMemcpyToSymbol(c_Kernel, hKernel, K * sizeof(float));
}


// ============================================================
// Dispatch helpers (NO templates inside extern "C")
// ============================================================
template<int R> void launch_horiz(
    const float* d_in, float* d_out, const float* d_kernel,
    int w, int h, int pitch, cudaStream_t s)
{
    dim3 block(32,8);
    dim3 grid((w+31)/32, (h+7)/8);
    horiz_shared<R><<<grid, block, 0, s>>>(d_in, d_out, d_kernel, w, h, pitch);
}

template<int R> void launch_vert(
    const float* d_in, float* d_out, const float* d_kernel,
    int w, int h, int pitch, cudaStream_t s)
{
    dim3 block(32,8);
    dim3 grid((w+31)/32, (h+7)/8);
    vert_shared<R><<<grid, block, 0, s>>>(d_in, d_out, d_kernel, w, h, pitch);
}


// ============================================================
// PUBLIC API — SAFE C LINKAGE
// ============================================================
extern "C"
void runHorizontalConvolution(const float* d_in, float* d_out,
                              const float* d_kernel, int w, int h,
                              int pitch, int R, cudaStream_t s)
{
    switch (R) {
        case 1: launch_horiz<1>(d_in, d_out, d_kernel, w, h, pitch, s); break;
        case 2: launch_horiz<2>(d_in, d_out, d_kernel, w, h, pitch, s); break;
        case 3: launch_horiz<3>(d_in, d_out, d_kernel, w, h, pitch, s); break;
        case 4: launch_horiz<4>(d_in, d_out, d_kernel, w, h, pitch, s); break;
        default:
            printf("Unsupported R=%d\n", R);
    }
}

extern "C"
void runVerticalConvolution(const float* d_in, float* d_out,
                            const float* d_kernel, int w, int h,
                            int pitch, int R, cudaStream_t s)
{
    switch (R) {
        case 1: launch_vert<1>(d_in, d_out, d_kernel, w, h, pitch, s); break;
        case 2: launch_vert<2>(d_in, d_out, d_kernel, w, h, pitch, s); break;
        case 3: launch_vert<3>(d_in, d_out, d_kernel, w, h, pitch, s); break;
        case 4: launch_vert<4>(d_in, d_out, d_kernel, w, h, pitch, s); break;
        default:
            printf("Unsupported R=%d\n", R);
    }
}