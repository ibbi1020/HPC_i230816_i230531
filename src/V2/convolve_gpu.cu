// convolve_gpu.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include "convolve_gpu.h"

// GPU kernel for horizontal convolution
__global__ void convolveHorizKernel(
    const float *imgin,
    const float *kernel_data,
    float *imgout,
    int ncols,
    int nrows,
    int kernel_width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // column
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // row
    
    if (i >= ncols || j >= nrows) return;
    
    int radius = kernel_width / 2;
    int out_idx = j * ncols + i;
    
    // Zero border columns
    if (i < radius || i >= ncols - radius) {
        imgout[out_idx] = 0.0f;
        return;
    }
    
    // Convolve middle columns
    float sum = 0.0f;
    for (int k = 0; k < kernel_width; k++) {
        int in_idx = j * ncols + (i - radius + k);
        sum += imgin[in_idx] * kernel_data[kernel_width - 1 - k];
    }
    imgout[out_idx] = sum;
}

// GPU kernel for vertical convolution
__global__ void convolveVertKernel(
    const float *imgin,
    const float *kernel_data,
    float *imgout,
    int ncols,
    int nrows,
    int kernel_width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // column
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // row
    
    if (i >= ncols || j >= nrows) return;
    
    int radius = kernel_width / 2;
    int out_idx = j * ncols + i;
    
    // Zero border rows
    if (j < radius || j >= nrows - radius) {
        imgout[out_idx] = 0.0f;
        return;
    }
    
    // Convolve middle rows
    float sum = 0.0f;
    for (int k = 0; k < kernel_width; k++) {
        int in_idx = (j - radius + k) * ncols + i;
        sum += imgin[in_idx] * kernel_data[kernel_width - 1 - k];
    }
    imgout[out_idx] = sum;
}

// Host wrapper functions that can be called from C code
extern "C" {

void launchConvolveHorizKernel(
    const float *d_imgin,
    const float *d_kernel_data,
    float *d_imgout,
    int ncols,
    int nrows,
    int kernel_width,
    int gridDimX,
    int gridDimY,
    int blockDimX,
    int blockDimY)
{
    dim3 gridDim(gridDimX, gridDimY);
    dim3 blockDim(blockDimX, blockDimY);
    convolveHorizKernel<<<gridDim, blockDim>>>(
        d_imgin, d_kernel_data, d_imgout, ncols, nrows, kernel_width);
}

void launchConvolveVertKernel(
    const float *d_imgin,
    const float *d_kernel_data,
    float *d_imgout,
    int ncols,
    int nrows,
    int kernel_width,
    int gridDimX,
    int gridDimY,
    int blockDimX,
    int blockDimY)
{
    dim3 gridDim(gridDimX, gridDimY);
    dim3 blockDim(blockDimX, blockDimY);
    convolveVertKernel<<<gridDim, blockDim>>>(
        d_imgin, d_kernel_data, d_imgout, ncols, nrows, kernel_width);
}

} // extern "C"