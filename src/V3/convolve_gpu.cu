// convolve_gpu.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include "convolve_gpu.h"

/*********************************************************************
 * SHARED MEMORY TILING CONFIGURATION
 *********************************************************************/
#define TILE_WIDTH 16
#define TILE_HEIGHT 16
#define MAX_KERNEL_RADIUS 10  // Supports up to 21-element kernels
#define MAX_KERNEL_WIDTH (2 * MAX_KERNEL_RADIUS + 1)
#define SHARED_MEM_HORIZ_WIDTH (TILE_WIDTH + 2 * MAX_KERNEL_RADIUS)
#define SHARED_MEM_VERT_HEIGHT (TILE_HEIGHT + 2 * MAX_KERNEL_RADIUS)

/*********************************************************************
 * SHARED MEMORY KERNELS (Optimized)
 *********************************************************************/

__global__ void convolveHorizShared(
    const float* input,
    const float* kernel_data,
    float* output,
    int width,
    int height,
    int kernel_width)
{
    // Shared memory for image tile (with halos)
    __shared__ float s_tile[TILE_HEIGHT][SHARED_MEM_HORIZ_WIDTH];
    
    // Shared memory for kernel
    __shared__ float s_kernel[MAX_KERNEL_WIDTH];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + tx;
    int row = blockIdx.y * TILE_HEIGHT + ty;
    int kernel_radius = kernel_width / 2;
    
    // ==== PHASE 1: Load kernel into shared memory ====
    if (tx == 0 && ty < kernel_width) {
        s_kernel[ty] = kernel_data[kernel_width - 1 - ty]; // Reverse for convolution
    }
    
    // ==== PHASE 2: Cooperatively load image tile with halos ====
    
    // Load main tile (with halo offset)
    int tile_col = tx + kernel_radius;
    if (col < width && row < height) {
        s_tile[ty][tile_col] = input[row * width + col];
    } else {
        s_tile[ty][tile_col] = 0.0f;  // Zero-pad out-of-bounds
    }
    
    // Load LEFT halo (first KERNEL_RADIUS threads)
    if (tx < kernel_radius) {
        int halo_col = col - kernel_radius;
        if (halo_col >= 0 && row < height) {
            s_tile[ty][tx] = input[row * width + halo_col];
        } else {
            s_tile[ty][tx] = 0.0f;  // Zero-pad left border
        }
    }
    
    // Load RIGHT halo (first KERNEL_RADIUS threads load right side)
    if (tx < kernel_radius) {
        int halo_col = col + TILE_WIDTH;
        int tile_idx = tx + TILE_WIDTH + kernel_radius;
        if (halo_col < width && row < height) {
            s_tile[ty][tile_idx] = input[row * width + halo_col];
        } else {
            s_tile[ty][tile_idx] = 0.0f;  // Zero-pad right border
        }
    }
    
    __syncthreads();  // Ensure all data loaded before computation
    
    // ==== PHASE 3: Compute convolution from shared memory ====
    if (col < width && row < height) {
        // Zero border pixels (outside kernel radius)
        if (col < kernel_radius || col >= width - kernel_radius) {
            output[row * width + col] = 0.0f;
        } else {
            float sum = 0.0f;
            for (int k = 0; k < kernel_width; k++) {
                sum += s_tile[ty][tx + k] * s_kernel[k];
            }
            output[row * width + col] = sum;
        }
    }
}

__global__ void convolveVertShared(
    const float* input,
    const float* kernel_data,
    float* output,
    int width,
    int height,
    int kernel_width)
{
    // Shared memory for image tile (with vertical halos)
    __shared__ float s_tile[SHARED_MEM_VERT_HEIGHT][TILE_WIDTH];
    
    // Shared memory for kernel
    __shared__ float s_kernel[MAX_KERNEL_WIDTH];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + tx;
    int row = blockIdx.y * TILE_HEIGHT + ty;
    int kernel_radius = kernel_width / 2;
    
    // ==== PHASE 1: Load kernel into shared memory ====
    if (tx == 0 && ty < kernel_width) {
        s_kernel[ty] = kernel_data[kernel_width - 1 - ty];
    }
    
    // ==== PHASE 2: Cooperatively load image tile with vertical halos ====
    
    // Load main tile (with halo offset)
    int tile_row = ty + kernel_radius;
    if (col < width && row < height) {
        s_tile[tile_row][tx] = input[row * width + col];
    } else {
        s_tile[tile_row][tx] = 0.0f;
    }
    
    // Load TOP halo (first KERNEL_RADIUS threads in y)
    if (ty < kernel_radius) {
        int halo_row = row - kernel_radius;
        if (halo_row >= 0 && col < width) {
            s_tile[ty][tx] = input[halo_row * width + col];
        } else {
            s_tile[ty][tx] = 0.0f;  // Zero-pad top border
        }
    }
    
    // Load BOTTOM halo (first KERNEL_RADIUS threads load bottom)
    if (ty < kernel_radius) {
        int halo_row = row + TILE_HEIGHT;
        int tile_idx = ty + TILE_HEIGHT + kernel_radius;
        if (halo_row < height && col < width) {
            s_tile[tile_idx][tx] = input[halo_row * width + col];
        } else {
            s_tile[tile_idx][tx] = 0.0f;  // Zero-pad bottom border
        }
    }
    
    __syncthreads();
    
    // ==== PHASE 3: Compute convolution from shared memory ====
    if (col < width && row < height) {
        // Zero border pixels
        if (row < kernel_radius || row >= height - kernel_radius) {
            output[row * width + col] = 0.0f;
        } else {
            float sum = 0.0f;
            for (int k = 0; k < kernel_width; k++) {
                sum += s_tile[ty + k][tx] * s_kernel[k];
            }
            output[row * width + col] = sum;
        }
    }
}

/*********************************************************************
 * NAIVE KERNELS (Original - kept for fallback)
 *********************************************************************/

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

/*********************************************************************
 * SHARED MEMORY WRAPPERS (Optimized)
 *********************************************************************/

void launchConvolveHorizShared(
    const float* d_input,
    const float* d_kernel,
    float* d_output,
    int width,
    int height,
    int kernel_width)
{
    // Validate kernel size
    if (kernel_width > MAX_KERNEL_WIDTH) {
        fprintf(stderr, "Error: kernel width %d exceeds MAX_KERNEL_WIDTH %d\n",
                kernel_width, MAX_KERNEL_WIDTH);
        return;
    }
    
    dim3 blockDim(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridDim((width + TILE_WIDTH - 1) / TILE_WIDTH,
                 (height + TILE_HEIGHT - 1) / TILE_HEIGHT);
    
    convolveHorizShared<<<gridDim, blockDim>>>(
        d_input, d_kernel, d_output,
        width, height, kernel_width
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in convolveHorizShared: %s\n",
                cudaGetErrorString(err));
    }
}

void launchConvolveVertShared(
    const float* d_input,
    const float* d_kernel,
    float* d_output,
    int width,
    int height,
    int kernel_width)
{
    if (kernel_width > MAX_KERNEL_WIDTH) {
        fprintf(stderr, "Error: kernel width %d exceeds MAX_KERNEL_WIDTH %d\n",
                kernel_width, MAX_KERNEL_WIDTH);
        return;
    }
    
    dim3 blockDim(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridDim((width + TILE_WIDTH - 1) / TILE_WIDTH,
                 (height + TILE_HEIGHT - 1) / TILE_HEIGHT);
    
    convolveVertShared<<<gridDim, blockDim>>>(
        d_input, d_kernel, d_output,
        width, height, kernel_width
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in convolveVertShared: %s\n",
                cudaGetErrorString(err));
    }
}

} // extern "C"