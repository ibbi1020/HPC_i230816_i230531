/*********************************************************************
 * subsample_cuda.cu
 *
 * GPU-accelerated image subsampling for pyramid construction
 *********************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include "subsample_cuda.h"

/*********************************************************************
 * Subsample kernel - takes every 2nd pixel from source image
 * 
 * Grid configuration: 2D blocks covering destination image
 * Each thread computes one pixel in destination (samples from source)
 *********************************************************************/
__global__ void subsampleKernel(
    const float* d_src,
    int src_width,
    int src_height,
    float* d_dst,
    int dst_width,
    int dst_height
) {
    // Destination coordinates
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary check
    if (dst_x >= dst_width || dst_y >= dst_height) return;
    
    // Source coordinates (sample every 2nd pixel)
    int src_x = dst_x * 2;
    int src_y = dst_y * 2;
    
    // Copy pixel from source to destination
    int dst_idx = dst_y * dst_width + dst_x;
    int src_idx = src_y * src_width + src_x;
    
    d_dst[dst_idx] = d_src[src_idx];
}

/*********************************************************************
 * Host wrapper for subsampling
 *********************************************************************/
extern "C"
void cudaSubsampleImage(
    const float* d_src,
    int src_width,
    int src_height,
    float* d_dst
) {
    // Destination dimensions (half of source)
    int dst_width = src_width / 2;
    int dst_height = src_height / 2;
    
    // Configure kernel launch (16x16 thread blocks)
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (dst_width + blockDim.x - 1) / blockDim.x,
        (dst_height + blockDim.y - 1) / blockDim.y
    );
    
    // Launch kernel
    subsampleKernel<<<gridDim, blockDim>>>(
        d_src, src_width, src_height,
        d_dst, dst_width, dst_height
    );
    
    // Check for errors (synchronous for debugging)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA subsample kernel error: %s\n", cudaGetErrorString(err));
    }
}
