/*********************************************************************
 * mineigenvalue_cuda.cu
 * 
 * CUDA implementation of minimum eigenvalue computation for corner detection
 * This is used in KLT feature selection to find good features to track.
 *********************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include "mineigenvalue_cuda.h"

/*********************************************************************
 * minEigenvalueKernel
 * 
 * CUDA kernel that computes minimum eigenvalue of structure tensor
 * for each pixel. This measures "cornerness" - how much the image
 * gradient changes in multiple directions.
 * 
 * Each thread processes one pixel and computes eigenvalue from
 * gradients in a window around that pixel.
 * 
 * Parameters:
 *   gradx - image gradient in x direction
 *   grady - image gradient in y direction
 *   eigenvalues - output array of eigenvalues (one per pixel)
 *   ncols - image width
 *   nrows - image height
 *   window_hw - half-width of window (e.g., 3 for 7x7 window)
 */
__global__ void minEigenvalueKernel(
    const float* gradx,
    const float* grady,
    float* eigenvalues,
    int ncols,
    int nrows,
    int window_hw)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if pixel is within image bounds
    if (x >= ncols || y >= nrows) {
        return;
    }
    
    // Check if pixel is too close to border (invalid)
    if (x < window_hw || x >= ncols - window_hw ||
        y < window_hw || y >= nrows - window_hw) {
        eigenvalues[y * ncols + x] = -1.0f;  // Mark as invalid
        return;
    }
    
    // Compute structure tensor elements over window
    // Structure tensor: | gxx  gxy |
    //                   | gxy  gyy |
    float gxx = 0.0f;
    float gxy = 0.0f;
    float gyy = 0.0f;
    
    for (int j = -window_hw; j <= window_hw; j++) {
        for (int i = -window_hw; i <= window_hw; i++) {
            int idx = (y + j) * ncols + (x + i);
            float gx = gradx[idx];
            float gy = grady[idx];
            
            gxx += gx * gx;
            gxy += gx * gy;
            gyy += gy * gy;
        }
    }
    
    // Compute minimum eigenvalue of 2x2 structure tensor
    // Formula: min_eigenvalue = (trace - sqrt(trace^2 - 4*det)) / 2
    // where trace = gxx + gyy, det = gxx*gyy - gxy^2
    float trace = gxx + gyy;
    float det = gxx * gyy - gxy * gxy;
    float discriminant = sqrtf(trace * trace - 4.0f * det);
    float min_eigenval = (trace - discriminant) * 0.5f;
    
    eigenvalues[y * ncols + x] = min_eigenval;
}

// C linkage for functions called from C code
extern "C" {

/*********************************************************************
 * cudaComputeMinEigenvalues
 * 
 * Host function that computes minimum eigenvalues for all pixels
 * using GPU acceleration. Handles memory allocation and transfer.
 * 
 * Parameters:
 *   gradx - host gradient array in x direction
 *   grady - host gradient array in y direction
 *   eigenvalues - host output array (must be pre-allocated)
 *   ncols - image width
 *   nrows - image height
 *   window_hw - half-width of window for eigenvalue computation
 */
void cudaComputeMinEigenvalues(const float* gradx, const float* grady, float* eigenvalues, int ncols, int nrows, int window_hw)
{
    // Device pointers
    float *d_gradx, *d_grady, *d_eigenvalues;
    
    // Calculate size
    size_t size = ncols * nrows * sizeof(float);
    
    // Allocate device memory
    cudaMalloc(&d_gradx, size);
    cudaMalloc(&d_grady, size);
    cudaMalloc(&d_eigenvalues, size);
    
    // Copy input data to device
    cudaMemcpy(d_gradx, gradx, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grady, grady, size, cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    // Use 16x16 thread blocks (256 threads per block)
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (ncols + blockSize.x - 1) / blockSize.x,
        (nrows + blockSize.y - 1) / blockSize.y
    );
    
    // Launch kernel
    minEigenvalueKernel<<<gridSize, blockSize>>>(
        d_gradx, d_grady, d_eigenvalues,
        ncols, nrows, window_hw
    );
    
    // Copy results back to host
    cudaMemcpy(eigenvalues, d_eigenvalues, size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_gradx);
    cudaFree(d_grady);
    cudaFree(d_eigenvalues);
}

/*********************************************************************
 * cudaComputeMinEigenvaluesWithGradients
 * 
 * Version that takes gradient images already on GPU (for future
 * optimization when gradients are computed on GPU).
 * 
 * Parameters:
 *   d_gradx - device gradient array in x direction (already on GPU)
 *   d_grady - device gradient array in y direction (already on GPU)
 *   eigenvalues - host output array
 *   ncols - image width
 *   nrows - image height
 *   window_hw - half-width of window
 */
void cudaComputeMinEigenvaluesWithGradients(const float* d_gradx, const float* d_grady, float* eigenvalues, int ncols, int nrows, int window_hw)
{
    float *d_eigenvalues;
    size_t size = ncols * nrows * sizeof(float);
    
    cudaMalloc(&d_eigenvalues, size);
    
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (ncols + blockSize.x - 1) / blockSize.x,
        (nrows + blockSize.y - 1) / blockSize.y
    );
    
    minEigenvalueKernel<<<gridSize, blockSize>>>(
        d_gradx, d_grady, d_eigenvalues,
        ncols, nrows, window_hw
    );
    
    cudaMemcpy(eigenvalues, d_eigenvalues, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_eigenvalues);
}

} 
