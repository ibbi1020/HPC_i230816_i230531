/*********************************************************************
 * interpolate_cuda.cu
 * 
 * CUDA implementation of bilinear interpolation for KLT tracking
 * This is a naive implementation that manually performs bilinear
 * interpolation without using texture memory.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "interpolate_cuda.h"

// Kernel stays in C++ namespace (kernels are always C++)
/*********************************************************************
 * naiveInterpolateKernel
 * 
 * CUDA kernel that performs bilinear interpolation for multiple points
 * Each thread processes one interpolation point.
 * 
 * Parameters:
 *   img - flattened image data (height x width)
 *   width - image width
 *   height - image height  
 *   coords - array of (x,y) coordinates [x0,y0, x1,y1, x2,y2, ...]
 *   results - output array for interpolated values
 *   numPoints - number of points to interpolate
 */
__global__ void naiveInterpolateKernel(
    const float* img,
    int width,
    int height,
    const float* coords,
    float* results,
    int numPoints)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numPoints) return;
    
    // Get x,y coordinates for this point
    float x = coords[idx * 2];
    float y = coords[idx * 2 + 1];
    
    // Get integer parts
    int xt = (int)x;
    int yt = (int)y;
    
    // Boundary check
    if (xt < 0 || yt < 0 || xt >= width - 1 || yt >= height - 1) {
        results[idx] = 0.0f;
        return;
    }
    
    // Get fractional parts
    float ax = x - xt;
    float ay = y - yt;
    
    // Get the four surrounding pixels
    int offset = yt * width + xt;
    float p00 = img[offset];           // (xt, yt)
    float p10 = img[offset + 1];       // (xt+1, yt)
    float p01 = img[offset + width];   // (xt, yt+1)
    float p11 = img[offset + width + 1]; // (xt+1, yt+1)
    
    // Bilinear interpolation
    results[idx] = (1 - ax) * (1 - ay) * p00 +
                   ax * (1 - ay) * p10 +
                   (1 - ax) * ay * p01 +
                   ax * ay * p11;
}

// C linkage for functions called from C code
extern "C" {

/*********************************************************************
 * cudaNaiveInterpolate
 * 
 * Host function that manages CUDA memory and kernel launch for
 * batch interpolation of multiple points.
 * 
 * Parameters:
 *   img - host image data (height x width, row-major)
 *   width - image width
 *   height - image height
 *   coords - host array of (x,y) coordinates [x0,y0, x1,y1, ...]
 *   results - host output array for interpolated values
 *   numPoints - number of points to interpolate
 */
void cudaNaiveInterpolate(const float* img, int width, int height,const float* coords, float* results, int numPoints)
{
    // Device pointers
    float *d_img, *d_coords, *d_results;
    
    // Calculate sizes
    size_t imgSize = width * height * sizeof(float);
    size_t coordsSize = numPoints * 2 * sizeof(float);
    size_t resultsSize = numPoints * sizeof(float);
    
    // Allocate device memory
    cudaMalloc(&d_img, imgSize);
    cudaMalloc(&d_coords, coordsSize);
    cudaMalloc(&d_results, resultsSize);
    
    // Copy data to device
    cudaMemcpy(d_img, img, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_coords, coords, coordsSize, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;
    
    naiveInterpolateKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_img, width, height, d_coords, d_results, numPoints);
    
    // Copy results back
    cudaMemcpy(results, d_results, resultsSize, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_img);
    cudaFree(d_coords);
    cudaFree(d_results);
}

/*********************************************************************
 * cudaNaiveInterpolatePersistent
 * 
 * Host function for persistent GPU memory (when processing multiple
 * batches with the same image). The caller manages device memory.
 * 
 * Parameters:
 *   d_img - device image data (already on GPU)
 *   width - image width
 *   height - image height
 *   coords - host array of (x,y) coordinates
 *   results - host output array
 *   numPoints - number of points to interpolate
 */
void cudaNaiveInterpolatePersistent(const float* d_img, int width, int height, const float* coords, float* results, int numPoints)
{
    float *d_coords, *d_results;
    
    size_t coordsSize = numPoints * 2 * sizeof(float);
    size_t resultsSize = numPoints * sizeof(float);
    
    cudaMalloc(&d_coords, coordsSize);
    cudaMalloc(&d_results, resultsSize);
    
    cudaMemcpy(d_coords, coords, coordsSize, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;
    
    naiveInterpolateKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_img, width, height, d_coords, d_results, numPoints);
    
    cudaMemcpy(results, d_results, resultsSize, cudaMemcpyDeviceToHost);
    
    cudaFree(d_coords);
    cudaFree(d_results);
}

}
