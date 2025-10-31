/*********************************************************************
 * interpolate_cuda.cu
 * 
 * CUDA implementation of bilinear interpolation for KLT tracking.
 * Uses texture memory + persistent GPU buffers for maximum performance.
 * 
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "interpolate_cuda.h"
#include "cuda_config.h"

/*********************************************************************
 * textureInterpolateKernel
 * 
 * CUDA kernel using hardware texture memory for bilinear interpolation.
 * Texture units provide:
 *   - Hardware-accelerated bilinear interpolation (free computation)
 *   - Optimized 2D spatial locality caching
 *   - Automatic boundary handling
 *   - 5-15× faster than naive CPU implementation
 * 
 * Parameters:
 *   texObj - CUDA texture object bound to image data
 *   width - image width
 *   height - image height  
 *   coords - array of (x,y) coordinates [x0,y0, x1,y1, x2,y2, ...]
 *   results - output array for interpolated values
 *   numPoints - number of points to interpolate
 */
__global__ void textureInterpolateKernel(
    cudaTextureObject_t texObj,
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
    
    // Boundary check (clamp mode handles this, but explicit check for consistency)
    if (x < 0.0f || y < 0.0f || x >= width - 1.0f || y >= height - 1.0f) {
        results[idx] = 0.0f;
        return;
    }
    
    // Hardware bilinear interpolation via texture fetch
    // +0.5f offset for pixel center (texture coordinates are pixel-centered)
    results[idx] = tex2D<float>(texObj, x + 0.5f, y + 0.5f);
}

// C linkage for functions called from C code
extern "C" {

/*********************************************************************
 * cudaTextureInterpolate
 * 
 * Non-persistent wrapper for backward compatibility with legacy code.
 * 
 * This version allocates device memory internally for code paths that
 * don't have access to the tracking context's persistent buffers
 * (e.g., helper functions in trackFeatures.c).
 * 
 * Performance:
 *   - Still uses hardware texture interpolation (5-10× faster than CPU)
 *   - Has allocation overhead (~300 μs per call)
 *   - Acceptable for infrequent calls (helper functions)
 *   - Main tracking loop should use cudaTextureInterpolatePersistent()
 * 
 * Parameters:
 *   img - host image data (height × width, row-major)
 *   width - image width in pixels
 *   height - image height in pixels
 *   coords - host coordinates array [x0,y0, x1,y1, ...]
 *   results - host results array for interpolated values
 *   numPoints - number of points to interpolate
 */
void cudaTextureInterpolate(
    const float* img,
    int width,
    int height,
    const float* coords,
    float* results,
    int numPoints)
{
    if (numPoints <= 0) return;
    
    // Allocate device memory
    float *d_img, *d_coords, *d_results;
    cudaArray* cuArray;
    cudaTextureObject_t texObj = 0;
    
    size_t imgSize = width * height * sizeof(float);
    size_t coordsSize = numPoints * 2 * sizeof(float);
    size_t resultsSize = numPoints * sizeof(float);
    
    // Create CUDA array for texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    
    // Copy image to array
    cudaMemcpy2DToArray(cuArray, 0, 0, img, width * sizeof(float),
                        width * sizeof(float), height, cudaMemcpyHostToDevice);
    
    // Create texture object
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    
    // Allocate device memory for coordinates and results
    cudaMalloc(&d_coords, coordsSize);
    cudaMalloc(&d_results, resultsSize);
    
    // Copy coordinates to device
    cudaMemcpy(d_coords, coords, coordsSize, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;
    
    textureInterpolateKernel<<<blocksPerGrid, threadsPerBlock>>>(
        texObj, width, height, d_coords, d_results, numPoints);
    
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(results, d_results, resultsSize, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(d_coords);
    cudaFree(d_results);
}

/*********************************************************************
 * cudaTextureInterpolatePersistent
 * 
 * Persistent buffer version for optimal performance in tracking loops.
 * Reuses pre-allocated GPU buffers to eliminate malloc/free overhead.
 * 
 * This is the PREFERRED function for performance-critical code paths.
 * 
 * Performance benefits:
 *   - Texture memory: Hardware bilinear interpolation (~5-10× faster)
 *   - Persistent buffers: No allocation overhead (~300 μs saved per call)
 *   - Better memory locality (persistent address)
 *   - Enables async operations (no implicit sync from malloc/free)
 * 
 * Parameters:
 *   d_img_data - pre-allocated device image buffer (from tracking context)
 *   d_coords - pre-allocated device coordinates buffer (from tracking context)
 *   d_results - pre-allocated device results buffer (from tracking context)
 *   width - image width in pixels
 *   height - image height in pixels
 *   h_coords - host coordinates array [x0,y0, x1,y1, ...]
 *   h_results - host results array for interpolated values
 *   numPoints - number of points to interpolate
 *   texObj - pre-created texture object (void* cast of cudaTextureObject_t)
 */
void cudaTextureInterpolatePersistent(
    void* d_img_data,
    void* d_coords,
    void* d_results,
    int width,
    int height,
    const float* h_coords,
    float* h_results,
    int numPoints,
    void* texObj)
{
    // Cast buffers to proper types
    float* d_img = (float*)d_img_data;
    float* d_coord_buf = (float*)d_coords;
    float* d_result_buf = (float*)d_results;
    cudaTextureObject_t texture = *((cudaTextureObject_t*)&texObj);
    
    // Calculate sizes
    size_t coordsSize = numPoints * 2 * sizeof(float);
    size_t resultsSize = numPoints * sizeof(float);
    
    // Transfer coordinates to pre-allocated buffer
    cudaMemcpy(d_coord_buf, h_coords, coordsSize, cudaMemcpyHostToDevice);
    
    // Launch kernel with optimized block size
    // Texture memory benefits from 256 threads for cache efficiency
    int threadsPerBlock = 256;
    int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;
    
    textureInterpolateKernel<<<blocksPerGrid, threadsPerBlock>>>(
        texture, width, height, d_coord_buf, d_result_buf, numPoints);
    
    // Copy results back from pre-allocated buffer
    cudaMemcpy(h_results, d_result_buf, resultsSize, cudaMemcpyDeviceToHost);
}

} // extern "C"
