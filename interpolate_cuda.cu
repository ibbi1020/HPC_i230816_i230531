/*********************************************************************
 * interpolate_cuda.cu
 * 
 * CUDA implementation of bilinear interpolation for KLT tracking.
 */

 #include <cuda_runtime.h>
 #include <stdio.h>
 #include "interpolate_cuda.h"
 
 /*********************************************************************
  * naiveInterpolateKernel
  *********************************************************************/
 __global__ void naiveInterpolateKernel(
     const float* img,
     int width,
     int height,
     const float* coords,
     float* results,
     int numPoints)
 {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
 
     if (idx >= numPoints)
         return;
 
     float x = coords[idx * 2];
     float y = coords[idx * 2 + 1];
 
     int xt = (int)x;
     int yt = (int)y;
 
     if (xt < 0 || yt < 0 || xt >= width - 1 || yt >= height - 1) {
         results[idx] = 0.0f;
         return;
     }
 
     float ax = x - xt;
     float ay = y - yt;
 
     int offset = yt * width + xt;
 
     float p00 = img[offset];
     float p10 = img[offset + 1];
     float p01 = img[offset + width];
     float p11 = img[offset + width + 1];
 
     results[idx] =
         (1 - ax) * (1 - ay) * p00 +
          ax      * (1 - ay) * p10 +
         (1 - ax) *      ay  * p01 +
          ax      *      ay  * p11;
 }
 
 /*********************************************************************
  * cudaNaiveInterpolate  (host wrapper)
  *********************************************************************/
 extern "C"
 void cudaNaiveInterpolate(
     const float* img,
     int width,
     int height,
     const float* coords,
     float* results,
     int numPoints)
 {
     float *d_img = nullptr, *d_coords = nullptr, *d_results = nullptr;
 
     size_t imgSize     = width * height * sizeof(float);
     size_t coordsSize  = numPoints * 2 * sizeof(float);
     size_t resultsSize = numPoints * sizeof(float);
 
     cudaMalloc(&d_img, imgSize);
     cudaMalloc(&d_coords, coordsSize);
     cudaMalloc(&d_results, resultsSize);
 
     cudaMemcpy(d_img, img, imgSize, cudaMemcpyHostToDevice);
     cudaMemcpy(d_coords, coords, coordsSize, cudaMemcpyHostToDevice);
 
     int block = 256;
     int grid = (numPoints + block - 1) / block;
 
     naiveInterpolateKernel<<<grid, block>>>(
         d_img, width, height, d_coords, d_results, numPoints);
 
     cudaMemcpy(results, d_results, resultsSize, cudaMemcpyDeviceToHost);
 
     cudaFree(d_img);
     cudaFree(d_coords);
     cudaFree(d_results);
 }
 
 /*********************************************************************
  * cudaNaiveInterpolatePersistent  (host wrapper)
  *********************************************************************/
 extern "C"
 void cudaNaiveInterpolatePersistent(
     const float* d_img,
     int width,
     int height,
     const float* coords,
     float* results,
     int numPoints)
 {
     float *d_coords = nullptr, *d_results = nullptr;
 
     size_t coordsSize  = numPoints * 2 * sizeof(float);
     size_t resultsSize = numPoints * sizeof(float);
 
     cudaMalloc(&d_coords, coordsSize);
     cudaMalloc(&d_results, resultsSize);
 
     cudaMemcpy(d_coords, coords, coordsSize, cudaMemcpyHostToDevice);
 
     int block = 256;
     int grid = (numPoints + block - 1) / block;
 
     naiveInterpolateKernel<<<grid, block>>>(
         d_img, width, height, d_coords, d_results, numPoints);
 
     cudaMemcpy(results, d_results, resultsSize, cudaMemcpyDeviceToHost);
 
     cudaFree(d_coords);
     cudaFree(d_results);
 }
 