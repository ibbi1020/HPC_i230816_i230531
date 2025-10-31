#ifndef _CONVOLVE_GPU_H_
#define _CONVOLVE_GPU_H_

#ifdef __cplusplus
extern "C" {
#endif

/*********************************************************************
 * CUDA Convolution Functions (Optimization 2 + 3)
 * 
 * All functions use:
 * - Shared memory tiling for image data (3-6× speedup)
 * - Constant memory for kernel weights (10-20% additional speedup)
 * 
 * These are the ONLY supported CUDA convolution functions.
 *********************************************************************/

// Upload Gaussian kernel weights to constant memory
void cudaSetGaussianKernel(const float* h_kernel, int width);
void cudaSetGaussianDerivKernel(const float* h_kernel, int width);

// Horizontal convolution (shared memory + constant memory)
void launchConvolveHorizSharedConstant(
    const float* d_imgin,
    float* d_imgout,
    int ncols,
    int nrows);

// Vertical convolution (shared memory + constant memory)
void launchConvolveVertSharedConstant(
    const float* d_imgin,
    float* d_imgout,
    int ncols,
    int nrows);

#ifdef __cplusplus
}
#endif

#endif // _CONVOLVE_GPU_H_