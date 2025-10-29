#ifndef CONVOLVE_GPU_H
#define CONVOLVE_GPU_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Upload kernel to constant memory (only works if compiled with -DUSE_CONST)
// If USE_CONST is not used, this function does nothing.
// K = kernel size = 2*R + 1
// ============================================================
void uploadKernelToConst(const float* hKernel, int K);


// ============================================================
// Horizontal convolution (one pass of separable blur)
// d_in, d_out are pitched GPU buffers
// pitch = pitch in ELEMENTS (not bytes)
// R = radius (kernel size = 2*R+1)
// stream = use 0 if you don't use CUDA streams
// ============================================================
void runHorizontalConvolution(const float* d_in,
                              float* d_out,
                              const float* d_kernel,
                              int width,
                              int height,
                              int pitch,        // in elements
                              int R,
                              cudaStream_t stream);


// ============================================================
// Vertical convolution (second pass of separable blur)
// Same parameters as horizontal
// ============================================================
void runVerticalConvolution(const float* d_in,
                            float* d_out,
                            const float* d_kernel,
                            int width,
                            int height,
                            int pitch,        // in elements
                            int R,
                            cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // CONVOLVE_GPU_H
