#ifndef _CONVOLVE_GPU_H_
#define _CONVOLVE_GPU_H_

#ifdef __cplusplus
extern "C" {
#endif

// Naive GPU kernel wrappers (original implementation)
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
    int blockDimY);

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
    int blockDimY);

// Shared memory optimized wrappers (Optimization 2)
void launchConvolveHorizShared(
    const float* d_input,
    const float* d_kernel,
    float* d_output,
    int width,
    int height,
    int kernel_width);

void launchConvolveVertShared(
    const float* d_input,
    const float* d_kernel,
    float* d_output,
    int width,
    int height,
    int kernel_width);

#ifdef __cplusplus
}
#endif

#endif // _CONVOLVE_GPU_H_