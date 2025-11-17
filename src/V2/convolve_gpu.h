#ifndef _CONVOLVE_GPU_H_
#define _CONVOLVE_GPU_H_

#ifdef __cplusplus
extern "C" {
#endif

// GPU kernel declarations (to be called from host code)
// Note: These are declared but defined in the .cu file
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

#ifdef __cplusplus
}
#endif

#endif // _CONVOLVE_GPU_H_