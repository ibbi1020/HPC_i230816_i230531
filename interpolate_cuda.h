#ifndef INTERPOLATE_CUDA_H
#define INTERPOLATE_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

void cudaNaiveInterpolate(
    const float* img,
    int width,
    int height,
    const float* coords,
    float* results,
    int numPoints);

void cudaNaiveInterpolatePersistent(
    const float* d_img,
    int width,
    int height,
    const float* coords,
    float* results,
    int numPoints);

#ifdef __cplusplus
}
#endif

#endif /* INTERPOLATE_CUDA_H */
