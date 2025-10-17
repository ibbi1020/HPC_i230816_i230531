#ifndef INTERPOLATE_CUDA_H
#define INTERPOLATE_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

/*********************************************************************
 * cudaNaiveInterpolate
 * 
 * Performs batch bilinear interpolation on the GPU (naive implementation)
 * Handles all GPU memory allocation and transfer internally.
 * 
 * Parameters:
 *   img - host image data (height x width, row-major)
 *   width - image width
 *   height - image height
 *   coords - host array of (x,y) coordinates [x0,y0, x1,y1, ...]
 *   results - host output array for interpolated values
 *   numPoints - number of points to interpolate
 */
void cudaNaiveInterpolate(
    const float* img,
    int width,
    int height,
    const float* coords,
    float* results,
    int numPoints);

/*********************************************************************
 * cudaNaiveInterpolatePersistent
 * 
 * Performs batch bilinear interpolation with pre-allocated GPU image.
 * Use this when processing multiple batches with the same image.
 * 
 * Parameters:
 *   d_img - device image data (already on GPU)
 *   width - image width
 *   height - image height
 *   coords - host array of (x,y) coordinates
 *   results - host output array
 *   numPoints - number of points to interpolate
 */
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
