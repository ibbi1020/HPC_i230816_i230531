/*********************************************************************
 * mineigenvalue_cuda.h
 * 
 * Header file for CUDA minimum eigenvalue computation
 *********************************************************************/

 #ifndef MINEIGENVALUE_CUDA_H
 #define MINEIGENVALUE_CUDA_H
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 /*********************************************************************
  * cudaComputeMinEigenvalues
  * 
  * Computes minimum eigenvalue of structure tensor for all pixels.
  * This measures corner strength for feature detection.
  * Handles all GPU memory allocation and transfer internally.
  * 
  * Parameters:
  *   gradx - host gradient array in x direction (ncols x nrows)
  *   grady - host gradient array in y direction (ncols x nrows)
  *   eigenvalues - host output array (ncols x nrows, pre-allocated)
  *   ncols - image width
  *   nrows - image height
  *   window_hw - half-width of window (e.g., 3 for 7x7 window)
  */
 void cudaComputeMinEigenvalues(
     const float* gradx,
     const float* grady,
     float* eigenvalues,
     int ncols,
     int nrows,
     int window_hw);
 
 /*********************************************************************
  * cudaComputeMinEigenvaluesWithGradients
  * 
  * Version that takes gradients already on GPU.
  * Use when gradient computation is also done on GPU.
  * 
  * Parameters:
  *   d_gradx - device gradient array in x direction (on GPU)
  *   d_grady - device gradient array in y direction (on GPU)
  *   eigenvalues - host output array (ncols x nrows, pre-allocated)
  *   ncols - image width
  *   nrows - image height
  *   window_hw - half-width of window
  */
 void cudaComputeMinEigenvaluesWithGradients(
     const float* d_gradx,
     const float* d_grady,
     float* eigenvalues,
     int ncols,
     int nrows,
     int window_hw);
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif /* MINEIGENVALUE_CUDA_H */