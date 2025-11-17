/*********************************************************************
 * subsample_cuda.h
 *
 * GPU-accelerated image subsampling for pyramid construction
 * 
 * Performs 2x downsampling by selecting every 2nd pixel
 *********************************************************************/

#ifndef _SUBSAMPLE_CUDA_H_
#define _SUBSAMPLE_CUDA_H_

#ifdef __cplusplus
extern "C" {
#endif

/*********************************************************************
 * cudaSubsampleImage
 * 
 * Subsample an image by a factor of 2 (takes every 2nd pixel)
 * 
 * INPUT:
 *   d_src: Device pointer to source image (width × height)
 *   src_width: Width of source image
 *   src_height: Height of source image
 *   d_dst: Device pointer to destination image ((width/2) × (height/2))
 *   
 * NOTE: Assumes d_src and d_dst are already allocated on device
 *********************************************************************/
void cudaSubsampleImage(
    const float* d_src,
    int src_width,
    int src_height,
    float* d_dst
);

#ifdef __cplusplus
}
#endif

#endif /* _SUBSAMPLE_CUDA_H_ */
