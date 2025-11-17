/*********************************************************************
 * convolve.c
 *********************************************************************/


#include <assert.h>
#include <math.h>
#include <stdlib.h> 

#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h" /* printing */
#include "cuda_config.h" /* CUDA optimization flags */

#ifdef USE_CUDA
#include "convolve_gpu.h"
#include <cuda_runtime.h>

/* Global tracking context for persistent buffer access */
/* This is set by _KLTSetGlobalTrackingContext() before pyramid operations */
KLT_TrackingContext _KLTGlobalTrackingContext = NULL;
#endif

#define MAX_KERNEL_WIDTH 71

typedef struct
{
  int width;
  float data[MAX_KERNEL_WIDTH];
} ConvolutionKernel;

/* Kernels */
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;

/*********************************************************************
 * _KLTToFloatImage
 *
 * Given a pointer to image data (probably unsigned chars), copy
 * data to a float image.
 */

void _KLTToFloatImage(
    KLT_PixelType *img,
    int ncols, int nrows,
    _KLT_FloatImage floatimg)
{
  KLT_PixelType *ptrend = img + ncols * nrows;
  float *ptrout = floatimg->data;

  /* Output image must be large enough to hold result */
  assert(floatimg->ncols >= ncols);
  assert(floatimg->nrows >= nrows);

  floatimg->ncols = ncols;
  floatimg->nrows = nrows;

  while (img < ptrend)
    *ptrout++ = (float)*img++;
}

/*********************************************************************
 * _computeKernels
 */

static void _computeKernels(
    float sigma,
    ConvolutionKernel *gauss,
    ConvolutionKernel *gaussderiv)
{
  const float factor = 0.01f; /* for truncating tail */
  int i;

  assert(MAX_KERNEL_WIDTH % 2 == 1);
  assert(sigma >= 0.0);

  /* Compute kernels, and automatically determine widths */
  {
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float)(sigma * exp(-0.5f));

    /* Compute gauss and deriv */
    for (i = -hw; i <= hw; i++)
    {
      gauss->data[i + hw] = (float)exp(-i * i / (2 * sigma * sigma));
      gaussderiv->data[i + hw] = -i * gauss->data[i + hw];
    }

    /* Compute widths */
    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabs(gauss->data[i + hw] / max_gauss) < factor;
         i++, gauss->width -= 2)
      ;
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabs(gaussderiv->data[i + hw] / max_gaussderiv) < factor;
         i++, gaussderiv->width -= 2)
      ;
    if (gauss->width == MAX_KERNEL_WIDTH ||
        gaussderiv->width == MAX_KERNEL_WIDTH)
      KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
               "a sigma of %f",
               MAX_KERNEL_WIDTH, sigma);
  }

  /* Shift if width less than MAX_KERNEL_WIDTH */
  for (i = 0; i < gauss->width; i++)
    gauss->data[i] = gauss->data[i + (MAX_KERNEL_WIDTH - gauss->width) / 2];
  for (i = 0; i < gaussderiv->width; i++)
    gaussderiv->data[i] = gaussderiv->data[i + (MAX_KERNEL_WIDTH - gaussderiv->width) / 2];
  /* Normalize gauss and deriv */
  {
    const int hw = gaussderiv->width / 2;
    float den;

    den = 0.0;
    for (i = 0; i < gauss->width; i++)
      den += gauss->data[i];
    for (i = 0; i < gauss->width; i++)
      gauss->data[i] /= den;
    den = 0.0;
    for (i = -hw; i <= hw; i++)
      den -= i * gaussderiv->data[i + hw];
    for (i = -hw; i <= hw; i++)
      gaussderiv->data[i + hw] /= den;
  }

  sigma_last = sigma;
}

/*********************************************************************
 * _KLTGetKernelWidths
 *
 */

void _KLTGetKernelWidths(
    float sigma,
    int *gauss_width,
    int *gaussderiv_width)
{
  _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}

/*********************************************************************
 * _convolveImageHoriz
 */

static void _convolveImageHoriz(
    _KLT_FloatImage imgin,
    ConvolutionKernel kernel,
    _KLT_FloatImage imgout)
{
  float *ptrrow = imgin->data;           /* Points to row's first pixel */
  register float *ptrout = imgout->data, /* Points to next output pixel */
      *ppp;
  register float sum;
  register int radius = kernel.width / 2;
  register int ncols = imgin->ncols, nrows = imgin->nrows;
  register int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);

  /* Must read from and write to different images */
  assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  /* For each row, do ... */
  for (j = 0; j < nrows; j++)
  {

    /* Zero leftmost columns */
    for (i = 0; i < radius; i++)
      *ptrout++ = 0.0;

    /* Convolve middle columns with kernel */
    for (; i < ncols - radius; i++)
    {
      ppp = ptrrow + i - radius;
      sum = 0.0;
      for (k = kernel.width - 1; k >= 0; k--)
        sum += *ppp++ * kernel.data[k];
      *ptrout++ = sum;
    }

    /* Zero rightmost columns */
    for (; i < ncols; i++)
      *ptrout++ = 0.0;

    ptrrow += ncols;
  }
}

/*********************************************************************
 * _convolveImageVert
 */

static void _convolveImageVert(
    _KLT_FloatImage imgin,
    ConvolutionKernel kernel,
    _KLT_FloatImage imgout)
{
  float *ptrcol = imgin->data;           /* Points to row's first pixel */
  register float *ptrout = imgout->data, /* Points to next output pixel */
      *ppp;
  register float sum;
  register int radius = kernel.width / 2;
  register int ncols = imgin->ncols, nrows = imgin->nrows;
  register int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);

  /* Must read from and write to different images */
  assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  /* For each column, do ... */
  for (i = 0; i < ncols; i++)
  {

    /* Zero topmost rows */
    for (j = 0; j < radius; j++)
    {
      *ptrout = 0.0;
      ptrout += ncols;
    }

    /* Convolve middle rows with kernel */
    for (; j < nrows - radius; j++)
    {
      ppp = ptrcol + ncols * (j - radius);
      sum = 0.0;
      for (k = kernel.width - 1; k >= 0; k--)
      {
        sum += *ppp * kernel.data[k];
        ppp += ncols;
      }
      *ptrout = sum;
      ptrout += ncols;
    }

    /* Zero bottommost rows */
    for (; j < nrows; j++)
    {
      *ptrout = 0.0;
      ptrout += ncols;
    }

    ptrcol++;
    ptrout -= nrows * ncols - 1;
  }
}

/*********************************************************************
 * _convolveSeparate
 */

#ifdef USE_CUDA
void _convolveSeparate(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
  /* NOTE: This function is called from _KLTComputeGradients and _KLTComputeSmoothedImage
   * which don't have access to tracking context. We pass tc through a global.
   * This is set in KLTSelectGoodFeatures and KLTTrackFeatures before pyramid building.
   */
  extern KLT_TrackingContext _KLTGlobalTrackingContext;
  KLT_TrackingContext tc = _KLTGlobalTrackingContext;
  
  int ncols = imgin->ncols;
  int nrows = imgin->nrows;
  size_t img_size = ncols * nrows * sizeof(float);
  size_t kernel_size_h = horiz_kernel.width * sizeof(float);
  size_t kernel_size_v = vert_kernel.width * sizeof(float);
  
  /* Device pointers */
  float *d_imgin, *d_tmpimg, *d_imgout;
  float *d_horiz_kernel, *d_vert_kernel;
  
  /* Allocate or reuse persistent buffers */
  if (tc && tc->gpu_buffers_initialized && tc->gpu_buffer_size >= img_size) {
    /* Reuse existing buffers (FAST PATH) */
    d_imgin  = (float*)tc->gpu_temp_float_img1;
    d_tmpimg = (float*)tc->gpu_temp_float_img2;
    d_imgout = (float*)tc->gpu_temp_float_img3;
  } else {
    /* First call or buffer too small - allocate new buffers */
    if (tc && tc->gpu_buffers_initialized) {
      /* Free old buffers that are too small */
      cudaFree(tc->gpu_temp_float_img1);
      cudaFree(tc->gpu_temp_float_img2);
      cudaFree(tc->gpu_temp_float_img3);
    }
    
    /* Allocate new buffers with 20% extra space to reduce reallocation */
    size_t alloc_size = img_size + img_size / 5;
    cudaMalloc((void**)&d_imgin, alloc_size);
    cudaMalloc((void**)&d_tmpimg, alloc_size);
    cudaMalloc((void**)&d_imgout, alloc_size);
    
    if (tc) {
      tc->gpu_temp_float_img1 = d_imgin;
      tc->gpu_temp_float_img2 = d_tmpimg;
      tc->gpu_temp_float_img3 = d_imgout;
      tc->gpu_buffer_size = alloc_size;
      tc->gpu_buffers_initialized = 1;
    }
  }
  
  /* Allocate kernel buffers (small, not worth persisting) */
  cudaMalloc((void**)&d_horiz_kernel, kernel_size_h);
  cudaMalloc((void**)&d_vert_kernel, kernel_size_v);
  
  /* Copy data to device */
  cudaMemcpy(d_imgin, imgin->data, img_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_horiz_kernel, horiz_kernel.data, kernel_size_h, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vert_kernel, vert_kernel.data, kernel_size_v, cudaMemcpyHostToDevice);
  
  /* Launch horizontal convolution with shared memory */
  launchConvolveHorizShared(d_imgin, d_horiz_kernel, d_tmpimg,
                            ncols, nrows, horiz_kernel.width);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Horizontal shared convolution kernel failed: %s\n", cudaGetErrorString(err));
  }
  
  /* Launch vertical convolution with shared memory */
  launchConvolveVertShared(d_tmpimg, d_vert_kernel, d_imgout,
                           ncols, nrows, vert_kernel.width);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Vertical shared convolution kernel failed: %s\n", cudaGetErrorString(err));
  }
  
  /* Single synchronization at the end (before copying back) */
  cudaDeviceSynchronize();
  
  /* Copy result back to host */
  cudaMemcpy(imgout->data, d_imgout, img_size, cudaMemcpyDeviceToHost);
  
  /* Free kernel buffers (small, allocated each time) */
  cudaFree(d_horiz_kernel);
  cudaFree(d_vert_kernel);
  
  /* Note: Do NOT free d_imgin, d_tmpimg, d_imgout - they are persistent! */
}
#else
/* CPU fallback version */
static void _convolveSeparate(
    _KLT_FloatImage imgin,
    ConvolutionKernel horiz_kernel,
    ConvolutionKernel vert_kernel,
    _KLT_FloatImage imgout)
{
  /* Temporary image */
  _KLT_FloatImage tmpimg;

  /* Allocate temporary image */
  tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);

  /* Do convolution */
  _convolveImageHoriz(imgin, horiz_kernel, tmpimg);
  _convolveImageVert(tmpimg, vert_kernel, imgout);

  /* Free memory */
  _KLTFreeFloatImage(tmpimg);
}

#endif

/*********************************************************************
 * _KLTSetGlobalTrackingContext
 * 
 * Sets the global tracking context for persistent GPU buffer access.
 * This must be called before any pyramid/convolution operations.
 */

void _KLTSetGlobalTrackingContext(KLT_TrackingContext tc)
{
#ifdef USE_CUDA
  extern KLT_TrackingContext _KLTGlobalTrackingContext;
  _KLTGlobalTrackingContext = tc;
#endif
}

/*********************************************************************
 * _KLTComputeGradients
 */

void _KLTComputeGradients(
    _KLT_FloatImage img,
    float sigma,
    _KLT_FloatImage gradx,
    _KLT_FloatImage grady)
{

  /* Output images must be large enough to hold result */
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  /* Compute kernels, if necessary */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
  _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);
}

/*********************************************************************
 * _KLTComputeSmoothedImage
 */

void _KLTComputeSmoothedImage(
    _KLT_FloatImage img,
    float sigma,
    _KLT_FloatImage smooth)
{
  /* Output image must be large enough to hold result */
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  /* Compute kernel, if necessary; gauss_deriv is not used */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}
