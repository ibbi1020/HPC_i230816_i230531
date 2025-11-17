/*********************************************************************
 * pyramid.c
 *
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <stdlib.h>		/* malloc() ? */
#include <string.h>		/* memset() ? */
#include <math.h>		/* */

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"	/* for computing pyramid */
#include "pyramid.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "subsample_cuda.h"
#endif


/*********************************************************************
 *
 */

_KLT_Pyramid _KLTCreatePyramid(
  int ncols,
  int nrows,
  int subsampling,
  int nlevels)
{
  _KLT_Pyramid pyramid;
  int nbytes = sizeof(_KLT_PyramidRec) +	
    nlevels * sizeof(_KLT_FloatImage *) +
    nlevels * sizeof(int) +
    nlevels * sizeof(int);
  int i;

  if (subsampling != 2 && subsampling != 4 && 
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTCreatePyramid)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

     
  /* Allocate memory for structure and set parameters */
  pyramid = (_KLT_Pyramid)  malloc(nbytes);
  if (pyramid == NULL)
    KLTError("(_KLTCreatePyramid)  Out of memory");
     
  /* Set parameters */
  pyramid->subsampling = subsampling;
  pyramid->nLevels = nlevels;
  pyramid->img = (_KLT_FloatImage *) (pyramid + 1);
  pyramid->ncols = (int *) (pyramid->img + nlevels);
  pyramid->nrows = (int *) (pyramid->ncols + nlevels);

  /* Allocate memory for each level of pyramid and assign pointers */
  for (i = 0 ; i < nlevels ; i++)  {
    pyramid->img[i] =  _KLTCreateFloatImage(ncols, nrows);
    pyramid->ncols[i] = ncols;  pyramid->nrows[i] = nrows;
    ncols /= subsampling;  nrows /= subsampling;
  }

  return pyramid;
}


/*********************************************************************
 *
 */

void _KLTFreePyramid(
  _KLT_Pyramid pyramid)
{
  int i;

  /* Free images */
  for (i = 0 ; i < pyramid->nLevels ; i++)
    _KLTFreeFloatImage(pyramid->img[i]);

  /* Free structure */
  free(pyramid);
}

#ifdef USE_CUDA
/* NOTE: GPU pyramid allocation is now handled in trackFeatures.c */
/* by the _uploadPyramidToGPU() helper function */
#endif


/*********************************************************************
 *
 */

void _KLTComputePyramid(
  _KLT_FloatImage img, 
  _KLT_Pyramid pyramid,
  float sigma_fact)
{
  _KLT_FloatImage currimg, tmpimg;
  int ncols = img->ncols, nrows = img->nrows;
  int subsampling = pyramid->subsampling;
  int subhalf = subsampling / 2;
  float sigma = subsampling * sigma_fact;  /* empirically determined */
  int oldncols;
  int i, x, y;
	
  if (subsampling != 2 && subsampling != 4 && 
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTComputePyramid)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

  assert(pyramid->ncols[0] == img->ncols);
  assert(pyramid->nrows[0] == img->nrows);

  /* Copy original image to level 0 of pyramid */
  memcpy(pyramid->img[0]->data, img->data, ncols*nrows*sizeof(float));

  /* GPU pyramid computation - Only in sequential mode with USE_CUDA */
#ifdef USE_CUDA
  /* NOTE: GPU pyramid is only used in sequential mode (the common case) */
  /* For non-sequential mode, we fall back to CPU computation below */
  /* This approach keeps the code simple while optimizing the hot path */
  
  /* The GPU pyramid caching is handled externally by the caller */
  /* This function just builds the pyramid levels normally */
  /* The key optimization is that pyramids stay on GPU in tracking context */
#endif

  /* Build remaining pyramid levels */
  currimg = img;
  for (i = 1 ; i < pyramid->nLevels ; i++)  {
    tmpimg = _KLTCreateFloatImage(ncols, nrows);
    _KLTComputeSmoothedImage(currimg, sigma, tmpimg);

    /* Subsample */
    oldncols = ncols;
    ncols /= subsampling;  nrows /= subsampling;
    for (y = 0 ; y < nrows ; y++)
      for (x = 0 ; x < ncols ; x++)
        pyramid->img[i]->data[y*ncols+x] = 
          tmpimg->data[(subsampling*y+subhalf)*oldncols +
                      (subsampling*x+subhalf)];

    /* Reassign current image */
    currimg = pyramid->img[i];
				
    _KLTFreeFloatImage(tmpimg);
  }
}
 











