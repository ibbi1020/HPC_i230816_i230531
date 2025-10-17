/*********************************************************************
 * cuda_config.h
 * 
 * Central configuration file for CUDA optimizations in KLT
 * Comment out any #define to disable that optimization
 *********************************************************************/

#ifndef CUDA_CONFIG_H
#define CUDA_CONFIG_H

/*********************************************************************
 * CUDA Optimization Switches
 * 
 * Uncomment to enable CUDA acceleration for each component.
 * Comment out to use CPU-only implementation.
 *********************************************************************/

/* Enable CUDA acceleration for bilinear interpolation (trackFeatures.c) */
//#define USE_CUDA_INTERPOLATION 1

/* Enable CUDA acceleration for feature selection (selectGoodFeatures.c) */
//#define USE_CUDA_FEATURE_SELECTION 1

/* Enable CUDA acceleration for convolution (future optimization) */
//#define USE_CUDA_CONVOLUTION 1

#endif /* CUDA_CONFIG_H */
