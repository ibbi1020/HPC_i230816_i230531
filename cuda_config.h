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
 * Master switches that enable ALL optimizations for each subsystem.
 * When enabled, uses the best-performing GPU implementation.
 * When disabled, falls back to CPU-only implementation.
 *********************************************************************/

#if defined(USE_CUDA_BUILD) && (USE_CUDA_BUILD)

/* Enable CUDA acceleration for bilinear interpolation
 * Includes: Texture memory + Persistent GPU buffers (Opt 1 + 5)
 */
#define USE_CUDA_INTERPOLATION 1

/* Enable CUDA acceleration for convolution
 * Includes: Shared memory tiling + Constant memory kernels (Opt 2 + 3)
 */
#define USE_CUDA_CONVOLUTION 1

/* Enable CUDA acceleration for feature selection (eigenvalue computation) */
#define USE_CUDA_FEATURE_SELECTION 1

#else

#undef USE_CUDA_INTERPOLATION
#undef USE_CUDA_CONVOLUTION
#undef USE_CUDA_FEATURE_SELECTION

#endif /* USE_CUDA_BUILD */

#endif /* CUDA_CONFIG_H */
