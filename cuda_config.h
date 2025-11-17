/*********************************************************************
 * cuda_config.h
 * 
 * Central configuration file for CUDA optimizations in KLT
 * 
 * To enable GPU acceleration: Compile with -DUSE_CUDA
 * To use CPU only: Compile without -DUSE_CUDA flag
 * 
 * Example:
 *   make USE_CUDA=1 clean lib example3   # GPU enabled
 *   make USE_CUDA=0 clean lib example3   # CPU only (default)
 *   make clean lib example3              # CPU only (default)
 *********************************************************************/

#ifndef CUDA_CONFIG_H
#define CUDA_CONFIG_H

/*********************************************************************
 * Single CUDA Switch
 * 
 * When USE_CUDA is defined (via compiler flag -DUSE_CUDA):
 *   - All CUDA-accelerated kernels will be used
 *   - Includes: interpolation, convolution (with shared memory), feature selection
 * 
 * When USE_CUDA is not defined:
 *   - CPU-only implementations will be used
 *   - No CUDA runtime dependencies
 *********************************************************************/

/* Note: USE_CUDA is now defined by the Makefile via -DUSE_CUDA flag
 * Do not manually define it here. Control it via:
 *   make USE_CUDA=1 example3
 */

#endif /* CUDA_CONFIG_H */
