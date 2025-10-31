######################################################################
# BUILD CONFIGURATION
# Set USE_CUDA=1 for GPU-accelerated build (requires CUDA toolkit)
# Set USE_CUDA=0 for CPU-only build (portable, no CUDA required)
######################################################################
USE_CUDA ?= 1

######################################################################
# Compilers
######################################################################
CC = gcc
NVCC = nvcc

######################################################################
# Compiler flags
######################################################################
FLAG1 = -DNDEBUG  # Remove asserts for performance
# FLAG2 = -DKLT_USE_QSORT  # Uncomment to use standard qsort()

# Base C flags
CFLAGS = -O3 $(FLAG1) $(FLAG2)

# Conditional CUDA configuration
ifeq ($(USE_CUDA), 1)
	# GPU build: Enable CUDA optimizations
	CFLAGS += -DUSE_CUDA_BUILD=1
	CFLAGS += -I/usr/local/cuda/include
    
    # CUDA compiler flags (Tesla T4: sm_75, V100: sm_70, A100: sm_80)
    CUDA_FLAGS = -O3 -arch=sm_75 \
                 -gencode=arch=compute_70,code=sm_70 \
                 -gencode=arch=compute_75,code=sm_75 \
                 -gencode=arch=compute_80,code=sm_80 \
                 -lineinfo
    
    # CUDA libraries
    CUDA_LIBS = -lcudart
    CUDA_LIBDIR = -L/usr/local/cuda/lib64
    
    # CUDA source files
    CUDA_SRCS = interpolate_cuda.cu mineigenvalue_cuda.cu convolve_gpu.cu
    CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
else
	# CPU-only build: No CUDA dependencies
	CFLAGS += -DUSE_CUDA_BUILD=0
    CUDA_FLAGS =
    CUDA_LIBS =
    CUDA_LIBDIR =
    CUDA_SRCS =
    CUDA_OBJS =
endif

######################################################################
# Source files
######################################################################
EXAMPLES = example1.c example2.c example3.c example4.c example5.c
ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c

######################################################################
# Library paths
######################################################################
LIB = -L/usr/local/lib -L/usr/lib -L.

.SUFFIXES: .c .o .cu

######################################################################
# Build Targets
######################################################################

all: lib $(EXAMPLES:.c=)

# C compilation (general rule)
%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

# Explicit dependencies for files that include cuda_config.h
convolve.o: convolve.c cuda_config.h convolve.h
	$(CC) -c $(CFLAGS) convolve.c -o convolve.o

trackFeatures.o: trackFeatures.c cuda_config.h
	$(CC) -c $(CFLAGS) trackFeatures.c -o trackFeatures.o

selectGoodFeatures.o: selectGoodFeatures.c cuda_config.h
	$(CC) -c $(CFLAGS) selectGoodFeatures.c -o selectGoodFeatures.o

# CUDA compilation (only if USE_CUDA=1)
ifeq ($(USE_CUDA), 1)

interpolate_cuda.o: interpolate_cuda.cu interpolate_cuda.h cuda_config.h
	$(NVCC) -c $(CUDA_FLAGS) interpolate_cuda.cu

mineigenvalue_cuda.o: mineigenvalue_cuda.cu mineigenvalue_cuda.h cuda_config.h
	$(NVCC) -c $(CUDA_FLAGS) mineigenvalue_cuda.cu

convolve_gpu.o: convolve_gpu.cu convolve_gpu.h cuda_config.h
	$(NVCC) -c $(CUDA_FLAGS) convolve_gpu.cu
endif

# Library build (includes CUDA objects only if USE_CUDA=1)
libklt.a: $(ARCH:.c=.o) $(CUDA_OBJS)
	rm -f libklt.a
	ar ruv libklt.a $(ARCH:.c=.o) $(CUDA_OBJS)

lib: libklt.a

# Example programs
example1: libklt.a
	$(CC) $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) $(CUDA_LIBDIR) $(CUDA_LIBS) -lm

example2: libklt.a
	$(CC) $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) $(CUDA_LIBDIR) $(CUDA_LIBS) -lm

example3: libklt.a
	$(CC) $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) $(CUDA_LIBDIR) $(CUDA_LIBS) -lm

example4: libklt.a
	$(CC) $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) $(CUDA_LIBDIR) $(CUDA_LIBS) -lm

example5: libklt.a
	$(CC) $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) $(CUDA_LIBDIR) $(CUDA_LIBS) -lm

######################################################################
# Utility Targets
######################################################################

depend:
	makedepend $(ARCH) $(EXAMPLES)

clean:
	rm -f *.o *.a $(EXAMPLES:.c=) libklt.a \
    	feat*.ppm features.ft features.txt gmon.out \
    	profile-*.txt profile-*.pdf profile-*.ncu-rep profile-*.nsys-rep

######################################################################
# Profiling Targets (GPU only - requires USE_CUDA=1)
######################################################################

ifeq ($(USE_CUDA), 1)

# Nsight Systems: Timeline view (when CPU and GPU are active)
nsys-example3: example3
	@echo "Profiling with Nsight Systems (timeline view)..."
	nsys profile --stats=true --force-overwrite=true --output=profile-example3-nsys ./example3
	@echo "✓ Generated: profile-example3-nsys.nsys-rep"
	@echo "✓ Generated: profile-example3-nsys.txt (text summary)"

# Nsight Compute: Detailed kernel metrics (occupancy, memory efficiency)
ncu-example3: example3
	@echo "Profiling kernels with Nsight Compute (detailed metrics)..."
	ncu --set full --export profile-example3-ncu --force-overwrite ./example3
	@echo "✓ Generated: profile-example3-ncu.ncu-rep"

# Nsight Compute: Quick profiling (basic metrics, faster)
ncu-quick-example3: example3
	@echo "Quick kernel profiling..."
	ncu --set basic --export profile-example3-ncu-quick --force-overwrite ./example3
	@echo "✓ Generated: profile-example3-ncu-quick.ncu-rep"

# Comprehensive profiling (nsys + ncu summary)
cuda-profile-example3: example3
	@echo "========================================"
	@echo "COMPREHENSIVE CUDA PROFILING"
	@echo "========================================"
	@nsys profile --stats=true --force-overwrite=true --output=profile-example3-nsys ./example3 > profile-example3-nsys.txt 2>&1
	@ncu --set basic --export profile-example3-ncu --force-overwrite ./example3 > profile-example3-ncu.txt 2>&1 || true
	@echo ""
	@echo "✓ PROFILING COMPLETE!"
	@echo "  • profile-example3-nsys.nsys-rep - Timeline (view with nsys-ui)"
	@echo "  • profile-example3-nsys.txt - Text summary"
	@echo "  • profile-example3-ncu.ncu-rep - Kernel metrics (view with ncu-ui)"

else

# CPU-only build: No CUDA profiling available
nsys-example3 ncu-example3 ncu-quick-example3 cuda-profile-example3:
	@echo "Error: CUDA profiling requires USE_CUDA=1"
	@echo "Rebuild with: make clean && make USE_CUDA=1 example3"
	@exit 1
endif

######################################################################
# Help Target
######################################################################

help:
	@echo "KLT Feature Tracker - Build System"
	@echo ""
	@echo "BUILD MODES:"
	@echo "  make [USE_CUDA=1]        Build with GPU acceleration (default)"
	@echo "  make USE_CUDA=0          Build CPU-only version (portable)"
	@echo ""
	@echo "TARGETS:"
	@echo "  make all                 Build library + all examples"
	@echo "  make lib                 Build libklt.a only"
	@echo "  make example3            Build example3 (primary benchmark)"
	@echo "  make clean               Remove all build artifacts"
	@echo ""
	@echo "PROFILING (GPU only):"
	@echo "  make nsys-example3       Timeline profiling (CPU+GPU activity)"
	@echo "  make ncu-example3        Detailed kernel metrics"
	@echo "  make cuda-profile-example3  Comprehensive profiling"
	@echo ""
	@echo "USAGE EXAMPLES:"
	@echo "  make clean && make example3              # GPU build (default)"
	@echo "  make clean && make USE_CUDA=0 example3   # CPU-only build"
	@echo "  make cuda-profile-example3               # Profile GPU kernels"

.PHONY: all lib clean depend help nsys-example3 ncu-example3 ncu-quick-example3 cuda-profile-example3