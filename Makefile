######################################################################
# Choose your compilers
CC = gcc
NVCC = nvcc
######################################################################
# CUDA compiler and flags
# Updated for Google Colab GPUs (Tesla T4: sm_75, V100: sm_70, A100: sm_80)
# Using sm_75 as default for T4, add multiple architectures for compatibility
NVCC = nvcc
CUDA_FLAGS = -O3 -arch=sm_75 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80
CUDA_LIBS = -lcudart
CUDA_LIBDIR = -L/usr/local/cuda/lib64

# CUDA Profiling flags
CUDA_PROFILE_FLAGS = -lineinfo -Xcompiler -rdynamic

######################################################################
# -DNDEBUG prevents the assert() statements from being included in 
# the code.  If you are having problems running the code, you might 
# want to comment this line to see if an assert() statement fires.
FLAG1 = -DNDEBUG
######################################################################
# -DKLT_USE_QSORT forces the code to use the standard qsort() 
# routine.  Otherwise it will use a quicksort routine that takes
# advantage of our specific data structure to greatly reduce the
# running time on some machines.  Uncomment this line if for some
# reason you are unhappy with the special routine.
# FLAG2 = -DKLT_USE_QSORT
######################################################################
# Add your favorite C flags here.
# Add CUDA include path so convolve.c can find cuda_runtime.h
CFLAGS = $(FLAG1) $(FLAG2) -I/usr/local/cuda/include
######################################################################
# CUDA flags
# GTX 960 is Maxwell architecture with compute capability 5.2
NVCCFLAGS = $(FLAG1) $(FLAG2) \
            -gencode arch=compute_50,code=sm_50 \
            -gencode arch=compute_50,code=compute_50 \
            -Xcompiler -fPIC \
            -rdc=true
######################################################################
# Library flags
LIB = -L/usr/local/lib -L/usr/lib -L. -L/usr/local/cuda/lib64
LIBS = -lm -lcudart
######################################################################
# Source and object files
EXAMPLES = example1.c example2.c example3.c example4.c example5.c

# C source files (convolve.c stays as .c)
C_ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
         storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c

# CUDA source files (new separate GPU file)
CUDA_ARCH = convolve_gpu.cu

EXAMPLES = example1.c example2.c example3.c example4.c example5.c
ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c
CUDA_SRCS = interpolate_cuda.cu mineigenvalue_cuda.cu convolve_gpu.cu
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
LIB = -L/usr/local/lib -L/usr/lib

.SUFFIXES:  .c .o .cu

######################################################################
# Targets

all: lib $(EXAMPLES:.c=)

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

.cu.o:
	$(NVCC) -c $(CUDA_FLAGS) $<

# Explicit rules for CUDA object files
interpolate_cuda.o: interpolate_cuda.cu interpolate_cuda.h cuda_config.h
	$(NVCC) -c $(CUDA_FLAGS) interpolate_cuda.cu

mineigenvalue_cuda.o: mineigenvalue_cuda.cu mineigenvalue_cuda.h cuda_config.h
	$(NVCC) -c $(CUDA_FLAGS) mineigenvalue_cuda.cu

convolve_gpu.o: convolve_gpu.cu convolve_gpu.h cuda_config.h
	$(NVCC) -c $(CUDA_FLAGS) convolve_gpu.cu

# Explicit library target so rules depending on 'libklt.a' can build it
libklt.a: $(ARCH:.c=.o) $(CUDA_OBJS)
	rm -f libklt.a
	ar ruv libklt.a $(ARCH:.c=.o) $(CUDA_OBJS)

# Keep 'lib' as an alias for convenience
lib: libklt.a

example1: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) $(CUDA_LIBDIR) $(CUDA_LIBS) -lm

example2: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) $(CUDA_LIBDIR) $(CUDA_LIBS) -lm

example3: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) $(CUDA_LIBDIR) $(CUDA_LIBS) -lm

example4: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) $(CUDA_LIBDIR) $(CUDA_LIBS) -lm

example5: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) $(CUDA_LIBDIR) $(CUDA_LIBS) -lm

######################################################################
# Utility targets
depend:
	makedepend $(C_ARCH) $(EXAMPLES)

clean:
	rm -f *.o *.a $(EXAMPLES:.c=) *.tar *.tar.gz libklt.a \
	      feat*.ppm features.ft features.txt gmon.out \
	      profile-*.txt profile-*.dot profile-*.pdf \
	      profile-*-gprof.txt profile-*-gprof.pdf profile-*-gprof.dot \
	      profile-*.ncu-rep profile-*.nsys-rep *.qdrep *.sqlite

######################################################################
# Profiling targets with gprof
# Compiles with profiling flags and generates text + PDF reports

# Build library with gprof profiling flags (CPU only, no CUDA)
gprof-lib: CFLAGS += -O1 -g -pg -fno-inline -fno-omit-frame-pointer
gprof-lib: $(ARCH:.c=.o)
	rm -f libklt.a
	ar ruv libklt.a $(ARCH:.c=.o)

# Build individual examples with gprof (without auto-running)
gprof-example1: gprof-lib
	$(CC) -O1 -g -pg -fno-inline -fno-omit-frame-pointer $(FLAG1) $(FLAG2) -o example1 example1.c -L. -lklt $(LIB) -lm -pg

gprof-example2: gprof-lib
	$(CC) -O1 -g -pg -fno-inline -fno-omit-frame-pointer $(FLAG1) $(FLAG2) -o example2 example2.c -L. -lklt $(LIB) -lm -pg

gprof-example3: gprof-lib
	$(CC) -O1 -g -pg -fno-inline -fno-omit-frame-pointer $(FLAG1) $(FLAG2) -o example3 example3.c -L. -lklt $(LIB) -lm -pg

gprof-example4: gprof-lib
	$(CC) -O1 -g -pg -fno-inline -fno-omit-frame-pointer $(FLAG1) $(FLAG2) -o example4 example4.c -L. -lklt $(LIB) -lm -pg

gprof-example5: gprof-lib
	$(CC) -O1 -g -pg -fno-inline -fno-omit-frame-pointer $(FLAG1) $(FLAG2) -o example5 example5.c -L. -lklt $(LIB) -lm -pg

# Build all examples with gprof
gprof-all: gprof-example1 gprof-example2 gprof-example3 gprof-example4 gprof-example5
	@echo "All examples built with gprof profiling!"

######################################################################
# Profile and generate reports (run program + create text + PDF)

# Profile example3 with gprof (text + PDF output)
profile-gprof-example3: gprof-example3
	@echo "Running example3 with gprof profiling..."
	./example3
	@echo "Generating gprof text profile..."
	gprof example3 gmon.out > profile-example3-gprof.txt
	@echo "Generating gprof PDF call graph..."
	@if [ -f ./gprof2pdf.sh ]; then \
		bash ./gprof2pdf.sh profile-example3-gprof.txt; \
		echo "✓ Generated: profile-example3-gprof.txt"; \
		echo "✓ Generated: profile-example3-gprof.pdf"; \
	else \
		echo "Warning: gprof2pdf.sh not found. Only text profile generated."; \
		echo "✓ Generated: profile-example3-gprof.txt"; \
	fi

# Profile example1 (text + PDF)
profile-gprof-example1: gprof-example1
	@echo "Running example1 with gprof profiling..."
	./example1
	gprof example1 gmon.out > profile-example1-gprof.txt
	@if [ -f ./gprof2pdf.sh ]; then bash ./gprof2pdf.sh profile-example1-gprof.txt; fi
	@echo "✓ Profile saved to profile-example1-gprof.txt (and .pdf if available)"

# Profile example2 (text + PDF)
profile-gprof-example2: gprof-example2
	@echo "Running example2 with gprof profiling..."
	./example2
	gprof example2 gmon.out > profile-example2-gprof.txt
	@if [ -f ./gprof2pdf.sh ]; then bash ./gprof2pdf.sh profile-example2-gprof.txt; fi
	@echo "✓ Profile saved to profile-example2-gprof.txt (and .pdf if available)"

# Profile example4 (text + PDF)
profile-gprof-example4: gprof-example4
	@echo "Running example4 with gprof profiling..."
	./example4
	gprof example4 gmon.out > profile-example4-gprof.txt
	@if [ -f ./gprof2pdf.sh ]; then bash ./gprof2pdf.sh profile-example4-gprof.txt; fi
	@echo "✓ Profile saved to profile-example4-gprof.txt (and .pdf if available)"

# Profile example5 (text + PDF)
profile-gprof-example5: gprof-example5
	@echo "Running example5 with gprof profiling..."
	./example5
	gprof example5 gmon.out > profile-example5-gprof.txt
	@if [ -f ./gprof2pdf.sh ]; then bash ./gprof2pdf.sh profile-example5-gprof.txt; fi
	@echo "✓ Profile saved to profile-example5-gprof.txt (and .pdf if available)"

# Profile all examples with gprof
profile-gprof-all: gprof-all
	@echo "Profiling all examples with gprof..."
	@for example in example1 example2 example3 example4 example5; do \
		echo "Profiling $$example..."; \
		./$$example; \
		gprof $$example gmon.out > profile-$$example-gprof.txt; \
		if [ -f ./gprof2pdf.sh ]; then bash ./gprof2pdf.sh profile-$$example-gprof.txt; fi; \
		echo "✓ $$example profiled"; \
	done
	@echo "All profiles generated!"

# Alias for quick access (most common use case)
profile-example3: profile-gprof-example3

######################################################################
# CUDA/GPU Profiling targets
# These profile GPU kernels AND CPU overhead (total execution time)
# Use these for GPU-accelerated code to see complete performance picture

# Profile with nvprof (GPU kernels + CUDA API + CPU overhead)
# This shows TOTAL time including CPU→GPU transfers, kernel execution, etc.
nvprof-example3: example3
	@echo "=========================================="
	@echo "Profiling example3 with nvprof..."
	@echo "This shows GPU kernel time + CUDA API time + memory transfers"
	@echo "=========================================="
	nvprof --print-summary --print-api-trace --print-summary --log-file profile-example3-nvprof.txt ./example3 2>&1 || nvprof --print-gpu-trace --log-file profile-example3-nvprof.txt ./example3 2>&1
	@echo ""
	@echo "✓ Generated: profile-example3-nvprof.txt"
	@if [ -f profile-example3-nvprof.txt ]; then \
		echo "Generating PDF visualization..."; \
		bash nvprof2pdf.sh profile-example3-nvprof.txt profile-example3-nvprof.pdf && \
		echo "✓ Generated: profile-example3-nvprof.pdf" || \
		echo "⚠ PDF generation failed (matplotlib required: pip install matplotlib)"; \
	fi
	@echo "View with: cat profile-example3-nvprof.txt"

# Profile with nvprof (simple version, GPU trace only)
nvprof-simple-example3: example3
	@echo "Profiling GPU kernels only..."
	nvprof --print-summary --log-file profile-example3-nvprof-simple.txt ./example3
	@echo "✓ Generated: profile-example3-nvprof-simple.txt"
	@if [ -f profile-example3-nvprof-simple.txt ]; then \
		echo "Generating PDF visualization..."; \
		bash nvprof2pdf.sh profile-example3-nvprof-simple.txt profile-example3-nvprof-simple.pdf && \
		echo "✓ Generated: profile-example3-nvprof-simple.pdf" || \
		echo "⚠ PDF generation failed (matplotlib required: pip install matplotlib)"; \
	fi

# Profile with Nsight Systems (timeline view of CPU + GPU)
# This is best for understanding execution flow and finding bottlenecks
nsys-example3: example3
	@echo "=========================================="
	@echo "Profiling with Nsight Systems (timeline)..."
	@echo "This shows when CPU and GPU are active over time"
	@echo "=========================================="
	nsys profile --stats=true --force-overwrite=true --output=profile-example3-nsys ./example3 > profile-example3-nsys.txt 2>&1 || echo "Warning: nsys not available. Install with: apt-get install nsight-systems-cli"
	@echo ""
	@echo "✓ Generated: profile-example3-nsys.nsys-rep (binary, view with nsys-ui)"
	@echo "✓ Generated: profile-example3-nsys.txt (text summary)"

# Profile with Nsight Compute (detailed kernel metrics)
# Use this to optimize individual kernel performance
ncu-example3: example3
	@echo "=========================================="
	@echo "Profiling kernels with Nsight Compute (detailed metrics)..."
	@echo "This shows kernel occupancy, memory efficiency, etc."
	@echo "=========================================="
	ncu --set full --export profile-example3-ncu --force-overwrite ./example3 || echo "Warning: ncu not available. Install with: apt-get install nsight-compute-cli"
	@echo ""
	@echo "✓ Generated: profile-example3-ncu.ncu-rep (view with ncu-ui)"

# Quick kernel profiling (faster, basic metrics)
ncu-quick-example3: example3
	@echo "Quick kernel profiling with Nsight Compute..."
	ncu --set basic --export profile-example3-ncu-quick --force-overwrite ./example3 || ncu --print-summary stdout ./example3 > profile-example3-ncu-quick.txt 2>&1
	@echo "✓ Generated: profile-example3-ncu-quick.ncu-rep"

# Comprehensive profiling (runs nvprof + nsys if available)
cuda-profile-example3: example3
	@echo "=========================================="
	@echo "COMPREHENSIVE CUDA PROFILING"
	@echo "This profiles EVERYTHING: CPU, GPU, memory transfers"
	@echo "=========================================="
	@echo ""
	@echo "1. Running nvprof (GPU + API trace)..."
	@nvprof --print-summary --print-api-trace --print-summary --log-file profile-example3-nvprof.txt ./example3 2>&1 || echo "nvprof failed, trying simple mode..." && nvprof --print-gpu-trace --log-file profile-example3-nvprof.txt ./example3 2>&1
	@echo ""
	@echo "2. Running nsys (if available)..."
	@nsys profile --stats=true --force-overwrite=true --output=profile-example3-nsys ./example3 > profile-example3-nsys-summary.txt 2>&1 || echo "nsys not available (optional)"
	@echo ""
	@echo "=========================================="
	@echo "✓ PROFILING COMPLETE!"
	@echo "=========================================="
	@echo "Generated files:"
	@echo "  • profile-example3-nvprof.txt - GPU kernels + API calls + timing"
	@echo "  • profile-example3-nvprof.pdf - GPU kernel visualization (if matplotlib available)"
	@echo "  • profile-example3-nsys.nsys-rep - Timeline (if nsys available)"
	@echo "  • profile-example3-nsys-summary.txt - Text summary (if nsys available)"

# Profile all examples with CUDA
cuda-profile-all: cuda-profile-example3
	@echo "All CUDA profiles generated!"

.PHONY: clean depend all lib gprof gprof-example1 gprof-example2 gprof-example3 gprof-example4 gprof-example5