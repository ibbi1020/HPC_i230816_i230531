######################################################################
# Compilers
CC   = gcc
NVCC = nvcc

######################################################################
# Global Flags
FLAG1 = -DNDEBUG

# Enable ALL optimization flags by default
CFLAGS = $(FLAG1) -I/usr/local/cuda/include -fPIC

NVCCFLAGS = $(FLAG1) \
            -gencode arch=compute_50,code=sm_50 \
            -gencode arch=compute_50,code=compute_50 \
            -Xcompiler -fPIC \
            -rdc=true \
            -DUSE_CUDA_FEATURE_SELECTION \
            -DUSE_CONST



######################################################################
# Library paths
LIB = -L/usr/local/lib -L/usr/lib -L.
LIBS = -lm -lcudart

######################################################################
# Source files
C_ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
         storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c

CUDA_SRCS = convolve_gpu.cu interpolate_cuda.cu mineigenvalue_cuda.cu
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

EXAMPLES = example1 example2 example3 example4 example5

######################################################################
# Rules
.SUFFIXES: .c .o .cu

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

.cu.o:
	$(NVCC) -c $(NVCCFLAGS) $< -o $@


######################################################################
# Build static library
lib: $(C_ARCH:.c=.o) $(CUDA_OBJS)
	rm -f libklt.a
	ar ruv libklt.a $(C_ARCH:.c=.o) $(CUDA_OBJS)

######################################################################
# Example programs — NVCC handles linking to CUDA libs
example%: libklt.a
	$(NVCC) -O3 -rdc=true -o $@ $@.c -L. -lklt -lm -lcudart


######################################################################
clean:
	rm -f *.o *.a $(EXAMPLES) *.tar *.tar.gz libklt.a \
	      feat*.ppm features.ft features.txt gmon.out \
	      profile-*.txt profile-*.pdf

.PHONY: clean all lib
