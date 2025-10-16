######################################################################
# Choose your compilers
CC = gcc
NVCC = nvcc
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

# All architecture files
ARCH = $(C_ARCH) $(CUDA_ARCH)

# Object files
C_OBJS = $(C_ARCH:.c=.o)
CUDA_OBJS = $(CUDA_ARCH:.cu=.o)
ALL_OBJS = $(C_OBJS) $(CUDA_OBJS)

######################################################################
# Targets

all: lib $(EXAMPLES:.c=)

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

lib: $(ALL_OBJS)
	rm -f libklt.a
	$(NVCC) $(NVCCFLAGS) -dlink convolve_gpu.o -o device_link.o
	ar ruv libklt.a $(ALL_OBJS) device_link.o
	ranlib libklt.a

example1: libklt.a
	$(NVCC) -O3 $(NVCCFLAGS) -o $@ $@.c convolve_gpu.cu $(LIB) -lklt -lm -lcudart $(LDFLAGS)

example2: libklt.a
	$(NVCC) -O3 $(NVCCFLAGS) -o $@ $@.c convolve_gpu.cu $(LIB) -lklt -lm -lcudart $(LDFLAGS)

example3: libklt.a
	$(NVCC) -O3 $(NVCCFLAGS) -o $@ $@.c convolve_gpu.cu $(LIB) -lklt -lm -lcudart $(LDFLAGS)

example4: libklt.a
	$(NVCC) -O3 $(NVCCFLAGS) -o $@ $@.c convolve_gpu.cu $(LIB) -lklt -lm -lcudart $(LDFLAGS)

example5: libklt.a
	$(NVCC) -O3 $(NVCCFLAGS) -o $@ $@.c convolve_gpu.cu $(LIB) -lklt -lm -lcudart $(LDFLAGS)

######################################################################
# Gprof profiling targets
gprof: CFLAGS=-O1 -g -pg -fno-inline -fno-omit-frame-pointer -Wall -Wfatal-errors $(FLAG1) $(FLAG2)
gprof: NVCCFLAGS=$(FLAG1) $(FLAG2) -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -Xcompiler "-O1 -g -pg -fno-inline -fno-omit-frame-pointer -Wall -Wfatal-errors"
gprof: LDFLAGS=-pg
gprof: clean lib $(EXAMPLES:.c=)
	@echo "Gprof-enabled executables built. Run one of the examples to generate gmon.out"

# Profile individual examples
gprof-example1: gprof
	./example1
	gprof -b example1 > profile-example1.txt
	./gprof2pdf.sh profile-example1.txt
	@echo "Profile saved to profile-example1.txt and profile-example1.pdf"

gprof-example2: gprof
	./example2
	gprof -b example2 > profile-example2.txt
	./gprof2pdf.sh profile-example2.txt
	@echo "Profile saved to profile-example2.txt and profile-example2.pdf"

gprof-example3: gprof
	./example3
	gprof -b example3 > profile-example3.txt
	./gprof2pdf.sh profile-example3.txt
	@echo "Profile saved to profile-example3.txt and profile-example3.pdf"

gprof-example4: gprof
	./example4
	gprof -b example4 > profile-example4.txt
	./gprof2pdf.sh profile-example4.txt
	@echo "Profile saved to profile-example4.txt and profile-example4.pdf"

gprof-example5: gprof
	./example5
	gprof -b example5 > profile-example5.txt
	./gprof2pdf.sh profile-example5.txt
	@echo "Profile saved to profile-example5.txt and profile-example5.pdf"

######################################################################
# Utility targets
depend:
	makedepend $(C_ARCH) $(EXAMPLES)

clean:
	rm -f *.o *.a $(EXAMPLES:.c=) *.tar *.tar.gz libklt.a device_link.o \
	      feat*.ppm features.ft features.txt gmon.out profile-*.txt profile-*.pdf

.PHONY: clean depend all lib gprof gprof-example1 gprof-example2 gprof-example3 gprof-example4 gprof-example5