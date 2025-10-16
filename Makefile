######################################################################
# Choose your compilers
CC = gcc
NVCC = nvcc
######################################################################
# CUDA compiler and flags
NVCC = nvcc
CUDA_FLAGS = -O3 -arch=sm_50
CUDA_LIBS = -lcudart
CUDA_LIBDIR = -L/usr/local/cuda/lib64

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
CUDA_SRCS = interpolate_cuda.cu mineigenvalue_cuda.cu
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

lib: $(ARCH:.c=.o) $(CUDA_OBJS)
	rm -f libklt.a
	ar ruv libklt.a $(ARCH:.c=.o) $(CUDA_OBJS)

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
	      feat*.ppm features.ft features.txt gmon.out profile-*.txt profile-*.dot profile-*.pdf

######################################################################
# Profiling targets with gprof
gprof-lib: CFLAGS += -O1 -g -pg -fno-inline -fno-omit-frame-pointer
gprof-lib: $(ARCH:.c=.o)
	rm -f libklt.a
	ar ruv libklt.a $(ARCH:.c=.o)

gprof-example1: gprof-lib
	$(CC) -O1 -g -pg -fno-inline -fno-omit-frame-pointer $(FLAG1) $(FLAG2) -o example1 example1.c -L. -lklt $(LIB) -lm -pg
	./example1
	gprof -b example1 > profile-example1.txt
	../05-gprof/gprof2pdf.sh profile-example1.txt
	@echo "Profile saved to profile-example1.txt and profile-example1.pdf"

gprof-example2: gprof-lib
	$(CC) -O1 -g -pg -fno-inline -fno-omit-frame-pointer $(FLAG1) $(FLAG2) -o example2 example2.c -L. -lklt $(LIB) -lm -pg
	./example2
	gprof -b example2 > profile-example2.txt
	../05-gprof/gprof2pdf.sh profile-example2.txt
	@echo "Profile saved to profile-example2.txt and profile-example2.pdf"

gprof-example3: gprof-lib
	$(CC) -O1 -g -pg -fno-inline -fno-omit-frame-pointer $(FLAG1) $(FLAG2) -o example3 example3.c -L. -lklt $(LIB) -lm -pg
	./example3
	gprof -b example3 > profile-example3.txt
	../05-gprof/gprof2pdf.sh profile-example3.txt
	@echo "Profile saved to profile-example3.txt and profile-example3.pdf"

gprof-example4: gprof-lib
	$(CC) -O1 -g -pg -fno-inline -fno-omit-frame-pointer $(FLAG1) $(FLAG2) -o example4 example4.c -L. -lklt $(LIB) -lm -pg
	./example4
	gprof -b example4 > profile-example4.txt
	../05-gprof/gprof2pdf.sh profile-example4.txt
	@echo "Profile saved to profile-example4.txt and profile-example4.pdf"

gprof-example5: gprof-lib
	$(CC) -O1 -g -pg -fno-inline -fno-omit-frame-pointer $(FLAG1) $(FLAG2) -o example5 example5.c -L. -lklt $(LIB) -lm -pg
	./example5
	gprof -b example5 > profile-example5.txt
	../05-gprof/gprof2pdf.sh profile-example5.txt
	@echo "Profile saved to profile-example5.txt and profile-example5.pdf"

gprof-all: gprof-example1 gprof-example2 gprof-example3 gprof-example4 gprof-example5
	@echo "All profiles generated!"



.PHONY: clean depend all lib gprof gprof-example1 gprof-example2 gprof-example3 gprof-example4 gprof-example5