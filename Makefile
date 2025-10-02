######################################################################
# Choose your favorite C compiler
CC = gcc
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
CFLAGS = $(FLAG1) $(FLAG2)
######################################################################
# Library flags
LIB = -L/usr/local/lib -L/usr/lib -L.
LIBS = -lm
######################################################################
# Source and object files
EXAMPLES = example1.c example2.c example3.c example4.c example5.c
ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c

######################################################################
# Targets
.SUFFIXES:  .c .o

all: lib $(EXAMPLES:.c=)

.c.o:
	$(CC) -c $(CFLAGS) $<

lib: $(ARCH:.c=.o)
	rm -f libklt.a
	ar ruv libklt.a $(ARCH:.c=.o)
	rm -f *.o

example1: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c $(LIB) -lklt $(LIBS) $(LDFLAGS)

example2: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c $(LIB) -lklt $(LIBS) $(LDFLAGS)

example3: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c $(LIB) -lklt $(LIBS) $(LDFLAGS)

example4: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c $(LIB) -lklt $(LIBS) $(LDFLAGS)

example5: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c $(LIB) -lklt $(LIBS) $(LDFLAGS)

######################################################################
# Gprof profiling targets
gprof: CFLAGS=-O1 -g -pg -fno-inline -fno-omit-frame-pointer -Wall -Wfatal-errors $(FLAG1) $(FLAG2)
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
	makedepend $(ARCH) $(EXAMPLES)

clean:
	rm -f *.o *.a $(EXAMPLES:.c=) *.tar *.tar.gz libklt.a \
	      feat*.ppm features.ft features.txt gmon.out profile-*.txt profile-*.pdf

.PHONY: clean depend all lib gprof gprof-example1 gprof-example2 gprof-example3 gprof-example4 gprof-example5
