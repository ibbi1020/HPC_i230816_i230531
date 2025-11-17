
CC = gcc

FLAG1 = -DNDEBUG

CFLAGS = $(FLAG1) $(FLAG2)

LIB = -L/usr/local/lib -L/usr/lib -L.
LIBS = -lm

EXAMPLES = example1.c example2.c example3.c example4.c example5.c
ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c


.SUFFIXES:  .c .o

all: lib $(EXAMPLES:.c=)

.c.o:
	$(CC) -c $(CFLAGS) $<

libklt.a: $(ARCH:.c=.o)
	rm -f libklt.a
	ar ruv libklt.a $(ARCH:.c=.o)
	rm -f *.o

# keep a backward-compatible `lib` name (used by the top-level `all` target)
lib: libklt.a

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