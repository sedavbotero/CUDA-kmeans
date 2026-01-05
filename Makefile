INCLUDEFLAGS=-I include

DEFINE=

CC=gcc
CFLAGS=-g -Wall $(INCLUDEFLAGS) $(DEFINE) -O3
LDLIBS=

NVCC=nvcc
NVCCFLAGS=-O3 -arch=sm_80 -g -G $(INCLUDEFLAGS) --std c++17 $(DEFINE) --extended-lambda

CFILES=$(wildcard src/*.c)
CUDAFILES=$(wildcard src/*.cu)
OBJ=$(patsubst src/%.c,build/%.o,$(CFILES))
OBJ+=$(patsubst src/%.cu,build/%.o,$(CUDAFILES))
CDEP=$(patsubst src/%.c,dependencies/%.d,$(CFILES))
CUDADEP=$(patsubst src/%.cu,dependencies/%.du,$(CUDAFILES))
TESTS=$(wildcard tests/src/*)
TESTOBJ=$(patsubst tests/src/%.cu,tests/build/%.o,$(patsubst tests/src/%.cpp,tests/build/%.o,$(TESTS)))

EXECNAME=bin/KMeans

all: $(CDEP) $(CUDADEP) build

.PHONY: clean all check build clean-all clean-lib clean-tests

clean-all: clean clean-lib clean-tests

clean:
	-rm -r build/ bin/ dependencies/ tests/build/ tests/test tests/test_outputs

clean-lib:
	cd lib && $(MAKE) clean

clean-tests:
	cd tests/end_to_end_tests && $(MAKE) clean

dependencies/%.d: src/%.c Makefile
	mkdir -p dependencies
	mkdir -p build
	echo -n "build/" >$@
	$(CC) $(CFLAGS) -M $< >>$@
	echo "\t$(CC) $(CFLAGS) $< -o build/$*.o -c" >>$@

dependencies/%.du: src/%.cu Makefile
	mkdir -p dependencies
	mkdir -p build
	echo -n "build/" >$@
	$(NVCC) $(NVCCFLAGS) -M $< >>$@
	echo "\t$(NVCC) $(NVCCFLAGS) $< -o build/$*.o -dc" >>$@

include $(CDEP) $(CUDADEP)

build: $(OBJ)
	mkdir -p bin
	 $(NVCC) $(OBJ) -o $(EXECNAME) $(LDLIBS) $(NVCCFLAGS)

run: build
	./$(EXECNAME)

TESTFLAGS=-I tests/include -I lib/usr/local/include
GTESTLIB=lib/usr/local/lib/libgtest.a lib/usr/local/lib/libgtest_main.a

libs:
	cd lib && $(MAKE)

tests/build/%.o:: tests/src/%.cu libs
	mkdir -p tests/build
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(TESTFLAGS) -dc $< -o $@

tests/build/%.o:: tests/src/%.cpp libs
	mkdir -p tests/build
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(TESTFLAGS) -c $< -o $@

tests/test: $(OBJ) $(TESTOBJ) libs
	echo $(TESTOBJ)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(LDLIBS) $(TESTOBJ) $(TESTFLAGS) \
		$(filter-out build/main.o, $(OBJ)) \
		$(GTESTLIB) \
		-o tests/test

check: tests/test build
	mkdir -p tests/test_outputs
	cd tests && ./test
	# cd tests/end_to_end_tests && $(MAKE) check
