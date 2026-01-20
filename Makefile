INCLUDEFLAGS=-I include

CC=gcc
CFLAGS=-g -Wall $(INCLUDEFLAGS) -O3
LDLIBS=

NVCC=nvcc
NVCCFLAGS=-O3 -arch=sm_80 -g -G $(INCLUDEFLAGS) --std c++17 --extended-lambda

CFILES=$(wildcard src/*.c)
CPPFILES=$(wildcard src/*.cpp)
CUDAFILES=$(wildcard src/*.cu)
OBJ=$(patsubst src/%.c,build/%.o,$(CFILES))
OBJ+=$(patsubst src/%.cpp,build/%.o,$(CPPFILES))
OBJ+=$(patsubst src/%.cu,build/%.o,$(CUDAFILES))
CDEP=$(patsubst src/%.c,dependencies/%.d,$(CFILES))
CPPDEP=$(patsubst src/%.cpp,dependencies/%.d,$(CPPFILES))
CUDADEP=$(patsubst src/%.cu,dependencies/%.du,$(CUDAFILES))

EXECNAME=bin/KMeans

all: $(CDEP) $(CUDADEP) build

.PHONY: clean all check build


clean:
	-rm -r build/ bin/ dependencies/ tests/test_output_files

dependencies/%.d: src/%.c Makefile
	mkdir -p dependencies
	mkdir -p build
	echo -n "build/" >$@
	$(CC) $(CFLAGS) -M $< >>$@
	echo "\t$(CC) $(CFLAGS) $< -o build/$*.o -c" >>$@

dependencies/%.d: src/%.cpp Makefile
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

include $(CDEP) $(CPPDEP) $(CUDADEP)

build: $(OBJ)
	mkdir -p bin
	 $(NVCC) $(OBJ) -o $(EXECNAME) $(LDLIBS) $(NVCCFLAGS)

check: build
	cd tests && ./test.bash
