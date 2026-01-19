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

check: build
	cd tests && ./test.bash
