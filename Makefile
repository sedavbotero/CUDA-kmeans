INCLUDEFLAGS=-I include

CC=gcc
CFLAGS=-g -Wall $(INCLUDEFLAGS) -O3
LDLIBS=

CUDA_PATH ?= /usr/local/cuda-13.1
CCCL_INCLUDE = $(CUDA_PATH)/targets/x86_64-linux/include/cccl

NVCC=nvcc
NVCCFLAGS=-O3 \
	-gencode arch=compute_60,code=sm_60 \
	-gencode arch=compute_75,code=sm_75 \
	-gencode arch=compute_80,code=sm_80 \
	  -g -G $(INCLUDEFLAGS) --std c++17 --extended-lambda -I $(CCCL_INCLUDE)

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
	printf "build/" >$@
	$(CC) $(CFLAGS) -M $< >>$@
	printf "\t$(CC) $(CFLAGS) $< -o build/$*.o -c\n" >>$@

dependencies/%.d: src/%.cpp Makefile
	mkdir -p dependencies
	mkdir -p build
	printf "build/" >$@
	$(CC) $(CFLAGS) -M $< >>$@
	printf "\t$(CC) $(CFLAGS) $< -o build/$*.o -c\n" >>$@

dependencies/%.du: src/%.cu Makefile
	mkdir -p dependencies
	mkdir -p build
	printf "build/" >$@
	$(NVCC) $(NVCCFLAGS) -M $< >>$@
	printf "\t$(NVCC) $(NVCCFLAGS) $< -o build/$*.o -dc\n" >>$@

include $(CDEP) $(CPPDEP) $(CUDADEP)

build: $(OBJ)
	mkdir -p bin
	 $(NVCC) $(OBJ) -o $(EXECNAME) $(LDLIBS) $(NVCCFLAGS)

check: build
	cd tests && ./test.bash
