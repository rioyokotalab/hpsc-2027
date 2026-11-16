# Fixed harness file: do not edit.
ARCH ?= sm_90

all: gemm reference

gemm: harness.cu kernel.cu inputs.h
	nvcc -O3 -arch=$(ARCH) harness.cu kernel.cu -o gemm

reference: reference.cu inputs.h
	nvcc -O3 -arch=$(ARCH) reference.cu -lcublas -o reference

clean:
	rm -f gemm reference
