# Fixed harness file: do not edit.
ARCH ?= sm_90

all: gemm reference gemm_bf16 reference_bf16

gemm: harness.cu kernel.cu inputs.h
	nvcc -O3 -arch=$(ARCH) harness.cu kernel.cu -o gemm

reference: reference.cu inputs.h
	nvcc -O3 -arch=$(ARCH) reference.cu -lcublas -o reference

gemm_bf16: harness_bf16.cu kernel_bf16.cu inputs.h
	nvcc -O3 -arch=$(ARCH) harness_bf16.cu kernel_bf16.cu -o gemm_bf16

reference_bf16: reference_bf16.cu inputs.h
	nvcc -O3 -arch=$(ARCH) reference_bf16.cu -lcublas -o reference_bf16

clean:
	rm -f gemm reference gemm_bf16 reference_bf16
