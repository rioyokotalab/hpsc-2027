# Lecture 4. Fetch the libraries once (see README.md), then: module load cuda; make
TK_DIR = thunderkittens
CUTLASS_DIR = cutlass
NVCC = nvcc -O3 -DNDEBUG -gencode arch=compute_90a,code=sm_90a -lcublas -lcuda
LEVELS = level_02 level_03 level_04 level_05 level_06 level_07 level_08

all: 00_cublas $(LEVELS) tk_h100 10_cutlass_gemm 30_linear_relu

00_cublas: 00_cublas.cu gemm_check.h
	$(NVCC) 00_cublas.cu -o 00_cublas

# ThunderKittens educational levels, timed and checked by our launch.cu
TKFLAGS = -DKITTENS_SM90 -std=c++20 --expt-extended-lambda --expt-relaxed-constexpr -I$(TK_DIR)/include -I$(TK_DIR)/prototype
build/%.cu: $(TK_DIR)/kernels/gemm/educational_h100/%.cu launch.cu gemm_check.h
	mkdir -p build && cp $< $@ && cp launch.cu gemm_check.h build/
level_%: build/level_%.cu
	$(NVCC) $(TKFLAGS) -DNAME='"TK level $*"' $< -o $@

# Exercise 4: cp build/level_08.cu level_08_any.cu, edit, then make level_08_any
level_08_any: level_08_any.cu launch.cu gemm_check.h
	$(NVCC) $(TKFLAGS) -DNAME='"level 08, any N"' level_08_any.cu -o level_08_any

# The full ThunderKittens H100 GEMM, with its own timing
tk_h100: $(TK_DIR)/kernels/gemm/bf16_h100/bf16_h100_gemm.cu
	$(NVCC) $(TKFLAGS) $< -o $@

# CUTLASS: change the tile with TILE="-DTM=128 -DTN=128 -DTK=64 -DCM=1 -DSTAGES=3"
TILE =
10_cutlass_gemm: 10_cutlass_gemm.cu gemm_check.h
	$(NVCC) -std=c++17 --expt-relaxed-constexpr $(TILE) -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include 10_cutlass_gemm.cu -o 10_cutlass_gemm

clean:
	rm -rf build 00_cublas $(LEVELS) level_08_any tk_h100 10_cutlass_gemm 30_linear_relu

# Homework: one fused kernel for y = relu(x W + b)
30_linear_relu: 30_linear_relu.cu
	$(NVCC) -std=c++17 --expt-relaxed-constexpr -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include 30_linear_relu.cu -o 30_linear_relu
