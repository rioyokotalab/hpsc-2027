ARCH = sm_90
NVCC = nvcc -O3 -arch=$(ARCH)

cuda: 00_hello 01_thread 02_block 03_if 04_atomic 05_memcpy 06_shared 07_devquery vecadd 09_threads

00_hello: 00_hello.cu
	$(NVCC) 00_hello.cu -o 00_hello
01_thread: 01_thread.cu
	$(NVCC) 01_thread.cu -o 01_thread
02_block: 02_block.cu
	$(NVCC) 02_block.cu -o 02_block
03_if: 03_if.cu
	$(NVCC) 03_if.cu -o 03_if
04_atomic: 04_atomic.cu
	$(NVCC) 04_atomic.cu -o 04_atomic
05_memcpy: 05_memcpy.cu
	$(NVCC) 05_memcpy.cu -o 05_memcpy
06_shared: 06_shared.cu
	$(NVCC) 06_shared.cu -o 06_shared
07_devquery: 07_devquery.cu
	$(NVCC) 07_devquery.cu -o 07_devquery
vecadd: 08_vecadd.cu
	$(NVCC) 08_vecadd.cu -o vecadd
09_threads: 09_threads.cu
	$(NVCC) 09_threads.cu -o 09_threads

# OpenACC: module purge; module load nvhpc  (nvhpc conflicts with the cuda module)
ACC = nvc++ -O3 -acc -gpu=cc90 -Minfo=accel

10_acc_vecadd: 10_acc_vecadd.cpp
	$(ACC) 10_acc_vecadd.cpp -o 10_acc_vecadd
11_acc_gang: 11_acc_gang.cpp
	$(ACC) 11_acc_gang.cpp -o 11_acc_gang
12_acc_data: 12_acc_data.cpp
	$(ACC) 12_acc_data.cpp -o 12_acc_data
13_acc_reduction: 13_acc_reduction.cpp
	$(ACC) 13_acc_reduction.cpp -o 13_acc_reduction
14_acc_matmul: 14_acc_matmul.cpp
	$(ACC) 14_acc_matmul.cpp -o 14_acc_matmul
15_acc_parallel: 15_acc_parallel.cpp
	$(ACC) 15_acc_parallel.cpp -o 15_acc_parallel
16_acc_kernels: 16_acc_kernels.cpp
	$(ACC) 16_acc_kernels.cpp -o 16_acc_kernels

openacc: 10_acc_vecadd 11_acc_gang 12_acc_data 13_acc_reduction 14_acc_matmul 15_acc_parallel 16_acc_kernels

clean:
	rm -f 00_hello 01_thread 02_block 03_if 04_atomic 05_memcpy 06_shared 07_devquery vecadd 09_threads 10_acc_vecadd 11_acc_gang 12_acc_data 13_acc_reduction 14_acc_matmul 15_acc_parallel 16_acc_kernels a.out
