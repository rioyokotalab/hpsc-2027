ARCH = sm_90
NVCC = nvcc -O3 -arch=$(ARCH)

00_hello: 00_hello.cu
	$(NVCC) 00_hello.cu
01_thread: 01_thread.cu
	$(NVCC) 01_thread.cu
02_block: 02_block.cu
	$(NVCC) 02_block.cu
03_if: 03_if.cu
	$(NVCC) 03_if.cu
04_atomic: 04_atomic.cu
	$(NVCC) 04_atomic.cu
05_memcpy: 05_memcpy.cu
	$(NVCC) 05_memcpy.cu
06_shared: 06_shared.cu
	$(NVCC) 06_shared.cu
07_devquery: 07_devquery.cu
	$(NVCC) 07_devquery.cu
vecadd: 08_vecadd.cu
	$(NVCC) 08_vecadd.cu -o vecadd
09_threads: 09_threads.cu
	$(NVCC) 09_threads.cu
