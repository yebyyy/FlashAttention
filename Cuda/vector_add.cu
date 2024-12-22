#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/timeb.h>
#include <device_launch_parameters.h>

__global__ void cuda_vector_add(int* out, int* a, int* b, int n) {
	int i = threadIdx.x;
	if (i < n) {
		out[i] = a[i] + b[i];
	}
}

void test_add_vector(int n) {
	int* a, * b, * out;
	int* d_a, * d_b, * d_out;    // d_ is for device
	a = (int *)malloc(n * sizeof(int));
	b = (int *)malloc(n * sizeof(int));
	out = (int *)malloc(n * sizeof(int));

	for(int i = 0; i < n; i++) {
		a[i] = rand() % 100;
		b[i] = rand() % 100;
	}

	//void** &d_a means to convert the address of d_a to void**
	//after the cudaMalloc, d_a will store the address of the memory on the GPU
	cudaError_t cudaStatus = cudaMalloc((void**)&d_a, sizeof(int) * n);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "a malloc failed");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_b, sizeof(int) * n);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "b malloc failed");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_out, sizeof(int) * n);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "out malloc failed");
		goto Error;
	}


	//copy the data from CPU to GPU
	cudaStatus = cudaMemcpy(d_a, a, sizeof(int) * n, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "a copy failed");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_b, b, sizeof(int) * n, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "b copy failed");
		goto Error;
	}

	//cudaEveent_t is used to record the time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	// lauch a kernel with 1 block and n threads
	cuda_vector_add <<<1, n >>> (d_out, d_a, d_b, n);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	printf("Time: %f ms\n", time);
	cudaStatus = cudaMemcpy(out, d_out, sizeof(int) * n, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "out copy failed");
		goto Error;
	}

	for (int i = 0; i < n; i++) {
		printf("%d + %d = %d\n", a[i], b[i], out[i]);
	}
	
Error:
	//cudafree frees the memory on the GPU, just like cudaMalloc allocates memory on the GPU
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_out);
	free(a);
	free(b);
	free(out);
}