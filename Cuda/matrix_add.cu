#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/timeb.h>
#include <device_launch_parameters.h>
#include <cmath>

__global__ void matrix_add(int *d_A, int *d_B, int *d_OUT, int num_col, int num_row) {
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int index = row * num_col + col;
	if (col < num_col && row < num_row) {
		d_OUT[index] = d_A[index] + d_B[index];
	}
}

void matrix_add(int num_col, int num_row) {
	int* A, * B, * OUT;
	int* d_A, * d_B, * d_OUT;
	int size = num_col * num_row;
	A = (int *) malloc(size * sizeof(int));
	B = (int *) malloc(size * sizeof(int));
	OUT = (int *) malloc(size * sizeof(int));

	for (int i = 0; i < size; i++) {
		A[i] = rand() % 100;
		B[i] = rand() % 100;
	}

	cudaError_t cudaStatus = cudaMalloc((void**)&d_A, sizeof(int) * size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "A malloc failed");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_B, sizeof(int) * size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "B malloc failed");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_OUT, sizeof(int) * size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "OUT malloc failed");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_A, A, sizeof(int) * size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "A copy failed");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_B, B, sizeof(int) * size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "B copy failed");
		goto Error;
	}
	

	dim3 grid(ceil(num_col / 32), ceil(num_row / 32));
	dim3 block(32, 32);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	matrix_add<<<grid, block>>>(d_A, d_B, d_OUT, num_col, num_row);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	float time;
	cudaEventElapsedTime(&time, start, stop);
	printf("Time: %f\n", time);
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaStatus = cudaMemcpy(OUT, d_OUT, sizeof(int) * size, cudaMemcpyHostToDevice);


Error:
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_OUT);
	free(A);
	free(B);
	free(OUT);
}