//
//  main2.c
//  Lab4
//
//  Created by Alan Weide on 4/10/16.
//  Copyright © 2016 weidea. All rights reserved.
//

#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define DIM 2048
#define NBLK 8
#define TPB 512

void init_matrices(double *A, double *C) {
	for (int i = 0; i < DIM; i++) {
		for (int j = 0; j < DIM; j++) {
			A[DIM * i + j] = 1.0 + ((double)rand() / RAND_MAX);
			C[DIM * i + j] = 0;
		}
	}
}

void compute_stuff(double *A, double *C, int wBlock, int hBlock) {
	for (int i = 0; i < DIM; i++) {
		for (int j = 0; j < DIM; j++) {
			for (int k = 0; k < DIM; k++) {
				C[DIM * i + j] += A[DIM * k + i] * A[DIM * k + j];
			}
		}
		
	}
}

#ifdef CUDA
__global__ void cuda_compute(double *A, double *C, int wBlock, int hBlock) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	
	for (int i = x * wBlock; i < DIM && i < (x + 1) * wBlock; i++) {
		for (int j = y * hBlock; j < DIM && j < (y + 1) * hBlock; j++) {
			for (int k = 0; k < DIM; k++) {
				C[DIM * i + j] += A[DIM * k + i] * A[DIM * k + j];
			}
		}
	}
}
#endif

void printMatrix (double *F) {
	for (int i = 0; i < DIM; i++) {
		printf("[");
		for (int j = 0; j < DIM; j++) {
			printf(" %7.3lf", F[DIM * i + j]);
		}
		printf(" ]\n");
	}
	
}

int main(int argc, const char * argv[]) {
	srand((uint32_t)time(NULL));
	size_t memSize = DIM * DIM * sizeof(double);
	double *A, *C;
	A = (double*) malloc(memSize);
	C = (double*) malloc(memSize);
	
	clock_t clock_start = clock();
	
	init_matrices(A, C);
	
#ifdef PRINT
	printMatrix(A);
	printf("---------------------------------------------------------------------------------------\n");
#endif
	
	int wBlock = 1 + DIM / NBLK;
	int hBlock = 1 + DIM / TPB;
	
#ifdef CUDA
	double *d_A, *d_C;
	cudaMalloc((void**) &d_A, memSize);
	cudaMalloc((void**) &d_C, memSize);
	cudaMemcpy(d_A, A, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, C, memSize, cudaMemcpyHostToDevice);
	dim3 dimGrid(NBLK);
	dim3 dimBlock(TPB);
#endif
	
	clock_t init_duration = clock() - clock_start;
	
#ifdef CUDA
	cuda_compute<<<dimGrid, dimBlock>>>(d_A, d_C, wBlock, hBlock);
	cudaThreadSynchronize();
	cudaMemcpy(C, d_C, memSize, cudaMemcpyDeviceToHost);
#else
	compute_stuff(A, C, wBlock, hBlock);
#endif
	
	clock_t total_duration = clock() - clock_start;
	
	double time_in_seconds = (total_duration - init_duration) / 1000000.0;
	
#ifdef PRINT
	printMatrix(C);
#endif
	
	printf("Computation duration: %lfs; Performance: %lf GFlops\n",
		   time_in_seconds, 1E-9 * 2.0 * DIM * DIM * DIM / time_in_seconds);
}