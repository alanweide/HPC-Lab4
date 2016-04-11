//
//  main.c
//  Lab4
//
//  Created by Alan Weide on 4/9/16.
//  Copyright © 2016 weidea. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define DIM 4097

#ifndef REPS
#define REPS 100
#endif

void initMatrix(double *F) {
	for (int i = 0; i < DIM; i++) {
		for (int j = 0; j < DIM; j++) {
			F[DIM * i + j] = 1.0 + ((double)rand() / RAND_MAX);
		}
	}
}

void printMatrix (double *F) {
	for (int i = 0; i < DIM; i++) {
		printf("[ ");
		for (int j = 0; j < DIM; j++) {
			printf(" %lf", F[DIM * i + j]);
		}
		printf(" ]\n");
	}
	
}

void computeStuff(double *F) {
	for (int k = 0; k < REPS; k++)  {
		for (int i = 1; i < DIM; i++) {
			for (int j = 0; j < DIM - 1; j++) {
				F[DIM * i + j] = F[DIM * (i-1) + (j+1)] + F[DIM * i + (j+1)];
			}
		}
	}
}

__global__ void cudaCompute(double* F) {
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	
	for (int i = bid + 1; i < DIM; i+=gridDim.x) {
		for (int j = tid; j < DIM-1; j+=blockDim.x) {
			F[DIM * i + j] = F[DIM * (i-1) + (j+1)] + F[DIM * i + (j+1)];
		}
	}
}

int main(int argc, const char * argv[]) {
	srand((uint32_t)time(NULL));
	size_t memSize = DIM * DIM * sizeof(double);
	double* F =(double*) malloc(memSize);
	
	clock_t clock_start = clock();

	initMatrix(F);
	
#ifdef CUDA
	double* d_F;
	cudaMalloc((void**) &d_F, memSize);
	cudaMemcpy(d_F, F, memSize, cudaMemcpyHostToDevice);
	int nBlocks = 1;
	int tpb = 1024;
	dim3 dimGrid(nBlocks);
	dim3 dimBlock(tpb);
#endif
	
	clock_t init_duration = clock() - clock_start;
	
#ifdef CUDA
	for (int i = 0; i < REPS; i++) {
		cudaCompute<<<dimGrid, dimBlock>>>(d_F);
		cudaThreadSynchronize();
	}
#else
	computeStuff(F);
#endif
	
	clock_t total_duration = clock() - clock_start;
	
#ifdef CUDA
	cudaMemcpy(F, d_F, memSize, cudaMemcpyDeviceToHost);
#endif
	
	double time_in_seconds = (total_duration - init_duration) / 1000000.0;
	
	printf("Computation duration: %lfs; Performance: %lf GFlops\n",
		   time_in_seconds,
		   1E-9 * (((long)REPS * (long)DIM * (long)DIM) / time_in_seconds));
}

