//
//  main3.c
//  Lab4
//
//  Created by Alan Weide on 4/11/16.
//  Copyright © 2016 weidea. All rights reserved.
//

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define DIM 1024
#define NBLK 1
#define TPB 32

void initMatrix(int *F) {
	for (int i = 0; i < DIM; i++) {
		for (int j = 0; j < DIM; j++) {
			F[DIM * i + j] = DIM*i + j;//(int)(1 + ((1000L * rand()) / RAND_MAX));
		}
	}
}

void printMatrix (int *F) {
	for (int i = 0; i < DIM; i++) {
		printf("[");
		for (int j = 0; j < DIM; j++) {
			printf(" %3d", F[DIM * i + j]);
		}
		printf(" ]\n");
	}
	
}

void computeStuff(int *F) {
	for (int i = 0; i < DIM; i++) {
		for (int j = 0; j < i; j++) {
			int tmp = F[DIM*j + i];
			F[DIM*j + i] = F[DIM*i + j];
			F[DIM*i + j] = tmp;
		}
	}
}

void verifyTranspose(int *expected, int *actual) {
	for (int i = 0; i < DIM; i++) {
		for (int j = 0; j <= i; j++) {
			if (expected[DIM*j + i] != actual[DIM*i + j]) {
				printf("Failure: actual[%d][%d] expected %d, but found %d\n", i, j, expected[DIM*j + i], actual[DIM*i + j]);
				return;
			}
		}
	}
	printf("Verified: Matrices are transposes of each other.\n");
}

#ifdef CUDA
__global__ void cuda_compute(int *F, int *startI, int *startJ, int spt) {
	int thrDone = 0;
	int count = 0;
	int tid = gridDim.x * blockIdx.x + threadIdx.x;
	
	for (int i = startI[tid]; i < DIM && !thrDone; i++) {
		for (int j = (i == startI[tid] ? startJ[tid] : 0); j < i && !thrDone; j++) {
			int tmp = F[DIM*j + i];
			F[DIM*j + i] = F[DIM*i + j];
			F[DIM*i + j] = tmp;
			count++;
			thrDone = (count >= spt);
		}
	}
}
#endif

int main(int argc, char* argv[]) {
	
	int nblk, tpb;
	if (argc == 3) {
		nblk = atoi(argv[1]);
		tpb = atoi(argv[2]);
	} else {
		nblk = NBLK;
		tpb = TPB;
	}
	
	srand((uint32_t)time(NULL));
	size_t memSize = DIM * DIM * sizeof(int);
	size_t thrArrSize = nblk * tpb * sizeof(int);
	int *F = (int*) malloc(memSize);
	int *computedF = (int*) malloc(memSize);
	
	clock_t clock_start = clock();
	
	initMatrix(F);

	int spt = 1 + (DIM * DIM) / 2 / (nblk * tpb);
	int *startI = (int*)malloc(thrArrSize);
	int *startJ = (int*)malloc(thrArrSize);
	
	// Computes the start i and j coordinates for each thread.
	
	/* 
	 * Thread division is based on the following indexing of the matrix:
	 * [  -  -  -  -  -  ... ]
	 * [  0  -  -  -  -  ... ]
	 * [  1  2  -  -  -  ... ]
	 * [  3  4  5  -  -  ... ]
	 * [  6  7  8  9  -  ... ]
	 * [ 10 11 12 13 14  ... ]
	 * [ 15 16 17 18 19  ... ]
	 * ...
	 * Thread 0 gets cells [0, spt), thread 1 [spt, 2*spt), and so on
	 */
	for (int i = 0; i < nblk * tpb; i++) {
		startI[i] = 1 + (((int)(sqrt(1 + 8 * i * spt)) - 1) / 2);
		startJ[i] = (i * spt) - ((startI[i] + 0) * (startI[i] - 1) / 2);
	}
	
#ifdef CUDA
	int *d_F, *d_i, *d_j;
	cudaMalloc((void**) &d_F, memSize);
	cudaMemcpy(d_F, F, memSize, cudaMemcpyHostToDevice);
	
	dim3 dimGrid(nblk);
	dim3 dimBlock(tpb);

	cudaMalloc((void**) &d_i, thrArrSize);
	cudaMalloc((void**) &d_j, thrArrSize);
	cudaMemcpy(d_i, startI, thrArrSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_j, startJ, thrArrSize, cudaMemcpyHostToDevice);
#endif
	
	clock_t init_duration = clock() - clock_start;
	
#ifdef CUDA
	cuda_compute<<<dimGrid, dimBlock>>>(d_F, d_i, d_j, spt);
	cudaThreadSynchronize();
#else
	// Serial
	computeStuff(F);
#endif
	
	clock_t total_duration = clock() - clock_start;
	
#ifdef CUDA
	cudaMemcpy(computedF, d_F, memSize, cudaMemcpyDeviceToHost);
#endif
	
	verifyTranspose(F, computedF);
	
	double time_in_seconds = (total_duration - init_duration) / 1000000.0;
	
	printf("\nComputation duration: %lfs; Performance: %lf GIops\n",
		   time_in_seconds,
		   1E-9 * (10 * (long)DIM * (long)DIM) / time_in_seconds);
}
