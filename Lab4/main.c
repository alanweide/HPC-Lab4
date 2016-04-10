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

#define DIM 4097
#define REPS 100
double F[DIM][DIM];

void initMatrix() {
//	F = malloc(dim * dim * sizeof(double));
	
	for (int i = 0; i < DIM; i++) {
		for (int j = 0; j < DIM; j++) {
			F[i][j] = 1.0 + ((double)rand() / RAND_MAX);
		}
	}
}

void cudaInit() {
	size_t memSize = DIM * DIM;
	cudaMalloc((void**) &F, memSize);
}

void printMatrix () {
	for (int i = 0; i < DIM; i++) {
		printf("[ ");
		for (int j = 0; j < DIM; j++) {
			printf(" %lf", F[i][j]);
		}
		printf(" ]\n");
	}
	
}

void computeStuff() {
	for (int k = 0; k < REPS; k++)  {
		for (int i = 1; i < DIM; i++) {
			for (int j = 0; j < DIM - 1; j++) {
				F[i][j] = F[i-1][j+1] + F[i][j+1];
			}
		}
	}
}

__global__ void cudaCompute(double** F) {
	
}

int main(int argc, const char * argv[]) {
	srand((uint32_t)time(NULL));
	initMatrix();
	cudaInit();
//	printMatrix();
	clock_t clock_start = clock();
	computeStuff();
	clock_t clock_duration = clock() - clock_start;
	printf("Computation duration: %.3lfs; Performance: %.3lf GFlops\n",
		   clock_duration / 1000000.0,
		   0.001 * REPS * (DIM * DIM) / clock_duration);
}

