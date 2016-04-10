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

const int dim = 4097;
const int reps = 100;
double F[dim][dim];

void initMatrix() {
//	F = malloc(dim * dim * sizeof(double));
	
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			F[i][j] = 1.0 + ((double)rand() / RAND_MAX);
		}
	}
}

void printMatrix () {
	for (int i = 0; i < dim; i++) {
		printf("[ ");
		for (int j = 0; j < dim; j++) {
			printf(" %lf", F[i][j]);
		}
		printf(" ]\n");
	}
	
}

void computeStuff() {
	for (int k = 0; k < reps; k++)  {
		for (int i = 1; i < dim; i++) {
			for (int j = 0; j < dim - 1; j++) {
				F[i][j] = F[i-1][j+1] + F[i][j+1];
			}
		}
	}
}

int main(int argc, const char * argv[]) {
	srand((uint32_t)time(NULL));
	initMatrix();
//	printMatrix();
	clock_t clock_start = clock();
	computeStuff();
	clock_t clock_duration = clock() - clock_start;
	printf("Computation duration: %.3lfs; Performance: %.3lf GFlops\n", clock_duration / 1000000.0, 0.001 * reps * (dim * dim) / clock_duration);
}

