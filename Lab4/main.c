//
//  main.c
//  Lab4
//
//  Created by Alan Weide on 4/9/16.
//  Copyright © 2016 weidea. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#include <cuda_builtin_vars.h>

void initMatrix(double** F, int dim) {
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			F[i][j] = 1.0 + ((double)rand() / RAND_MAX);
		}
	}
}

void printMatrix (double** F, int dim) {
	for (int i = 0; i < dim; i++) {
		printf("[ ");
		for (int j = 0; j < dim; j++) {
			printf(" %lf", F[i][j]);
		}
		printf(" ]\n");
	}
	
}

void computeStuff(double** F, int dim) {
	for (int k = 0; k < 100; k++)  {
		for (int i = 1; i < dim; i++) {
			for (int j = 0; j < dim - 1; j++) {
				F[i][j] = F[i-1][j+1] + F[i][j+1];
			}
		}
	}
}

int main(int argc, const char * argv[]) {
	int dim = 4097;
	double** F = malloc(dim * dim * sizeof(double));
	srand(time(NULL));
	
//	initMatrix(F, dim);
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			F[i][j] = 1.0 + ((double)rand() / RAND_MAX);
		}
	}
	
//	printMatrix(F, dim);
	
	computeStuff(F, dim);
}

