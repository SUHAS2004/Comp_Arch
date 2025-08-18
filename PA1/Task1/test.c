#include <stdio.h>
#include <stdlib.h>         // for malloc, free, atoi
#include <time.h>           // for time()
#include <chrono>	
#include <xmmintrin.h> 		// for SSE
#include <immintrin.h>		// for AVX

#include "helper.h" 

void naive_mat_mul(double *A, double *B, double *C, int size) {
   for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j = j + 1) {
			__m256d sum = _mm256_setzero_pd();  // initialize the sum array to zero
			for (int k = 0; k < size; k = k + 4) { // the sum array will take 
				__m256d A1 = _mm256_loadu_pd(&A[i*size + k]); // Load four elements from i th row of A at once
				double B0[4] = {B[k*size + j],B[ (k+1)*size + j],B[ (k+2)*size + j],B[ (k+3)*size + j]}; // Load four elements from j th column of B at once
				__m256d B1 = _mm256_loadu_pd(B0);
				sum = _mm256_fmadd_pd(A1,B1,sum);
			}
			// Horizontal sum of the 4 products
			double result_array[4];
			_mm256_storeu_pd(result_array,sum);
			for (int k = 0; k <4; k = k + 1) {
				C[i * size + j] += result_array[k];  
			}
		}
                
    } 
}
int main(int argc, char **argv) {

	if ( argc <= 1 ) {
		printf("Usage: %s <matrix_dimension>\n", argv[0]);
		return 0;
	}

	else {
		int size = atoi(argv[1]);

		double *A = (double *)malloc(size * size * sizeof(double));
		double *B = (double *)malloc(size * size * sizeof(double));
		double *C = (double *)calloc(size * size, sizeof(double));

		// initialize random seed
		srand(time(NULL));

		// initialize matrices A and B with random values
		initialize_matrix(A, size, size);
		initialize_matrix(B, size, size);

		// perform normal matrix multiplication
		auto start = std::chrono::high_resolution_clock::now();
		naive_mat_mul(A, B, C, size);
		auto end = std::chrono::high_resolution_clock::now();
		auto time_naive_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Normal matrix multiplication took %ld ms to execute \n\n", time_naive_mat_mul);

		// free allocated memory
		free(A);
		free(B);
		free(C);

		return 0;
	}
}
