#pragma once 

#include <cuComplex.h>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

double complex_abs(cuDoubleComplex x);

double matrix_diff_norm(
    int m, int n,
    const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* B, int ldb
);

void print_matrix(const char* name, int m, int n,
                  const cuDoubleComplex* A, int lda);


#if defined(__cplusplus)
}
#endif /* __cplusplus */