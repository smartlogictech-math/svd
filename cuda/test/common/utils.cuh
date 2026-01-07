#pragma once 

#include <cuComplex.h>
#include <stdbool.h>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

double complex_abs(cuDoubleComplex x);

double matrix_diff_norm(
    int m, int n,
    const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* B, int ldb
);

bool matrix_allclose(
    const cuDoubleComplex* A,
    const cuDoubleComplex* B,
    int m, int n,
    double atol = 1e-12,
    double rtol = 1e-12
);

void print_matrix(const char* name, int m, int n,
                  const cuDoubleComplex* A, int lda);


#if defined(__cplusplus)
}
#endif /* __cplusplus */