#include "utils.cuh"

#include <cuComplex.h>
#include <stdio.h>

double complex_abs(cuDoubleComplex x) {
    return hypot(x.x, x.y);
}

double matrix_diff_norm(
    int m, int n,
    const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* B, int ldb
) {
    double max_err = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            cuDoubleComplex d = cuCsub(A[j*lda + i], B[j*ldb + i]);
            double e = complex_abs(d);
            if (e > max_err) max_err = e;
        }
    }
    return max_err;
}

void print_matrix(const char* name, int m, int n,
                  const cuDoubleComplex* A, int lda) {
    printf("%s =\n", name);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf(" (%+.3e,%+.3e)", A[j*lda + i].x, A[j*lda + i].y);
        }
        printf("\n");
    }
}
