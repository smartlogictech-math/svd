#include "utils.cuh"

#include <cuComplex.h>
#include <stdio.h>
#include <cmath>
#include <algorithm>

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

// 计算复数的绝对值 |z|
static inline double cabs_complex(const cuDoubleComplex& z) {
    return hypot(cuCreal(z), cuCimag(z));
}

// 判断两个复数是否接近
static inline bool complexAllClose(
    const cuDoubleComplex& a,
    const cuDoubleComplex& b,
    double atol,
    double rtol
) {
    double diff = cabs_complex( cuCsub(a, b) );
    double an   = cabs_complex(a);
    double bn   = cabs_complex(b);
    return diff <= (atol + rtol * std::max(an, bn));
}

// A(m×n) 与 B(m×n) 判断是否接近
bool matrix_allclose(
    const cuDoubleComplex* A,
    const cuDoubleComplex* B,
    int m, int n,
    double atol,
    double rtol
) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;
            if (!complexAllClose(A[idx], B[idx], atol, rtol)) {
                printf("[matrixAllClose] mismatch at (%d,%d): "
                       "A=(%.16f, %.16f), B=(%.16f, %.16f)\n",
                       i, j,
                       cuCreal(A[idx]), cuCimag(A[idx]),
                       cuCreal(B[idx]), cuCimag(B[idx]));
                return false;
            }
        }
    }
    return true;
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
