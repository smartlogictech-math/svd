#pragma once 
#include "cublas_v2.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

void zlacgv_gpu(int n, cuDoubleComplex* d_x, int incx, cudaStream_t stream);

void zlarfg_gpu(
    cublasHandle_t handle,
    int n,
    cuDoubleComplex *d_alpha,
    cuDoubleComplex *d_x,
    int incx,
    cuDoubleComplex *d_tau);

enum HouseholderSide {
    HOUSEHOLDER_LEFT,
    HOUSEHOLDER_RIGHT
};

void zlarf_gpu(
    cublasHandle_t handle,
    HouseholderSide side,
    int m, int n,
    const cuDoubleComplex* d_v, int incv,
    cuDoubleComplex tau,
    cuDoubleComplex* d_C, int ldc,
    cuDoubleComplex* d_work
);

/**
 * @brief Apply Householder reflector H = I - tau * v v^H to matrix C
 *        v[0] is implicit 1, v[1..] are stored in d_v[1..]
 *
 * @param handle   cuBLAS handle
 * @param side     'L' = left multiply H*C, 'R' = right multiply C*H
 * @param m        number of rows of C
 * @param n        number of columns of C
 * @param d_v      device pointer to full v, length >= 1 + (max(m,n)-1)*abs(incv)
 * @param incv     stride of v
 * @param tau      scalar tau
 * @param d_C      device pointer to matrix C (column-major)
 * @param ldc      leading dimension of C
 * @param d_work   device workspace, length >= max(m,n)
 */
void zlarf1f_gpu(
    cublasHandle_t handle,
    char side,
    int m, int n,
    const cuDoubleComplex* d_v, int incv,
    cuDoubleComplex tau,
    cuDoubleComplex* d_C, int ldc,
    cuDoubleComplex* d_work
);

#if defined(__cplusplus)
}
#endif /* __cplusplus */