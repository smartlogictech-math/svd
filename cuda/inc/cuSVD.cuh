#pragma once 
#include "cublas_v2.h"

#if defined(__cplusplus)
extern "C" {
#endif

/******************************************************************************
 * 1. Scaling
 *    修改矩阵 A，使其范数在合理范围内： A = A / scale
 ******************************************************************************/
/**
 * @brief Compute the maximum absolute value (entry-wise max norm)
 *        of a complex matrix A stored on the GPU.
 *
 * This routine finds the maximum value of |A[i,j]| over all matrix entries.
 * For a complex number z = x + i*y, the absolute value is sqrt(x*x + y*y).
 *
 * The matrix A is on the GPU. The kernel scans all elements, performs
 * parallel reduction, and stores the resulting maximum absolute value
 * into the host pointer hMaxabs. A device workspace buffer "work" is used
 * for partial reductions.
 *
 * @param[in] m
 *      Number of rows of matrix A. Must be m >= 0.
 *
 * @param[in] n
 *      Number of columns of matrix A. Must be n >= 0.
 *
 * @param[in] dA
 *      Pointer to device memory containing matrix A (column-major).
 *      Must store at least lda * n elements of type cuDoubleComplex.
 *
 * @param[in] lda
 *      Leading dimension of A. Must satisfy lda >= max(1, m).
 *
 * @param[out] hMaxabs
 *      Pointer to host memory where the resulting maximum absolute value
 *      will be written. The function writes one double value.
 *
 * @param[in,out] dwork
 *      Device workspace used for partial reductions.
 *      Size requirements depend on implementation (typically several
 *      blocks of temporary maxima).
 *
 * @param[in] stream
 *      CUDA stream where the kernels will be launched.
 *
 * @return
 *      - CUDA_SUCCESS on success
 *      - CUDA error code if any GPU operation fails
 *
 * @note
 * - This function is commonly used in SVD/QR algorithms to determine
 *   a scaling factor for numerical stability.
 * - The final result is copied asynchronously to hMaxabs using the given stream.
 * - work is used only on the device; hMaxabs must be host-accessible.
 */
void zlange_maxabs_gpu(
    int m, int n,
    const cuDoubleComplex* dA, int lda,
    double *hMaxabs,
    double *dwork,
    cudaStream_t stream);
/**
 * @brief Scale a dense complex matrix A on GPU:  A = (*dScale) * A.
 *
 * This routine multiplies every entry of an m-by-n complex matrix A by the
 * scalar value stored in device memory at `dScale`.  The matrix A must be
 * stored in column-major layout with leading dimension `lda`.
 *
 * This is a simplified GPU version of LAPACK's ZLASCL.  Only general dense
 * matrices are supported and the routine performs a single scaling pass
 * without overflow/underflow protection.
 *
 * Differences from LAPACK ZLASCL:
 * - Only general full matrices are supported (TYPE = 'G').
 * - No CFROM/CTO multi-step scaling; uses only a single scalar *dScale.
 * - No band, triangular, Hessenberg, or special structured matrix support.
 * - Does not handle safe scaling to avoid overflow or underflow.
 *
 * Typical usage:
 * 1. Compute max absolute value of A.
 * 2. Write (1.0 / maxabs) to a device pointer `dScale`.
 * 3. Call this routine to normalize A on the GPU.
 *
 * @param[in] m
 *     Number of rows of A.
 *
 * @param[in] n
 *     Number of columns of A.
 *
 * @param[in] dScale
 *     Device pointer to a single double value holding the scale factor.
 *     Each A(i,j) becomes (*dScale) * A(i,j).
 *
 * @param[in,out] dA
 *     Device pointer to an m-by-n complex matrix stored in column-major
 *     format with leading dimension `lda`.
 *
 * @param[in] lda
 *     Leading dimension of A. Must satisfy lda >= max(1, m).
 *
 * @param[in] stream
 *     CUDA stream in which the scaling kernel will be launched.
 *
 * @return void
 */
void zlascl_gpu(
    int m, int n,
    const double* dScale,
    cuDoubleComplex* dA, int lda,
    cudaStream_t stream);

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