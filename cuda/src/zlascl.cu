#include "cuSVD.cuh"

#include <cublas_v2.h>

/**
 * @brief Scale a matrix A by a device pointer scalar dScale: A = dScale * A
 *
 * This uses cuBLAS zdscal for each column of A.
 *
 * @param m       Number of rows of the matrix A
 * @param n       Number of columns of the matrix A
 * @param dScale  Device pointer to scalar value
 * @param dA      Device pointer to matrix A (column-major)
 * @param lda     Leading dimension of A
 * @param stream  CUDA stream to use
 */
void zlascl_gpu(
    int m, int n,
    const double* dScale,
    cuDoubleComplex* dA, int lda,
    cudaStream_t stream)
{
    if (m <= 0 || n <= 0) return;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    for (int col = 0; col < n; ++col) {
        cuDoubleComplex* col_ptr = dA + col * lda;
        // cuBLAS zdscal: scale a vector of m elements by real scalar dScale
        cublasZdscal(handle, m, dScale, col_ptr, 1);
    }

    cublasDestroy(handle);
}