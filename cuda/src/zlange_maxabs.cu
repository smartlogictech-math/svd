#include "cuSVD.cuh"

void blasDamax(
    cudaStream_t stream,
    int n,
    const double *dx, int incx,
    double *dret);

void blasZamax(
    cudaStream_t stream,
    int n,
    const cuDoubleComplex *dx, int incx,
    double *dret);

/**
 * @brief Compute max absolute value of a complex matrix on GPU.
 *
 * The "max-abs" is defined column-wise as:
 *      max_{i,j}  max(|Re(A[i,j])|, |Im(A[i,j])|)
 *
 * Steps:
 *   1. For each column j, call blasZamax on A(:,j)
 *      results are stored in device array 'dwork[j]'.
 *   2. Call blasDamax on dwork[0..n-1] to obtain the global max.
 *   3. Copy scalar result to hMaxabs on host.
 *
 * @param m Number of rows of A.
 * @param n Number of columns of A.
 * @param dA Device pointer to matrix A.
 * @param lda Leading dimension of A.
 * @param hMaxabs Host pointer to output scalar.
 * @param dwork Device workspace, length >= n.
 * @param stream CUDA stream to use.
 */
void zlange_maxabs_gpu(
    int m, int n,
    const cuDoubleComplex* dA, int lda,
    double *hMaxabs,
    double *dwork,
    cudaStream_t stream)
{
    // Each column: Zamax
    for (int j = 0; j < n; j++) {
        const cuDoubleComplex* colPtr = dA + j * lda;

        // dwork[j] = max_i max(|Re|,|Im|)
        blasZamax(
            stream,
            m,
            colPtr,
            1,
            dwork + j);
    }

    // Global max among all columns: Damax
    double *d_out = dwork + n;  // reuse workspace or ensure pointer, but here we assume dwork[n] is free
    blasDamax(
        stream,
        n,
        dwork,
        1,
        d_out);

    // Copy scalar back to host
    cudaMemcpyAsync(
        hMaxabs,
        d_out,
        sizeof(double),
        cudaMemcpyDeviceToHost,
        stream);

    cudaStreamSynchronize(stream);
}