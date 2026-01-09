#include "cuSVD.cuh"

void zgebd2_gpu(
    cublasHandle_t handle,
    int m, int n,
    cuDoubleComplex* dA, int lda,
    double* hD,
    double* hE,
    cuDoubleComplex* hTauQ,
    cuDoubleComplex* hTauP,
    cuDoubleComplex* dwork
)
{
    int k = min(m, n);

    for (int i = 0; i < k; ++i) {

        // ===== Left reflector =====
        zlarfg_gpu(
            handle,
            m - i,
            &dA[i + i*lda],
            &dA[(i+1) + i*lda],
            1,
            &hTauQ[i]
        );

        // save hD
        cudaMemcpy(&hD[i], &dA[i + i*lda],
                   sizeof(double), cudaMemcpyDeviceToHost);

        if (i < n-1) {
            // apply from left
            zlarf_gpu(
                handle,
                HOUSEHOLDER_LEFT,
                m - i,
                n - i - 1,
                &dA[i + i*lda],
                1,
                cuConj(hTauQ[i]),
                &dA[i + (i+1)*lda],
                lda,
                dwork
            );
        }

        // ===== Right reflector =====
        if (i < n-1) {
            zlarfg_gpu(
                handle,
                n - i - 1,
                &dA[i + (i+1)*lda],
                &dA[i + (i+2)*lda],
                lda,
                &hTauP[i]
            );

            cudaMemcpy(&hE[i], &dA[i + (i+1)*lda],
                       sizeof(double), cudaMemcpyDeviceToHost);

            if (i < m-1) {
                zlarf_gpu(
                    handle,
                    HOUSEHOLDER_RIGHT,
                    m - i - 1,
                    n - i - 1,
                    &dA[i + (i+1)*lda],
                    lda,
                    hTauP[i],
                    &dA[(i+1) + (i+1)*lda],
                    lda,
                    dwork
                );
            }
        }
    }
}
