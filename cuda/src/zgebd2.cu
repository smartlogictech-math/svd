#include "cuSVD.cuh"

void zgebd2_gpu(
    cublasHandle_t      handle,
    int                 m,
    int                 n,
    cuDoubleComplex*    d_A,        // device matrix A
    int                 lda,
    double*             h_D,        // host diag
    double*             h_E,        // host super/subdiag
    cuDoubleComplex*    h_tauQ,     // host tauQ
    cuDoubleComplex*    h_tauP,     // host tauP
    cuDoubleComplex*    d_work      // device workspace
)
{
    cudaStream_t stream;
    cublasGetStream(handle, &stream);

    // ============================================================
    //  Case 1: m >= n  —— Reduce to upper bidiagonal
    // ============================================================
    if (m >= n)
    {
        for (int ii = 0; ii < n; ii++)
        {
            int rows = m - ii;      // remaining rows
            int cols = n - ii - 1;  // columns right of ii

            // ----------------------------------------------------
            // 1. H(ii): Generate Householder reflector for column
            //    H(ii) annihilates A(ii+1:m, ii)
            // ----------------------------------------------------
            cuDoubleComplex* d_Aii   = d_A + ii + ii * lda;       // A(ii,ii)
            cuDoubleComplex* d_Acol  = d_A + (ii+1) + ii * lda;   // A(ii+1:m,ii)
            zlarfg_gpu(
                handle,
                rows,
                d_Aii,
                d_Acol,
                1,
                &h_tauQ[ii]
            );
            // save diagonal element
            cuDoubleComplex h_alpha;
            cudaMemcpy(&h_alpha, d_Aii, sizeof(h_alpha), cudaMemcpyDeviceToHost);
            h_D[ii] = cuCreal(h_alpha);

            // ----------------------------------------------------
            // Apply H(ii)^H to A(ii:m, ii+1:n) from the LEFT
            // ----------------------------------------------------
            if (cols > 0)
            {
                cuDoubleComplex* d_Aright = d_A + ii + (ii+1) * lda;
                zlarf1f_gpu(
                    handle,
                    'L',
                    rows,
                    cols,
                    d_Aii,            // v(0) = A(ii,ii)
                    1,
                    cuConj(h_tauQ[ii]),
                    d_Aright,
                    lda,
                    d_work
                );
                cudaStreamSynchronize(stream);
            }

            // ----------------------------------------------------
            // 2. G(ii): Reflector for row ii
            //    annihilates A(ii, ii+2:n)
            // ----------------------------------------------------
            if (cols > 0)
            {
                cuDoubleComplex* d_Ai_ip1 = d_A + ii + (ii+1) * lda;      // A(ii,ii+1)
                cuDoubleComplex* d_Ai_ip2 = d_A + ii + (ii+2) * lda;      // A(ii,ii+2:n)

                // conjugate row segment
                zlacgv_gpu(cols, d_Ai_ip1, lda, stream);
                cudaStreamSynchronize(stream);

                zlarfg_gpu(
                    handle,
                    cols,
                    d_Ai_ip1,
                    d_Ai_ip2,
                    lda,
                    &h_tauP[ii]
                );
                cudaStreamSynchronize(stream);

                // save superdiagonal
                cuDoubleComplex h_alpha2;
                cudaMemcpy(&h_alpha2, d_Ai_ip1, sizeof(h_alpha2), cudaMemcpyDeviceToHost);
                h_E[ii] = cuCreal(h_alpha2);

                // apply G(ii) from RIGHT to A(ii+1:m, ii+1:n)
                if (ii < m - 1)
                {
                    int rows2 = m - (ii + 1);
                    cuDoubleComplex* d_Asub = d_A + (ii+1) + (ii+1) * lda;

                    zlarf1f_gpu(
                        handle,
                        'R',
                        rows2,
                        cols,
                        d_Ai_ip1,
                        lda,
                        h_tauP[ii],
                        d_Asub,
                        lda,
                        d_work
                    );
                    cudaStreamSynchronize(stream);
                }
                // restore conjugation
                zlacgv_gpu(cols, d_Ai_ip1, lda, stream);
                cudaStreamSynchronize(stream);
                cudaMemcpy(d_Ai_ip1, &h_alpha2, sizeof(h_alpha2), cudaMemcpyHostToDevice);
            }
            else
            {
                h_tauP[ii] = make_cuDoubleComplex(0.0, 0.0);
            }
        }
    }

    // ============================================================
    //  Case 2: m < n  —— Reduce to lower bidiagonal
    // ============================================================
    else {
        /// 暂时对标cuSolver,不支持m<n
        return;
    }
}
