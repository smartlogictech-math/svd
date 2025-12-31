#include "cuSVD.cuh"
#include <cuComplex.h>

void zlarf_gpu(
    cublasHandle_t handle,
    HouseholderSide side,
    int m, int n,
    const cuDoubleComplex* d_v, int incv,
    cuDoubleComplex tau,
    cuDoubleComplex* d_C, int ldc,
    cuDoubleComplex* d_work
)
{
    if (cuCreal(tau) == 0.0 && cuCimag(tau) == 0.0)
        return;

    const cuDoubleComplex one  = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
    const cuDoubleComplex neg_tau = make_cuDoubleComplex(-tau.x, -tau.y);

    if (side == HOUSEHOLDER_LEFT) {
        // -------------------------------------------------
        // w = v^H C   (size n)
        // -------------------------------------------------
        // gemv: w = conj(v)^T * C
        cublasZgemv(
            handle,
            CUBLAS_OP_C,   // v^H
            m, n,
            &one,
            d_C, ldc,
            d_v, incv,
            &zero,
            d_work, 1
        );

        // -------------------------------------------------
        // C = C - tau * v * w^T
        // -------------------------------------------------
        // gerc: C += alpha * v * w^H
        cublasZgerc(
            handle,
            m, n,
            &neg_tau,
            d_v, incv,
            d_work, 1,
            d_C, ldc
        );
    }
    else {
        // -------------------------------------------------
        // w = C v   (size m)
        // -------------------------------------------------
        cublasZgemv(
            handle,
            CUBLAS_OP_N,
            m, n,
            &one,
            d_C, ldc,
            d_v, incv,
            &zero,
            d_work, 1
        );

        // -------------------------------------------------
        // C = C - tau * w * v^H
        // -------------------------------------------------
        cublasZgerc(
            handle,
            m, n,
            &neg_tau,
            d_work, 1,
            d_v, incv,
            d_C, ldc
        );
    }
}