#include <cuda_runtime.h>
#include <cublas_v2.h>

void zlarfg_gpu(
    cublasHandle_t handle,
    int n,
    cuDoubleComplex* d_alpha,
    cuDoubleComplex* d_x,
    int incx,
    cuDoubleComplex* d_tau
)
{
    if (n <= 1) {
        cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
        cudaMemcpy(d_tau, &zero, sizeof(zero), cudaMemcpyHostToDevice);
        return;
    }

    double xnorm;
    cublasDznrm2(handle, n-1, d_x, incx, &xnorm);

    cuDoubleComplex alpha;
    cudaMemcpy(&alpha, d_alpha, sizeof(alpha), cudaMemcpyDeviceToHost);

    cuDoubleComplex beta, tau;

    if (xnorm == 0.0 && cuCimag(alpha) == 0.0) {
        tau  = make_cuDoubleComplex(0.0, 0.0);
        beta = alpha;
    } else {
        double alpha_abs = cuCabs(alpha);
        double beta_real =
            -copysign(hypot(alpha_abs, xnorm), cuCreal(alpha));

        beta = make_cuDoubleComplex(beta_real, 0.0);

        tau = cuCdiv(
            cuCsub(beta, alpha),
            beta
        );

        cuDoubleComplex scale = cuCdiv(
            make_cuDoubleComplex(1.0, 0.0),
            cuCsub(alpha, beta)
        );

        cublasZscal(handle, n-1, &scale, d_x, incx);
    }

    cudaMemcpy(d_alpha, &beta, sizeof(beta), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tau,   &tau,  sizeof(tau),  cudaMemcpyHostToDevice);
}
