#include <cuda_runtime.h>
#include <cublas_v2.h>

/**
*  ZLARFG generates a complex elementary reflector H of order n, such
*  that
* 
*        H**H * ( alpha ) = ( beta ),   H**H * H = I.
*               (   x   )   (   0  )
* 
*  where alpha and beta are scalars, with beta real, and x is an
*  (n-1)-element complex vector. H is represented in the form
* 
*        H = I - tau * ( 1 ) * ( 1 v**H ) ,
*                      ( v )
* 
*  where tau is a complex scalar and v is a complex (n-1)-element
*  vector. Note that H is not hermitian.
* 
*  If the elements of x are all zero and alpha is real, then tau = 0
*  and H is taken to be the unit matrix.
* 
*  Otherwise  1 <= real(tau) <= 2  and  abs(tau-1) <= 1 .
 */

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
