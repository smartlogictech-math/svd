#include "cuBlasEx.cuh"

__global__ void zaxpyc_kernel(
    int n,
    cuDoubleComplex alpha,
    const cuDoubleComplex* x, int incx,
    cuDoubleComplex* y, int incy
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        cuDoubleComplex x_conj = cuConj(x[idx * incx]);
        y[idx * incy] = cuCadd(y[idx * incy], cuCmul(alpha, x_conj));
    }
}

void zaxpyc(
    int n,
    cuDoubleComplex alpha,
    const cuDoubleComplex* x, int incx,
    cuDoubleComplex* y, int incy,
    cudaStream_t stream
) {
    if (n <= 0) return;
    if (cuCreal(alpha) == 0.0 && cuCimag(alpha) == 0.0) return;

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    zaxpyc_kernel<<<blocks, threads, 0, stream>>>(
        n, alpha, x, incx, y, incy
    );
}