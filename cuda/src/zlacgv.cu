#include "cuSVD.cuh"

// -------------------- CUDA kernel --------------------
__global__ void zlacgv_kernel(cuDoubleComplex* x, int incx, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx * incx] = cuConj(x[idx * incx]);
    }
}

// -------------------- GPU interface --------------------
/**
 * zlacgv_gpu - in-place conjugate of a complex vector
 *
 * @param n      number of elements
 * @param d_x    device pointer to vector x
 * @param incx   stride between elements
 * @param handle cuBLAS handle to get associated stream
 */
void zlacgv_gpu(int n, cuDoubleComplex* d_x, int incx, cudaStream_t stream) {
    if (n <= 0) return;

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    zlacgv_kernel<<<blocks, threads, 0, stream>>>(d_x, incx, n);
}
