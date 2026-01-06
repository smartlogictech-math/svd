#include "cuBlasEx.cuh"
#include "cuSVD.cuh"
#include <stdio.h>

// -------------------- Implementation --------------------
void zlarf1f_gpu(
    cublasHandle_t handle,
    char side,
    int m, int n,
    const cuDoubleComplex* d_v, int incv,
    cuDoubleComplex tau,
    cuDoubleComplex* d_C, int ldc,
    cuDoubleComplex* d_work
) {
    if (cuCreal(tau) == 0.0 && cuCimag(tau) == 0.0) return;
    if (m <= 0 || n <= 0) return;

    const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex minus_tau = make_cuDoubleComplex(-tau.x, -tau.y);

    // -------------------- 获取 handle 当前流 --------------------
    cudaStream_t stream;
    cublasGetStream(handle, &stream);

    if (side == 'L' || side == 'l') {
        // -------------------- Left multiply H * C --------------------
        if (m == 1) {
            cuDoubleComplex scale = cuCsub(one, tau);
            cublasZscal(handle, n, &scale, d_C, ldc); // scale first row
            return;
        }

        // 1️⃣ work = C(1,:)  (copy first row)
        cublasZcopy(handle, n, d_C, ldc, d_work, 1);

        // 2️⃣ work = conj(work)  (in-place)
        zlacgv_gpu(n, d_work, 1, stream);

        // 2️⃣ work += C[1:m,:]^H * v_2
        cublasZgemv(handle, CUBLAS_OP_C, m - 1, n,
                    &one, d_C + 1, ldc, d_v + 1, incv, &one, d_work, 1);

        // 3️⃣ C(1,:) -= tau * conj(work)
        zaxpyc(n, minus_tau, d_work, 1, d_C, ldc, stream);

        // 4️⃣ C(2:m,:) -= tau * v_2 * work^H
        cublasZgerc(handle, m - 1, n, &minus_tau, d_v + 1, incv, d_work, 1, d_C + 1, ldc);

    } else if (side == 'R' || side == 'r') {
        // -------------------- Right multiply C * H --------------------
        if (n == 1) {
            cuDoubleComplex scale = cuCsub(one, tau);
            cublasZscal(handle, m, &scale, d_C, 1);  // scale first column
            return;
        }

        // 1️⃣ work = C[:,2:n] * v_2
        cublasZgemv(handle, CUBLAS_OP_N, m, n-1,
                    &one, d_C + ldc, ldc, d_v + 1, incv, &zero, d_work, 1);

        // 2️⃣ work += C[:,1]  (v[0] contribution)
        cublasZaxpy(handle, m, &one, d_C, 1, d_work, 1);

        // 3️⃣ C[:,1] -= tau * work
        cublasZaxpy(handle, m, &minus_tau, d_work, 1, d_C, 1);

        // 4️⃣ C[:,2:n] -= tau * work * v_2^H
        cublasZgerc(handle, m, n-1, &minus_tau, d_work, 1, d_v + 1, incv, d_C + ldc, ldc);

    } else {
        fprintf(stderr, "[zlarf1f_gpu] Error: side must be 'L' or 'R'\n");
    }
}