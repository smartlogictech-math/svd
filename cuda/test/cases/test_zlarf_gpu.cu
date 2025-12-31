#include "cuSVD.cuh"
#include "../common/common.cuh"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void test_zlarf_left(int n)
{
    assert(n >= 2);

    // =========================
    // Host data
    // =========================
    cuDoubleComplex* h_y   = (cuDoubleComplex*)malloc(n * sizeof(*h_y));
    cuDoubleComplex* h_y0  = (cuDoubleComplex*)malloc(n * sizeof(*h_y0));
    cuDoubleComplex* h_v   = (cuDoubleComplex*)malloc(n * sizeof(*h_v));
    cuDoubleComplex* h_Hy  = (cuDoubleComplex*)malloc(n * sizeof(*h_Hy));
    cuDoubleComplex* h_res = (cuDoubleComplex*)malloc(n * sizeof(*h_res));

    for (int i = 0; i < n; ++i) {
        h_y[i] = make_cuDoubleComplex(
            drand48() - 0.5,
            drand48() - 0.5
        );
        h_y0[i] = h_y[i];
    }

    // =========================
    // Device data
    // =========================
    cuDoubleComplex *d_alpha, *d_x, *d_tau;
    cuDoubleComplex *d_y, *d_v, *d_work;

    cudaMalloc(&d_alpha, sizeof(*d_alpha));
    cudaMalloc(&d_x, (n - 1) * sizeof(*d_x));
    cudaMalloc(&d_tau, sizeof(*d_tau));
    cudaMalloc(&d_y, n * sizeof(*d_y));
    cudaMalloc(&d_v, n * sizeof(*d_v));
    cudaMalloc(&d_work, n * sizeof(*d_work));

    cudaMemcpy(d_alpha, &h_y[0], sizeof(*d_alpha), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, &h_y[1], (n - 1) * sizeof(*d_x), cudaMemcpyHostToDevice);

    // =========================
    // cuBLAS handle
    // =========================
    cublasHandle_t handle;
    cublasCreate(&handle);

    // =========================
    // Generate Householder reflector
    // =========================
    zlarfg_gpu(handle, n, d_alpha, d_x, 1, d_tau);

    cuDoubleComplex beta, tau;
    cudaMemcpy(&beta, d_alpha, sizeof(beta), cudaMemcpyDeviceToHost);
    cudaMemcpy(&tau,  d_tau,   sizeof(tau),  cudaMemcpyDeviceToHost);

    // build v on host, then copy to device
    h_v[0] = make_cuDoubleComplex(1.0, 0.0);
    cudaMemcpy(&h_v[1], d_x, (n - 1) * sizeof(*h_v), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_v, h_v, n * sizeof(*d_v), cudaMemcpyHostToDevice);

    // ============================================================
    // Check 1: H^H y = [beta, 0, ..., 0]
    // ============================================================
    cudaMemcpy(d_y, h_y0, n * sizeof(*d_y), cudaMemcpyHostToDevice);

    zlarf_gpu(
        handle,
        HOUSEHOLDER_LEFT,
        n, 1,
        d_v, 1,
        cuConj(tau),      // H^H
        d_y, n,
        d_work
    );

    cudaMemcpy(h_Hy, d_y, n * sizeof(*h_Hy), cudaMemcpyDeviceToHost);

    // tail norm
    double tail_norm = 0.0;
    for (int i = 1; i < n; ++i) {
        tail_norm += cuCreal(cuCmul(cuConj(h_Hy[i]), h_Hy[i]));
    }
    tail_norm = sqrt(tail_norm);

    double y_norm = znrm2_host(n, h_y0);

    // head error
    double head_err =
        cuCabs(cuCsub(h_Hy[0], beta)) / cuCabs(beta);

    printf("[Check1] tail/y = %.3e, head_err = %.3e\n",
           tail_norm / y_norm, head_err);

    assert(tail_norm / y_norm < 1e-12);
    assert(head_err < 1e-12);

    // ============================================================
    // Check 2: H^H H y = y   (unitarity)
    // ============================================================
    cudaMemcpy(d_y, h_y0, n * sizeof(*d_y), cudaMemcpyHostToDevice);

    // y1 = H y
    zlarf_gpu(
        handle,
        HOUSEHOLDER_LEFT,
        n, 1,
        d_v, 1,
        tau,
        d_y, n,
        d_work
    );

    // y2 = H^H y1
    zlarf_gpu(
        handle,
        HOUSEHOLDER_LEFT,
        n, 1,
        d_v, 1,
        cuConj(tau),
        d_y, n,
        d_work
    );

    cudaMemcpy(h_Hy, d_y, n * sizeof(*h_Hy), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        h_res[i] = cuCsub(h_Hy[i], h_y0[i]);
    }

    double rel_err =
        znrm2_host(n, h_res) / znrm2_host(n, h_y0);

    printf("[Check2] H^H H y relative error = %.3e\n", rel_err);
    assert(rel_err < 1e-12);

    // =========================
    // Cleanup
    // =========================
    free(h_y);
    free(h_y0);
    free(h_v);
    free(h_Hy);
    free(h_res);

    cudaFree(d_alpha);
    cudaFree(d_x);
    cudaFree(d_tau);
    cudaFree(d_y);
    cudaFree(d_v);
    cudaFree(d_work);

    cublasDestroy(handle);
}
int main()
{
    test_zlarf_left(65535);
    printf("zlarf_gpu test passed.\n");
    return 0;
}
