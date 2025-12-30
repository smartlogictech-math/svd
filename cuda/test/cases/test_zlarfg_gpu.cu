#include "cuSVD.cuh"
#include "../common/common.cuh"

#include <assert.h>
#include <stdio.h>

void test_zlarfg_gpu(int n)
{
    assert(n >= 2);

    // ---------- host data ----------
    cuDoubleComplex* h_y     = (cuDoubleComplex*)malloc(n * sizeof(*h_y));
    cuDoubleComplex* h_y0    = (cuDoubleComplex*)malloc(n * sizeof(*h_y0));
    cuDoubleComplex* h_Hy    = (cuDoubleComplex*)malloc(n * sizeof(*h_Hy));
    cuDoubleComplex* h_v     = (cuDoubleComplex*)malloc(n * sizeof(*h_v));

    // random input
    for (int i = 0; i < n; ++i) {
        h_y[i] = make_cuDoubleComplex(
            drand48() - 0.5,
            drand48() - 0.5
        );
        h_y0[i] = h_y[i];
    }

    // ---------- device data ----------
    cuDoubleComplex *d_alpha, *d_x, *d_tau;
    cudaMalloc(&d_alpha, sizeof(*d_alpha));
    cudaMalloc(&d_x,     (n-1) * sizeof(*d_x));
    cudaMalloc(&d_tau,   sizeof(*d_tau));

    cudaMemcpy(d_alpha, &h_y[0], sizeof(*d_alpha), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,     &h_y[1], (n-1)*sizeof(*d_x), cudaMemcpyHostToDevice);

    // ---------- call under test ----------
    cublasHandle_t handle;
    cublasCreate(&handle);

    zlarfg_gpu(handle, n, d_alpha, d_x, 1, d_tau);

    // ---------- copy back ----------
    cuDoubleComplex beta, tau;
    cudaMemcpy(&beta, d_alpha, sizeof(beta), cudaMemcpyDeviceToHost);
    cudaMemcpy(&tau,  d_tau,   sizeof(tau),  cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_v[1], d_x, (n-1)*sizeof(*d_x), cudaMemcpyDeviceToHost);

    h_v[0] = make_cuDoubleComplex(1.0, 0.0);

    // ---------- checks ----------
    // check beta real
    printf("Im(beta) = %.3e\n", fabs(cuCimag(beta)));

    // apply H
    apply_householder(n, tau, h_v, h_y0, h_Hy);

    // target = [beta, 0, 0, ...]
    cuDoubleComplex* h_target =
        (cuDoubleComplex*)calloc(n, sizeof(*h_target));
    h_target[0] = beta;

    cuDoubleComplex* h_res =
        (cuDoubleComplex*)malloc(n * sizeof(*h_res));

    for (int i = 0; i < n; ++i) {
        h_res[i] = h_Hy[i];
    }
    h_res[0] = cuCsub(h_res[0], beta);  // Hy[0] - beta

    double rel_err =
        znrm2_host(n, h_res) / znrm2_host(n, h_y0);


    printf("relative error = %.3e\n", rel_err);

    assert(rel_err < 1e-12);

    // ---------- cleanup ----------
    free(h_y);
    free(h_y0);
    free(h_Hy);
    free(h_v);
    free(h_target);
    free(h_res);

    cudaFree(d_alpha);
    cudaFree(d_x);
    cudaFree(d_tau);
    cublasDestroy(handle);
}

int main()
{
    test_zlarfg_gpu(2);
    printf("All zlarfg_gpu tests passed.\n");
    return 0;
}
