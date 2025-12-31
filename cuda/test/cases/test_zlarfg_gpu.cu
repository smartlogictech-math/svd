#include "cuSVD.cuh"
#include "../common/common.cuh"

#include <assert.h>
#include <stdio.h>

static void check(const int n, const cuDoubleComplex *y, const cuDoubleComplex beta, const cuDoubleComplex tau, const cuDoubleComplex *v){
    // 1. check beta
    printf("imag(beta) = %.3e\n",cuCimag(beta));
    assert(cuCimag(beta) == 0.0);

    // 2.  H**H * ( y ) = ( beta 0)
    cuDoubleComplex* Hy = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
    apply_householder(n, cuConj(tau), v, y, Hy);
    // target = [beta, 0, 0, ...]
    cuDoubleComplex* target =(cuDoubleComplex*)calloc(n, sizeof(cuDoubleComplex));
    target[0] = beta;

    cuDoubleComplex* res =(cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
    for (int i = 0; i < n; ++i) {
        res[i] = Hy[i];
    }
    res[0] = cuCsub(res[0], beta);  // Hy[0] - beta
    double hy_rel_err =znrm2_host(n, res) / znrm2_host(n, y);
    printf("H**H * y relative error = %.3e\n", hy_rel_err);
    assert(hy_rel_err < 1e-12);
    
    /// 3. H * H**H * y = y
    cuDoubleComplex *HHy = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
    apply_householder(n, tau, v, Hy, HHy);
    for (int i = 0; i < n; ++i) {
        res[i] = cuCsub(HHy[i], y[i]);
    }
    double hhy_rel_err =znrm2_host(n, res) / znrm2_host(n, y);
    printf("H * H**H * y relative error = %.3e\n", hhy_rel_err);
    assert(hhy_rel_err < 1e-12);

    free(Hy);
    free(target);
    free(res);
    free(HHy);
}

void test_zlarfg_gpu(int n)
{
    assert(n >= 2);

    // ---------- host data ----------
    cuDoubleComplex* h_y     = (cuDoubleComplex*)malloc(n * sizeof(*h_y));
    cuDoubleComplex* h_y0    = (cuDoubleComplex*)malloc(n * sizeof(*h_y0));
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

    check(n, h_y, beta, tau, h_v);

    // ---------- cleanup ----------
    free(h_y);
    free(h_y0);
    free(h_v);

    cudaFree(d_alpha);
    cudaFree(d_x);
    cudaFree(d_tau);
    cublasDestroy(handle);
}

int main()
{
    test_zlarfg_gpu(65535);
    printf("All zlarfg_gpu tests passed.\n");
    return 0;
}
