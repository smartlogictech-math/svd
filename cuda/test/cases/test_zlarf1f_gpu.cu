#include "cuSVD.cuh"

#include "../common/host.cuh"
#include "../common/utils.cuh"

#include <stdio.h>
#include <assert.h>

void test_small_left(cublasHandle_t handle)
{
    printf("== test_small_left_right ==\n");

    const int m = 3, n = 2, ldc = m;
    cuDoubleComplex C[m * n] = {
        {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}};

    // Householder vector v = [1, -2+i, 3-i]
    cuDoubleComplex v[m] = {
        {1, 0}, {-2, 1}, {3, -1}};
    cuDoubleComplex tau = make_cuDoubleComplex(0.0625, -0.0625);

    cuDoubleComplex C_ref[m * n], C_gpu[m * n];

    // host golden
    zlarf_host('L', m, n, v, 1, tau, C, ldc, C_ref);

    // device
    cuDoubleComplex *d_C, *d_v, *d_work;
    cudaMalloc(&d_C, sizeof(C));
    cudaMalloc(&d_v, sizeof(v));
    cudaMalloc(&d_work, sizeof(cuDoubleComplex) * n);

    cudaMemcpy(d_C, C, sizeof(C), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, sizeof(v), cudaMemcpyHostToDevice);

    zlarf1f_gpu(handle, 'L', m, n, d_v, 1, tau, d_C, ldc, d_work);
    cudaMemcpy(C_gpu, d_C, sizeof(C), cudaMemcpyDeviceToHost);
    double err = matrix_diff_norm(m, n, C_ref, ldc, C_gpu, ldc);
    printf("Left err = %.3e\n", err);
    assert(err < 1e-12);

    zlarf1f_gpu(handle, 'L', m, n, d_v, 1, cuConj(tau), d_C, ldc, d_work);
    cudaMemcpy(C_gpu, d_C, sizeof(C), cudaMemcpyDeviceToHost);
    err = matrix_diff_norm(m, n, C, ldc, C_gpu, ldc);
    printf("Left orth err = %.3e\n", err);
    assert(err < 1e-12);

    cudaFree(d_C);
    cudaFree(d_v);
    cudaFree(d_work);
}

void test_small_right(cublasHandle_t handle)
{
    printf("== test_small_right ==\n");

    const int m = 3, n = 3, ldc = m;

    cuDoubleComplex C[m * n] = {
        {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}};

    // Householder vector v = [1, -2+i, 3-i]
    cuDoubleComplex v[n] = {
        {1, 0}, {-2, 1}, {3, -1}};
    cuDoubleComplex tau = make_cuDoubleComplex(0.0625, -0.0625);

    cuDoubleComplex C_ref[m * n], C_gpu[m * n];

    // ---------- host reference ----------
    zlarf_host('R', m, n, v, 1, tau, C, ldc, C_ref);

    // ---------- device ----------
    cuDoubleComplex *d_C, *d_v, *d_work;
    cudaMalloc(&d_C, sizeof(C));
    cudaMalloc(&d_v, sizeof(v));
    cudaMalloc(&d_work, sizeof(cuDoubleComplex) * m);

    cudaMemcpy(d_C, C, sizeof(C), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, sizeof(v), cudaMemcpyHostToDevice);

    zlarf1f_gpu(handle, 'R', m, n, d_v, 1, tau, d_C, ldc, d_work);
    cudaMemcpy(C_gpu, d_C, sizeof(C), cudaMemcpyDeviceToHost);
    double err = matrix_diff_norm(m, n, C_ref, ldc, C_gpu, ldc);
    printf("Right multiply err = %.3e\n", err);
    assert(err < 1e-12);

    zlarf1f_gpu(handle, 'R', m, n, d_v, 1, cuConj(tau), d_C, ldc, d_work);
    cudaMemcpy(C_gpu, d_C, sizeof(C), cudaMemcpyDeviceToHost);
    err = matrix_diff_norm(m, n, C, ldc, C_gpu, ldc);
    printf("Right orth err = %.3e\n", err);
    assert(err < 1e-12);

    cudaFree(d_C);
    cudaFree(d_v);
    cudaFree(d_work);
}

void test_large_random(cublasHandle_t handle, char side)
{
    printf("== test_large_random (%c) ==\n", side);

    const int m = 512, n = 768, ldc = m;
    const int k = (side == 'L') ? m : n;

    const size_t mtSz = sizeof(cuDoubleComplex) * m * n;
    const size_t vSz = sizeof(cuDoubleComplex) * k;

    cuDoubleComplex *C0 = new cuDoubleComplex[m * n];
    cuDoubleComplex *C1 = new cuDoubleComplex[m * n];
    cuDoubleComplex *v = new cuDoubleComplex[k];

    for (int i = 0; i < m * n; ++i)
        C0[i] = make_cuDoubleComplex(drand48(), drand48());
    for (int i = 0; i < k; ++i)
        v[i] = make_cuDoubleComplex(drand48(), drand48());

    cuDoubleComplex tau;
    zlarfg_host(k, v, v + 1, 1, &tau);

    cuDoubleComplex *d_C, *d_v, *d_work;
    cudaMalloc(&d_C, mtSz);
    cudaMalloc(&d_v, vSz);
    cudaMalloc(&d_work, vSz);

    cudaMemcpy(d_C, C0, mtSz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, vSz, cudaMemcpyHostToDevice);

    zlarf1f_gpu(handle, side, m, n, d_v, 1, tau, d_C, ldc, d_work);
    zlarf1f_gpu(handle, side, m, n, d_v, 1, cuConj(tau), d_C, ldc, d_work);
    cudaMemcpy(C1, d_C, mtSz, cudaMemcpyDeviceToHost);
    double err = matrix_diff_norm(m, n, C0, ldc, C1, ldc);
    printf("Orth err = %.3e\n", err);
    assert(err < 1e-12);

    cudaFree(d_C);
    cudaFree(d_v);
    cudaFree(d_work);

    delete[] C0;
    delete[] C1;
    delete[] v;
}

void test_tau_zero(cublasHandle_t handle)
{
    printf("== test_tau_zero ==\n");

    const int m = 3, n = 4;
    cuDoubleComplex C0[m * n] = {
        {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 9}, {9, 1}, {1, 1}, {2, 2}, {3, 3}};

    cuDoubleComplex v[3] = {{1, 0}, {2, 1}, {3, -1}};
    cuDoubleComplex tau = make_cuDoubleComplex(0, 0);

    // GPU
    cuDoubleComplex *d_C, *d_v, *d_work;
    cudaMalloc(&d_C, sizeof(C0));
    cudaMalloc(&d_v, sizeof(v));
    cudaMalloc(&d_work, m * sizeof(cuDoubleComplex));
    cudaMemcpy(d_C, C0, sizeof(C0), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, sizeof(v), cudaMemcpyHostToDevice);

    zlarf1f_gpu(handle, 'L', m, n, d_v, 1, tau, d_C, m, d_work);

    cuDoubleComplex C_gpu[m * n];
    cudaMemcpy(C_gpu, d_C, sizeof(C_gpu), cudaMemcpyDeviceToHost);

    // 断言
    assert(matrix_allclose(C_gpu, C0, m, n));

    cudaFree(d_C);
    cudaFree(d_v);
    cudaFree(d_work);
}

void test_incv_gt_1(cublasHandle_t handle)
{
    printf("== test_incv_gt_1 ==\n");

    const int m = 3, n = 3;

    cuDoubleComplex C0[m * n] = {
        {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 8}, {9, 9}};
    cuDoubleComplex C_ref[m * n];
    memcpy(C_ref, C0, sizeof(C0));

    // v stored with incv=2
    cuDoubleComplex v_store[6] = {
        {1, 0}, {9, 9}, // v[0]
        {-2, 1}, {9, 9}, // v[1]
        {3, -1}, {9, 9} // v[2]
    };

    cuDoubleComplex tau = make_cuDoubleComplex(0.0625, -0.0625);

    zlarf_host('L', m, n, v_store, 2, tau, C0, m, C_ref);

    cuDoubleComplex *d_C, *d_v, *d_w;
    cudaMalloc(&d_C, sizeof(C0));
    cudaMalloc(&d_v, sizeof(v_store));
    cudaMalloc(&d_w, n * sizeof(cuDoubleComplex));
    cudaMemcpy(d_C, C0, sizeof(C0), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v_store, sizeof(v_store), cudaMemcpyHostToDevice);

    zlarf1f_gpu(handle, 'L', m, n, d_v, 2, tau, d_C, m, d_w);

    cuDoubleComplex C_gpu[m * n];
    cudaMemcpy(C_gpu, d_C, sizeof(C0), cudaMemcpyDeviceToHost);

    assert(matrix_allclose(C_gpu, C_ref, m, n));

    cudaFree(d_C);
    cudaFree(d_v);
    cudaFree(d_w);
}

int main()
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    test_small_left(handle);
    test_small_right(handle);
    test_large_random(handle, 'L');
    test_large_random(handle, 'R');
    test_tau_zero(handle);
    test_incv_gt_1(handle);

    cublasDestroy(handle);
    printf("All zlarf1f tests PASSED.\n");
    return 0;
}
