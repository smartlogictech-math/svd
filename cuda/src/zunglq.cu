#include "cuSVD.cuh"

void zunglq_gpu(
    int m, int n, int k,   // P^H = m x n
    cuDoubleComplex* A, int lda,
    const cuDoubleComplex* TAUP,
    cuDoubleComplex* work,
    cudaStream_t stream){}