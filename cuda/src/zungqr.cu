#include "cuSVD.cuh"
void zungqr_gpu(
    int m, int n, int k,   // Q = m x n, k = number of Householder vectors
    cuDoubleComplex* A, int lda,
    const cuDoubleComplex* TAUQ,
    cuDoubleComplex* work,
    cudaStream_t stream){}