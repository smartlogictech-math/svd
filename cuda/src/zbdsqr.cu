#include "cuSVD.cuh"

void zbdsqr_gpu(
    char uplo,                 // 'U'
    int n,                     // k = min(m,n)
    int ncvt,                  // n or 0
    int nru,                   // m or 0
    double* hD,          // host: diagonal, length n
    double* hE,          // host: off diagonal, length n-1
    cuDoubleComplex* dVT,      // device (may be NULL)
    int ldvt,
    cuDoubleComplex* dU,       // device (may be NULL)
    int ldu,
    double* dWork,             // device workspace, >= 4*n
    int* devInfo,               // device int
    cudaStream_t stream){}