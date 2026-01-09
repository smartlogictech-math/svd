#pragma once

#include <cuComplex.h>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

void zaxpyc(
    int n,
    const cuDoubleComplex *alpha,
    const cuDoubleComplex* x, int incx,
    cuDoubleComplex* y, int incy,
    cudaStream_t stream
);

#if defined(__cplusplus)
}
#endif /* __cplusplus */