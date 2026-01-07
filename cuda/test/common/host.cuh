#pragma once
#include <cuComplex.h>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

void zlarfg_host(
    int n,
    cuDoubleComplex* alpha,       // v[0], in/out
    cuDoubleComplex* x,           // v[1:], length n-1, in/out
    int incx,
    cuDoubleComplex* tau          // output
);

void zlarfv_host(
    int n,
    cuDoubleComplex tau,
    const cuDoubleComplex *v,
    const cuDoubleComplex *y,
    cuDoubleComplex *Hy);
void zlarf_host(
    char side,              // 'L' or 'R'
    int m, int n,
    const cuDoubleComplex* v, int incv,
    cuDoubleComplex tau,
    const cuDoubleComplex* C, int ldc,
    cuDoubleComplex* outC
);

double znrm2_host(int n, const cuDoubleComplex *x);

#if defined(__cplusplus)
}
#endif /* __cplusplus */