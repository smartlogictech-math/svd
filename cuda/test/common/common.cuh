#pragma once
#include <cuComplex.h>

void apply_householder(
    int n,
    cuDoubleComplex tau,
    const cuDoubleComplex *v,
    const cuDoubleComplex *y,
    cuDoubleComplex *Hy);

extern double znrm2_host(int n, const cuDoubleComplex *x);