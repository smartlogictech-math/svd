#pragma once
#include <cuComplex.h>

typedef enum {
    HOUSEHOLDER_LEFT,   // y <- H^H y
    HOUSEHOLDER_RIGHT   // y <- H y
} householder_side_t;

void apply_householder(
    int n,
    cuDoubleComplex tau,
    const cuDoubleComplex *v,
    const cuDoubleComplex *y,
    cuDoubleComplex *Hy,
    householder_side_t side);

extern double znrm2_host(int n, const cuDoubleComplex *x);