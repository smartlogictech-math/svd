#pragma once 

#include "cublas_v2.h"

extern void zlarfg_gpu(
    cublasHandle_t handle,
    int n,
    cuDoubleComplex *d_alpha,
    cuDoubleComplex *d_x,
    int incx,
    cuDoubleComplex *d_tau);