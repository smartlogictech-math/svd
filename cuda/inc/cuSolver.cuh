#pragma once
#include <cuda_runtime.h>
#include <cusolverDn.h>     // 使用官方 cuSOLVER 的所有类型定义
#include <cusolver_common.h>

#ifdef __cplusplus
extern "C" {
#endif

cusolverStatus_t
solverDnZgebrd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *Lwork );

cusolverStatus_t
solverDnZgebrd(cusolverDnHandle_t handle,
           int m,
           int n,
           cuDoubleComplex *A,
           int lda,
           double *D,
           double *E,
           cuDoubleComplex *TAUQ,
           cuDoubleComplex *TAUP,
           cuDoubleComplex *Work,
           int Lwork,
           int *devInfo );

cusolverStatus_t
solverDnZgesvd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *lwork );

// full SVD
cusolverStatus_t
solverDnZgesvd(
    cusolverDnHandle_t handle,
    signed char jobu,
    signed char jobvt,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda,
    double *S,
    cuDoubleComplex *U,
    int ldu,
    cuDoubleComplex *VT,
    int ldvt,
    cuDoubleComplex *work,
    int lwork,
    double *rwork,
    int *devInfo );

#ifdef __cplusplus
}
#endif
