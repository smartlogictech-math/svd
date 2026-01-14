#include "cuSolver.cuh"
#include "cuSVD.cuh"

cusolverStatus_t
solverDnZgebrd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *Lwork )
{
    if (!Lwork || m < 0 || n < 0 || (m < n)) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }

    int maxmn = (m >= n) ? m : n;   ///< zlarf/zlarf1f 临时存储gemv结果

    *Lwork = maxmn;

    return CUSOLVER_STATUS_SUCCESS;
}

static cusolverStatus_t check_gebrd_arguments(
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
    int *devInfo )
{
    int h_info = 0;   // host-side temporary

    if (devInfo == NULL) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }

    // LAPACK 规则：lda >= max(1, m)
    if (m < 0) {
        h_info = -1;  // LAPACK 序号从 1 开始
    } else if (n < 0) {
        h_info = -2;
    } else if (A == NULL) {
        h_info = -3;
    } else if (lda < ((m > 1) ? m : 1)) {
        h_info = -4;
    } else if (D == NULL) {
        h_info = -5;
    } else if (E == NULL) {
        h_info = -6;
    } else if (TAUQ == NULL) {
        h_info = -7;
    } else if (TAUP == NULL) {
        h_info = -8;
    } else if (Work == NULL) {
        h_info = -9;
    } else if (Lwork < 1) {
        h_info = -10;
    }

    if (h_info != 0) {
        cudaMemcpy(devInfo, &h_info, sizeof(int), cudaMemcpyHostToDevice);
        return CUSOLVER_STATUS_INVALID_VALUE;
    }

    // 通过检查 → devInfo = 0
    h_info = 0;
    cudaMemcpy(devInfo, &h_info, sizeof(int), cudaMemcpyHostToDevice);
    return CUSOLVER_STATUS_SUCCESS;
}

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
           int *devInfo )
{
    cusolverStatus_t status = check_gebrd_arguments(
        m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo
    );

    if (status != CUSOLVER_STATUS_SUCCESS) {
        return status;   // 参数非法，devInfo 已经被写入
    }

    zgebd2_gpu(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work);

    return CUSOLVER_STATUS_SUCCESS;
}