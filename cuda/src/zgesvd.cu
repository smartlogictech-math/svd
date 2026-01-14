#include "cuSolver.cuh"
#include "cuSVD.cuh"

#include <stdio.h>

cusolverStatus_t
solverDnZgesvd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *lwork )
{
    if (!lwork || m < 0 || n < 0 || (m < n)) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }

    int gebrdLwork = 0;
    cusolverStatus_t status;

    status = solverDnZgebrd_bufferSize(handle, m, n, &gebrdLwork);
    if(CUSOLVER_STATUS_SUCCESS != status){
        return status;
    }else {
        *lwork = gebrdLwork;
    }

    return CUSOLVER_STATUS_SUCCESS;
}

#include <cuda_runtime.h>
#include <cusolverDn.h>

//
//  参数检查：ZGESVD (与 cuSolverDnZgesvd 对齐)
//
//  devInfo 必须指向 device-side int
//
static cusolverStatus_t check_zgesvd_arguments(
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
    int *devInfo)
{
    int info = 0;

    if (devInfo == NULL) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }

    // ----- 检查 jobu jobvt -----
    bool valid_jobu  = (jobu  == 'A' || jobu  == 'S' || jobu  == 'O' || jobu  == 'N');
    bool valid_jobvt = (jobvt == 'A' || jobvt == 'S' || jobvt == 'O' || jobvt == 'N');

    if (!valid_jobu) {
        info = -1;  // jobu invalid (lapack index = 1)
    } else if (!valid_jobvt) {
        info = -2;  // jobvt invalid
    } else if (m < 0) {
        info = -3;
    } else if ((n < 0) || (m < n)) {
        info = -4;
    } else if (A == NULL) {
        info = -5;
    } else if (lda < ((m > 1) ? m : 1)) {
        info = -6;
    } else if (S == NULL) {
        info = -7;
    } else if (U == NULL && jobu != 'N') {    // jobu=N 时可以不需要 U
        info = -8;
    } else if (ldu < ((jobu=='N') ? 1 : m)) {
        info = -9;
    } else if (VT == NULL && jobvt != 'N') {
        info = -10;
    } else if (ldvt < ((jobvt=='N') ? 1 : ((jobvt=='A') ? n : (m < n ? m : n)))) {
        // For ldvt: LAPACK rules
        info = -11;
    } else if (work == NULL) {
        info = -12;
    } else if (lwork < m) {
        info = -13;
    } else if (rwork == NULL) {
        info = -14;
    }

    if (info != 0) {
        cudaMemcpy(devInfo, &info, sizeof(int), cudaMemcpyHostToDevice);
        return CUSOLVER_STATUS_INVALID_VALUE;
    }

    // 正常：devInfo = 0
    info = 0;
    cudaMemcpy(devInfo, &info, sizeof(int), cudaMemcpyHostToDevice);
    return CUSOLVER_STATUS_SUCCESS;
}

// -------------------------------------------------------------
//  完整 SVD 实现（流程完整，kernel 空壳）
// -------------------------------------------------------------
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
    cuDoubleComplex *work,      // complex workspace
    int lwork,
    double *rwork,              // real workspace (5*k)
    int *devInfo)
{
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;

    status = check_zgesvd_arguments(
        jobu, jobvt, m, n, A, lda, S, U, ldu,
        VT, ldvt, work, lwork, rwork, devInfo
    );

    if (status != CUSOLVER_STATUS_SUCCESS) {
        return status;   // devInfo 已经写入错误码
    }

    cudaStream_t stream;
    cusolverDnGetStream(handle, &stream);    ///< 外网cuda实现，暂不链接cuSolver库

    // ---------------------------
    // 确定 k = min(m,n)
    // ---------------------------
    int k = (m < n ? m : n);

    cuDoubleComplex* w_gemv = work;     // gemv buffer: max(m,n)

    double* rw_bdsqr = rwork;

    // ---------------------------
    // 创建 cuBLAS
    // ---------------------------
    cublasHandle_t blasHandle;
    cublasCreate(&blasHandle);
    cublasSetStream(blasHandle, stream);
    /// 分配内存
    double *dScale;
    cudaMalloc(&dScale, sizeof(double));
    double *hD, *hE;
    cuDoubleComplex *hTauQ, *hTauP;
    cudaMallocHost(&hD, sizeof(double) * k);
    cudaMallocHost(&hE, sizeof(double) * (k - 1));
    cudaMallocHost(&hTauQ, sizeof(cuDoubleComplex) * k);
    cudaMallocHost(&hTauP, sizeof(cuDoubleComplex) * k);


    // **********************************************************************
    // 1. 缩放矩阵 |A|max = maxabs
    // **********************************************************************
    double *rw_maxabs = rwork;   ///< zamax result: n
    double maxabs = 0.0;
    zlange_maxabs_gpu(m, n, A, lda, &maxabs, rw_maxabs, stream);
    cudaStreamSynchronize(stream);
    if (maxabs == 0.0) {
        cudaMemsetAsync(S, 0, sizeof(double)*k, stream);
        *devInfo = 0;
        return CUSOLVER_STATUS_SUCCESS;
    }
    
    double hScale = 1.0 / maxabs;
    cudaMemcpyAsync(dScale, &hScale, sizeof(double), cudaMemcpyHostToDevice, stream);

    zlascl_gpu(m, n, dScale, A, lda, stream);
    cudaStreamSynchronize(stream);

    // **********************************************************************
    // 2. ZGEBD2：A → Bidiagonal (D,E)
    // **********************************************************************

    cuDoubleComplex *dgebrdWork = work;

    int gebrdInfo = 0;
    status = solverDnZgebrd(
                handle, 
                m, n,
                A, lda,
                hD, hE,
                hTauQ, hTauP,
                dgebrdWork,
                lwork,
                &gebrdInfo
            );
    if(CUSOLVER_STATUS_SUCCESS != status) {
        printf("solverDnZgebrd return status=%d, gebrdInfo=%d\n", status, gebrdInfo);
        return status;
    }

    // **********************************************************************
    // 3. 生成 U = Q
    // **********************************************************************
    if (jobu == 'S' || jobu == 'A') {
        // NOTE：此处按 LAPACK 的方式从 A 中恢复 Q
        // 实际 GPU 最终实现你可能需要接入 ZUNMBR，而不是直接 ZUNGQR。
        zungqr_gpu(
            m,       // rows of U
            (jobu=='A' ? m : k),
            k,
            A, lda,
            hTauQ,
            w_gemv,
            stream);

        cudaMemcpyAsync(
            U, A,
            sizeof(cuDoubleComplex)*ldu*((jobu=='A')?m:k),
            cudaMemcpyDeviceToDevice,
            stream);
    }

    // **********************************************************************
    // 4. 生成 VT = P^H
    // **********************************************************************
    if (jobvt == 'S' || jobvt == 'A') {
        zunglq_gpu(
            (jobvt=='A'?n:k),
            n,
            k,
            A, lda,
            hTauP,
            w_gemv,
            stream);

        cudaMemcpyAsync(
            VT, A,
            sizeof(cuDoubleComplex)*ldvt*((jobvt=='A')?n:k),
            cudaMemcpyDeviceToDevice,
            stream);
    }

    // **********************************************************************
    // 5. BDSQR：二维对角 D,E → SVD
    // **********************************************************************
    int ncvt = (jobvt=='A' || jobvt=='S') ? n : 0;
    int nru  = (jobu =='A' || jobu =='S') ? m : 0;

    char uplo = 'U';    // 因为仅支持m>=n，所以固定为U
    zbdsqr_gpu(
        uplo,
        k,
        ncvt,
        nru,
        hD,
        hE,
        VT, ldvt,
        U, ldu,
        rw_bdsqr,
        devInfo,
        stream
    );

    // **********************************************************************
    // 6. 反缩放奇异值
    // **********************************************************************
    cudaStreamSynchronize(stream);
    cudaMemcpyAsync(dScale, &maxabs, sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(S, hD, sizeof(double)*k, cudaMemcpyHostToDevice, stream);
    // no need to sync, because handle uses stream
    cublasDscal(blasHandle, k, dScale, S, 1);
    cudaStreamSynchronize(stream);

    // **********************************************************************
    // 完成
    // **********************************************************************
    cublasDestroy(blasHandle);

    cudaFree(dScale);
    cudaFreeHost(hD);
    cudaFreeHost(hE);
    cudaFreeHost(hTauQ);
    cudaFreeHost(hTauP);

    return CUSOLVER_STATUS_SUCCESS;
}