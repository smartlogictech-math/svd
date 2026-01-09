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
    if (!lwork || m < 0 || n < 0) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }

    int k = (m < n) ? m : n;

    // cuSOLVER / LAPACK requirement
    // LWORK >= 2*k + max(m,n)
    int lw = 2 * k + ((m > n) ? m : n);

    *lwork = (lw > 1 ? lw : 1);

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
    if (!devInfo) return CUSOLVER_STATUS_INVALID_VALUE;
    cudaStream_t stream;
    // cusolverDnGetStream(handle, &stream);    ///< 外网cuda实现，暂不链接cuSolver库
    cudaStreamCreate(&stream);

    // ---------------------------
    // 确定 k = min(m,n)
    // ---------------------------
    int k = (m < n ? m : n);

    // ---------------------------
    // 检查 workspace
    // ---------------------------
    int lwork_req = (m > n) ? m : n;
    if (lwork < lwork_req) return CUSOLVER_STATUS_INVALID_VALUE;
    if (!rwork) return CUSOLVER_STATUS_INVALID_VALUE;

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

    zgebd2_gpu(
        blasHandle,
        m, n,
        A, lda,
        hD, hE,
        hTauQ, hTauP,
        w_gemv);

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

    char uplo = 'U';    // 固定为 U，不要根据 m,n 判断
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
    cudaStreamDestroy(stream);

    cudaFree(dScale);
    cudaFreeHost(hD);
    cudaFreeHost(hE);
    cudaFreeHost(hTauQ);
    cudaFreeHost(hTauP);

    return CUSOLVER_STATUS_SUCCESS;
}