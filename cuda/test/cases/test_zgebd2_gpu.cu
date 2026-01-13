#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cmath>
#include "cuSVD.cuh"

using cuDoubleComplex = cuDoubleComplex;

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = (call); \
    if(status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "CUBLAS error at " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_CUSOLVER(call) do { \
    cusolverStatus_t status = (call); \
    if(status != CUSOLVER_STATUS_SUCCESS) { \
        std::cerr << "CUSOLVER error at " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// ---------------- helpers ----------------
cuDoubleComplex make_complex(double r, double i) {
    return make_cuDoubleComplex(r,i);
}

void h2d(cuDoubleComplex* d, const std::vector<cuDoubleComplex>& h) {
    CHECK_CUDA(cudaMemcpy(d,h.data(),sizeof(cuDoubleComplex)*h.size(),cudaMemcpyHostToDevice));
}

void d2h(std::vector<cuDoubleComplex>& h, cuDoubleComplex* d) {
    CHECK_CUDA(cudaMemcpy(h.data(),d,sizeof(cuDoubleComplex)*h.size(),cudaMemcpyDeviceToHost));
}

double max_rel_error_robust(const std::vector<double>& ref,
                            const std::vector<double>& my)
{
    const double eps = 1e-14;  // 最小量防止除零
    const int n = ref.size();

    double max_err = 0.0;

    for (int i = 0; i < n; ++i) {
        double r = ref[i];
        double m = my[i];
        double diff = fabs(r - m);

        // 如果参考值很小，用绝对误差，否则用相对误差
        double rel = (fabs(r) > eps) ? diff / fabs(r) : diff;

        if (rel > max_err) max_err = rel;
    }
    return max_err;
}

double max_rel_error_cu_robust(const std::vector<cuDoubleComplex>& ref,
                               const std::vector<cuDoubleComplex>& my)
{
    const double eps = 1e-14;
    const int n = ref.size();

    double max_err = 0.0;

    for (int i = 0; i < n; ++i) {
        double r_re = cuCreal(ref[i]);
        double r_im = cuCimag(ref[i]);
        double m_re = cuCreal(my[i]);
        double m_im = cuCimag(my[i]);

        double diff_re = fabs(r_re - m_re);
        double diff_im = fabs(r_im - m_im);

        double rel_re = (fabs(r_re) > eps) ? diff_re / fabs(r_re) : diff_re;
        double rel_im = (fabs(r_im) > eps) ? diff_im / fabs(r_im) : diff_im;

        double rel = std::max(rel_re, rel_im);
        if (rel > max_err) max_err = rel;
    }
    return max_err;
}

// ---------------- matrix generator ----------------
enum class MatrixType { HAND_UPPER, HAND_LOWER, RANDOM };

void generate_matrix(MatrixType type, int m, int n, std::vector<cuDoubleComplex>& hA) {
    hA.resize(m*n);
    if(type==MatrixType::HAND_UPPER) {
        // small m>n example, fixed numbers for hand check
        double val = 1.0;
        for(int j=0;j<n;j++)
            for(int i=0;i<m;i++)
                hA[j*m+i] = make_complex(val++, 1.0);
    } else if(type==MatrixType::HAND_LOWER) {
        // small m<n example, fixed numbers for hand check
        double val = 1.0;
        for(int j=0;j<n;j++)
            for(int i=0;i<m;i++)
                hA[j*m+i] = make_complex(val++,0);
    } else { // RANDOM
        for(int j=0;j<n;j++)
            for(int i=0;i<m;i++)
                hA[j*m+i] = make_complex(((double)rand()/RAND_MAX)*2-1,
                                          ((double)rand()/RAND_MAX)*2-1);
    }
}

void print_bidiag_compare(
    int m,
    int n,
    int idx,
    const std::vector<double>& D_ref,
    const std::vector<double>& D_my,
    const std::vector<double>& E_ref,
    const std::vector<double>& E_my,
    const std::vector<cuDoubleComplex>& TauQ_ref,
    const std::vector<cuDoubleComplex>& TauQ_my,
    const std::vector<cuDoubleComplex>& TauP_ref,
    const std::vector<cuDoubleComplex>& TauP_my,
    const std::vector<cuDoubleComplex>& A_ref,
    const std::vector<cuDoubleComplex>& A_my
) {
    int min_mn = std::min(m, n);

    printf("\n=============================\n");
    printf("Bidiagonal Column %d Compare\n", idx);
    printf("=============================\n");

    printf("\n-- D --\n");
    for (int i = 0; i < min_mn; i++) {
        printf("i=%2d  ref=%+.20e  my=%+.20e  diff=%+.20e\n",
               i, D_ref[i], D_my[i], D_my[i] - D_ref[i]);
    }

    printf("\n-- E --\n");
    for (int i = 0; i < min_mn - 1; i++) {
        printf("i=%2d  ref=%+.20e  my=%+.20e  diff=%+.20e\n",
               i, E_ref[i], E_my[i], E_my[i] - E_ref[i]);
    }

    printf("\n-- TauQ --\n");
    for (int i = 0; i < min_mn; i++) {
        printf("i=%2d  ref=(%+.20e,%+.20e)  my=(%+.20e,%+.20e)\n",
               i,
               TauQ_ref[i].x, TauQ_ref[i].y,
               TauQ_my[i].x, TauQ_my[i].y);
    }

    printf("\n-- TauP --\n");
    for (int i = 0; i < min_mn; i++) {
        printf("i=%2d  ref=(%+.20e,%+.20e)  my=(%+.20e,%+.20e)\n",
               i,
               TauP_ref[i].x, TauP_ref[i].y,
               TauP_my[i].x, TauP_my[i].y);
    }

    printf("\n-- Left Householder vectors v(i:m,i) --\n");
    for (int col = 0; col < min_mn; col++) {
        printf("v column %d :\n", col);
        for (int row = col; row < m; row++) {
            cuDoubleComplex vr = A_ref[row + col * m];
            cuDoubleComplex vm = A_my[row + col * m];
            printf("   row %d: ref=(%+.20e,%+.20e)  my=(%+.20e,%+.20e)\n",
                   row,
                   vr.x, vr.y,
                   vm.x, vm.y);
        }
    }

    printf("\n-- Right Householder vectors u(i,i+1:n) --\n");
    for (int col = 0; col < min_mn; col++) {
        printf("u row %d :\n", col);
        for (int col2 = col + 1; col2 < n; col2++) {
            cuDoubleComplex ur = A_ref[col + col2 * m];
            cuDoubleComplex um = A_my[col + col2 * m];
            printf("   col %d: ref=(%+.20e,%+.20e)  my=(%+.20e,%+.20e)\n",
                   col2,
                   ur.x, ur.y,
                   um.x, um.y);
        }
    }

    printf("====================================\n\n");
}

// ---------------- general test ----------------
void test_zgebd2(MatrixType type, int m, int n)
{
    const int lda     = m;
    const int min_mn  = std::min(m, n);

    // ---------------------------------------------------------------------
    // 1. Generate host test matrix
    // ---------------------------------------------------------------------
    std::vector<cuDoubleComplex> h_A;
    generate_matrix(type, m, n, h_A);

    // =====================================================================
    //      Our implementation (zgebd2_gpu)
    // =====================================================================
    cuDoubleComplex* d_A_my  = nullptr;
    cuDoubleComplex* d_work  = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_A_my, sizeof(cuDoubleComplex) * m * n));
    CHECK_CUDA(cudaMalloc((void**)&d_work,  sizeof(cuDoubleComplex) * std::max(m, n)));
    cuDoubleComplex *h_A_data = h_A.data();
    printf("h_A_data=%p\n", h_A_data);

    h2d(d_A_my, h_A);  // upload input matrix

    std::vector<double>          h_D_my(min_mn), h_E_my(min_mn - 1);
    std::vector<cuDoubleComplex> h_TauQ_my(min_mn), h_TauP_my(min_mn);

    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    // ---- run our implementation ----
    zgebd2_gpu(
        cublasHandle,
        m, n,
        d_A_my, lda,
        h_D_my.data(), h_E_my.data(),
        h_TauQ_my.data(), h_TauP_my.data(),
        d_work
    );

    // 拷贝 A_my 回 host（Householder vectors）
    std::vector<cuDoubleComplex> hA_my(m * n);
    CHECK_CUDA(cudaMemcpy(
        hA_my.data(), d_A_my,
        sizeof(cuDoubleComplex) * m * n,
        cudaMemcpyDeviceToHost
    ));

    // =====================================================================
    //      Reference implementation (cuSOLVER ZGEBRD)
    // =====================================================================
    cuDoubleComplex* d_A_ref   = nullptr;
    double*          d_D_ref   = nullptr;
    double*          d_E_ref   = nullptr;
    cuDoubleComplex* d_TauQ_ref = nullptr;
    cuDoubleComplex* d_TauP_ref = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_A_ref,    sizeof(cuDoubleComplex) * m * n));
    CHECK_CUDA(cudaMalloc((void**)&d_D_ref,    sizeof(double) * min_mn));
    CHECK_CUDA(cudaMalloc((void**)&d_E_ref,    sizeof(double) * (min_mn - 1)));
    CHECK_CUDA(cudaMalloc((void**)&d_TauQ_ref, sizeof(cuDoubleComplex) * min_mn));
    CHECK_CUDA(cudaMalloc((void**)&d_TauP_ref, sizeof(cuDoubleComplex) * min_mn));

    h2d(d_A_ref, h_A);   // load the same input matrix

    cusolverDnHandle_t cusolverHandle;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverHandle));

    // Query workspace
    int lwork_ref = 0;
    CHECK_CUSOLVER(cusolverDnZgebrd_bufferSize(cusolverHandle, m, n, &lwork_ref));

    cuDoubleComplex* d_work_ref = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_work_ref, sizeof(cuDoubleComplex) * lwork_ref));

    int* d_info = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_info, sizeof(int)));

    // ---- run cuSOLVER ZGEBRD ----
    CHECK_CUSOLVER(
        cusolverDnZgebrd(
            cusolverHandle,
            m, n,
            d_A_ref, lda,
            d_D_ref, d_E_ref,
            d_TauQ_ref, d_TauP_ref,
            d_work_ref, lwork_ref,
            d_info
        )
    );

    // check info
    int h_info = 0;
    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        std::cerr << "[cusolverDnZgebrd] info = " << h_info << std::endl;
    }

    // Reference outputs
    std::vector<double>          h_D_ref(min_mn), h_E_ref(min_mn - 1);
    std::vector<cuDoubleComplex> h_TauQ_ref(min_mn), h_TauP_ref(min_mn);

    CHECK_CUDA(cudaMemcpy(h_D_ref.data(),    d_D_ref,    sizeof(double)*min_mn,
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_E_ref.data(),    d_E_ref,    sizeof(double)*(min_mn-1),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_TauQ_ref.data(), d_TauQ_ref, sizeof(cuDoubleComplex)*min_mn,
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_TauP_ref.data(), d_TauP_ref, sizeof(cuDoubleComplex)*min_mn,
                          cudaMemcpyDeviceToHost));

    // 拷贝 A_ref 回 host（Householder vectors / reduced form）
    std::vector<cuDoubleComplex> hA_ref(m * n);
    CHECK_CUDA(cudaMemcpy(
        hA_ref.data(), d_A_ref,
        sizeof(cuDoubleComplex) * m * n,
        cudaMemcpyDeviceToHost
    ));

    // =====================================================================
    // 5. Print summary
    // =====================================================================
    const char* type_str =
        (type == MatrixType::HAND_UPPER ? "HAND_UPPER" :
         type == MatrixType::HAND_LOWER ? "HAND_LOWER" : "RANDOM");

    std::cout << "Test " << type_str << "  " << m << "x" << n
              << " | D err = "    << max_rel_error_robust(h_D_ref,    h_D_my)
              << " | E err = "    << max_rel_error_robust(h_E_ref,    h_E_my)
              << " | TauQ err = " << max_rel_error_cu_robust(h_TauQ_ref, h_TauQ_my)
              << " | TauP err = " << max_rel_error_cu_robust(h_TauP_ref, h_TauP_my)
              << std::endl;

    // =====================================================================
    // 6. Detailed dump: call your print_bidiag_compare
    // =====================================================================
#if 1
    print_bidiag_compare(
        m, n, /* idx */ -1,
        h_D_ref,    h_D_my,
        h_E_ref,    h_E_my,
        h_TauQ_ref, h_TauQ_my,
        h_TauP_ref, h_TauP_my,
        hA_ref,     hA_my
    );
#endif
    // =====================================================================
    // 7. Cleanup
    // =====================================================================
    CHECK_CUDA(cudaFree(d_A_my));
    CHECK_CUDA(cudaFree(d_work));

    CHECK_CUDA(cudaFree(d_A_ref));
    CHECK_CUDA(cudaFree(d_D_ref));
    CHECK_CUDA(cudaFree(d_E_ref));
    CHECK_CUDA(cudaFree(d_TauQ_ref));
    CHECK_CUDA(cudaFree(d_TauP_ref));
    CHECK_CUDA(cudaFree(d_work_ref));
    CHECK_CUDA(cudaFree(d_info));

    CHECK_CUBLAS(cublasDestroy(cublasHandle));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverHandle));
}

int main(int argc,char* argv[]) {
    srand(1234);

    // small upper bidiagonal, m>n
    test_zgebd2(MatrixType::HAND_UPPER, 3, 3);

    // m<n, no supported
    // test_zgebd2(MatrixType::HAND_LOWER, 3, 4);

    // random matrix if user specifies m,n
    if(argc>=3){
        int m = atoi(argv[1]);
        int n = atoi(argv[2]);
        test_zgebd2(MatrixType::RANDOM, m,n);
    }
    return 0;
}
