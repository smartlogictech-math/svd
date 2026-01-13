#include "cuSVD.cuh"

#include <cfloat>
#include <algorithm>

// Helper: stable hypotenuse for three numbers
inline double dlapy3(double a, double b, double c) {
    a = fabs(a); b = fabs(b); c = fabs(c);
    double w = std::max({a,b,c});
    if (w == 0.0) return 0.0;
    a /= w; b /= w; c /= w;
    return w * sqrt(a*a + b*b + c*c);
}

// LAPACK-style safe division for complex numbers
inline cuDoubleComplex zladiv(const cuDoubleComplex &x, const cuDoubleComplex &y) {
    double a = cuCreal(x), b = cuCimag(x);
    double c = cuCreal(y), d = cuCimag(y);
    double r, den;
    if (fabs(c) >= fabs(d)) {
        r = d/c;
        den = c + d*r;
        return make_cuDoubleComplex((a + b*r)/den, (b - a*r)/den);
    } else {
        r = c/d;
        den = c*r + d;
        return make_cuDoubleComplex((a*r + b)/den, (b*r - a)/den);
    }
}

// ZLARFG for GPU
void zlarfg_gpu(
    cublasHandle_t handle,
    int n,
    cuDoubleComplex* d_alpha, // device pointer
    cuDoubleComplex* d_x,     // device pointer
    int incx,
    cuDoubleComplex* h_tau    // host output
) {
    if (n <= 0) {
        *h_tau = make_cuDoubleComplex(0.0,0.0);
        return;
    }

    const double one = 1.0;

    // 1. Compute ||x||_2
    double xnorm = 0.0;
    cublasDznrm2(handle, n-1, d_x, incx, &xnorm);

    // 2. Copy alpha to host
    cuDoubleComplex alpha;
    cudaMemcpy(&alpha, d_alpha, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    double alphr = cuCreal(alpha);
    double alphi = cuCimag(alpha);

    if (xnorm == 0.0 && alphi == 0.0) {
        *h_tau = make_cuDoubleComplex(0.0, 0.0);
    } else {
        // General case
        cuDoubleComplex tau;
        double beta_real = -copysign(dlapy3(alphr, alphi, xnorm), alphr);
        // 3. Machine safe min
        const double safmin = DBL_MIN / DBL_EPSILON; // LAPACK: dlamch('S') / dlamch('E')
        const double rsafmn = one / safmin;

        // 4. Scale if beta is too small
        int knt = 0;
        if(fabs(beta_real) < safmin){
            while (fabs(beta_real) < safmin && knt < 20) {
                knt = knt + 1;
                cublasZdscal(handle, n - 1, &rsafmn, d_x, incx);
                beta_real *= rsafmn;
                alphr *= rsafmn;
                alphi *= rsafmn;
            }
            cublasDznrm2(handle, n-1, d_x, incx, &xnorm);
            alpha = make_cuDoubleComplex(alphr, alphi);
            beta_real = -copysign(dlapy3(alphr, alphi, xnorm), alphr);
        }

        // 5. Compute tau
        tau = make_cuDoubleComplex((beta_real - alphr)/beta_real, -alphi/beta_real);

        // 6. Scale x = x / (alpha - beta)
        cuDoubleComplex alphambeta = make_cuDoubleComplex(alphr - beta_real, alphi);
        cuDoubleComplex scalefac = zladiv(make_cuDoubleComplex(one,0.0), alphambeta);
        if (n > 1)
            cublasZscal(handle, n-1, &scalefac, d_x, incx);

        // 7. Rescale beta if we had scaling loop
        for (int j=0; j<knt; j++)
            beta_real *= safmin;
        
        cuDoubleComplex beta = make_cuDoubleComplex(beta_real, 0.0);
        // 8. Copy back results
        cudaMemcpy(d_alpha, &beta, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        *h_tau = tau;
    }
}
