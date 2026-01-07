#include "host.cuh"

#include <stdio.h>

void zlarfg_host(
    int n,
    cuDoubleComplex* alpha,       // v[0], in/out
    cuDoubleComplex* x,           // v[1:], length n-1, in/out
    int incx,
    cuDoubleComplex* tau          // output
)
{
    if (n <= 1) {
        *tau = make_cuDoubleComplex(0.0, 0.0);
        return;
    }

    //------------------------------------------------------------------
    // Compute xnorm = ||x|| (host version of cublasDznrm2)
    //------------------------------------------------------------------
    double xnorm = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        cuDoubleComplex xi = x[i * incx];
        xnorm = hypot(xnorm, cuCabs(xi));
    }

    //------------------------------------------------------------------
    // Fetch alpha
    //------------------------------------------------------------------
    cuDoubleComplex a = *alpha;

    cuDoubleComplex beta, tau_val;

    //------------------------------------------------------------------
    // Case: reflector = I
    //------------------------------------------------------------------
    if (xnorm == 0.0 && cuCimag(a) == 0.0) {
        tau_val = make_cuDoubleComplex(0.0, 0.0);
        beta = a;
    } 
    //------------------------------------------------------------------
    // General case: construct Householder reflector
    //------------------------------------------------------------------
    else {
        double a_abs = cuCabs(a);

        // beta = - sign(real(a)) * sqrt(|a|^2 + ||x||^2)
        double beta_real = -copysign(
            hypot(a_abs, xnorm),
            cuCreal(a)
        );
        beta = make_cuDoubleComplex(beta_real, 0.0);

        // tau = (beta - alpha) / beta
        tau_val = cuCdiv(
            cuCsub(beta, a),
            beta
        );

        // scale = 1 / (alpha - beta)
        cuDoubleComplex denom = cuCsub(a, beta);
        cuDoubleComplex scale =
            cuCdiv(make_cuDoubleComplex(1.0, 0.0), denom);

        // scale x
        for (int i = 0; i < n - 1; ++i) {
            x[i * incx] = cuCmul(x[i * incx], scale);
        }
    }

    //------------------------------------------------------------------
    // Write results back
    //------------------------------------------------------------------
    *alpha = beta;
    *tau   = tau_val;
}

void zlarfv_host(
    int n,
    cuDoubleComplex tau,
    const cuDoubleComplex* v,
    const cuDoubleComplex* y,
    cuDoubleComplex* Hy
)
{
    cuDoubleComplex dot = make_cuDoubleComplex(0.0, 0.0);

    for (int i = 0; i < n; ++i) {
        dot = cuCadd(dot, cuCmul(cuConj(v[i]), y[i]));
    }

    for (int i = 0; i < n; ++i) {
        Hy[i] = cuCsub(y[i], cuCmul(tau, cuCmul(v[i], dot)));
    }
}

void zlarf_host(
    char side,              // 'L' or 'R'
    int m, int n,
    const cuDoubleComplex* v, int incv,
    cuDoubleComplex tau,
    const cuDoubleComplex* C, int ldc,
    cuDoubleComplex* outC
) {
    if (cuCreal(tau) == 0.0 && cuCimag(tau) == 0.0) {
        // outC = C
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                outC[j*ldc + i] = C[j*ldc + i];
        return;
    }

    if (side == 'L' || side == 'l') {
        // outC = (I - tau v v^H) C

        for (int j = 0; j < n; ++j) {
            // dot = v^H * C(:,j)
            cuDoubleComplex dot = make_cuDoubleComplex(0.0, 0.0);
            for (int i = 0; i < m; ++i) {
                dot = cuCadd(dot,
                             cuCmul(cuConj(v[i*incv]),
                                    C[j*ldc + i]));
            }

            // outC(:,j) = C(:,j) - tau * v * dot
            for (int i = 0; i < m; ++i) {
                outC[j*ldc + i] =
                    cuCsub(C[j*ldc + i],
                           cuCmul(tau,
                                  cuCmul(v[i*incv], dot)));
            }
        }

    } else if (side == 'R' || side == 'r') {
        // outC = C (I - tau v v^H)

        for (int i = 0; i < m; ++i) {
            // dot = C(i,:) * v
            cuDoubleComplex dot = make_cuDoubleComplex(0.0, 0.0);
            for (int j = 0; j < n; ++j) {
                dot = cuCadd(dot,
                             cuCmul(C[j*ldc + i],
                                    v[j*incv]));
            }

            // outC(i,:) = C(i,:) - tau * dot * v^H
            for (int j = 0; j < n; ++j) {
                outC[j*ldc + i] =
                    cuCsub(C[j*ldc + i],
                           cuCmul(tau,
                                  cuCmul(dot,
                                         cuConj(v[j*incv]))));
            }
        }

    } else {
        fprintf(stderr, "[zlarf_host] side must be 'L' or 'R'\n");
    }
}
