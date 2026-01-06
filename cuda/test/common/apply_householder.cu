#include "host.cuh"

#include <stdio.h>

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
