#include "common.cuh"

void apply_householder(
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
