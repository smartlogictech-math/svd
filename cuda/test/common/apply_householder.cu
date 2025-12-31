#include "common.cuh"

void apply_householder(
    int n,
    cuDoubleComplex tau,
    const cuDoubleComplex* v,
    const cuDoubleComplex* y,
    cuDoubleComplex* Hy,
    householder_side_t side
)
{
    cuDoubleComplex dot = make_cuDoubleComplex(0.0, 0.0);

    for (int i = 0; i < n; ++i) {
        dot = cuCadd(dot, cuCmul(cuConj(v[i]), y[i]));
    }

    cuDoubleComplex alpha =
        (side == HOUSEHOLDER_LEFT) ? cuConj(tau) : tau;

    for (int i = 0; i < n; ++i) {
        Hy[i] = cuCsub(y[i], cuCmul(alpha, cuCmul(v[i], dot)));
    }
}
