#include <cuComplex.h>

double znrm2_host(int n, const cuDoubleComplex* x)
{
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double re = cuCreal(x[i]);
        double im = cuCimag(x[i]);
        sum += re*re + im*im;
    }
    return sqrt(sum);
}
