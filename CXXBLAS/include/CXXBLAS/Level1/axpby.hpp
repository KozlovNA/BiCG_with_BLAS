#ifndef CXXBLAS_LEVEL1_AXPBY_HPP
#define CXXBLAS_LEVEL1_AXPBY_HPP
namespace BLAS
{
    // Adds x scaled by alpha to y:
    //       y = \alpha x + \beta y
    template <class DataType>
        void axpby(const CXXBLAS_INT &n,
                  const DataType &alpha,
                  const DataType *x, const CXXBLAS_INT &incx,
                  const DataType &beta,
                  DataType *y, const CXXBLAS_INT &incy)
        {
#pragma omp parallel for
            for (CXXBLAS_INT i = 0; i < n; i++)
            {
                y[i * incy] *= beta;
                y[i * incy] += x[i * incx] * alpha;
            }
        }
}

#endif
