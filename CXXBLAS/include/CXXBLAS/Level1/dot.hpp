#ifndef CXXBLAS_LEVEL1_DOT_HPP
#define CXXBLAS_LEVEL1_DOT_HPP

namespace BLAS
{
    // Computes dot product of conjugated strided vector x by strided vector y
    template <class DataType>
        DataType dot(const CXXBLAS_INT &n,
                  const DataType *x, const CXXBLAS_INT &incx,
                  const DataType *y, const CXXBLAS_INT &incy)
        {
            DataType r(0.0);
#pragma omp parallel
            {
                DataType lr(0.0);
#pragma omp for nowait
                for (CXXBLAS_INT i = 0; i < n; i++)
                {
                    if constexpr (std::is_same_v<DataType, std::complex<float>> ||
                                  std::is_same_v<DataType, std::complex<double>>)
                    {lr += std::conj(x[i * incx]) * y[i * incy];}
                    else 
                    {lr += x[i * incx] * y[i * incy];}
                }
#pragma omp critical
                {
                    r += lr;
                }
            }
            return r;
        }
}

#endif
