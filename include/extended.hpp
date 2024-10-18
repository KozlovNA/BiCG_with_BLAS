#ifndef EXTENDED_HPP
#define EXTENDED_HPP

#include<complex>

template<class T>
struct ExtendedTypesMap
{
    typedef void TYPE;
};

template <>
struct ExtendedTypesMap<float> { typedef double TYPE; };

template <>
struct ExtendedTypesMap<double> { typedef long double TYPE; };

template <>
struct ExtendedTypesMap<std::complex<float>> { typedef std::complex<double> TYPE; };

template <>
struct ExtendedTypesMap<std::complex<double>> { typedef std::complex<long double> TYPE; };

template<class T>
using is_complex = std::is_same<std::complex<T>, T>;

template<class T>
using extended = typename ExtendedTypesMap<T>::TYPE;

template <class T>
typename std::enable_if<!is_complex<T>::value, T>::type conj(T a)
{
  return a;
}

template <class T>
typename std::enable_if<is_complex<T>::value, std::complex<T>>::type conj(std::complex<T> a)
{
  return std::conj(a);
}


template <class T>
static inline extended<T> edotc(std::size_t n, const T *x, int64_t incx, const T *y, int64_t incy)
{
using eT = extended<T>;
eT s = 0;
for (std::size_t i = 0; i < n; ++i)
{
s += conj(static_cast<eT>(x[i * incx])) * static_cast<eT>(y[i * incy]);
}
return s;
}

#endif