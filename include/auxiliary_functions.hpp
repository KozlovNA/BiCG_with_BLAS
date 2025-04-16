#include<fstream>
#include<BlasLapackInterface.hpp>
#include<memory>
#include<vector>
#include<algorithm>
#include<CXXBLAS.hpp>
#include<assert.h>

#ifndef AUXILIARLY_FUNCTIONS_HPP
#define AUXILIARLY_FUNCTIONS_HPP



// Solves system A X = B with least squares method
// A - skinny tall m x n matrix 
// B - rhs matrix m x s
template <class T>
void qr_solve(int         matrix_layout, 
              int32_t     m, 
              int32_t     n, 
              int32_t     s,
              T          *A,
              T          *B)
{
  assert(m >= n);
  char Ad = 'T';
  if constexpr (std::is_same_v<std::complex<float>, T> ||
                std::is_same_v<std::complex<double>, T> ||
                std::is_same_v<std::complex<long double>, T>)
  {
    Ad = 'C';
  }
  if (matrix_layout == LAPACK_COL_MAJOR){
    std::unique_ptr<T[]> TAU(new T[n]);
    LAPACKE::geqrf(LAPACK_COL_MAJOR, m, n, A, m, TAU.get());
    LAPACKE::mqr(LAPACK_COL_MAJOR, 'L', Ad, m, s, n, A, m, TAU.get(), B, m);
    LAPACKE::trtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N', n, s, A, m, B, m); 
  }
  if (matrix_layout == LAPACK_ROW_MAJOR){
    std::unique_ptr<T[]> TAU(new T[n]);
    LAPACKE::geqrf(LAPACK_ROW_MAJOR, m, n, A, n, TAU.get());
    LAPACKE::mqr(LAPACK_ROW_MAJOR, 'L', Ad, m, s, n, A, n, TAU.get(), B, s);
    LAPACKE::trtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'N', n, s, A, n, B, s); 
  }
}

template <class T>
void qr_solve(int         matrix_layout, 
              int32_t     m, 
              int32_t     n, 
              int32_t     s,
              T          *A,
              T          *B,
              std::ofstream &out,
              int k,
              char norm)
{
  assert(m >= n);
  assert(matrix_layout==LAPACK_COL_MAJOR);
  char Ad = 'T';
  if constexpr (std::is_same_v<std::complex<float>, T> ||
                std::is_same_v<std::complex<double>, T> ||
                std::is_same_v<std::complex<long double>, T>)
  {
    Ad = 'C';
  }
  if (matrix_layout == LAPACK_COL_MAJOR){
    std::unique_ptr<T[]> TAU(new T[n]);
    LAPACKE::geqrf(LAPACK_COL_MAJOR, m, n, A, m, TAU.get());
    LAPACKE::mqr(LAPACK_COL_MAJOR, 'L', Ad, m, s, n, A, m, TAU.get(), B, m);
    trcon_write<T>(k, A, n, m, out, norm);
    LAPACKE::trtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N', n, s, A, m, B, m); 
  }
} 


//Block generalization of AT::matvec() 
template<class AT, class VT1, class VT2>
void bmatvec(const AT      &A,
             const VT1     &X,
             const int     &N,
             const int     &s,
             VT2           &Res)
{
  using T = std::decay<decltype(*X.begin())>::type;
  std::vector<T> rtmp(N);
  std::vector<T> xtmp(N);

  for(int i = 0; i < s; i++)
  {
    std::copy(X.begin()+i*N, X.begin()+(i+1)*N, xtmp.begin());
    A.template matvec(xtmp, N, rtmp);
    std::copy(rtmp.begin(), rtmp.end(), Res.begin() + i*N);
  }
}

template<class AT, class VT1, class VT2>
void bcmatvec(const AT      &A,
                 const VT1     &X,
                 const int     &N,
                 const int     &s,
                 VT2           &Res)
{
  using T = std::decay<decltype(*X.begin())>::type;
  std::vector<T> rtmp(N);
  std::vector<T> xtmp(N);

  for(int i = 0; i < s; i++)
  {
    std::copy(X.begin()+i*N, X.begin()+(i+1)*N, xtmp.begin());
    A.template cmatvec(xtmp, N, rtmp);
    std::copy(rtmp.begin(), rtmp.end(), Res.begin() + i*N);
  }
}

//Counts maximum euclidian norm of a column in matrix Mat 
float max2norm(int matrix_layout,
               int N, int s,
               const float *Mat)
{
  assert(matrix_layout == LAPACK_COL_MAJOR);
  std::vector<float> R_norms(s, 0);
  for (int i = 0; i < s; i++)
  {
    R_norms[i] = BLAS::nrm2(N, Mat + i*N,1);
  }
  float res_2norm_max = *std::max_element(R_norms.begin(), R_norms.end());
  return res_2norm_max;
}

//Counts maximum euclidian norm of a column in matrix Mat 
double max2norm(int matrix_layout,
               int N, int s,
               const double *Mat)
{
  assert(matrix_layout == LAPACK_COL_MAJOR);
  std::vector<double> R_norms(s, 0);
  for (int i = 0; i < s; i++)
  {
    R_norms[i] = BLAS::nrm2(N, Mat + i*N,1);
  }
  double res_2norm_max = *std::max_element(R_norms.begin(), R_norms.end());
  return res_2norm_max;
}

//Counts maximum euclidian norm of a column in matrix Mat 
float max2norm(int matrix_layout,
               int N, int s,
               const std::complex<float> *Mat)
{
  assert(matrix_layout == LAPACK_COL_MAJOR);
  std::vector<float> R_norms(s, 0);
  for (int i = 0; i < s; i++)
  {
    R_norms[i] = BLAS::nrm2(N, Mat + i*N,1);
  }
  float res_2norm_max = *std::max_element(R_norms.begin(), R_norms.end());
  return res_2norm_max;
}

//Counts maximum euclidian norm of a column in matrix Mat 
double max2norm(int matrix_layout,
               int N, int s,
               const std::complex<double> *Mat)
{
  assert(matrix_layout == LAPACK_COL_MAJOR);
  std::vector<double> R_norms(s, 0);
  for (int i = 0; i < s; i++)
  {
    R_norms[i] = BLAS::nrm2(N, Mat + i*N,1);
  }
  double res_2norm_max = *std::max_element(R_norms.begin(), R_norms.end());
  return res_2norm_max;
}

template<typename T>
void matrix_with_singular_values(T* s, int m, int n){
  
}

#endif