#include<fstream>
#include<vector>
#include<assert.h>
#include<BlasLapackInterface.hpp>


#ifndef STABILITY_CHECL_HPP
#define STABILITY_CHECK_HPP

// writes condition number of triangular matrix into file out
template<class T>
void trcon_write(int k, T *A, int n, int m, std::ofstream &out, char norm)
{ 
  double trcon_v = 2.0;
  LAPACKE::trcon(LAPACK_COL_MAJOR, norm, 'U', 'N', n, A, m, &trcon_v);
  out << k << "," << trcon_v << "\n";
} 

// counts condition number of general matrix A relative to some norm
double gecon_v (std::complex<double> *A, int s, char norm)
{
  std::vector<std::complex<double>> A_clone = {A, A+s*s};
  std::vector<int> Piv(s);
  double rcond = 2.0;
  double anorm = LAPACKE_zlange(LAPACK_COL_MAJOR, norm, s, s, A_clone.data(), s);
  LAPACKE_zgetrf(LAPACK_COL_MAJOR, s,s,A_clone.data(), s, Piv.data());
  LAPACKE_zgecon(LAPACK_COL_MAJOR, norm, s, A_clone.data(), s, anorm, &rcond);
  return rcond;
}
float gecon_v (std::complex<float> *A, int s, char norm)
{
  std::vector<std::complex<float>> A_clone = {A, A+s*s};
  std::vector<int> Piv(std::min(s,s));
  float rcond = 2.0;
  float anorm = LAPACKE_clange(LAPACK_COL_MAJOR, norm, s, s, A_clone.data(), s);
  LAPACKE_cgetrf(LAPACK_COL_MAJOR, s,s,A_clone.data(), s, Piv.data());
  LAPACKE_cgecon(LAPACK_COL_MAJOR, norm, s, A_clone.data(), s, anorm, &rcond);
  return rcond;
}

//counts minimal singular value
float nrmminsv (std::complex<float> *A, int m, int n)
{
  assert(m>0&n>0);
  std::vector<std::complex<float>> A_clone = {A, A + m*n};
  std::vector<float> s(std::min(m, n));
  std::vector<std::complex<float>> U(m * m);
  std::vector<std::complex<float>> VT(n * n);
  char jobz = 'N';
  int lda = m;
  int ldu = m;
  int ldvt = n;
  LAPACKE_cgesdd(LAPACK_COL_MAJOR, jobz, m, n, A_clone.data(), lda, s.data(), U.data(), ldu, VT.data(), ldvt);
  return s[std::min(m,n)-1];
}

double nrmminsv (std::complex<double> *A, int m, int n)
{
  assert(m>0&n>0);
  std::vector<std::complex<double>> A_clone = {A, A + m*n};
  std::vector<double> s(std::min(m, n));
  std::vector<std::complex<double>> U(m * m);
  std::vector<std::complex<double>> VT(n * n);
  char jobz = 'N';
  int lda = m;
  int ldu = m;
  int ldvt = n;
  LAPACKE_zgesdd(LAPACK_COL_MAJOR, jobz, m, n, A_clone.data(), lda, s.data(), U.data(), ldu, VT.data(), ldvt);
  return s[std::min(m,n)-1];
}

//counts maximal singular value
float nrmmaxsv (std::complex<float> *A, int m, int n)
{
  std::vector<std::complex<float>> A_clone = {A, A + m*n};
  std::vector<float> s(std::min(m, n));
  std::vector<std::complex<float>> U(m * m);
  std::vector<std::complex<float>> VT(n * n);
  char jobz = 'N';
  int lda = m;
  int ldu = m;
  int ldvt = n;
  LAPACKE_cgesdd(LAPACK_COL_MAJOR, jobz, m, n, A_clone.data(), lda, s.data(), U.data(), ldu, VT.data(), ldvt);
  return s[0];
}

double nrmmaxsv (std::complex<double> *A, int m, int n)
{
  std::vector<std::complex<double>> A_clone = {A, A + m*n};
  std::vector<double> s(std::min(m, n));
  std::vector<std::complex<double>> U(m * m);
  std::vector<std::complex<double>> VT(n * n);
  char jobz = 'N';
  int lda = m;
  int ldu = m;
  int ldvt = n;
  LAPACKE_zgesdd(LAPACK_COL_MAJOR, jobz, m, n, A_clone.data(), lda, s.data(), U.data(), ldu, VT.data(), ldvt);
  return s[0];
}

#endif