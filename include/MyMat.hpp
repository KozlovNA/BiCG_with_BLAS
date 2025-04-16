#ifndef MYMAT_HPP
#define MYMAT_HPP

#include <Eigen/Dense>

class MyMatrixXd: public Eigen::MatrixXd 
{
public:

  using Eigen::MatrixXd::MatrixXd;

  template<class VT, class RT>
  void matvec(const VT   &x,
              const int  &n,
              RT         &res) const
  {
    
    for (int i = 0; i < n; i++)
    {
      double sum = 0;

      for(int j = 0; j < n; j++)
      {
        sum += (*this).coeff(i,j)*(*(x.begin()+j));
      }

      *(res.begin() + i) = sum;
    }

  }

};

template<typename T>
class MyMatrixX: public Eigen::MatrixX<T> 
{
public:

  using Eigen::MatrixX<T>::MatrixX;

  template<class VT, class RT>
  void matvec(const VT   &x,
              const int  &n,
              RT         &res) const
  {
    T one = 1;
    T zero = 0;
    BLAS::gemv('N', n, n, one, (*this).data(), n, x.data(), 1, zero, res.data(), 1);
  }
  
  template<class VT, class RT>
  void cmatvec(const VT   &x,
               const int  &n,
               RT         &res) const
  {
    T one = 1;
    T zero = 0;
    BLAS::gemv('C', n, n, one, (*this).data(), n, x.data(), 1, zero, res.data(), 1);
  }

};

template<typename ScalarType>
class MyMatrix: public Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> 
{
public:

  using Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>::Matrix;

  template<class VT, class RT>
  void matvec(const VT   &x,
              const int  &n,
              RT         &res) const
  {
    ScalarType one = 1;
    ScalarType zero = 0;
    BLAS::gemv('N', n, n, one, (*this).data(), n, x.data(), 1, zero, res.data(), 1);
  }
  
  template<class VT, class RT>
  void cmatvec(const VT   &x,
               const int  &n,
               RT         &res) const
  {
    ScalarType one = 1;
    ScalarType zero = 0;
    BLAS::gemv('C', n, n, one, (*this).data(), n, x.data(), 1, zero, res.data(), 1);
  }

};

#endif