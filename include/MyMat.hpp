#ifndef MYMAT_H
#define MYMAT_H

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

#endif