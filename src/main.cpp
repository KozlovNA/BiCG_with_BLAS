#include<iostream>
#include<cblas.h>
#include<Eigen/Dense>
#include<vector>
#include<BCG.hpp>
#include<MyMat.hpp>
int main()
{
  using VectorType = Eigen::VectorXd;
  using MatrixType = MyMatrixXd;

  MatrixType A(3,3);
  A << 1, 2, 3,
       4, 5, 6,
       7, 8, 9;

  VectorType b(3);
  b << 1, 2, 3;

  VectorType x(3);
  x << 4, 5, 6;

  bcg<MatrixType, VectorType>(A, b, x);
  
  return 0;
}