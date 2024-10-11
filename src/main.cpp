#include<iostream>
#include<cblas.h>
#include<Eigen/Dense>
#include<vector>
#include<BCG.hpp>
#include<MyMat.hpp>
int main()
{
  //-----------------------------------------------
  //simplest test on 3x3 matrix
  using VectorType = Eigen::VectorXd;
  using MatrixType = MyMatrixXd;

  MatrixType A(3,3);
  A << 2, 3, 5,
       3, 7, 4,
       1, 2, 2;

  VectorType b(3);
  b << 10, 3, 3;

  VectorType x(3);
  x << 0, 0, 0;

  bcg<MatrixType, VectorType>(A, b, x);
  std::cout << x << "\n"; 
  //precise solution: 3, -2, 2
  //-----------------------------------------------

  return 0;
}