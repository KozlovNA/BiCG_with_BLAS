#include<iostream>
#include<cblas.h>
#include<Eigen/Dense>
#include<vector>
#include<BCG.hpp>
#include<MyMat.hpp>
#include<fbinio.hpp>
int main()
{
  //-----------------------------------------------
  //simplest test on 3x3 matrix
  // using VectorType = Eigen::VectorXcf;
  // using MatrixType = MyMatrixXcf;

  // MatrixType A(3,3);
  // A << 2, 3, 5,
  //      3, 7, 4,
  //      1, 2, 2;

  // VectorType b(3);
  // b << 10, 3, 3;

  // VectorType x(3);
  // x << 0, 0, 0;

  // bcg<MatrixType, VectorType>(A, b, x, 0.1);
  // std::cout << x << "\n"; 
  //precise solution: 3, -2, 2
  //-----------------------------------------------
  //test on complex matrix 14k x 14k and 1 rhs
  Eigen::MatrixXcf rhs(3, 3); // "3" is arbitrary, in read_binary() function it will
  Eigen::MatrixXcf A(3,3);    // be resized

  read_binary("/home/starman/rhs_alm_722.dat", rhs);
  read_binary("/home/starman/mat_alm_full.dat", A);
  Eigen::VectorXcf b = rhs.col(0);
  std::cout << "A is " << A.rows() << "x" << A.cols() << "\n\n";
  std::cout << "b is " << b.rows() << "x" << b.cols() << "\n\n";
  Eigen::VectorXcf x = Eigen::VectorXcf::Zero(b.rows()); 
  bcg<MyMatrixXcf, Eigen::VectorXcf>(A, b, x, 0.01);
  write_binary("../output/BiCGSTAB_solution.dat", x, b.rows(), b.cols());
  //-----------------------------------------------


  return 0;
}