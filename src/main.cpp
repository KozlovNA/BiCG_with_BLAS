#include<iostream>
#include<cblas.h>
#include<Eigen/Dense>
#include<vector>
#include<BCG.hpp>
#include<MyMat.hpp>
#include<fbinio.hpp>
#include<BBCG.hpp>
int main()
{
  //-----------------------------------------------
  //! simplest test on 3x3 matrix
  // using VectorType = Eigen::VectorXd;
  // using MatrixType = MyMatrixX<double>;

  // MatrixType A(3,3);
  // A << 2, 3, 5,
  //      3, 7, 4,
  //      1, 2, 2;

  // VectorType b(6);
  // b << 10, 3, 3, 0, 1, 0;

  // VectorType x(6);
  // x << 0, 0, 0, 0, 0, 0;

  // bbcg<MatrixType, VectorType>(A, b,3,2, x, 0.001);
  // std::cout << x << "\n"; 
  // precise solution: 3, -2, 2
  //-----------------------------------------------
  //! simplest test on 3x3 matrix
  // using VectorType = Eigen::VectorXcf;
  // using MatrixType = MyMatrixX<std::complex<float>>;
  // using T = std::complex<float>;

  // MatrixType A(2,2);
  // A << T(1, 1), T(-1, 0),
  //      T(1,-1), T(1, 1);

  // VectorType b(2);
  // b << T(0,1), T(1,0);

  // VectorType x(2);
  // x << T(0,0), T(0,0);

  // bbcg<MatrixType, VectorType>(A, b, 2, 1, x, 0.1);
  // std::cout << x << "\n"; 
  //precise solution: 3, -2, 2
  //-----------------------------------------------
  //! test on complex matrix 14k x 14k and 1 rhs with single precision
  // using VectorType = Eigen::VectorXcf;
  // using MatrixType = MyMatrixX<std::complex<float>>;
  // MatrixType rhs(3, 3); // "3" is arbitrary, in read_binary() function it will
  // MatrixType A(3,3);    // be resized

  // read_binary("/home/starman/rhs_alm_722.dat", rhs);
  // read_binary("/home/starman/mat_alm_full.dat", A);
  // VectorType b = rhs.col(0);
  // std::cout << "A is " << A.rows() << "x" << A.cols() << "\n\n";
  // std::cout << "b is " << b.rows() << "x" << b.cols() << "\n\n";
  // VectorType x = VectorType::Zero(b.rows()); 
  // bcg<MatrixType, VectorType>(A, b, x, 0.01);
  // write_binary("../output/BiCGSTAB_solution.dat", x, b.rows(), b.cols());
  //-----------------------------------------------
  //! test on complex matrix 14k x 14k and 1 rhs with double precision
  // using MatrixType = MyMatrixX<std::complex<float>>;
  // using MatrixType2 = MyMatrixX<std::complex<double>>;
  // MatrixType rhs(3, 3); // "3" is arbitrary, in read_binary() function it will
  // MatrixType A_d(3,3);    // be resized

  // read_binary("/home/starman/rhs_alm_722.dat", rhs);
  // read_binary("/home/starman/mat_alm_full.dat", A_d);
  // MatrixType2 A = A_d.cast<std::complex<double>>();  
  // Eigen::VectorXcd b = rhs.col(0).cast<std::complex<double>>();
  // std::cout << "A is " << A.rows() << "x" << A.cols() << "\n\n";
  // std::cout << "b is " << b.rows() << "x" << b.cols() << "\n\n";
  // Eigen::VectorXcd x = Eigen::VectorXcd::Zero(b.rows()); 
  // ebcg<MatrixType2, Eigen::VectorXcd>(A, b, x, 0.01);
  // write_binary("../output/BiCGSTAB_solution.dat", x, b.rows(), b.cols());
  //-----------------------------------------------
  //! bbcg test on complex matrix 14k x 14k and 1 rhs with single precision
  using RHSType = Eigen::MatrixXcf;
  using MatrixType = MyMatrixX<std::complex<float>>;
  using VectorType = Eigen::VectorXcf;
  RHSType rhs(3, 3); // "3" is arbitrary, in read_binary() function it will
  MatrixType A(3,3);    // be resized

  read_binary("/home/starman/rhs_alm_722.dat", rhs);
  read_binary("/home/starman/mat_alm_full.dat", A);
  // VectorType c1 = rhs.col(0);
  // VectorType c2 = rhs.col(1);
  RHSType B = rhs.leftCols(2);//(rhs.rows(), 2);
  // B.col(0) = c1;
  // B.col(1) = c2;
  std::cout << "A is " << A.rows() << "x" << A.cols() << "\n\n";
  std::cout << "b is " << B.rows() << "x" << B.cols() << "\n\n";
  int N = B.rows();
  int s = B.cols();
  RHSType X = RHSType::Zero(N, s);

  VectorType X_v = X.reshaped();
  VectorType B_v = B.reshaped();

  bbcg<MatrixType, VectorType>(A, B_v,N,s, X_v, 0.01);
  // write_binary("../output/BiCGSTAB_solution.dat", X, B.rows(), B.cols());
  return 0;
}