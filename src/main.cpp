#include<iostream>
#include<cblas.h>
#include<Eigen/Dense>
#include<vector>
#include<BCGStab.hpp>
#include<MyMat.hpp>
#include<fbinio.hpp>
#include<BBCGSR.hpp>
// #include<BBCGStab.hpp>
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

  // bbcg<MatrixType, VectorType>(A, b,3, 2, x, 0.01);
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
  // VectorType b = rhs.col(240);
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
  // using RHSType = Eigen::MatrixXcf;
  // using MatrixType = MyMatrixX<std::complex<float>>;
  // using VectorType = Eigen::VectorXcf;
  // RHSType rhs(3, 3); // "3" is arbitrary, in read_binary() function it will
  // MatrixType A(3,3);    // be resized

  // read_binary("/home/starman/rhs_alm_722.dat", rhs);
  // read_binary("/home/starman/mat_alm_full.dat", A);
  // RHSType B = rhs.leftCols(3);
  // std::cout << "A is " << A.rows() << "x" << A.cols() << "\n\n";
  // std::cout << "b is " << B.rows() << "x" << B.cols() << "\n\n";
  // int N = B.rows();
  // int s = B.cols();
  // RHSType X = RHSType::Zero(N, s);

  // VectorType X_v = X.reshaped();
  // VectorType B_v = B.reshaped();

  // bbcgsr<MatrixType, VectorType>(A, B_v,N,s, X_v, 0.01);
  // write_binary("../output/BiCGSTAB_solution.dat", X, B.rows(), B.cols());
  //-----------------------------------------------
  //! bbcg test on complex matrix 14k x 14k and 1 rhs with single precision
  // using RHSType = Eigen::MatrixXcf;
  // using MatrixType = MyMatrixX<std::complex<float>>;
  // using VectorType = Eigen::VectorXcf;
  // RHSType rhs(3, 3); // "3" is arbitrary, in read_binary() function it will
  // MatrixType A(3,3);    // be resized

  // read_binary("/home/starman/rhs_alm_722.dat", rhs);
  // read_binary("/home/starman/mat_alm_full.dat", A);
  // VectorType c1 = rhs.col(1);
  // VectorType c2 = rhs.col(91);
  // VectorType c3 = rhs.col(181);
  // VectorType c4 = rhs.col(271);

  // RHSType B(rhs.rows(), 3);// = rhs.leftCols(2);//
  // B.col(0) = c1;
  // B.col(1) = c2;
  // B.col(2) = c3;
  // B.col(3) = c4;

  // std::cout << "A is " << A.rows() << "x" << A.cols() << "\n\n";
  // std::cout << "b is " << B.rows() << "x" << B.cols() << "\n\n";
  // int N = B.rows();
  // int s = B.cols();
  // RHSType X = RHSType::Zero(N, s);

  // VectorType X_v = X.reshaped();
  // VectorType B_v = B.reshaped();

  // bbcg<MatrixType, VectorType>(A, B_v,N,s, X_v, 0.01);
  // write_binary("../output/BiCGSTAB_solution.dat", X, B.rows(), B.cols());
  //-----------------------------------------------
  //! bbcg test on complex matrix 14k x 14k and 3 rhs with single precision
  // using RHSType = Eigen::MatrixXcf;
  // using MatrixType = MyMatrixX<std::complex<float>>;
  // using VectorType = Eigen::VectorXcf;
  // RHSType rhs(3, 3); // "3" is arbitrary, in read_binary() function it will
  // MatrixType A(3,3);    // be resized

  // read_binary("/home/starman/rhs_alm_722.dat", rhs);
  // read_binary("/home/starman/mat_alm_full.dat", A);
  
  // int s = 3;
  // int interval = 361/s;
  // int N = rhs.rows();
  // RHSType B(N, s);
  // std::cout << "picked columns: "; 
  // for (int i = 0; i < s; i++)
  // {
  //   std::cout << i*360/s << ", ";
  //   B.col(i) = rhs.col(i*360/s);
  // }
  // std::cout << "\n\n";

  // std::cout << "A is " << A.rows() << "x" << A.cols() << "\n\n";
  // std::cout << "b is " << B.rows() << "x" << B.cols() << "\n\n";
  // RHSType X = RHSType::Zero(N, s);

  // VectorType X_v = X.reshaped();
  // VectorType B_v = B.reshaped();

  // bbcgsr<MatrixType, VectorType>(A, B_v,N,s, X_v, 0.01);
  //-----------------------------------------------
  //! bbcgsr test on complex matrix 14k x 14k and 3 rhs with double precision
  using RHSType = Eigen::MatrixXcf;
  using MatrixType = MyMatrixX<std::complex<float>>;
  using MatrixType2 = MyMatrixX<std::complex<double>>;
  using VectorType = Eigen::VectorXcf;
  using VectorType2 = Eigen::VectorXcd;

  RHSType rhs_s(3, 3); // "3" is arbitrary, in read_binary() function it will
  MatrixType A_s(3,3);    // be resized

  read_binary("/home/starman/rhs_alm_722.dat", rhs_s);
  read_binary("/home/starman/mat_alm_full.dat", A_s);
  
  MatrixType2 A = A_s.cast<std::complex<double>>();
  MatrixType2 rhs = rhs_s.cast<std::complex<double>>();

  int s = 3;
  int interval = 361/s;
  int N = rhs.rows();
  MatrixType2 B(N, s);
  std::cout << "picked columns: "; 
  for (int i = 0; i < s; i++)
  {
    std::cout << i*360/s << ", ";
    B.col(i) = rhs.col(i*360/s);
  }
  std::cout << "\n\n";

  std::cout << "A is " << A.rows() << "x" << A.cols() << "\n\n";
  std::cout << "b is " << B.rows() << "x" << B.cols() << "\n\n";
  MatrixType2 X = MatrixType2::Zero(N, s);

  VectorType2 X_v = X.reshaped();
  VectorType2 B_v = B.reshaped();

  bbcgsr<MatrixType2, VectorType2>(A, B_v,N,s, X_v, 0.01);
  //-----------------------------------------------
  //! bbcgsr simplest test on 3x3 matrix
  /*
  using VectorType = Eigen::VectorXcf;
  using MatrixType = MyMatrixX<std::complex<float>>;
  using T = std::complex<float>;
  MatrixType A(3,3);
  A << T(2,0), T(3,0), T(5,0),
       T(3,0), T(7,0), T(4,0),
       T(1,0), T(2,0), T(2,0);

  VectorType b(6);
  b << T(10,0), T(3,0), T(3,0), T(0,0), T(1,0), T(0,0);

  VectorType x(6);
  x << T(0,0), T(0,0), T(0,0), T(0,0), T(0,0), T(0,0);

  bbcgsr<MatrixType, VectorType>(A, b, 3, 2, x, 0.0001);
  std::cout << x << "\n"; 
  */
  // precise solution: 3, -2, 2; 4, -1, -1 
  //-----------------------------------------------

  return 0;
}