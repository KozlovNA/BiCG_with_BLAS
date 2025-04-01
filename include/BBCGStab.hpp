#ifndef BBCGS_HPP
#define BBCGS_HPP

#include<vector>
#include<CXXBLAS.hpp>
// #include<lapacke.h>
#include<memory>
#include<chrono>
#include<fstream>
// #include<cblas.h>
#include<BlasLapackInterface.hpp>
#include<auxiliary_functions.hpp>

//------------------------
// Block BCGSTAB
//------------------------
template<class AT, class VT>
void bbcgs(const AT      &A,
          const VT      &B,
          const int     &N,
          const int     &s,
          VT            &X,
          const double  &eps)
{
  assert(N >= s);

  using T = std::decay<decltype(*X.begin())>::type;
  T one = 1;
  T m_one = -1;
  T zero = 0;

  //prepare outputs
  int matvec_count = 0;
  double rk_max2norm_rel = 0;
  
  std::ofstream logs("../output/bbsgs/rrqr_361_rhs_20_picked.csv", std::ios::out | std::ios::trunc);
  logs << "k,res_max2norm_rel,matvec_count\n";

  auto start = std::chrono::high_resolution_clock::now();

  //initializing algorythm
  std::unique_ptr<T[]> tau(new T[s]);
  std::vector<T> Rk(N*s);
  bmatvec(A, X, N, s, Rk);
  matvec_count+=s;

  BLAS::rscal(N*s, -1, Rk.data(),1);
  BLAS::axpy(N*s, one, B.data(),1, Rk.data(),1);
  // LAPACKE::geqrf(LAPACK_COL_MAJOR, N,s,Rk.data(),N,tau.get());
  // LAPACKE::gqr(LAPACK_COL_MAJOR, N,s,s,Rk.data(),N,tau.get());

  std::vector<T> Pk(N*s);
  BLAS::copy(N*s, Rk.data(), 1, Pk.data(), 1);

  std::vector<T> R0c(N*s);
  BLAS::copy(N*s, Rk.data(), 1, R0c.data(), 1);

  //variables needed in main loop
  std::vector<T> Vk(N*s);
  
  std::vector<T> alpha(s*s);
  std::vector<T> alpha_system(s*s);

  std::vector<T> Tk(N*s);
  T omegak;

  //TODO: alpha_system = beta system, so change qr_solve to utilize QR that you already found


  //main loop
  for (int k = 0; k < (N+s-1)/s; k++)
  {
    //Pk -> Pk * U^{-1}
    // LAPACKE::geqrf(LAPACK_COL_MAJOR, N,s,Pk.data(),N,tau.get());
    // LAPACKE::gqr(LAPACK_COL_MAJOR, N,s,s,Pk.data(),N,tau.get());
    
    //V_k = A P_k
    bmatvec(A, Pk, N, s, Vk);
    

    //alpha_system = R_0c**H V_k
    CBLAS::gemm(CblasRowMajor, CblasConjNoTrans, CblasTrans, 
                s, s, N, 
                &one, R0c.data(), N, Vk.data(), N, 
                &zero, alpha_system.data(), s);

    //alpha_rhs = R0с**H R_k
    CBLAS::gemm(CblasRowMajor, CblasConjNoTrans, CblasTrans, 
                s, s, N, 
                &one, R0c.data(), N, Rk.data(), N,
                &zero, alpha.data(), s);
  //   //------check--------
  // // if (k = 0){
  //// std::cout << "\ncheck=";
  //// for (auto v: Vk)
  //// {
  // //   std::cout << v << " ";
  // // }
  // // std::cout << "\n\n";//}
  // // //---------------------  
    //solve (R0c**H Vk) alpha_k = R0c**H Rk 
    qr_solve<T>(LAPACK_ROW_MAJOR, s, s, s, alpha_system.data(), alpha.data());


    //S_k = R_k - V_k alpha_k
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasTrans, 
                N, s, s,
                &m_one, Vk.data(), N, alpha.data(), s,
                &one, Rk.data(), N);

    //output 1/2
    matvec_count+=s;
    rk_max2norm_rel = max2norm(CblasColMajor, N, s, Rk.data())/
                     max2norm(CblasColMajor, N, s,R0c.data());
    std::cout << "step: " << float(k) + 0.5
              << ", ||Sk||_max2norm / ||R0||_max2norm = " << rk_max2norm_rel
              << "\n\n";
    logs << float(k) + 0.5 << ',' 
         << rk_max2norm_rel << ',' 
         << matvec_count << '\n';
    if (k % 5 == 0) logs.flush();
    if (rk_max2norm_rel < eps){
      CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasTrans, 
                  N, s, s,
                  &one, Pk.data(), N, alpha.data(), s,
                  &one, X.data(), N);
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << "\n\n" << "total time: "
                << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()
                << " mcs\n\n";
      break;
    }
    //------

    //T_k = A S_k                
    bmatvec(A, Rk, N,s, Tk);
    //omega_k = <T_k, S_k>_F / <T_k, T_k>_F            
    omegak = BLAS::dotc(N*s, Tk.data(), 1, Rk.data(), 1)/BLAS::dotc(N*s, Tk.data(), 1, Tk.data(), 1);
    //X_(k+1) = X_k + P_k alpha_k
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasTrans, 
                N, s, s,
                &one, Pk.data(), N, alpha.data(), s,
                &one, X.data(), N);
    //X_(k+1) += omega_k S_k               
    BLAS::axpy(N*s, omegak, Rk.data(), 1, X.data(), 1);
    //R_(k+1) = Sk - omega_k T_k
    BLAS::axpy(N*s, -omegak, Tk.data(), 1, Rk.data(), 1);

    //output 1/2
    matvec_count+=s;
    rk_max2norm_rel = max2norm(CblasColMajor, N, s, Rk.data())/
                     max2norm(CblasColMajor, N, s,R0c.data());
    std::cout << "step: " << k + 1
              << ", ||Sk||_max2norm / ||R0||_max2norm = " << rk_max2norm_rel
              << "\n\n";
    logs << k + 1 << ',' 
         << rk_max2norm_rel << ',' 
         << matvec_count << '\n';
    if (k % 5 == 0) logs.flush();
    if (rk_max2norm_rel < eps){
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << "\n\n" << "total time: "
                << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()
                << " mcs\n\n";
      break;
    }
    //------

    //beta_system = alpha_system
    CBLAS::gemm(CblasRowMajor, CblasConjNoTrans, CblasTrans, 
                s, s, N, 
                &one, R0c.data(), N, Vk.data(), N, 
                &zero, alpha_system.data(), s);
    //beta_rhs = - R0с**H T_k
    CBLAS::gemm(CblasRowMajor, CblasConjNoTrans, CblasTrans, 
                s, s, N, 
                &m_one, R0c.data(), N, Tk.data(), N,
                &zero, alpha.data(), s);
    qr_solve<T>(LAPACK_ROW_MAJOR, s, s, s, alpha_system.data(), alpha.data());
    //P_k = R_(k+1) + (P(k) - omega_k V_k) * beta_k
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasTrans, 
                N, s, s,
                &one, Pk.data(), N, alpha.data(), s,
                &zero, Tk.data(), N);
    BLAS::copy(N*s, Tk.data(),1, Pk.data(),1);
    T m_omegak = -omegak;
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasTrans, 
                N, s, s,
                &m_omegak, Vk.data(), N, alpha.data(), s,
                &one, Pk.data(), N);
    BLAS::axpy(N*s, one,Rk.data(),1, Pk.data(),1);              
} 


}

//-------------------
// auxilary functions
//-------------------

// template<class AT, class VT1, class VT2>
// void bmatvec(const AT      &A,
//              const VT1     &X,
//              const int     &N,
//              const int     &s,
//              VT2           &Res)
// {
//   using T = std::decay<decltype(*X.begin())>::type;
//   std::vector<T> rtmp(N);
//   std::vector<T> xtmp(N);

//   for(int i = 0; i < s; i++)
//   {
//     std::copy(X.begin()+i*N, X.begin()+(i+1)*N, xtmp.begin());
//     A.template matvec(xtmp, N, rtmp);
//     std::copy(rtmp.begin(), rtmp.end(), Res.begin() + i*N);
//   }
// }


// template <class T>
// void qr (T *Q, T *R, T *A, const size_t m, const size_t n) {
//     assert(m >= n);
//     std::size_t k = std::max(std::size_t(1), std::min(m, n)); // The number of elementary reflectors

//     std::unique_ptr<T[]> tau(new T[k]); // Scalars that define elementary reflectors

//     LAPACKE::geqrf(LAPACK_COL_MAJOR, m, n, A, m, tau.get());

//     // Generate R matrix
//     for (std::size_t i(0); i < n; ++i) {
//         std::memset(R + i * m + i + 1,0 , (m-i-1) * sizeof(T));
//         std::memcpy(R + i * m, A + i * m, (i+1) * sizeof(T));
//     }

//     // Generate Q matrix
//     LAPACKE::gqr(LAPACK_COL_MAJOR, m, k, k, A, m, tau.get());

//     if(m == n) {
//         std::memcpy(Q, A, sizeof(T) * (m * m));
//     } else {
//         for(std::size_t i(0); i < k; ++i) {
//             std::memcpy(Q + i * m, A + i * m, sizeof(T) * (m));
//         }
//     }
// }

// // Solves system A X = B with least squares method
// // A - skinny tall m x n matrix 
// // B - rhs matrix m x s
// template <class T>
// void qr_solve(int         matrix_layout, 
//               int32_t     m, 
//               int32_t     n, 
//               int32_t     s,
//               T          *A,
//               T          *B              )
// {
//   char Ad = 'T';
//   if constexpr (std::is_same_v<std::complex<float>, T> ||
//                 std::is_same_v<std::complex<double>, T> ||
//                 std::is_same_v<std::complex<long double>, T>)
//   {
//     Ad = 'C';
//   }
//   // if (matrix_layout == LAPACK_COL_MAJOR){

//   // std::unique_ptr<T[]> TAU(new T[n]);

//   // //performing QR factorization and store it in A in a packed form
//   // LAPACKE::geqrf(LAPACK_COL_MAJOR, m, n, A, m, TAU);
//   // //substituting B with Q**H B 
//   // LAPACKE::mqr(LAPACK_COL_MAJOR, 'L', Ad, m, s, n, A, m, TAU, B, m);

//   // //cutting the system to get RX = Q_cut**H*B
//   // for (int i = 0; i < n; i++)
//   //   std::memcpy(A + i*n, A + i*m, (i+1)*sizeof(T));
//   // for (int i = 0; i < s; i++)
//   //   std::memcpy(B + i*n, B + i*m, (n)*sizeof(T));

//   // std::memcpy(Eigen_R.data(), R.get(), (n*n)*sizeof(T));
//   // LAPACKE::trtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N', n, s, R.get(), n, QhB.get(), n); 
//   //}
//   assert(matrix_layout == LAPACK_ROW_MAJOR);
//   std::unique_ptr<T[]> TAU(new T[n]);
//   LAPACKE::geqrf(LAPACK_ROW_MAJOR, m, n, A, n, TAU.get());
//   LAPACKE::mqr(LAPACK_ROW_MAJOR, 'L', Ad, m, s, n, A, n, TAU.get(), B, s);
//   LAPACKE::trtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'N', n, s, A, n, B, s); 
// } 

// float max2norm(int matrix_layout,
//                int N, int s,
//                const float *Mat)
// {
//   assert(matrix_layout == LAPACK_COL_MAJOR);
//   std::vector<float> R_norms(s, 0);
//   for (int i = 0; i < s; i++)
//   {
//     R_norms[i] = BLAS::nrm2(N, Mat + i*N,1);
//   }
//   float res_2norm_max = *std::max_element(R_norms.begin(), R_norms.end());
//   return res_2norm_max;
// }

// double max2norm(int matrix_layout,
//                int N, int s,
//                const double *Mat)
// {
//   assert(matrix_layout == LAPACK_COL_MAJOR);
//   std::vector<double> R_norms(s, 0);
//   for (int i = 0; i < s; i++)
//   {
//     R_norms[i] = BLAS::nrm2(N, Mat + i*N,1);
//   }
//   double res_2norm_max = *std::max_element(R_norms.begin(), R_norms.end());
//   return res_2norm_max;
// }

// float max2norm(int matrix_layout,
//                int N, int s,
//                const std::complex<float> *Mat)
// {
//   assert(matrix_layout == LAPACK_COL_MAJOR);
//   std::vector<float> R_norms(s, 0);
//   for (int i = 0; i < s; i++)
//   {
//     R_norms[i] = BLAS::nrm2(N, Mat + i*N,1);
//   }
//   float res_2norm_max = *std::max_element(R_norms.begin(), R_norms.end());
//   return res_2norm_max;
// }

// double max2norm(int matrix_layout,
//                int N, int s,
//                const std::complex<double> *Mat)
// {
//   assert(matrix_layout == LAPACK_COL_MAJOR);
//   std::vector<double> R_norms(s, 0);
//   for (int i = 0; i < s; i++)
//   {
//     R_norms[i] = BLAS::nrm2(N, Mat + i*N,1);
//   }
//   double res_2norm_max = *std::max_element(R_norms.begin(), R_norms.end());
//   return res_2norm_max;
// }
#endif