#ifndef BBCG_HPP
#define BBCG_HPP

#include<vector>
#include<string>
#include<complex>
#include<CXXBLAS.hpp>
// #include<lapacke.h>
#include<memory>
#include<chrono>
#include<fstream>
// #include<cblas.h>
#include<BlasLapackInterface.hpp>

//------------------------
// Block BCGSTAB
//------------------------
template<class AT, class VT>
void bbcgsr(const AT      &A,
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
  
  std::ofstream logs("../output/BBCGSR_logs.csv", std::ios::out | std::ios::trunc);
  logs << "k,res_max2norm_rel,matvec_count\n";

  std::ofstream alpha_trcon1_out("../output/alpha_rcon_i.csv", std::ios::out | std::ios::trunc);
  alpha_trcon1_out << "k,alpha_trcon1\n";

  std::ofstream beta_trcon1_out("../output/beta_rcon_i.csv", std::ios::out | std::ios::trunc);
  beta_trcon1_out << "k,beta_trcon1\n";

  auto start = std::chrono::high_resolution_clock::now();

  //variables needed in main loop
  std::vector<T> Vk(N*s);
  
  std::vector<T> alpha(s*s);
  std::vector<T> alpha_system(s*s);

  std::vector<T> Tk(N*s);
  T omegak;
  T sum_omegak;

  std::vector<T> Sk(N*s);
  std::vector<T> Reor_helper(N*s);
  std::vector<T> Wk(N*s);

  //initializing algorythm
  //R_0 = B - A X 
  std::vector<T> Rk(N*s);
  bmatvec(A, X, N, s, Rk);
  matvec_count+=s;

  //orthorgonizing R_0
  std::unique_ptr<T[]> tau(new T[s]);
  BLAS::rscal(N*s, -1.0, Rk.data(),1);
  BLAS::axpy(N*s, one, B.data(),1, Rk.data(),1);
  LAPACKE::geqrf(LAPACK_COL_MAJOR, N,s,Rk.data(),N,tau.get());
  LAPACKE::gqr(LAPACK_COL_MAJOR, N,s,s,Rk.data(),N,tau.get());

  //R0c = R_0 
  std::vector<T> R0c(N*s);
  BLAS::copy(N*s, Rk.data(), 1, R0c.data(), 1);
  // for(int i = 0; i < N*s; i++)
  // {
  //   R0c[i] = std::conj(R0c[i]);  
  // }

  //P_0 = R_0
  std::vector<T> Pk(N*s);
  BLAS::copy(N*s, Rk.data(), 1, Pk.data(), 1);
  //P^hat = A**H R0c
  std::vector<T> P_hat(N*s);
  bcmatvec(A, R0c, N,s, P_hat);

  //TODO: alpha_system = beta system, so change qr_solve to utilize QR that you already found


  //main loop
  for (int k = 0; k < (N+s-1)/s; k++)
  {
    //Pk -> Pk * U^{-1}
    // LAPACKE::geqrf(LAPACK_COL_MAJOR, N,s,Pk.data(),N,tau.get());
    // LAPACKE::gqr(LAPACK_COL_MAJOR, N,s,s,Pk.data(),N,tau.get());
    //obtain (P^hat**H P_k)**-1
    std::vector<T> Eines(s*s);
    for (int i = 0; i < s; i++) Eines[s*i + i] = 1;
    CBLAS::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                s, s, N,
                &one, P_hat.data(), N, Pk.data(), N,
                &zero, alpha_system.data(), s);
    qr_solve<T>(LAPACK_COL_MAJOR, s,s,s, alpha_system.data(), Eines.data());
    //P_k = P_k (P^hat**H P_k)**-1
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                N,s,s,
                &one, Pk.data(),N, Eines.data(),s,
                &zero, Tk.data(), N);
    BLAS::copy(N*s, Tk.data(),1, Pk.data(),1);            
    //V_k = A P_k
    bmatvec(A, Pk, N,s, Vk);
    
    //calculating alpha with reorthogonalization
    //alpha_system = R_0c**H V_k
    CBLAS::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, 
                s, s, N, 
                &one, R0c.data(), N, Vk.data(), N, 
                &zero, alpha_system.data(), s);
    //alpha_rhs = R0с**H R_k
    CBLAS::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, 
                s, s, N, 
                &one, R0c.data(), N, Rk.data(), N,
                &zero, alpha.data(), s);
    //solve (R0c**H Vk) alpha_k = R0c**H Rk 
    qr_solve<T>(LAPACK_COL_MAJOR, s, s, s, alpha_system.data(), alpha.data());

    //S_k = R_k - V_k alpha_k
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                N, s, s,
                &m_one, Vk.data(), N, alpha.data(), s,
                &one, Rk.data(), N);
    //X += P_k alpha
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                N, s, s,
                &one, Pk.data(), N, alpha.data(), s,
                &one, X.data(), N);  

    //alpha_system = R_0c**H V_k
    CBLAS::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, 
                s, s, N, 
                &one, R0c.data(), N, Vk.data(), N, 
                &zero, alpha_system.data(), s);
    //alpha_rhs = R0с**H S_k
    CBLAS::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, 
                s, s, N, 
                &one, R0c.data(), N, Rk.data(), N,
                &zero, alpha.data(), s);
    //solve (R0c**H Vk) alpha_k = R0c**H Rk 
    qr_solve<T>(LAPACK_COL_MAJOR, s, s, s, alpha_system.data(), alpha.data(), alpha_trcon1_out, k, 'I');
    //X += P_k alpha
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                N, s, s,
                &one, Pk.data(), N, alpha.data(), s,
                &one, X.data(), N);
    //S_k = R_k - V_k alpha_k
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                N, s, s,
                &m_one, Vk.data(), N, alpha.data(), s,
                &one, Rk.data(), N);            
  //   //------check--------
  // // if (k = 0){
  //// std::cout << "\ncheck=";
  //// for (auto v: Vk)
  //// {
  // //   std::cout << v << " ";
  // // }
  // // std::cout << "\n\n";//}
  // // //---------------------  

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
    if (rk_max2norm_rel < eps){
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << "\n\n" << "total time: "
                << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()
                << " mcs\n\n";
      break;
    }
    //------

    //T_k = A S_k                
    bmatvec(A, Rk, N,s, Tk);

    //calculating omega with reorthogonalization
    BLAS::copy(N*s, Rk.data(),1, Sk.data(),1);
    //omega_k = <T_k, S_k>_F / <T_k, T_k>_F            
    omegak = BLAS::dotc(N*s, Tk.data(), 1, Rk.data(), 1)/
             BLAS::dotc(N*s, Tk.data(), 1, Tk.data(), 1);
    sum_omegak = omegak;
    //X_(k+1) += omega_k S_k               
    BLAS::axpy(N*s, omegak, Sk.data(), 1, X.data(), 1);
    //R_(k+1) = Sk - omega_k T_k
    BLAS::axpy(N*s, -omegak, Tk.data(), 1, Rk.data(), 1);
    //omega_k = <T_k, R_{k+1}>_F / <T_k, T_k>_F            
    omegak = BLAS::dotc(N*s, Tk.data(), 1, Rk.data(), 1)/
             BLAS::dotc(N*s, Tk.data(), 1, Tk.data(), 1);
    sum_omegak+=omegak;
    //X_(k+1) += omega_k S_k               
    BLAS::axpy(N*s, omegak, Sk.data(), 1, X.data(), 1);
    //R_(k+1) -= omega_k T_k
    BLAS::axpy(N*s, -omegak, Tk.data(), 1, Rk.data(), 1);

    //output 1
    matvec_count+=s;
    rk_max2norm_rel = max2norm(CblasColMajor, N, s, Rk.data())/
                     max2norm(CblasColMajor, N, s,R0c.data());
    std::cout << "step: " << k + 1
              << ", ||Sk||_max2norm / ||R0||_max2norm = " << rk_max2norm_rel
              << "\n\n";
    logs << k + 1 << ',' 
         << rk_max2norm_rel << ',' 
         << matvec_count << '\n';
    if (rk_max2norm_rel < eps){
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << "\n\n" << "total time: "
                << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()
                << " mcs\n\n";
      break;
    }
    //------

    //calculating beta with reorthogonalization
    //beta_system = alpha_system
    CBLAS::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, 
                s, s, N, 
                &one, R0c.data(), N, Vk.data(), N, 
                &zero, alpha_system.data(), s);
    //beta_rhs = - R0с**H T_k
    CBLAS::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, 
                s, s, N, 
                &m_one, R0c.data(), N, Tk.data(), N,
                &zero, alpha.data(), s);
    qr_solve<T>(LAPACK_COL_MAJOR, s, s, s, alpha_system.data(), alpha.data());
    //P_{k+1} = S_k + P_k beta_k
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                N, s, s,
                &one, Pk.data(), N, alpha.data(), s,
                &zero, Reor_helper.data(), N);
    BLAS::axpy(N*s, 1.0,Sk.data(),1, Reor_helper.data(),1);
    //W_k = T_k + V_k beta_k
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                N, s, s,
                &one, Vk.data(), N, alpha.data(), s,
                &zero, Wk.data(), N);
    BLAS::axpy(N*s, 1.0, Tk.data(),1, Wk.data(),1);
    //beta_system = alpha_system
    CBLAS::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, 
                s, s, N, 
                &one, R0c.data(), N, Vk.data(), N, 
                &zero, alpha_system.data(), s);
    //beta_rhs = - R0с**H T_k
    CBLAS::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, 
                s, s, N, 
                &m_one, R0c.data(), N, Wk.data(), N,
                &zero, alpha.data(), s);
    qr_solve<T>(LAPACK_COL_MAJOR, s, s, s, alpha_system.data(), alpha.data(), beta_trcon1_out, k, '1');
    //P_{k+1} += Pk beta_k
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                N, s, s,
                &one, Pk.data(), N, alpha.data(), s,
                &one, Reor_helper.data(), N);
    //W_k += V_k beta_k            
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                N, s, s,
                &one, Vk.data(), N, alpha.data(), s,
                &one, Wk.data(), N); 
    //P_{k+1} -= omega_k W_k                
    BLAS::axpy(N*s, -sum_omegak,Wk.data(),1, Reor_helper.data(),1);     
    BLAS::copy(N*s, Reor_helper.data(), 1, Pk.data(),1);         
} 

}

//-------------------
// auxilary functions
//-------------------

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

template<class T>
void trcon_write(int k, T *A, int n, int m, std::ofstream &out, char norm)
{ 
  double trcon_v = 2.0;
  LAPACKE::trcon(LAPACK_COL_MAJOR, norm, 'U', 'N', n, A, m, &trcon_v);
  out << k << "," << trcon_v << "\n";
} 

#endif