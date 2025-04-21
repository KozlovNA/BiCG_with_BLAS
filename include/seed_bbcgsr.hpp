#include<vector>
#include<string>
#include<complex>
#include<algorithm>
#include<fstream>
#include<memory>
#include<assert.h>
#include<chrono>
#include<CXXBLAS.hpp>
#include<BlasLapackInterface.hpp>
#include<auxiliary_functions.hpp>
#include<complex_map_type.hpp>

#ifndef SEED_HPP
#define SEED_HPP
//------------------------
// Block BCGSTAB
//------------------------
template<class AT, class VT>
void sbbcgsr(const AT      &A,
          const VT      &B,
          const VT      &B_other,
          const int     &N,
          const int     &s,
          const int     &s_other,
          VT            &X,
          VT            &X_other,
          const double  &eps)
{
  assert(N >= s);

  using T = std::decay<decltype(*X.begin())>::type;
  using U = typename map_type<T>::type;
  T one = 1;
  T m_one = -1;
  T zero = 0;

  //----prepare outputs----//
  int matvec_count = 0;
  double rk_max2norm_rel = 0;
  std::ofstream logs("/home/starman/Projects/INM/BiCG_with_BLAS/output/sbbcgsr/principal test.csv", std::ios::out | std::ios::trunc);
  logs << "k,res_max2norm_rel,matvec_count\n";

  // std::ofstream alpha_trcon1_out("../output/alpha_rcon_i.csv", std::ios::out | std::ios::trunc);
  // alpha_trcon1_out << "k,alpha_trcon1\n";

  // std::ofstream beta_trcon1_out("../output/beta_rcon_i.csv", std::ios::out | std::ios::trunc);
  // beta_trcon1_out << "k,beta_trcon1\n";

  // std::ofstream PhatPk_gecon_out("../output/PhatPk_gecon_1.csv", std::ios::out | std::ios::trunc);
  // PhatPk_gecon_out << "k,gecon\n";

  // std::ofstream omega_module_out("../output/omega_module.csv", std::ios::out | std::ios::trunc);
  // omega_module_out << "k,module\n";

  // std::ofstream omega_real_out("../output/omega_real.csv", std::ios::out | std::ios::trunc);
  // omega_real_out << "k,real\n";  

  // std::ofstream R_nrmminsv("../output/R_nrmminsv.csv", std::ios::out | std::ios::trunc);
  // R_nrmminsv << "k,v\n";

  // std::ofstream R_nrmmaxsv("../output/R_nrmmaxsv.csv", std::ios::out | std::ios::trunc);
  // R_nrmmaxsv << "k,v\n";

  // std::ofstream P_nrmsv("../output/P_nrmsv.csv", std::ios::out | std::ios::trunc);
  // P_nrmsv << "k,min,max\n";

  // std::ofstream Phat_nrmsv("../output/Phat_nrmsv.csv", std::ios::out | std::ios::trunc);
  // Phat_nrmsv << "k,min,max\n";

  auto start = std::chrono::high_resolution_clock::now();

  //----variables needed in main loop---//
  std::vector<T> Vk(N*s);
  
  std::vector<T> alpha(s*s);
  std::vector<T> alpha_other(s*s_other);
  std::vector<T> alpha_system(s*s);

  std::vector<T> Tk(N*s);
  std::vector<T> Tk_other(N*s_other);
  T omegak;
  T omegak_other;
  T sum_omegak;
  T sum_omegak_other;

  std::vector<T> Sk(N*s);
  std::vector<T> Sk_other(N*s);
  std::vector<T> Reor_helper(N*s);
  std::vector<T> Wk(N*s);
  std::vector<T> Rk(N*s);
  std::vector<T> Rk_other(N*s_other);
  std::vector<T> R0(N*s);
  std::vector<T> R0_other(N*s_other);
  std::vector<T> R0c(N*s);
  std::vector<T> Pk(N*s);
  std::vector<T> P_hat(N*s);

  
  
  //----INITIALIZING ALGORITHM-----//
  
  bmatvec(A, X, N, s, Rk);                              //R_0 = B - A X 
  BLAS::rscal(N*s, -1.0, Rk.data(),1);
  BLAS::axpy(N*s, one, B.data(),1, Rk.data(),1);
  bmatvec(A, X_other, N, s_other, Rk_other);                  //R_0_other = B_other - A X_other
  BLAS::rscal(N*s_other, -1.0, Rk_other.data(),1);
  BLAS::axpy(N*s_other, one, B_other.data(),1, Rk_other.data(),1);
  
  matvec_count+=s;  // output
  // /*
  BLAS::copy(N*s, Rk.data(), 1, R0c.data(), 1);   //R0c = R_0
  BLAS::copy(N*s, Rk.data(), 1, R0.data(), 1);    //R0 = R_0 
  BLAS::copy(N*s_other, Rk_other.data(), 1, R0_other.data(), 1);    //R0_other = R_0 
  BLAS::copy(N*s, Rk.data(), 1, Pk.data(), 1);    //P_0 = R_0
  // bcmatvec(A, R0c, N,s, P_hat);                   //P^hat = A**H R0c
  // */
  //----orthorgonizing R_0c----//
  // /*
  std::unique_ptr<T[]> tau(new T[s]);
  LAPACKE::geqrf(LAPACK_COL_MAJOR, N,s,R0c.data(),N,tau.get());
  LAPACKE::gqr(LAPACK_COL_MAJOR, N,s,s,R0c.data(),N,tau.get());
  // */
  //----P_hat = QR, P_hat -> Q; R0c -> R0c R**-1----//
  /*
  std::unique_ptr<T[]> tau_hat(new T[N]);
  T trsm_one(1.0);
  LAPACKE::geqrf(LAPACK_COL_MAJOR, N, s, P_hat.data(), N, tau_hat.get());      //P_hat = QR
  CBLAS::trsm<T>(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, 
                 N,s, &trsm_one,                                               //R0c -> R0c R**-1
                 P_hat.data(),N,
                 R0c.data(),N);
  bcmatvec(A, R0c, N,s, P_hat);                                                //P^hat = A**H R0c
  */

  //TODO: alpha_system = beta system, so change qr_solve to utilize QR that you already found
  //main loop
  for (int k = 0; k < (N+s-1)/s; k++)
  {
    //----Pk -> Pk * U^{-1}----//
    // /*
    std::unique_ptr<T[]> tau_PkUm1(new T[s]);
    LAPACKE::geqrf(LAPACK_COL_MAJOR, N,s,Pk.data(),N,tau_PkUm1.get());
    LAPACKE::gqr(LAPACK_COL_MAJOR, N,s,s,Pk.data(),N,tau_PkUm1.get());
    // */

    //----output minimal singular value of R_k----//
    /*
    R_nrmminsv << k << "," << nrmminsv(Rk.data(), N, s) << "\n";  
    R_nrmmaxsv << k << "," << nrmmaxsv(Rk.data(), N, s) << "\n";  
    output minimal and maximal singular value of R_k
    P_nrmsv << k << "," << nrmminsv(Pk.data(), N, s) << "," << nrmmaxsv(Pk.data(), N, s) << "\n";  
    */

    //-----obtain (P^hat**H P_k)**-1----//
    /*
    std::vector<T> Eines(s*s, 0);                                              //rhs = I     
    for (int i = 0; i < s; i++) Eines[s*i + i] = 1;
    CBLAS::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans,                   //system = P^hat**H P_k
                s, s, N,
                &one, P_hat.data(), N, Pk.data(), N,
                &zero, alpha_system.data(), s);
    
    //PhatPk_gecon_out << k << "," << gecon_v(alpha_system.data(), s,'1') << "\n";  //output

    qr_solve<T>(LAPACK_COL_MAJOR, s,s,s, alpha_system.data(), Eines.data());   //solve  (P^hat**H P_k) Z = I
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,                     //P_k -> P_k (P^hat**H P_k)**-1
                N,s,s,
                &one, Pk.data(),N, Eines.data(),s,
                &zero, Tk.data(), N);
    BLAS::copy(N*s, Tk.data(),1, Pk.data(),1);  
    */              

    // Phat_nrmsv << k << "," << nrmminsv(Pk.data(), N, s) << "," << nrmmaxsv(Pk.data(), N, s) << "\n"; //output 

    bmatvec(A, Pk, N,s, Vk); //V_k = A P_k

    //----calculating alpha with reorthogonalization----//
    CBLAS::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans,   //alpha_system = R_0c**H V_k
                s, s, N, 
                &one, R0c.data(), N, Vk.data(), N, 
                &zero, alpha_system.data(), s);
                
    std::vector<T> Eines(s*s, 0);                                              //rhs = I
    for (int i = 0; i < s; i++) Eines[s*i + i] = 1;

    qr_solve<T>(LAPACK_COL_MAJOR, s, s, s, alpha_system.data(), Eines.data()); //solve (R0c**H Vk) sol = I 
    
    std::vector<T> alpha_tmp(N*s, 0);
    std::vector<T> alpha_other_tmp(N*s_other, 0);

    CBLAS::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans,   //alpha_rhs = R0с**H R_k
                s, s, N, 
                &one, R0c.data(), N, Rk.data(), N,
                &zero, alpha_tmp.data(), s);
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,     //alpha = ...
                s, s, s, 
                &one, Eines.data(), s, alpha_tmp.data(), s,
                &zero, alpha.data(), s);

    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,     //S_k = R_k - V_k alpha_k
                N, s, s,
                &m_one, Vk.data(), N, alpha.data(), s,
                &one, Rk.data(), N);

    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,     //X += P_k alpha_k
                N, s, s,
                &one, Pk.data(), N, alpha.data(), s,
                &one, X.data(), N);  

    CBLAS::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans,   //alpha_rhs = R0с**H S_k
                s, s, N, 
                &one, R0c.data(), N, Rk.data(), N,
                &zero, alpha_tmp.data(), s);

    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,     //alpha = ...
                s, s, s, 
                &one, Eines.data(), s, alpha_tmp.data(), s,
                &zero, alpha.data(), s);

    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,     //X += P_k alpha
                N, s, s,
                &one, Pk.data(), N, alpha.data(), s,
                &one, X.data(), N);
                
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,     //S_k = R_k - V_k alpha_k
                N, s, s,
                &m_one, Vk.data(), N, alpha.data(), s,
                &one, Rk.data(), N);
//other residuals
    std::vector<T> alpha_other_system(N*s, 0);
    BLAS::copy(N*s, Vk.data(),1, alpha_other_system.data(),1); //alpha_other_system = Vk

    BLAS::copy(N*s_other, Rk_other.data(), 1, alpha_other_tmp.data(), 1); //alpha_other_rhs = R_k_other

    qr_solve<T>(LAPACK_COL_MAJOR, N, s, s_other, alpha_other_system.data(), alpha_other_tmp.data()); //solve MSE V_k alpha_other_k = R_other_k
    
    for (int i = 0; i < s; i++) {                           //alpha_other = ...
      for (int j = 0; j < s_other; j++) {
        alpha_other[i*s + j] = alpha_other_tmp[i*N + j];   
      }
    }

    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,     //S_k_other = R_k_other - V_k alpha_k_other
                N, s_other, s,
                &m_one, Vk.data(), N, alpha_other.data(), s,
                &one, Rk_other.data(), N);

    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,     //X_other += P_k alpha_k_other
                N, s_other, s,
                &one, Pk.data(), N, alpha_other.data(), s,
                &one, X_other.data(), N);  

  //   //------check--------
  // // if (k = 0){
  //// std::cout << "\ncheck=";
  //// for (auto v: Vk)
  //// {
  // //   std::cout << v << " ";
  // // }
  // // std::cout << "\n\n";//}
  // // //---------------------  

    //----output 1/2----//
    matvec_count+=s;
    rk_max2norm_rel = std::max(max2norm(CblasColMajor, N, s, Rk.data()),max2norm(CblasColMajor, N, s_other, Rk_other.data()))/
                      std::max(max2norm(CblasColMajor, N, s, R0.data()),max2norm(CblasColMajor, N, s_other, R0_other.data()));
    // rk_max2norm_rel = max2norm(CblasColMajor, N, s, Rk.data())/
                      // max2norm(CblasColMajor, N, s,R0.data());
    std::cout << "step: " << float(k) + 0.5
              << ", ||Sk||_max2norm / ||R0||_max2norm = " << rk_max2norm_rel
              << "\n\n";
    logs << float(k) + 0.5 << ',' 
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
    //------------------//
                
    bmatvec(A, Rk, N,s, Tk);  //T_k = A S_k
    bmatvec(A, Rk_other, N,s_other, Tk_other);  //T_other_k = A S_other_k

    //----calculating omega with reorthogonalization----//
    BLAS::copy(N*s, Rk.data(),1, Sk.data(),1); 
    BLAS::copy(N*s_other, Rk_other.data(),1, Sk_other.data(),1);            
    omegak = BLAS::dotc(N*s, Tk.data(), 1, Rk.data(), 1)/   //omega_k = <T_k, S_k>_F / <T_k, T_k>_F
             BLAS::dotc(N*s, Tk.data(), 1, Tk.data(), 1);
    omegak_other = BLAS::dotc(N*s_other, Tk_other.data(), 1, Rk_other.data(), 1)/   //omega_k = <T_k, S_k>_F / <T_k, T_k>_F
                   BLAS::dotc(N*s_other, Tk_other.data(), 1, Tk_other.data(), 1);
    sum_omegak = omegak;             
    sum_omegak_other = omegak_other;             
    BLAS::axpy(N*s, omegak, Sk.data(), 1, X.data(), 1);     //X_(k+1) += omega_k S_k
    BLAS::axpy(N*s_other, omegak_other, Sk_other.data(), 1, X_other.data(), 1);     //X_other_(k+1) += omega_k S_other_k  
    BLAS::axpy(N*s, -omegak, Tk.data(), 1, Rk.data(), 1);   //R_(k+1) = Sk - omega_k T_k           
    BLAS::axpy(N*s_other, -omegak_other, Tk_other.data(), 1, Rk_other.data(), 1);   //R_other_(k+1) = Sk_other - omega_other_k T_other_k           
    omegak = BLAS::dotc(N*s, Tk.data(), 1, Rk.data(), 1)/   //omega_k = <T_k, R_{k+1}>_F / <T_k, T_k>_F 
             BLAS::dotc(N*s, Tk.data(), 1, Tk.data(), 1);
    omegak_other = BLAS::dotc(N*s_other, Tk_other.data(), 1, Rk_other.data(), 1)/   //omega_other_k = <T_other_k, R_other_{k+1}>_F / <T_other_k, T_other_k>_F 
                   BLAS::dotc(N*s_other, Tk_other.data(), 1, Tk_other.data(), 1);
    sum_omegak+=omegak;
    sum_omegak_other+=omegak_other;

    // omega_module_out << k << ","                                            //output 
    //                  << std::abs(sum_omegak)*LAPACKE::lange(LAPACK_COL_MAJOR, 'f', N, s, Tk.data(), N)
    //                  << "\n";
    // omega_real_out << k << "," << std::abs(std::real(sum_omegak)) << "\n";  //output                                                
                   
    BLAS::axpy(N*s, omegak, Sk.data(), 1, X.data(), 1);     //X_(k+1) += omega_k S_k                    
    BLAS::axpy(N*s, -omegak, Tk.data(), 1, Rk.data(), 1);   //R_(k+1) -= omega_k T_k
    BLAS::axpy(N*s_other, omegak_other, Sk_other.data(), 1, X_other.data(), 1);     //X_(k+1) += omega_k S_k OTHER                    
    BLAS::axpy(N*s_other, -omegak_other, Tk_other.data(), 1, Rk_other.data(), 1);   //R_(k+1) -= omega_k T_k OTHER

    //----output 1----//
    matvec_count+=s;
    rk_max2norm_rel = std::max(max2norm(CblasColMajor, N, s, Rk.data()),max2norm(CblasColMajor, N, s_other, Rk_other.data()))/
                      std::max(max2norm(CblasColMajor, N, s,R0.data()), max2norm(CblasColMajor, N, s_other,R0_other.data()));
    // rk_max2norm_rel = max2norm(CblasColMajor, N, s, Rk.data())/
                      // max2norm(CblasColMajor, N, s, R0.data());
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
    //----------------//

    //----calculating beta with reorthogonalization----//
    CBLAS::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans,      //beta_system = alpha_system
                s, s, N, 
                &one, R0c.data(), N, Vk.data(), N, 
                &zero, alpha_system.data(), s);
    CBLAS::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans,      //beta_rhs = - R0с**H T_k
                s, s, N, 
                &m_one, R0c.data(), N, Tk.data(), N,
                &zero, alpha.data(), s);
    qr_solve<T>(LAPACK_COL_MAJOR, s, s, s, alpha_system.data(), alpha.data());
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,         //P_{k+1} = S_k + P_k beta_k
                N, s, s,
                &one, Pk.data(), N, alpha.data(), s,
                &zero, Reor_helper.data(), N);
    BLAS::axpy(N*s, 1.0,Sk.data(),1, Reor_helper.data(),1);
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,         //W_k = T_k + V_k beta_k
                N, s, s,
                &one, Vk.data(), N, alpha.data(), s,
                &zero, Wk.data(), N);
    BLAS::axpy(N*s, 1.0, Tk.data(),1, Wk.data(),1);
    CBLAS::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans,       //beta_system = alpha_system
                s, s, N, 
                &one, R0c.data(), N, Vk.data(), N, 
                &zero, alpha_system.data(), s);
    CBLAS::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans,       //beta_rhs = - R0с**H T_k
                s, s, N, 
                &m_one, R0c.data(), N, Wk.data(), N,
                &zero, alpha.data(), s);
    qr_solve<T>(LAPACK_COL_MAJOR, s, s, s, alpha_system.data(), alpha.data());
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,         //P_{k+1} += Pk beta_k
                N, s, s,
                &one, Pk.data(), N, alpha.data(), s,
                &one, Reor_helper.data(), N);           
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,         //W_k += V_k beta_k 
                N, s, s,
                &one, Vk.data(), N, alpha.data(), s,
                &one, Wk.data(), N);                
    BLAS::axpy(N*s, -sum_omegak,Wk.data(),1, Reor_helper.data(),1);//P_{k+1} -= omega_k W_k      
    BLAS::copy(N*s, Reor_helper.data(), 1, Pk.data(),1);         
} 

}




#endif