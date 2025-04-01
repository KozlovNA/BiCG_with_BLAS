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

#ifndef BBCGSR_HPP
#define BBCGSR_HPP


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
  using U = typename map_type<T>::type;
  T one = 1;
  T m_one = -1;
  T zero = 0;

  //----prepare outputs----//
  int matvec_count = 0;
  double rk_max2norm_rel = 0;
  
  std::ofstream logs("/home/starman/Projects/INM/BiCG_with_BLAS/output/bbcgsr/rrqr_361_rhs_15_picked.csv", std::ios::out | std::ios::trunc);
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
  std::vector<T> alpha_system(s*s);

  std::vector<T> Tk(N*s);
  T omegak;
  T sum_omegak;

  std::vector<T> Sk(N*s);
  std::vector<T> Reor_helper(N*s);
  std::vector<T> Wk(N*s);
  std::vector<T> Rk(N*s);
  std::vector<T> R0(N*s);
  std::vector<T> R0c(N*s);
  std::vector<T> Pk(N*s);
  std::vector<T> P_hat(N*s);

  //----INITIALIZING ALGORITHM-----//
  
  bmatvec(A, X, N, s, Rk);                        //R_0 = B - A X 
  BLAS::rscal(N*s, -1.0, Rk.data(),1);
  BLAS::axpy(N*s, one, B.data(),1, Rk.data(),1);
  matvec_count+=s;  // output
  //----orthorgonizing R_0----//
  /*
  std::unique_ptr<T[]> tau(new T[s]);
  LAPACKE::geqrf(LAPACK_COL_MAJOR, N,s,Rk.data(),N,tau.get());
  LAPACKE::gqr(LAPACK_COL_MAJOR, N,s,s,Rk.data(),N,tau.get());
  */
  // /*
  BLAS::copy(N*s, Rk.data(), 1, R0c.data(), 1);   //R0c = R_0
  BLAS::copy(N*s, Rk.data(), 1, R0.data(), 1);   //R0 = R_0 
  BLAS::copy(N*s, Rk.data(), 1, Pk.data(), 1);    //P_0 = R_0
  bcmatvec(A, R0c, N,s, P_hat);                   //P^hat = A**H R0c
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
    /*
    std::unique_ptr<T[]> tau_PkUm1(new T[s]);
    LAPACKE::geqrf(LAPACK_COL_MAJOR, N,s,Pk.data(),N,tau_PkUm1.get());
    LAPACKE::gqr(LAPACK_COL_MAJOR, N,s,s,Pk.data(),N,tau_PkUm1.get());
    */

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
    CBLAS::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans,   //alpha_rhs = R0с**H R_k
                s, s, N, 
                &one, R0c.data(), N, Rk.data(), N,
                &zero, alpha.data(), s);
    qr_solve<T>(LAPACK_COL_MAJOR, s, s, s, alpha_system.data(), alpha.data()); //solve (R0c**H Vk) alpha_k = R0c**H Rk 

    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,     //S_k = R_k - V_k alpha_k
                N, s, s,
                &m_one, Vk.data(), N, alpha.data(), s,
                &one, Rk.data(), N);
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,     //X += P_k alpha_k
                N, s, s,
                &one, Pk.data(), N, alpha.data(), s,
                &one, X.data(), N);  
    CBLAS::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans,   //alpha_system = R_0c**H V_k
                s, s, N, 
                &one, R0c.data(), N, Vk.data(), N, 
                &zero, alpha_system.data(), s);
    CBLAS::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans,   //alpha_rhs = R0с**H S_k
                s, s, N, 
                &one, R0c.data(), N, Rk.data(), N,
                &zero, alpha.data(), s);
    qr_solve<T>(LAPACK_COL_MAJOR, s, s, s, alpha_system.data(), alpha.data()); //solve (R0c**H Vk) alpha_k = R0c**H Rk 
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,     //X += P_k alpha
                N, s, s,
                &one, Pk.data(), N, alpha.data(), s,
                &one, X.data(), N);
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,     //S_k = R_k - V_k alpha_k
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

    //----output 1/2----//
    matvec_count+=s;
    rk_max2norm_rel = max2norm(CblasColMajor, N, s, Rk.data())/
                     max2norm(CblasColMajor, N, s,R0.data());
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

    //----calculating omega with reorthogonalization----//
    BLAS::copy(N*s, Rk.data(),1, Sk.data(),1);            
    omegak = BLAS::dotc(N*s, Tk.data(), 1, Rk.data(), 1)/   //omega_k = <T_k, S_k>_F / <T_k, T_k>_F
             BLAS::dotc(N*s, Tk.data(), 1, Tk.data(), 1);
    sum_omegak = omegak;             
    BLAS::axpy(N*s, omegak, Sk.data(), 1, X.data(), 1);     //X_(k+1) += omega_k S_k  
    BLAS::axpy(N*s, -omegak, Tk.data(), 1, Rk.data(), 1);   //R_(k+1) = Sk - omega_k T_k           
    omegak = BLAS::dotc(N*s, Tk.data(), 1, Rk.data(), 1)/   //omega_k = <T_k, R_{k+1}>_F / <T_k, T_k>_F 
             BLAS::dotc(N*s, Tk.data(), 1, Tk.data(), 1);
    sum_omegak+=omegak;

    // omega_module_out << k << ","                                            //output 
    //                  << std::abs(sum_omegak)*LAPACKE::lange(LAPACK_COL_MAJOR, 'f', N, s, Tk.data(), N)
    //                  << "\n";
    // omega_real_out << k << "," << std::abs(std::real(sum_omegak)) << "\n";  //output                                                
                   
    BLAS::axpy(N*s, omegak, Sk.data(), 1, X.data(), 1);     //X_(k+1) += omega_k S_k                    
    BLAS::axpy(N*s, -omegak, Tk.data(), 1, Rk.data(), 1);   //R_(k+1) -= omega_k T_k

    //----output 1----//
    matvec_count+=s;
    rk_max2norm_rel = max2norm(CblasColMajor, N, s, Rk.data())/
                     max2norm(CblasColMajor, N, s,R0.data());
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