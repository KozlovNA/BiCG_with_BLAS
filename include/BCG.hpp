#ifndef BCG_HPP
#define BCG_HPP

#include<vector>
#include<CXXBLAS.hpp>
#include<chrono>
#include<fstream>

template<class MT, class VT>
void bcg (const MT      &A,
          const VT      &b,
          VT            &x,
          const double  &eps)
{
  using T = std::decay<decltype(*x.begin())>::type;

  int n = std::distance(x.begin(), x.end());
  T one = 1;
  T m_one = -1;


  //initializing algorythm 
  std::vector<T> rk(n);
  A.template matvec<VT, std::vector<T>>(x, n, rk);
  BLAS::rscal(n, -1.0,rk.data(),1);
  BLAS::axpy(n, one,b.data(),1, rk.data(),1);

  std::vector<T> pk(n);
  BLAS::copy(n, rk.data(),1, pk.data(),1);

  std::vector<T> r0c(n);
  BLAS::copy(n, rk.data(),1, r0c.data(),1);

  //variables needed in main loop
  std::vector<T> vk(n);
  T alphak;
  std::vector<T> tk(n);
  T wk;
  T betak;
  T betak_denom;
  T omegak;

  //prepare outputs
  int matvec_count = 0;
  double rk_norm_sq_rel = 0;
  
  std::ofstream logs("../output/BCGSTAB_logs.csv", std::ios::out | std::ios::trunc);
  logs << "k,res_2norm_rel,matvec_count\n";

  auto start = std::chrono::high_resolution_clock::now();

  //main loop
  for (int k = 0; k < n; k++)
  {
    A.template matvec<std::vector<T>, std::vector<T>>(pk, n, vk);
    betak_denom = BLAS::dotc(n, r0c.data(),1, rk.data(),1);
    alphak = betak_denom /
             BLAS::dotc(n, r0c.data(),1, vk.data(),1);
    BLAS::axpy(n, -alphak,vk.data(),1, rk.data(),1);

    //output 1/2
    matvec_count++;
    rk_norm_sq_rel = BLAS::nrm2(n, rk.data(), 1)/
                     BLAS::nrm2(n, r0c.data(), 1);
    std::cout << "step: " << float(k) + 0.5
              << ", (sk,sk)^1/2 / (r0,r0)^1/2 = " << rk_norm_sq_rel
              << "\n\n";
    logs << float(k) + 0.5 << ',' 
         << rk_norm_sq_rel << ',' 
         << matvec_count << '\n';
    if (rk_norm_sq_rel < eps){
      BLAS::axpy(n, alphak,pk.data(),1, x.data(),1);
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << "\n\n" << "total time: "
                << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()
                << " mcs\n\n";
      break;
    }
    //------
    
    A.template matvec<std::vector<T>, std::vector<T>>(rk, n, tk);
    omegak = BLAS::dotc(n, tk.data(),1, rk.data(),1) /
             BLAS::dotc(n, tk.data(),1, tk.data(),1);
    BLAS::axpy(n, alphak,pk.data(),1, x.data(),1);
    BLAS::axpy(n, omegak,rk.data(),1, x.data(),1);
    BLAS::axpy(n, -omegak,tk.data(),1, rk.data(),1);

    //output 1
    matvec_count++;
    rk_norm_sq_rel = BLAS::nrm2(n, rk.data(), 1)/
                     BLAS::nrm2(n, r0c.data(), 1);
    std::cout << "step: " << k + 1
              << ", (rk,rk)^1/2 / (r0,r0)^1/2 = " << rk_norm_sq_rel
              << "\n\n";
    logs << k + 1 << ',' 
         << rk_norm_sq_rel << ',' 
         << matvec_count << '\n';
    if (rk_norm_sq_rel < eps)
    {
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "\n\n" << "total time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()
              << " mcs\n\n";   
    break;
    }
    //------

    betak = BLAS::dotc(n, r0c.data(),1, rk.data(),1) * alphak /
            betak_denom / omegak;
    BLAS::scal(n, betak,pk.data(),1);
    BLAS::axpy(n, one,rk.data(),1, pk.data(),1);
    BLAS::axpy(n, -betak*omegak,vk.data(),1, pk.data(),1);
  }
}

#endif