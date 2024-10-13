#ifndef BCG_H
#define BCG_H

#include<vector>
#include<CXXBLAS.hpp>

template<class MT, class VT>
void bcg (const MT  &A,
          const VT  &b,
          VT        &x)
{
  using T = std::decay<decltype(*x.begin())>::type;

  int n = std::distance(x.begin(), x.end());
  T one = 1;
  T m_one = -1;


  //initializing algorythm 
  std::vector<T> rk(n);
  A.template matvec<VT, std::vector<T>>(x, n, rk);
  BLAS::axpby(n, one,b.data(),1, m_one,rk.data(),1);

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

  //main loop
  for (int k = 0; k < n; k++)
  {
    A.template matvec<std::vector<T>, std::vector<T>>(pk, n, vk);
    betak_denom = BLAS::dot(n, r0c.data(),1, rk.data(),1);
    alphak = betak_denom /
             BLAS::dot(n, r0c.data(),1, vk.data(),1);
    BLAS::axpby(n, -alphak,vk.data(),1, one,rk.data(),1);
    A.template matvec<std::vector<T>, std::vector<T>>(rk, n, tk);
    omegak = BLAS::dot(n, tk.data(),1, rk.data(),1) /
             BLAS::dot(n, tk.data(),1, tk.data(),1);
    BLAS::axpby(n, alphak,pk.data(),1, one,x.data(),1);
    BLAS::axpby(n, omegak,rk.data(),1, one,x.data(),1);
    BLAS::axpby(n, -omegak,tk.data(),1, one,rk.data(),1);
    betak = BLAS::dot(n, r0c.data(),1, rk.data(),1) * alphak /
            betak_denom / omegak;
    BLAS::axpby(n, one,rk.data(),1, betak,pk.data(),1);
    BLAS::axpby(n, -betak*omegak,vk.data(),1, one,pk.data(),1);
  }
}

#endif