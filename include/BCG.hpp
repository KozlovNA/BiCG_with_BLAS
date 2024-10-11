#ifndef BCG_H
#define BCG_H

#include<vector>

template<class MT, class VT>
void bcg (const MT  &A,
          const VT  &b,
          VT        &x)
{
  using T = std::decay<decltype(*x.begin())>::type;

  int n = std::distance(x.begin(), x.end());

  //initializing algorythm 
  std::vector<T> rk(n);
  A.template matvec<VT, std::vector<T>>(x, n, rk);
  cblas_daxpby(n, 1.0,b.data(),1, -1.0,rk.data(),1);

  std::vector<T> pk(n);
  cblas_dcopy(n, rk.data(),1, pk.data(),1);

  std::vector<T> r0c(n);
  cblas_dcopy(n, rk.data(),1, r0c.data(),1);

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
    betak_denom = cblas_ddot(n, r0c.data(),1, rk.data(),1);
    alphak = betak_denom /
             cblas_ddot(n, r0c.data(),1, vk.data(),1);
    cblas_daxpby(n, -alphak,vk.data(),1, 1.0,rk.data(),1);
    A.template matvec<std::vector<T>, std::vector<T>>(rk, n, tk);
    omegak = cblas_ddot(n, tk.data(),1, rk.data(),1) /
             cblas_ddot(n, tk.data(),1, tk.data(),1);
    cblas_daxpby(n, alphak,pk.data(),1, 1.0,x.data(),1);
    cblas_daxpby(n, omegak,rk.data(),1, 1.0,x.data(),1);
    cblas_daxpby(n, -omegak,tk.data(),1, 1.0,rk.data(),1);
    betak = cblas_ddot(n, r0c.data(),1, rk.data(),1) * alphak /
            betak_denom / omegak;
    cblas_daxpby(n, 1.0,rk.data(),1, betak,pk.data(),1);
    cblas_daxpby(n, -betak*omegak,vk.data(),1, 1.0,pk.data(),1);
  }
}

#endif