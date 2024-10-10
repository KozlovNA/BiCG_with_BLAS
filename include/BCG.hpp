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

  std::vector<T> r0(n);
  A.template matvec<VT, std::vector<T>>(x, n, r0);
  cblas_daxpby(n,1,b.data(),1, -1,r0.data(),1);
  
}

#endif