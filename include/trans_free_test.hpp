#include<iostream>
#include<BlasLapackInterface.hpp>

#ifndef PDE_TEST_HPP
#define PDE_TEST_HPP

namespace trans_free_test {

template<typename T>
void define_matrix(T *A, int N) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            *(A + i*N + j) = 0;
        }
    }
    for(int i = 0; i < N-1; i++) { *(A + i*N + i+1) = -1; }
    for(int i = 1; i < N;   i++) { *(A + i*N + i-1) = -2; }
    for(int i = 0; i < N;   i++) { *(A + i*N + i  ) =  2; }
}

template<typename T>
void define_solution(T *v, int N) {
    for(int i = 0; i < N; i++) {*(v + i) = (i < N/2) ? i : N - i;}
}

template<typename T>
void generate_rhs(T* A, T *x, T *rhs, int N) {
    T one(1);
    T zero(0);
    CBLAS::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                N,1,N,
                &one, A,N, x,N,
                &zero, rhs, N);
}

}

#endif