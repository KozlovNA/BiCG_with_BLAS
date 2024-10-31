#ifndef BLASLAPACKINTERFACE_HPP
#define BLASLAPACKINTERFACE_HPP

#include<lapacke.h>
#include<cblas.h>

//-------------------
//LAPACK interface
//-------------------
namespace LAPACKE{

void geqrf(int        matrix_layout, 
           int32_t    m,
           int32_t    n,
           float     *a,
           int32_t    lda,
           float     *tau)
{LAPACKE_sgeqrf(matrix_layout, m, n, a, lda, tau);}

void geqrf(int        matrix_layout, 
           int32_t    m,
           int32_t    n,
           double    *a,
           int32_t    lda,
           double    *tau)
{LAPACKE_dgeqrf(matrix_layout, m, n, a, lda, tau);}

void geqrf(int                     matrix_layout, 
           int32_t                 m,
           int32_t                 n,
           std::complex<float>    *a,
           int32_t                 lda,
           std::complex<float>    *tau)
{LAPACKE_cgeqrf(matrix_layout, m, n, a, lda, tau);}

void geqrf(int                     matrix_layout, 
           int32_t                 m,
           int32_t                 n,
           std::complex<double>   *a,
           int32_t                 lda,
           std::complex<double>   *tau)
{LAPACKE_zgeqrf(matrix_layout, m, n, a, lda, tau);}

void gqr(int matrix_layout, 
         int32_t m,
         int32_t n,
         int32_t k, 
         float *a, 
         int32_t lda, 
         const float *tau)
{LAPACKE_sorgqr(matrix_layout, m, n, k, a, lda, tau);}

void gqr(int matrix_layout, 
         int32_t m,
         int32_t n,
         int32_t k, 
         double *a, 
         int32_t lda, 
         const double *tau)
{LAPACKE_dorgqr(matrix_layout, m, n, k, a, lda, tau);}

void gqr(int matrix_layout, 
         int32_t m,
         int32_t n,
         int32_t k, 
         std::complex<float> *a, 
         int32_t lda, 
         const std::complex<float> *tau)
{LAPACKE_cungqr(matrix_layout, m, n, k, a, lda, tau);}

void gqr(int matrix_layout, 
         int32_t m,
         int32_t n,
         int32_t k, 
         std::complex<double> *a, 
         int32_t lda, 
         const std::complex<double> *tau)
{LAPACKE_zungqr(matrix_layout, m, n, k, a, lda, tau);}

void trtrs( int            matrix_layout, 
            char           uplo,
            char           trans,
            char           diag,
            int32_t        n, 
            int32_t        nrhs, 
            const float   *a, 
            int32_t        lda, 
            float         *b, 
            int32_t        ldb             )
{LAPACKE_strtrs(matrix_layout, uplo, trans, diag, n, nrhs, a, lda, b, ldb);}

void trtrs( int            matrix_layout, 
            char           uplo,
            char           trans,
            char           diag,
            int32_t        n, 
            int32_t        nrhs, 
            const double  *a, 
            int32_t        lda, 
            double        *b, 
            int32_t        ldb             )
{LAPACKE_dtrtrs(matrix_layout, uplo, trans, diag, n, nrhs, a, lda, b, ldb);}

void trtrs( int                          matrix_layout, 
            char                         uplo,
            char                         trans,
            char                         diag,
            int32_t                      n, 
            int32_t                      nrhs, 
            const std::complex<float>   *a, 
            int32_t                      lda, 
            std::complex<float>         *b, 
            int32_t                      ldb             )
{LAPACKE_ctrtrs(matrix_layout, uplo, trans, diag, n, nrhs, a, lda, b, ldb);}

void trtrs( int                           matrix_layout, 
            char                          uplo,
            char                          trans,
            char                          diag,
            int32_t                       n, 
            int32_t                       nrhs, 
            const std::complex<double>   *a, 
            int32_t                       lda, 
            std::complex<double>         *b, 
            int32_t                       ldb             )
{LAPACKE_ztrtrs(matrix_layout, uplo, trans, diag, n, nrhs, a, lda, b, ldb);}

void mqr(int matrix_layout, 
         char side,
         char trans, 
         int32_t m, 
         int32_t n, 
         int32_t k, 
         const double *a, 
         int32_t lda, 
         const double *tau, 
         double *c, 
         int32_t ldc)
{LAPACKE_dormqr(matrix_layout, side, trans, m, n, k, a, lda, tau, c, ldc);}

void mqr(int            matrix_layout, 
         char           side,
         char           trans, 
         int32_t        m, 
         int32_t        n, 
         int32_t        k, 
         const float   *a, 
         int32_t        lda, 
         const float   *tau, 
         float         *c, 
         int32_t        ldc             )
{LAPACKE_sormqr(matrix_layout, side, trans, m, n, k, a, lda, tau, c, ldc);}

void mqr(int                          matrix_layout, 
         char                         side,
         char                         trans, 
         int32_t                      m, 
         int32_t                      n, 
         int32_t                      k, 
         const std::complex<float>   *a, 
         int32_t                      lda, 
         const std::complex<float>   *tau, 
         std::complex<float>         *c, 
         int32_t                      ldc             )
{LAPACKE_cunmqr(matrix_layout, side, trans, m, n, k, a, lda, tau, c, ldc);}

void mqr(int                           matrix_layout, 
         char                          side,
         char                          trans, 
         int32_t                       m, 
         int32_t                       n, 
         int32_t                       k, 
         const std::complex<double>   *a, 
         int32_t                       lda, 
         const std::complex<double>   *tau, 
         std::complex<double>         *c, 
         int32_t                       ldc             )
{LAPACKE_zunmqr(matrix_layout, side, trans, m, n, k, a, lda, tau, c, ldc);}

float lange(int            matrix_layout, 
            char           norm, 
            int32_t        m, 
            int32_t        n, 
            const float   *a,
            int32_t        lda              )
{return LAPACKE_slange(matrix_layout, norm, m, n, a, lda);}

double lange(int            matrix_layout, 
             char           norm, 
             int32_t        m, 
             int32_t        n, 
             const double  *a,
             int32_t        lda              )
{return LAPACKE_dlange(matrix_layout, norm, m, n, a, lda);}

float lange(int                          matrix_layout, 
            char                         norm, 
            int32_t                      m, 
            int32_t                      n, 
            const std::complex<float>   *a,
            int32_t                      lda              )
{return LAPACKE_clange(matrix_layout, norm, m, n, a, lda);}

double lange(int                          matrix_layout, 
             char                         norm, 
             int32_t                      m, 
             int32_t                      n, 
             const std::complex<double>  *a,
             int32_t                      lda             )
{return LAPACKE_zlange(matrix_layout, norm, m, n, a, lda);}

}

namespace CBLAS
{ 
void gemm(CBLAS_ORDER        Order, 
          CBLAS_TRANSPOSE    TransA, 
          CBLAS_TRANSPOSE    TransB, 
          blasint            M, 
          blasint            N, 
          blasint            K, 
          const float       *alpha, 
          const float       *A, 
          blasint            lda, 
          const float       *B, 
          blasint            ldb, 
          const float       *beta, 
          float             *C, 
          blasint            ldc                  )
{cblas_sgemm(Order, TransA, TransB, M, N, K, *alpha, A, lda, B, ldb, *beta, C, ldc);}

void gemm(CBLAS_ORDER        Order, 
          CBLAS_TRANSPOSE    TransA, 
          CBLAS_TRANSPOSE    TransB, 
          blasint            M, 
          blasint            N, 
          blasint            K, 
          const double      *alpha, 
          const double      *A, 
          blasint            lda, 
          const double      *B, 
          blasint            ldb, 
          const double      *beta, 
          double            *C, 
          blasint            ldc                  )
{cblas_dgemm(Order, TransA, TransB, M, N, K, *alpha, A, lda, B, ldb, *beta, C, ldc);}

void gemm(CBLAS_ORDER                     Order, 
          CBLAS_TRANSPOSE                 TransA, 
          CBLAS_TRANSPOSE                 TransB, 
          blasint                         M, 
          blasint                         N, 
          blasint                         K, 
          const std::complex<float>      *alpha, 
          const std::complex<float>      *A, 
          blasint                         lda, 
          const std::complex<float>      *B, 
          blasint                         ldb, 
          const std::complex<float>      *beta, 
          std::complex<float>            *C, 
          blasint                         ldc                  )
{cblas_cgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);}

void gemm(CBLAS_ORDER                     Order, 
          CBLAS_TRANSPOSE                 TransA, 
          CBLAS_TRANSPOSE                 TransB, 
          blasint                         M, 
          blasint                         N, 
          blasint                         K, 
          const std::complex<double>     *alpha, 
          const std::complex<double>     *A, 
          blasint                         lda, 
          const std::complex<double>     *B, 
          blasint                         ldb, 
          const std::complex<double>     *beta, 
          std::complex<double>           *C, 
          blasint                         ldc                  )
{cblas_zgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);}

void trmm(CBLAS_ORDER         Order, 
          CBLAS_SIDE          Side, 
          CBLAS_UPLO          Uplo,
          CBLAS_TRANSPOSE     TransA, 
          CBLAS_DIAG          Diag, 
          blasint             M, 
          blasint             N,
          float              *alpha,
          const float        *A,
          blasint             lda, 
          float              *B, 
          blasint             ldb      ) 
{cblas_strmm(Order, Side, Uplo, TransA, Diag, M, N, *alpha, A, lda, B, ldb);}

void trmm(CBLAS_ORDER         Order, 
          CBLAS_SIDE          Side, 
          CBLAS_UPLO          Uplo,
          CBLAS_TRANSPOSE     TransA, 
          CBLAS_DIAG          Diag, 
          blasint             M, 
          blasint             N,
          double             *alpha,
          const double       *A,
          blasint             lda, 
          double             *B, 
          blasint             ldb      ) 
{cblas_dtrmm(Order, Side, Uplo, TransA, Diag, M, N, *alpha, A, lda, B, ldb);}

void trmm(CBLAS_ORDER                   Order, 
          CBLAS_SIDE                    Side, 
          CBLAS_UPLO                    Uplo,
          CBLAS_TRANSPOSE               TransA, 
          CBLAS_DIAG                    Diag, 
          blasint                       M, 
          blasint                       N,
          std::complex<float>          *alpha,
          const std::complex<float>    *A,
          blasint                       lda, 
          std::complex<float>          *B, 
          blasint                       ldb      ) 
{cblas_ctrmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);}

void trmm(CBLAS_ORDER                   Order, 
          CBLAS_SIDE                    Side, 
          CBLAS_UPLO                    Uplo,
          CBLAS_TRANSPOSE               TransA, 
          CBLAS_DIAG                    Diag, 
          blasint                       M, 
          blasint                       N,
          std::complex<double>         *alpha,
          const std::complex<double>   *A,
          blasint                       lda, 
          std::complex<double>         *B, 
          blasint                       ldb      ) 
{cblas_ztrmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);}

}

#endif