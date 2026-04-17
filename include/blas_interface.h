#ifndef AMIGO_BLAS_INTERFACE_H
#define AMIGO_BLAS_INTERFACE_H

#include <complex>

extern "C" {

// Find the argmax
extern int idamax_(const int* n, const double* a, const int* inc);

// Copy values y = x
extern void dcopy_(const int* n, const double* x, const int* incx, double* y,
                   const int* incy);

// Swap values
extern void dswap_(const int* n, double* x, const int* incx, double* y,
                   const int* incy);

// Scale the entries in x by alpha
extern void dscal_(const int* n, const double* alpha, double* x,
                   const int* incx);

// Compute C := alpha*A*A**T + beta*C or C := alpha*A**T*A + beta*C
extern void dsyrk_(const char* uplo, const char* trans, const int* n,
                   const int* k, const double* alpha, const double* a,
                   const int* lda, const double* beta, double* c,
                   const int* ldc);

// Compute C := alpha*op( A )*op( B ) + beta*C,
extern void dgemm_(const char* ta, const char* tb, const int* m, const int* n,
                   const int* k, const double* alpha, const double* a,
                   const int* lda, const double* b, const int* ldb,
                   const double* beta, double* c, const int* ldc);

// Compute y = alpha*op(A)*x + beta*y
extern void dgemv_(const char* trans, const int* m, const int* n,
                   const double* alpha, const double* A, const int* lda,
                   const double* x, const int* incx, const double* beta,
                   double* y, const int* incy);

// Solve A*x = b or A^T*x = b where A is in packed format
extern void dtpsv_(const char* uplo, const char* transa, const char* diag,
                   const int* n, const double* a, double* x, const int* incx);

// Solve A*x = b or A^T*x = b where A is in packed format
extern void dtptrs_(const char* uplo, const char* transa, const char* diag,
                    const int* n, const int* nrhs, const double* a, double* b,
                    const int* ldb, int* info);

// Solve the equations op( A )*X = alpha*B or X*op( A ) = alpha*B,
extern void dtrsm_(const char* side, const char* uplo, const char* transa,
                   const char* diag, const int* m, const int* n,
                   const double* alpha, const double* A, const int* lda,
                   double* B, const int* ldb);

// Factorization of positive definite matrix in packed storage
extern void dpptrf_(const char* c, const int* n, double* ap, int* info);

// Factorization of a positive definite matrix in general storage
extern void dpotrf_(const char* uplo, const int* n, double* a, const int* lda,
                    int* info);

// Symmetric LDL factorization
extern void dsytrf_(const char* uplo, const int* n, double* A, const int* lda,
                    int* ipiv, double* work, const int* lwork, int* info);

// Solve the system with the LDL factorization
extern void dsytrs_(const char* uplo, const int* n, const int* nrhs,
                    const double* a, const int* lda, const int* ipiv, double* b,
                    const int* ldb, int* info);

// Find the argmax
extern int izamax_(const int* n, const std::complex<double>* a, const int* inc);

// Copy values y = x
extern void zcopy_(const int* n, const std::complex<double>* x, const int* incx,
                   std::complex<double>* y, const int* incy);

// Swap values
extern void zswap_(const int* n, std::complex<double>* x, const int* incx,
                   std::complex<double>* y, const int* incy);

// Scale the entries in x by alpha
extern void zscal_(const int* n, const std::complex<double>* alpha,
                   std::complex<double>* x, const int* incx);

// Compute C := alpha*A*A**T + beta*C or C := alpha*A**T*A + beta*C
extern void zsyrk_(const char* uplo, const char* trans, const int* n,
                   const int* k, const std::complex<double>* alpha,
                   const std::complex<double>* a, const int* lda,
                   const std::complex<double>* beta, std::complex<double>* c,
                   const int* ldc);

// Compute C := alpha*op( A )*op( B ) + beta*C,
extern void zgemm_(const char* ta, const char* tb, const int* m, const int* n,
                   const int* k, const std::complex<double>* alpha,
                   const std::complex<double>* a, const int* lda,
                   const std::complex<double>* b, const int* ldb,
                   const std::complex<double>* beta, std::complex<double>* c,
                   const int* ldc);

// Compute y = alpha*op(A)*x + beta*y
extern void zgemv_(const char* trans, const int* m, const int* n,
                   const std::complex<double>* alpha,
                   const std::complex<double>* A, const int* lda,
                   const double* x, const int* incx,
                   const std::complex<double>* beta, std::complex<double>* y,
                   const int* incy);

// Solve A*x = b or A^T*x = b where A is in packed format
extern void ztpsv_(const char* uplo, const char* transa, const char* diag,
                   const int* n, const std::complex<double>* a,
                   std::complex<double>* x, const int* incx);

// Solve A*x = b or A^T*x = b where A is in packed format
extern void ztptrs_(const char* uplo, const char* transa, const char* diag,
                    const int* n, const int* nrhs,
                    const std::complex<double>* a, std::complex<double>* b,
                    const int* ldb, int* info);

// Solve the equations op( A )*X = alpha*B or X*op( A ) = alpha*B,
extern void ztrsm_(const char* side, const char* uplo, const char* transa,
                   const char* diag, const int* m, const int* n,
                   const std::complex<double>* alpha,
                   const std::complex<double>* A, const int* lda,
                   std::complex<double>* B, const int* ldb);

// Factorization of positive definite matrix in packed storage
extern void zpptrf_(const char* c, const int* n, std::complex<double>* ap,
                    int* info);

// Factorization of a positive definite matrix in general storage
extern void zpotrf_(const char* uplo, const int* n, double* a, const int* lda,
                    int* info);

// Symmetric LDL factorization
extern void zsytrf_(const char* uplo, const int* n, double* A, const int* lda,
                    int* ipiv, double* work, const int* lwork, int* info);

// Solve the system with the LDL factorization
extern void zsytrs_(const char* uplo, const int* n, const int* nrhs,
                    const double* a, const int* lda, const int* ipiv, double* b,
                    const int* ldb, int* info);
}

namespace amigo {

template <typename T>
inline int blas_imax(int n, const T* a, int inc) {
  if constexpr (std::is_same<T, double>::value) {
    return idamax_(&n, a, &inc) - 1;
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    return izamax_(&n, a, &inc) - 1;
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_imax only supports double and std::complex<double>");
    return 0;
  }
}

template <typename T>
inline void blas_copy(int n, const T* x, int incx, T* y, int incy) {
  if constexpr (std::is_same<T, double>::value) {
    return dcopy_(&n, x, &incx, y, &incy);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    return zcopy_(&n, x, &incx, y, &incy);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_copy only supports double and std::complex<double>");
  }
}

template <typename T>
inline void blas_swap(int n, T* x, int incx, T* y, int incy) {
  if constexpr (std::is_same<T, double>::value) {
    return dswap_(&n, x, &incx, y, &incy);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    return zswap_(&n, x, &incx, y, &incy);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_swap only supports double and std::complex<double>");
  }
}

template <typename T>
inline void blas_scal(int n, T alpha, T* x, int incx) {
  if constexpr (std::is_same<T, double>::value) {
    return dscal_(&n, &alpha, x, &incx);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    return zscal_(&n, &alpha, x, &incx);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_scal only supports double and std::complex<double>");
  }
}

template <typename T>
inline void blas_syrk(const char* uplo, const char* trans, int n, int k,
                      T alpha, const T* a, int lda, T beta, T* c, int ldc) {
  if constexpr (std::is_same<T, double>::value) {
    dsyrk_(uplo, trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    zsyrk_(uplo, trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_syrk only supports double and std::complex<double>");
  }
}

template <typename T>
inline void blas_gemm(const char* ta, const char* tb, int m, int n, int k,
                      T alpha, const T* a, int lda, const T* b, int ldb, T beta,
                      T* c, int ldc) {
  if constexpr (std::is_same<T, double>::value) {
    dgemm_(ta, tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    zgemm_(ta, tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_gemm only supports double and std::complex<double>");
  }
}

template <typename T>
inline void blas_gemv(const char* ta, int m, int n, T alpha, const T* a,
                      int lda, const T* x, int incx, T beta, T* y, int incy) {
  if constexpr (std::is_same<T, double>::value) {
    dgemv_(ta, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    zgemv_(ta, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_gemv only supports double and std::complex<double>");
  }
}

template <typename T>
inline void blas_tpsv(const char* uplo, const char* transa, const char* diag,
                      int n, const T* a, T* x, int incx) {
  if constexpr (std::is_same<T, double>::value) {
    dtpsv_(uplo, transa, diag, &n, a, x, &incx);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    ztpsv_(uplo, transa, diag, &n, a, x, &incx);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_tpsv only supports double and std::complex<double>");
  }
}

template <typename T>
inline void blas_tptrs(const char* uplo, const char* transa, const char* diag,
                       int n, int nrhs, const T* a, T* x, int ldx, int* info) {
  if constexpr (std::is_same<T, double>::value) {
    dtptrs_(uplo, transa, diag, &n, &nrhs, a, x, &ldx, info);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    ztptrs_(uplo, transa, diag, &n, &nrhs, a, x, &ldx, info);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_tptrs only supports double and std::complex<double>");
  }
}

template <typename T>
inline void blas_trsm(const char* side, const char* uplo, const char* transa,
                      const char* diag, int m, int n, T alpha, const T* A,
                      int lda, T* B, int ldb) {
  if constexpr (std::is_same<T, double>::value) {
    dtrsm_(side, uplo, transa, diag, &m, &n, &alpha, A, &lda, B, &ldb);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    ztrsm_(side, uplo, transa, diag, &m, &n, &alpha, A, &lda, B, &ldb);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_trsm only supports double and std::complex<double>");
  }
}

template <typename T>
inline void lapack_pptrf(const char* c, int n, T* ap, int* info) {
  if constexpr (std::is_same<T, double>::value) {
    dpptrf_(c, &n, ap, info);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    zpptrf_(c, &n, ap, info);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "lapack_pptrf only supports double and std::complex<double>");
  }
}

template <typename T>
inline void lapack_sytrf(const char* uplo, int n, T* a, int lda, int* ipiv,
                         T* work, int lwork, int* info) {
  if constexpr (std::is_same<T, double>::value) {
    dsytrf_(uplo, &n, a, &lda, ipiv, work, &lwork, info);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    zsytrf_(uplo, &n, a, &lda, ipiv, work, &lwork, info);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "lapack_sytrf only supports double and std::complex<double>");
  }
}

template <typename T>
inline void lapack_sytrs(const char* uplo, int n, int nrhs, const T* a, int lda,
                         const int* ipiv, T* b, int ldb, int* info) {
  if constexpr (std::is_same<T, double>::value) {
    dsytrs_(uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    zsytrs_(uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "lapack_sytrs only supports double and std::complex<double>");
  }
}

template <typename T>
inline void lapack_potrf(const char* uplo, int n, T* a, int lda, int* info) {
  if constexpr (std::is_same<T, double>::value) {
    dpotrf_(uplo, &n, a, &lda, info);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    zpotrf_(uplo, &n, a, &lda, info);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "lapack_sytrf only supports double and std::complex<double>");
  }
}

}  // namespace amigo

#endif  // AMIGO_BLAS_INTERFACE_H