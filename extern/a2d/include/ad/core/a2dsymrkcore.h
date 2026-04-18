#ifndef A2D_SYMMAT_RK_CORE_H
#define A2D_SYMMAT_RK_CORE_H

#include "../../a2ddefs.h"

namespace A2D {

/*
  Compute the following:

  if op(A) == NORMAL:     S = A * A^{T}
  if op(A) == TRANSPOSE:  S = A^{T} * A
*/
template <typename T, int N, int K, MatOp op = MatOp::NORMAL,
          bool additive = false>
A2D_FUNCTION void SymMatRKCore(const T A[], T S[]) {
  if constexpr (op == MatOp::NORMAL) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j <= i; j++) {
        const T* a = &A[K * i];
        const T* b = &A[K * j];

        T val = 0.0;
        for (int k = 0; k < K; k++) {
          val += a[0] * b[0];
          a++, b++;
        }
        if constexpr (additive) {
          S[0] += val;
        } else {
          S[0] = val;
        }
        S++;
      }
    }
  } else {
    for (int i = 0; i < K; i++) {
      for (int j = 0; j <= i; j++) {
        const T* a = &A[i];
        const T* b = &A[j];

        T val = 0.0;
        for (int k = 0; k < N; k++) {
          val += a[0] * b[0];
          a += K, b += K;
        }
        if constexpr (additive) {
          S[0] += val;
        } else {
          S[0] = val;
        }
        S++;
      }
    }
  }
}

/*
  Compute the following:

  if op == NORMAL:     S = alpha * A * A^{T}
  if op == TRANSPOSE:  S = alpha * A^{T} * A
*/
template <typename T, int N, int K, MatOp op = MatOp::NORMAL,
          bool additive = false>
A2D_FUNCTION void SymMatRKCoreScale(const T alpha, const T A[], T S[]) {
  if constexpr (op == MatOp::NORMAL) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j <= i; j++) {
        const T* a = &A[K * i];
        const T* b = &A[K * j];

        T val = 0.0;
        for (int k = 0; k < K; k++) {
          val += a[0] * b[0];
          a++, b++;
        }
        if constexpr (additive) {
          S[0] += alpha * val;
        } else {
          S[0] = alpha * val;
        }
        S++;
      }
    }
  } else {
    for (int i = 0; i < K; i++) {
      for (int j = 0; j <= i; j++) {
        const T* a = &A[i];
        const T* b = &A[j];

        T val = 0.0;
        for (int k = 0; k < N; k++) {
          val += a[0] * b[0];
          a += K, b += K;
        }
        if constexpr (additive) {
          S[0] += alpha * val;
        } else {
          S[0] = alpha * val;
        }
        S++;
      }
    }
  }
}

/*
  Compute the following:

  if op == NORMAL:       S = A * B^{T} + B * A^{T}
  else op == TRANSPOSE:  S = A^{T} * B + B^{T} * A^{T}
*/
template <typename T, int N, int K, MatOp op = MatOp::NORMAL,
          bool additive = false>
A2D_FUNCTION void SymMatR2KCore(const T A[], const T B[], T S[]) {
  if constexpr (op == MatOp::NORMAL) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j <= i; j++) {
        T val = 0.0;

        const T* a = &A[K * i];
        const T* b = &B[K * j];
        for (int k = 0; k < K; k++) {
          val += a[0] * b[0];
          a++, b++;
        }

        a = &A[K * j];
        b = &B[K * i];
        for (int k = 0; k < K; k++) {
          val += a[0] * b[0];
          a++, b++;
        }

        if constexpr (additive) {
          S[0] += val;
        } else {
          S[0] = val;
        }
        S++;
      }
    }
  } else {
    for (int i = 0; i < K; i++) {
      for (int j = 0; j <= i; j++) {
        T val = 0.0;

        const T* a = &A[i];
        const T* b = &B[j];
        for (int k = 0; k < N; k++) {
          val += a[0] * b[0];
          a += K, b += K;
        }

        a = &A[j];
        b = &B[i];
        for (int k = 0; k < N; k++) {
          val += a[0] * b[0];
          a += K, b += K;
        }

        if constexpr (additive) {
          S[0] += val;
        } else {
          S[0] = val;
        }
        S++;
      }
    }
  }
}

/*
  Compute the following:

  if op == NORMAL:       S = alpha * (A * B^{T} + B * A^{T})
  else op == TRANSPOSE:  S = alpha * (A^{T} * B + B^{T} * A^{T})
*/
template <typename T, int N, int K, MatOp op = MatOp::NORMAL,
          bool additive = false>
A2D_FUNCTION void SymMatR2KCoreScale(const T alpha, const T A[], const T B[],
                                     T S[]) {
  if constexpr (op == MatOp::NORMAL) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j <= i; j++) {
        T val = 0.0;

        const T* a = &A[K * i];
        const T* b = &B[K * j];
        for (int k = 0; k < K; k++) {
          val += a[0] * b[0];
          a++, b++;
        }

        a = &A[K * j];
        b = &B[K * i];
        for (int k = 0; k < K; k++) {
          val += a[0] * b[0];
          a++, b++;
        }

        if constexpr (additive) {
          S[0] += alpha * val;
        } else {
          S[0] = alpha * val;
        }
        S++;
      }
    }
  } else {
    for (int i = 0; i < K; i++) {
      for (int j = 0; j <= i; j++) {
        T val = 0.0;

        const T* a = &A[i];
        const T* b = &B[j];
        for (int k = 0; k < N; k++) {
          val += a[0] * b[0];
          a += K, b += K;
        }

        a = &A[j];
        b = &B[i];
        for (int k = 0; k < N; k++) {
          val += a[0] * b[0];
          a += K, b += K;
        }

        if constexpr (additive) {
          S[0] += alpha * val;
        } else {
          S[0] = alpha * val;
        }
        S++;
      }
    }
  }
}

template <typename T, int N, int K, MatOp op = MatOp::NORMAL>
A2D_FUNCTION void SymMatRKCoreReverse(const T A[], const T Sb[], T Ab[]) {
  if constexpr (op == MatOp::NORMAL) {
    // Ab = Sb * A
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < K; j++) {
        int k = 0;
        const T* s = &Sb[i * (i + 1) / 2];
        const T* a = &A[j];

        T val = 0.0;
        for (; k < i; k++) {
          val += s[0] * a[0];
          a += K, s++;
        }

        for (; k < N; k++) {
          val += s[0] * a[0];
          a += K, s += k + 1;
        }

        val += A[K * i + j] * Sb[i + i * (i + 1) / 2];

        Ab[0] += val;
        Ab++;
      }
    }
  } else {  // op == MatOp::TRANSPOSE
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < K; j++) {
        int k = 0;
        const T* a = &A[K * i];
        const T* s = &Sb[j * (j + 1) / 2];

        T val = 0.0;
        for (; k < j; k++) {
          val += s[0] * a[0];
          a++, s++;
        }

        for (; k < K; k++) {
          val += s[0] * a[0];
          a++, s += k + 1;
        }

        val += A[K * i + j] * Sb[j + j * (j + 1) / 2];

        Ab[0] += val;
        Ab++;
      }
    }
  }
}

template <typename T, int N, int K, MatOp op = MatOp::NORMAL>
A2D_FUNCTION void SymMatRKCoreReverseScale(const T alpha, const T A[],
                                           const T Sb[], T Ab[]) {
  if constexpr (op == MatOp::NORMAL) {
    // Ab = Sb * A
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < K; j++) {
        int k = 0;
        const T* s = &Sb[i * (i + 1) / 2];
        const T* a = &A[j];

        T val = 0.0;
        for (; k < i; k++) {
          val += s[0] * a[0];
          a += K, s++;
        }

        for (; k < N; k++) {
          val += s[0] * a[0];
          a += K, s += k + 1;
        }

        val += A[K * i + j] * Sb[i + i * (i + 1) / 2];

        Ab[0] += alpha * val;
        Ab++;
      }
    }
  } else {  // op == MatOp::TRANSPOSE
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < K; j++) {
        int k = 0;
        const T* a = &A[K * i];
        const T* s = &Sb[j * (j + 1) / 2];

        T val = 0.0;
        for (; k < j; k++) {
          val += s[0] * a[0];
          a++, s++;
        }

        for (; k < K; k++) {
          val += s[0] * a[0];
          a++, s += k + 1;
        }

        val += A[K * i + j] * Sb[j + j * (j + 1) / 2];

        Ab[0] += alpha * val;
        Ab++;
      }
    }
  }
}

}  // namespace A2D

#endif  // A2D_SYMMAT_RK_CORE_H