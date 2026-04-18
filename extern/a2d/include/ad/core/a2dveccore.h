#ifndef A2D_VEC_CORE_H
#define A2D_VEC_CORE_H

#include "../../a2ddefs.h"

namespace A2D {

template <typename T>
A2D_FUNCTION void VecCrossCore(const T x[], const T y[], T v[]) {
  v[0] = x[1] * y[2] - x[2] * y[1];
  v[1] = x[2] * y[0] - x[0] * y[2];
  v[2] = x[0] * y[1] - x[1] * y[0];
}

template <typename T>
A2D_FUNCTION void VecCrossCoreAdd(const T x[], const T y[], T v[]) {
  v[0] += x[1] * y[2] - x[2] * y[1];
  v[1] += x[2] * y[0] - x[0] * y[2];
  v[2] += x[0] * y[1] - x[1] * y[0];
}

template <typename T, int size>
A2D_FUNCTION void VecZeroCore(T A[]) {
  for (int i = 0; i < size; i++) {
    A[0] = T(0.0);
    A++;
  }
}

template <typename T, int size>
A2D_FUNCTION void VecCopyCore(const T A[], T C[]) {
  for (int i = 0; i < size; i++) {
    C[0] = A[0];
    C++, A++;
  }
}

template <typename T, int size>
A2D_FUNCTION void VecScaleCore(const T alpha, const T A[], T C[]) {
  for (int i = 0; i < size; i++) {
    C[0] = alpha * A[0];
    C++, A++;
  }
}

template <typename T, int size>
A2D_FUNCTION void VecAddCore(const T A[], T C[]) {
  for (int i = 0; i < size; i++) {
    C[0] += A[0];
    C++, A++;
  }
}

template <typename T, int size>
A2D_FUNCTION void VecAddCore(const T alpha, const T A[], T C[]) {
  for (int i = 0; i < size; i++) {
    C[0] += alpha * A[0];
    C++, A++;
  }
}

template <typename T, int size>
A2D_FUNCTION T VecDotCore(const T A[], const T B[]) {
  T dot = 0.0;
  for (int i = 0; i < size; i++) {
    dot += A[0] * B[0];
    A++, B++;
  }
  return dot;
}

template <typename T, int size>
A2D_FUNCTION void VecSumCore(const T A[], const T B[], T C[]) {
  for (int i = 0; i < size; i++) {
    C[0] = A[0] + B[0];
    C++, A++, B++;
  }
}

template <typename T, int size>
A2D_FUNCTION void VecSumCore(const T alpha, const T A[], const T beta,
                             const T B[], T C[]) {
  for (int i = 0; i < size; i++) {
    C[0] = alpha * A[0] + beta * B[0];
    C++, A++, B++;
  }
}

template <typename T, int size>
A2D_FUNCTION void VecHadamardCore(const T A[], const T B[], T C[]) {
  for (int i = 0; i < size; i++) {
    C[0] = A[0] * B[0];
    C++, A++, B++;
  }
}

template <typename T, int size>
A2D_FUNCTION void VecHadamardDoubleCore(const T A[], const T Aseed[],
                                        const T B[], const T Bseed[], T C[]) {
  for (int i = 0; i < size; i++) {
    // C[0] = A[0] * B[0];
    C[0] = Aseed[0] * B[0] + Bseed[0] * A[0];
    C++, A++, B++, Aseed++, Bseed++;
  }
}

template <typename T, int size>
A2D_FUNCTION void VecHadamardSingleCore(const T A[], const T Bseed[], T C[]) {
  for (int i = 0; i < size; i++) {
    C[0] = A[0] * Bseed[0];
    C++, A++, Bseed++;
  }
}

template <typename T, int size>
A2D_FUNCTION void VecHadamardAddCore(const T A[], const T Bseed[], T C[]) {
  for (int i = 0; i < size; i++) {
    C[0] += A[0] * Bseed[0];
    C++, A++, Bseed++;
  }
}

template <typename T, int size>
A2D_FUNCTION void VecSumCoreAdd(const T A[], const T B[], T C[]) {
  for (int i = 0; i < size; i++) {
    C[0] += A[0] + B[0];
    C++, A++, B++;
  }
}

template <typename T, int size>
A2D_FUNCTION void VecSumCoreAdd(const T alpha, const T A[], const T beta,
                                const T B[], T C[]) {
  for (int i = 0; i < size; i++) {
    C[0] += alpha * A[0] + beta * B[0];
    C++, A++, B++;
  }
}

/*
  Compute the outer product of two vectors

  A = alpha * x * y^{T}

  Or

  A = A + alpha * x * y^{T}
*/
template <typename T, int M, int N, bool additive = false>
A2D_FUNCTION void VecOuterCore(const T x[], const T y[], T A[]) {
  if constexpr (additive) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        A[N * i + j] += x[i] * y[j];
      }
    }

  } else {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        A[N * i + j] += x[i] * y[j];
      }
    }
  }
}

template <typename T, int M, int N, bool additive = false>
A2D_FUNCTION void VecOuterCore(const T alpha, const T x[], const T y[], T A[]) {
  if constexpr (additive) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        A[N * i + j] += alpha * x[i] * y[j];
      }
    }

  } else {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        A[N * i + j] += alpha * x[i] * y[j];
      }
    }
  }
}

/*
 * Diagonal-preserving symmetrization of the vec-vec outer product:
 *
 * Ret = x y^T + y x^T - diag(x) * diag(y)
 * */
template <typename T, int N, bool additive = false>
A2D_FUNCTION void DiagonalPreservingVecSymOuterCore(const T x[], const T y[],
                                                    T S[]) {
  if constexpr (additive) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < i; j++) {
        S[0] += x[i] * y[j] + x[j] * y[i];
        S++;
      }
      S[0] += x[i] * y[i];
      S++;
    }
  } else {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < i; j++) {
        S[0] = x[i] * y[j] + x[j] * y[i];
        S++;
      }
      S[0] = x[i] * y[i];
      S++;
    }
  }
}

template <typename T, int N, bool additive = false>
A2D_FUNCTION void VecSymOuterCore(const T x[], T S[]) {
  if constexpr (additive) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j <= i; j++) {
        S[0] += x[i] * x[j];
        S++;
      }
    }
  } else {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j <= i; j++) {
        S[0] = x[i] * x[j];
        S++;
      }
    }
  }
}

template <typename T, int N, bool additive = false>
A2D_FUNCTION void VecSymOuterCore(const T alpha, const T x[], T S[]) {
  if constexpr (additive) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j <= i; j++) {
        S[0] += alpha * x[i] * x[j];
        S++;
      }
    }
  } else {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j <= i; j++) {
        S[0] = alpha * x[i] * x[j];
        S++;
      }
    }
  }
}

}  // namespace A2D

#endif  // A2D_VEC_CORE_H
