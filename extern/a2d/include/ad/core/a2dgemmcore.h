#ifndef A2D_GEMMCORE_H
#define A2D_GEMMCORE_H

#include <stdexcept>

#include "../../a2ddefs.h"

namespace A2D {

template <bool B, int i, int j>
struct int_conditional {};

template <int i, int j>
struct int_conditional<true, i, j> {
  static constexpr int value = i;
};

template <int i, int j>
struct int_conditional<false, i, j> {
  static constexpr int value = j;
};

template <typename T, MatOp opA = MatOp::NORMAL, MatOp opB = MatOp::NORMAL>
A2D_FUNCTION void MatMatMultCore3x3(const T A[], const T B[], T C[]) {
  if constexpr (opA == MatOp::NORMAL and opB == MatOp::NORMAL) {
    C[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
    C[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
    C[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];
    C[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
    C[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
    C[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];
    C[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
    C[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
    C[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
  } else if constexpr (opA == MatOp::TRANSPOSE and opB == MatOp::NORMAL) {
    C[0] = A[0] * B[0] + A[3] * B[3] + A[6] * B[6];
    C[1] = A[0] * B[1] + A[3] * B[4] + A[6] * B[7];
    C[2] = A[0] * B[2] + A[3] * B[5] + A[6] * B[8];
    C[3] = A[1] * B[0] + A[4] * B[3] + A[7] * B[6];
    C[4] = A[1] * B[1] + A[4] * B[4] + A[7] * B[7];
    C[5] = A[1] * B[2] + A[4] * B[5] + A[7] * B[8];
    C[6] = A[2] * B[0] + A[5] * B[3] + A[8] * B[6];
    C[7] = A[2] * B[1] + A[5] * B[4] + A[8] * B[7];
    C[8] = A[2] * B[2] + A[5] * B[5] + A[8] * B[8];
  } else if constexpr (opA == MatOp::NORMAL and opB == MatOp::TRANSPOSE) {
    C[0] = A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
    C[1] = A[0] * B[3] + A[1] * B[4] + A[2] * B[5];
    C[2] = A[0] * B[6] + A[1] * B[7] + A[2] * B[8];
    C[3] = A[3] * B[0] + A[4] * B[1] + A[5] * B[2];
    C[4] = A[3] * B[3] + A[4] * B[4] + A[5] * B[5];
    C[5] = A[3] * B[6] + A[4] * B[7] + A[5] * B[8];
    C[6] = A[6] * B[0] + A[7] * B[1] + A[8] * B[2];
    C[7] = A[6] * B[3] + A[7] * B[4] + A[8] * B[5];
    C[8] = A[6] * B[6] + A[7] * B[7] + A[8] * B[8];
  } else if constexpr (opA == MatOp::TRANSPOSE and opB == MatOp::TRANSPOSE) {
    C[0] = A[0] * B[0] + A[3] * B[1] + A[6] * B[2];
    C[1] = A[0] * B[3] + A[3] * B[4] + A[6] * B[5];
    C[2] = A[0] * B[6] + A[3] * B[7] + A[6] * B[8];
    C[3] = A[1] * B[0] + A[4] * B[1] + A[7] * B[2];
    C[4] = A[1] * B[3] + A[4] * B[4] + A[7] * B[5];
    C[5] = A[1] * B[6] + A[4] * B[7] + A[7] * B[8];
    C[6] = A[2] * B[0] + A[5] * B[1] + A[8] * B[2];
    C[7] = A[2] * B[3] + A[5] * B[4] + A[8] * B[5];
    C[8] = A[2] * B[6] + A[5] * B[7] + A[8] * B[8];
  }
}

template <typename T, MatOp opA = MatOp::NORMAL, MatOp opB = MatOp::NORMAL>
A2D_FUNCTION void MatMatMultCore3x3Scale(T scalar, const T A[], const T B[],
                                         T C[]) {
  if constexpr (opA == MatOp::NORMAL and opB == MatOp::NORMAL) {
    C[0] = scalar * (A[0] * B[0] + A[1] * B[3] + A[2] * B[6]);
    C[1] = scalar * (A[0] * B[1] + A[1] * B[4] + A[2] * B[7]);
    C[2] = scalar * (A[0] * B[2] + A[1] * B[5] + A[2] * B[8]);
    C[3] = scalar * (A[3] * B[0] + A[4] * B[3] + A[5] * B[6]);
    C[4] = scalar * (A[3] * B[1] + A[4] * B[4] + A[5] * B[7]);
    C[5] = scalar * (A[3] * B[2] + A[4] * B[5] + A[5] * B[8]);
    C[6] = scalar * (A[6] * B[0] + A[7] * B[3] + A[8] * B[6]);
    C[7] = scalar * (A[6] * B[1] + A[7] * B[4] + A[8] * B[7]);
    C[8] = scalar * (A[6] * B[2] + A[7] * B[5] + A[8] * B[8]);
  } else if constexpr (opA == MatOp::TRANSPOSE and opB == MatOp::NORMAL) {
    C[0] = scalar * (A[0] * B[0] + A[3] * B[3] + A[6] * B[6]);
    C[1] = scalar * (A[0] * B[1] + A[3] * B[4] + A[6] * B[7]);
    C[2] = scalar * (A[0] * B[2] + A[3] * B[5] + A[6] * B[8]);
    C[3] = scalar * (A[1] * B[0] + A[4] * B[3] + A[7] * B[6]);
    C[4] = scalar * (A[1] * B[1] + A[4] * B[4] + A[7] * B[7]);
    C[5] = scalar * (A[1] * B[2] + A[4] * B[5] + A[7] * B[8]);
    C[6] = scalar * (A[2] * B[0] + A[5] * B[3] + A[8] * B[6]);
    C[7] = scalar * (A[2] * B[1] + A[5] * B[4] + A[8] * B[7]);
    C[8] = scalar * (A[2] * B[2] + A[5] * B[5] + A[8] * B[8]);
  } else if constexpr (opA == MatOp::NORMAL and opB == MatOp::TRANSPOSE) {
    C[0] = scalar * (A[0] * B[0] + A[1] * B[1] + A[2] * B[2]);
    C[1] = scalar * (A[0] * B[3] + A[1] * B[4] + A[2] * B[5]);
    C[2] = scalar * (A[0] * B[6] + A[1] * B[7] + A[2] * B[8]);
    C[3] = scalar * (A[3] * B[0] + A[4] * B[1] + A[5] * B[2]);
    C[4] = scalar * (A[3] * B[3] + A[4] * B[4] + A[5] * B[5]);
    C[5] = scalar * (A[3] * B[6] + A[4] * B[7] + A[5] * B[8]);
    C[6] = scalar * (A[6] * B[0] + A[7] * B[1] + A[8] * B[2]);
    C[7] = scalar * (A[6] * B[3] + A[7] * B[4] + A[8] * B[5]);
    C[8] = scalar * (A[6] * B[6] + A[7] * B[7] + A[8] * B[8]);
  } else if constexpr (opA == MatOp::TRANSPOSE and opB == MatOp::TRANSPOSE) {
    C[0] = scalar * (A[0] * B[0] + A[3] * B[1] + A[6] * B[2]);
    C[1] = scalar * (A[0] * B[3] + A[3] * B[4] + A[6] * B[5]);
    C[2] = scalar * (A[0] * B[6] + A[3] * B[7] + A[6] * B[8]);
    C[3] = scalar * (A[1] * B[0] + A[4] * B[1] + A[7] * B[2]);
    C[4] = scalar * (A[1] * B[3] + A[4] * B[4] + A[7] * B[5]);
    C[5] = scalar * (A[1] * B[6] + A[4] * B[7] + A[7] * B[8]);
    C[6] = scalar * (A[2] * B[0] + A[5] * B[1] + A[8] * B[2]);
    C[7] = scalar * (A[2] * B[3] + A[5] * B[4] + A[8] * B[5]);
    C[8] = scalar * (A[2] * B[6] + A[5] * B[7] + A[8] * B[8]);
  }
}

template <typename T, MatOp opA = MatOp::NORMAL, MatOp opB = MatOp::NORMAL>
A2D_FUNCTION void MatMatMultCore3x3Add(const T A[], const T B[], T C[]) {
  if constexpr (opA == MatOp::NORMAL and opB == MatOp::NORMAL) {
    C[0] += A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
    C[1] += A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
    C[2] += A[0] * B[2] + A[1] * B[5] + A[2] * B[8];
    C[3] += A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
    C[4] += A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
    C[5] += A[3] * B[2] + A[4] * B[5] + A[5] * B[8];
    C[6] += A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
    C[7] += A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
    C[8] += A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
  } else if constexpr (opA == MatOp::TRANSPOSE and opB == MatOp::NORMAL) {
    C[0] += A[0] * B[0] + A[3] * B[3] + A[6] * B[6];
    C[1] += A[0] * B[1] + A[3] * B[4] + A[6] * B[7];
    C[2] += A[0] * B[2] + A[3] * B[5] + A[6] * B[8];
    C[3] += A[1] * B[0] + A[4] * B[3] + A[7] * B[6];
    C[4] += A[1] * B[1] + A[4] * B[4] + A[7] * B[7];
    C[5] += A[1] * B[2] + A[4] * B[5] + A[7] * B[8];
    C[6] += A[2] * B[0] + A[5] * B[3] + A[8] * B[6];
    C[7] += A[2] * B[1] + A[5] * B[4] + A[8] * B[7];
    C[8] += A[2] * B[2] + A[5] * B[5] + A[8] * B[8];
  } else if constexpr (opA == MatOp::NORMAL and opB == MatOp::TRANSPOSE) {
    C[0] += A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
    C[1] += A[0] * B[3] + A[1] * B[4] + A[2] * B[5];
    C[2] += A[0] * B[6] + A[1] * B[7] + A[2] * B[8];
    C[3] += A[3] * B[0] + A[4] * B[1] + A[5] * B[2];
    C[4] += A[3] * B[3] + A[4] * B[4] + A[5] * B[5];
    C[5] += A[3] * B[6] + A[4] * B[7] + A[5] * B[8];
    C[6] += A[6] * B[0] + A[7] * B[1] + A[8] * B[2];
    C[7] += A[6] * B[3] + A[7] * B[4] + A[8] * B[5];
    C[8] += A[6] * B[6] + A[7] * B[7] + A[8] * B[8];
  } else if constexpr (opA == MatOp::TRANSPOSE and opB == MatOp::TRANSPOSE) {
    C[0] += A[0] * B[0] + A[3] * B[1] + A[6] * B[2];
    C[1] += A[0] * B[3] + A[3] * B[4] + A[6] * B[5];
    C[2] += A[0] * B[6] + A[3] * B[7] + A[6] * B[8];
    C[3] += A[1] * B[0] + A[4] * B[1] + A[7] * B[2];
    C[4] += A[1] * B[3] + A[4] * B[4] + A[7] * B[5];
    C[5] += A[1] * B[6] + A[4] * B[7] + A[7] * B[8];
    C[6] += A[2] * B[0] + A[5] * B[1] + A[8] * B[2];
    C[7] += A[2] * B[3] + A[5] * B[4] + A[8] * B[5];
    C[8] += A[2] * B[6] + A[5] * B[7] + A[8] * B[8];
  }
}

template <typename T, MatOp opA = MatOp::NORMAL, MatOp opB = MatOp::NORMAL>
A2D_FUNCTION void MatMatMultCore3x3ScaleAdd(T scalar, const T A[], const T B[],
                                            T C[]) {
  if constexpr (opA == MatOp::NORMAL and opB == MatOp::NORMAL) {
    C[0] += scalar * (A[0] * B[0] + A[1] * B[3] + A[2] * B[6]);
    C[1] += scalar * (A[0] * B[1] + A[1] * B[4] + A[2] * B[7]);
    C[2] += scalar * (A[0] * B[2] + A[1] * B[5] + A[2] * B[8]);
    C[3] += scalar * (A[3] * B[0] + A[4] * B[3] + A[5] * B[6]);
    C[4] += scalar * (A[3] * B[1] + A[4] * B[4] + A[5] * B[7]);
    C[5] += scalar * (A[3] * B[2] + A[4] * B[5] + A[5] * B[8]);
    C[6] += scalar * (A[6] * B[0] + A[7] * B[3] + A[8] * B[6]);
    C[7] += scalar * (A[6] * B[1] + A[7] * B[4] + A[8] * B[7]);
    C[8] += scalar * (A[6] * B[2] + A[7] * B[5] + A[8] * B[8]);
  } else if constexpr (opA == MatOp::TRANSPOSE and opB == MatOp::NORMAL) {
    C[0] += scalar * (A[0] * B[0] + A[3] * B[3] + A[6] * B[6]);
    C[1] += scalar * (A[0] * B[1] + A[3] * B[4] + A[6] * B[7]);
    C[2] += scalar * (A[0] * B[2] + A[3] * B[5] + A[6] * B[8]);
    C[3] += scalar * (A[1] * B[0] + A[4] * B[3] + A[7] * B[6]);
    C[4] += scalar * (A[1] * B[1] + A[4] * B[4] + A[7] * B[7]);
    C[5] += scalar * (A[1] * B[2] + A[4] * B[5] + A[7] * B[8]);
    C[6] += scalar * (A[2] * B[0] + A[5] * B[3] + A[8] * B[6]);
    C[7] += scalar * (A[2] * B[1] + A[5] * B[4] + A[8] * B[7]);
    C[8] += scalar * (A[2] * B[2] + A[5] * B[5] + A[8] * B[8]);
  } else if constexpr (opA == MatOp::NORMAL and opB == MatOp::TRANSPOSE) {
    C[0] += scalar * (A[0] * B[0] + A[1] * B[1] + A[2] * B[2]);
    C[1] += scalar * (A[0] * B[3] + A[1] * B[4] + A[2] * B[5]);
    C[2] += scalar * (A[0] * B[6] + A[1] * B[7] + A[2] * B[8]);
    C[3] += scalar * (A[3] * B[0] + A[4] * B[1] + A[5] * B[2]);
    C[4] += scalar * (A[3] * B[3] + A[4] * B[4] + A[5] * B[5]);
    C[5] += scalar * (A[3] * B[6] + A[4] * B[7] + A[5] * B[8]);
    C[6] += scalar * (A[6] * B[0] + A[7] * B[1] + A[8] * B[2]);
    C[7] += scalar * (A[6] * B[3] + A[7] * B[4] + A[8] * B[5]);
    C[8] += scalar * (A[6] * B[6] + A[7] * B[7] + A[8] * B[8]);
  } else if constexpr (opA == MatOp::TRANSPOSE and opB == MatOp::TRANSPOSE) {
    C[0] += scalar * (A[0] * B[0] + A[3] * B[1] + A[6] * B[2]);
    C[1] += scalar * (A[0] * B[3] + A[3] * B[4] + A[6] * B[5]);
    C[2] += scalar * (A[0] * B[6] + A[3] * B[7] + A[6] * B[8]);
    C[3] += scalar * (A[1] * B[0] + A[4] * B[1] + A[7] * B[2]);
    C[4] += scalar * (A[1] * B[3] + A[4] * B[4] + A[7] * B[5]);
    C[5] += scalar * (A[1] * B[6] + A[4] * B[7] + A[7] * B[8]);
    C[6] += scalar * (A[2] * B[0] + A[5] * B[1] + A[8] * B[2]);
    C[7] += scalar * (A[2] * B[3] + A[5] * B[4] + A[8] * B[5]);
    C[8] += scalar * (A[2] * B[6] + A[5] * B[7] + A[8] * B[8]);
  }
}

/**
 * @brief mat-mat multiplication C = alpha * Op(A) * Op(B), where op is normal
 * or transpose
 *
 * Note: consistency of the dimensions are not checked so check them when
 * calling!
 *
 * @tparam T: scalar type
 * @tparam Anrows: number of rows of A
 * @tparam Ancols: number of cols of A
 * @tparam Bnrows: number of rows of B
 * @tparam Bncols: number of cols of B
 * @tparam Cnrows: number of rows of C
 * @tparam Cncols: number of cols of C
 * @tparam opA: transpose A or not
 * @tparam opB: transpose B or not
 * @tparam additive: true for increment (C += ..), false for assignment (C = ..)
 *
 * @param[in] A: Anrows-by-Ancols matrix
 * @param[in] B: Bnrows-by-Bncols matrix
 * @param[in, out] C: Cnrows-by-Cncols matrix
 * @param[in] alpha: the scalar
 */
template <typename T, int Anrows, int Ancols, int Bnrows, int Bncols,
          int Cnrows, int Cncols, MatOp opA = MatOp::NORMAL,
          MatOp opB = MatOp::NORMAL, bool additive = false>
A2D_FUNCTION void MatMatMultCoreGeneral(const T A[], const T B[], T C[]) {
  // Op(A) is M-by-P, Op(B) is P-by-N, C is M-by-N
  constexpr int M = Cnrows;
  constexpr int N = Cncols;
  // constexpr int P =
  //    int_conditional<opA == MatOp::NORMAL, Ancols, Anrows>::value;

  if constexpr (opA == MatOp::NORMAL) {
    if (opB == MatOp::NORMAL) {
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++, C++) {
          const T *a = &A[Ancols * i];
          const T *aend = a + Ancols;
          const T *b = &B[j];

          T value = T(0.0);
          for (; a < aend; a++, b += Bncols) {
            value += a[0] * b[0];
          }

          if constexpr (additive) {
            C[0] += value;
          } else {
            C[0] = value;
          }
        }
      }
    } else {  // opB == MatOp::TRANSPOSE
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++, C++) {
          const T *a = &A[Ancols * i];
          const T *aend = a + Ancols;
          const T *b = &B[Bncols * j];

          T value = T(0.0);
          for (; a < aend; a++, b++) {
            value += a[0] * b[0];
          }

          if constexpr (additive) {
            C[0] += value;
          } else {
            C[0] = value;
          }
        }
      }
    }
  } else {  // opA == MatOp::TRANSPOSE
    if (opB == MatOp::NORMAL) {
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++, C++) {
          const T *a = &A[i];
          const T *b = &B[j];
          const T *bend = b + Bnrows * Bncols;

          T value = T(0.0);
          for (; b < bend; a += Ancols, b += Bncols) {
            value += a[0] * b[0];
          }

          if constexpr (additive) {
            C[0] += value;
          } else {
            C[0] = value;
          }
        }
      }
    } else {  // opB == MatOp::TRANSPOSE
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++, C++) {
          const T *a = &A[i];
          const T *b = &B[Bncols * j];
          const T *bend = b + Bncols;

          T value = T(0.0);
          for (; b < bend; a += Ancols, b++) {
            value += a[0] * b[0];
          }

          if constexpr (additive) {
            C[0] += value;
          } else {
            C[0] = value;
          }
        }
      }
    }
  }
}

/**
 * @brief mat-mat multiplication C = alpha * Op(A) * Op(B), where op is
 * normal or transpose
 *
 * Note: consistency of the dimensions are not checked so check them when
 * calling!
 *
 * @tparam T: scalar type
 * @tparam Anrows: number of rows of A
 * @tparam Ancols: number of cols of A
 * @tparam Bnrows: number of rows of B
 * @tparam Bncols: number of cols of B
 * @tparam Cnrows: number of rows of C
 * @tparam Cncols: number of cols of C
 * @tparam opA: transpose A or not
 * @tparam opB: transpose B or not
 * @tparam additive: true for increment (C += ..), false for assignment (C =
 * ..)
 *
 * @param[in] A: Anrows-by-Ancols matrix
 * @param[in] B: Bnrows-by-Bncols matrix
 * @param[in, out] C: Cnrows-by-Cncols matrix
 * @param[in] alpha: the scalar
 */
template <typename T, int Anrows, int Ancols, int Bnrows, int Bncols,
          int Cnrows, int Cncols, MatOp opA = MatOp::NORMAL,
          MatOp opB = MatOp::NORMAL, bool additive = false>
A2D_FUNCTION void MatMatMultScaleCoreGeneral(const T alpha, const T A[],
                                             const T B[], T C[]) {
  // Op(A) is M-by-P, Op(B) is P-by-N, C is M-by-N
  constexpr int M = Cnrows;
  constexpr int N = Cncols;
  // constexpr int P =
  //    int_conditional<opA == MatOp::NORMAL, Ancols, Anrows>::value;

  if constexpr (opA == MatOp::NORMAL) {
    if (opB == MatOp::NORMAL) {
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++, C++) {
          const T *a = &A[Ancols * i];
          const T *aend = a + Ancols;
          const T *b = &B[j];

          T value = T(0.0);
          for (; a < aend; a++, b += Bncols) {
            value += a[0] * b[0];
          }

          if constexpr (additive) {
            C[0] += alpha * value;
          } else {
            C[0] = alpha * value;
          }
        }
      }
    } else {  // opB == MatOp::TRANSPOSE
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++, C++) {
          const T *a = &A[Ancols * i];
          const T *aend = a + Ancols;
          const T *b = &B[Bncols * j];

          T value = T(0.0);
          for (; a < aend; a++, b++) {
            value += a[0] * b[0];
          }

          if constexpr (additive) {
            C[0] += alpha * value;
          } else {
            C[0] = alpha * value;
          }
        }
      }
    }
  } else {  // opA == MatOp::TRANSPOSE
    if (opB == MatOp::NORMAL) {
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++, C++) {
          const T *a = &A[i];
          const T *b = &B[j];
          const T *bend = b + Bnrows * Bncols;

          T value = T(0.0);
          for (; b < bend; a += Ancols, b += Bncols) {
            value += a[0] * b[0];
          }

          if constexpr (additive) {
            C[0] += alpha * value;
          } else {
            C[0] = alpha * value;
          }
        }
      }
    } else {  // opB == MatOp::TRANSPOSE
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++, C++) {
          const T *a = &A[i];
          const T *b = &B[Bncols * j];
          const T *bend = b + Bncols;

          T value = T(0.0);
          for (; b < bend; a += Ancols, b++) {
            value += a[0] * b[0];
          }

          if constexpr (additive) {
            C[0] += alpha * value;
          } else {
            C[0] = alpha * value;
          }
        }
      }
    }
  }
}

/**
 * @brief matrix-matrix multiplication C = alpha * Op(A) * Op(B), where op
 * is normal (nominal) or transpose
 *
 * @note This function acts as a dispatcher for specialized versions of the
 * matrix multiplication function, falling back on the general
 * implementation. This also verifies the dimensions of the inputs and
 * outputs are appropriate for the multiplication to be executed.
 *
 * @tparam T: scalar type
 * @tparam Anrows: number of rows of A
 * @tparam Ancols: number of cols of A
 * @tparam Bnrows: number of rows of B
 * @tparam Bncols: number of cols of B
 * @tparam Cnrows: number of rows of C
 * @tparam Cncols: number of cols of C
 * @tparam opA: transpose A or not
 * @tparam opB: transpose B or not
 * @tparam additive: true for increment (C += ..), false for assignment (C =
 * ..)
 * @tparam scale: true if result should be scaled by alpha before output
 *
 * @param[in] A: Anrows-by-Ancols matrix
 * @param[in] B: Bnrows-by-Bncols matrix
 * @param[in, out] C: Cnrows-by-Cncols matrix
 * @param[in] alpha: the scalar
 */
template <typename T, int Anrows, int Ancols, int Bnrows, int Bncols,
          int Cnrows, int Cncols, MatOp opA = MatOp::NORMAL,
          MatOp opB = MatOp::NORMAL, bool additive = false>
A2D_FUNCTION void MatMatMultCore(const T A[], const T B[], T C[]) {
  // Check if shapes are consistent
  if constexpr (opA == MatOp::TRANSPOSE && opB == MatOp::TRANSPOSE) {
    static_assert(Anrows == Bncols && Ancols == Cnrows && Bnrows == Cncols,
                  "Matrix dimensions must agree.");
  } else if constexpr (opA == MatOp::TRANSPOSE) {
    static_assert(Anrows == Bnrows && Ancols == Cnrows && Bncols == Cncols,
                  "Matrix dimensions must agree.");
  } else if constexpr (opB == MatOp::TRANSPOSE) {
    static_assert(Ancols == Bncols && Anrows == Cnrows && Bnrows == Cncols,
                  "Matrix dimensions must agree.");
  } else {
    static_assert(Ancols == Bnrows && Anrows == Cnrows && Bncols == Cncols,
                  "Matrix dimensions must agree.");
  }

  if constexpr (Anrows == 3 && Ancols == 3 && Bnrows == 3 && Bncols == 3) {
    if constexpr (additive) {
      MatMatMultCore3x3Add<T, opA, opB>(A, B, C);
    } else {
      MatMatMultCore3x3<T, opA, opB>(A, B, C);
    }
  } else {  // The general fallback implmentation
    MatMatMultCoreGeneral<T, Anrows, Ancols, Bnrows, Bncols, Cnrows, Cncols,
                          opA, opB, additive>(A, B, C);
  }
}

template <typename T, int Anrows, int Ancols, int Bnrows, int Bncols,
          int Cnrows, int Cncols, MatOp opA = MatOp::NORMAL,
          MatOp opB = MatOp::NORMAL, bool additive = false>
inline void MatMatMultScaleCore(T alpha, const T A[], const T B[], T C[]) {
  // Check if shapes are consistent
  if constexpr (opA == MatOp::TRANSPOSE && opB == MatOp::TRANSPOSE) {
    static_assert(Anrows == Bncols && Ancols == Cnrows && Bnrows == Cncols,
                  "Matrix dimensions must agree.");
  } else if constexpr (opA == MatOp::TRANSPOSE) {
    static_assert(Anrows == Bnrows && Ancols == Cnrows && Bncols == Cncols,
                  "Matrix dimensions must agree.");
  } else if constexpr (opB == MatOp::TRANSPOSE) {
    static_assert(Ancols == Bncols && Anrows == Cnrows && Bnrows == Cncols,
                  "Matrix dimensions must agree.");
  } else {
    static_assert(Ancols == Bnrows && Anrows == Cnrows && Bncols == Cncols,
                  "Matrix dimensions must agree.");
  }

  if constexpr (Anrows == 3 && Ancols == 3 && Bnrows == 3 && Bncols == 3) {
    if constexpr (additive) {
      MatMatMultCore3x3ScaleAdd<T, opA, opB>(alpha, A, B, C);
    } else {
      MatMatMultCore3x3Scale<T, opA, opB>(alpha, A, B, C);
    }
  } else {  // The general fallback implmentation
    MatMatMultScaleCoreGeneral<T, Anrows, Ancols, Bnrows, Bncols, Cnrows,
                               Cncols, opA, opB, additive>(alpha, A, B, C);
  }
}

template <typename T>
A2D_FUNCTION void SMatSMatMultCore2x2(const T SA[], const T SB[], T C[]) {
  C[0] = SA[0] * SB[0] + SA[1] * SB[1];
  C[1] = SA[0] * SB[1] + SA[1] * SB[2];
  C[2] = SA[1] * SB[0] + SA[2] * SB[1];
  C[3] = SA[1] * SB[1] + SA[2] * SB[2];
}

template <typename T>
A2D_FUNCTION void SMatSMatMultCore2x2Add(const T SA[], const T SB[], T C[]) {
  C[0] += SA[0] * SB[0] + SA[1] * SB[1];
  C[1] += SA[0] * SB[1] + SA[1] * SB[2];
  C[2] += SA[1] * SB[0] + SA[2] * SB[1];
  C[3] += SA[1] * SB[1] + SA[2] * SB[2];
}

template <typename T>
A2D_FUNCTION void SMatSMatMultCore2x2Scale(const T scalar, const T SA[],
                                           const T SB[], T C[]) {
  C[0] = scalar * (SA[0] * SB[0] + SA[1] * SB[1]);
  C[1] = scalar * (SA[0] * SB[1] + SA[1] * SB[2]);
  C[2] = scalar * (SA[1] * SB[0] + SA[2] * SB[1]);
  C[3] = scalar * (SA[1] * SB[1] + SA[2] * SB[2]);
}

template <typename T>
A2D_FUNCTION void SMatSMatMultCore3x3(const T SA[], const T SB[], T C[]) {
  C[0] = SA[0] * SB[0] + SA[1] * SB[1] + SA[3] * SB[3];
  C[1] = SA[0] * SB[1] + SA[1] * SB[2] + SA[3] * SB[4];
  C[2] = SA[0] * SB[3] + SA[1] * SB[4] + SA[3] * SB[5];
  C[3] = SA[1] * SB[0] + SA[2] * SB[1] + SA[4] * SB[3];
  C[4] = SA[1] * SB[1] + SA[2] * SB[2] + SA[4] * SB[4];
  C[5] = SA[1] * SB[3] + SA[2] * SB[4] + SA[4] * SB[5];
  C[6] = SA[3] * SB[0] + SA[4] * SB[1] + SA[5] * SB[3];
  C[7] = SA[3] * SB[1] + SA[4] * SB[2] + SA[5] * SB[4];
  C[8] = SA[3] * SB[3] + SA[4] * SB[4] + SA[5] * SB[5];
}

template <typename T>
A2D_FUNCTION void SMatSMatMultCore3x3Add(const T SA[], const T SB[], T C[]) {
  C[0] += SA[0] * SB[0] + SA[1] * SB[1] + SA[3] * SB[3];
  C[1] += SA[0] * SB[1] + SA[1] * SB[2] + SA[3] * SB[4];
  C[2] += SA[0] * SB[3] + SA[1] * SB[4] + SA[3] * SB[5];
  C[3] += SA[1] * SB[0] + SA[2] * SB[1] + SA[4] * SB[3];
  C[4] += SA[1] * SB[1] + SA[2] * SB[2] + SA[4] * SB[4];
  C[5] += SA[1] * SB[3] + SA[2] * SB[4] + SA[4] * SB[5];
  C[6] += SA[3] * SB[0] + SA[4] * SB[1] + SA[5] * SB[3];
  C[7] += SA[3] * SB[1] + SA[4] * SB[2] + SA[5] * SB[4];
  C[8] += SA[3] * SB[3] + SA[4] * SB[4] + SA[5] * SB[5];
}

template <typename T>
A2D_FUNCTION void SMatSMatMultCore3x3Scale(const T scalar, const T SA[],
                                           const T SB[], T C[]) {
  C[0] = scalar * (SA[0] * SB[0] + SA[1] * SB[1] + SA[3] * SB[3]);
  C[1] = scalar * (SA[0] * SB[1] + SA[1] * SB[2] + SA[3] * SB[4]);
  C[2] = scalar * (SA[0] * SB[3] + SA[1] * SB[4] + SA[3] * SB[5]);
  C[3] = scalar * (SA[1] * SB[0] + SA[2] * SB[1] + SA[4] * SB[3]);
  C[4] = scalar * (SA[1] * SB[1] + SA[2] * SB[2] + SA[4] * SB[4]);
  C[5] = scalar * (SA[1] * SB[3] + SA[2] * SB[4] + SA[4] * SB[5]);
  C[6] = scalar * (SA[3] * SB[0] + SA[4] * SB[1] + SA[5] * SB[3]);
  C[7] = scalar * (SA[3] * SB[1] + SA[4] * SB[2] + SA[5] * SB[4]);
  C[8] = scalar * (SA[3] * SB[3] + SA[4] * SB[4] + SA[5] * SB[5]);
}

template <typename T>
A2D_FUNCTION void SMatSMatMultCore2x2ScaleAdd(const T scalar, const T SA[],
                                              const T SB[], T C[]) {
  C[0] += scalar * (SA[0] * SB[0] + SA[1] * SB[1]);
  C[1] += scalar * (SA[0] * SB[1] + SA[1] * SB[2]);
  C[2] += scalar * (SA[1] * SB[0] + SA[2] * SB[1]);
  C[3] += scalar * (SA[1] * SB[1] + SA[2] * SB[2]);
}

template <typename T>
A2D_FUNCTION void SMatSMatMultCore3x3ScaleAdd(const T scalar, const T SA[],
                                              const T SB[], T C[]) {
  C[0] += scalar * (SA[0] * SB[0] + SA[1] * SB[1] + SA[3] * SB[3]);
  C[1] += scalar * (SA[0] * SB[1] + SA[1] * SB[2] + SA[3] * SB[4]);
  C[2] += scalar * (SA[0] * SB[3] + SA[1] * SB[4] + SA[3] * SB[5]);
  C[3] += scalar * (SA[1] * SB[0] + SA[2] * SB[1] + SA[4] * SB[3]);
  C[4] += scalar * (SA[1] * SB[1] + SA[2] * SB[2] + SA[4] * SB[4]);
  C[5] += scalar * (SA[1] * SB[3] + SA[2] * SB[4] + SA[4] * SB[5]);
  C[6] += scalar * (SA[3] * SB[0] + SA[4] * SB[1] + SA[5] * SB[3]);
  C[7] += scalar * (SA[3] * SB[1] + SA[4] * SB[2] + SA[5] * SB[4]);
  C[8] += scalar * (SA[3] * SB[3] + SA[4] * SB[4] + SA[5] * SB[5]);
}

template <typename T, int Anrows, bool additive = false>
A2D_FUNCTION void SMatSMatMultCoreGeneral(const T SA[], const T SB[], T C[]) {
  for (int i = 0; i < Anrows; i++) {
    for (int k = 0; k < Anrows; k++) {
      T value = T(0.0);
      for (int j = 0; j < Anrows; j++) {
        value += SA[i >= j ? j + i * (i + 1) / 2 : i + j * (j + 1) / 2] *
                 SB[j >= k ? k + j * (j + 1) / 2 : j + k * (k + 1) / 2];
      }
      if (additive) {
        C[i * Anrows + k] += value;
      } else {
        C[i * Anrows + k] = value;
      }
    }
  }
}

template <typename T, int Anrows, bool additive = false>
A2D_FUNCTION void SMatSMatMultScaleCoreGeneral(const T scalar, const T SA[],
                                               const T SB[], T C[]) {
  for (int i = 0; i < Anrows; i++) {
    for (int k = 0; k < Anrows; k++) {
      T value = T(0.0);
      for (int j = 0; j < Anrows; j++) {
        value += SA[i >= j ? j + i * (i + 1) / 2 : i + j * (j + 1) / 2] *
                 SB[j >= k ? k + j * (j + 1) / 2 : j + k * (k + 1) / 2];
      }
      if (additive) {
        C[i * Anrows + k] += scalar * value;
      } else {
        C[i * Anrows + k] = scalar * value;
      }
    }
  }
}

template <typename T, int Anrows, int Bnrows, int Cnrows, int Cncols,
          bool additive = false>
A2D_FUNCTION void SMatSMatMultCore(const T SA[], const T SB[], T C[]) {
  // Check if shapes are consistent
  static_assert((Anrows == Bnrows and Bnrows == Cnrows and Cnrows == Cncols),
                "Matrix dimensions must agree.");

  if constexpr (Anrows == 2) {
    if constexpr (additive) {
      SMatSMatMultCore2x2Add(SA, SB, C);
    } else {
      SMatSMatMultCore2x2(SA, SB, C);
    }
  } else if constexpr (Anrows == 3) {
    if constexpr (additive) {
      SMatSMatMultCore3x3Add(SA, SB, C);
    } else {
      SMatSMatMultCore3x3(SA, SB, C);
    }
  } else {
    SMatSMatMultCoreGeneral<T, Anrows, additive>(SA, SB, C);
  }
}

template <typename T, int Anrows, int Bnrows, int Cnrows, int Cncols,
          bool additive = false>
A2D_FUNCTION void SMatSMatMultScaleCore(const T alpha, const T SA[],
                                        const T SB[], T C[]) {
  // Check if shapes are consistent
  static_assert((Anrows == Bnrows and Bnrows == Cnrows and Cnrows == Cncols),
                "Matrix dimensions must agree.");

  if constexpr (Anrows == 2) {
    if constexpr (additive) {
      SMatSMatMultCore2x2ScaleAdd(alpha, SA, SB, C);
    } else {
      SMatSMatMultCore2x2Scale(alpha, SA, SB, C);
    }
  } else if constexpr (Anrows == 3) {
    if constexpr (additive) {
      SMatSMatMultCore3x3ScaleAdd(alpha, SA, SB, C);
    } else {
      SMatSMatMultCore3x3Scale(alpha, SA, SB, C);
    }
  } else {
    SMatSMatMultScaleCoreGeneral<T, Anrows, additive>(alpha, SA, SB, C);
  }
}

template <typename T, MatOp opB = MatOp::NORMAL>
A2D_FUNCTION void SMatMatMultCore2x2(const T S[], const T B[], T C[]) {
  if constexpr (opB == MatOp::NORMAL) {
    C[0] = S[0] * B[0] + S[1] * B[2];
    C[1] = S[0] * B[1] + S[1] * B[3];
    C[2] = S[1] * B[0] + S[2] * B[2];
    C[3] = S[1] * B[1] + S[2] * B[3];
  } else if constexpr (opB == MatOp::TRANSPOSE) {
    C[0] = S[0] * B[0] + S[1] * B[1];
    C[1] = S[0] * B[2] + S[1] * B[3];
    C[2] = S[1] * B[0] + S[2] * B[1];
    C[3] = S[1] * B[2] + S[2] * B[3];
  }
}

template <typename T, MatOp opB = MatOp::NORMAL>
A2D_FUNCTION void SMatMatMultCore2x2Scale(const T scalar, const T S[],
                                          const T B[], T C[]) {
  if constexpr (opB == MatOp::NORMAL) {
    C[0] = scalar * (S[0] * B[0] + S[1] * B[2]);
    C[1] = scalar * (S[0] * B[1] + S[1] * B[3]);
    C[2] = scalar * (S[1] * B[0] + S[2] * B[2]);
    C[3] = scalar * (S[1] * B[1] + S[2] * B[3]);
  } else if constexpr (opB == MatOp::TRANSPOSE) {
    C[0] = scalar * (S[0] * B[0] + S[1] * B[1]);
    C[1] = scalar * (S[0] * B[2] + S[1] * B[3]);
    C[2] = scalar * (S[1] * B[0] + S[2] * B[1]);
    C[3] = scalar * (S[1] * B[2] + S[2] * B[3]);
  }
}

template <typename T, MatOp opB = MatOp::NORMAL>
A2D_FUNCTION void SMatMatMultCore2x2Add(const T S[], const T B[], T C[]) {
  if constexpr (opB == MatOp::NORMAL) {
    C[0] += S[0] * B[0] + S[1] * B[2];
    C[1] += S[0] * B[1] + S[1] * B[3];
    C[2] += S[1] * B[0] + S[2] * B[2];
    C[3] += S[1] * B[1] + S[2] * B[3];
  } else if constexpr (opB == MatOp::TRANSPOSE) {
    C[0] += S[0] * B[0] + S[1] * B[1];
    C[1] += S[0] * B[2] + S[1] * B[3];
    C[2] += S[1] * B[0] + S[2] * B[1];
    C[3] += S[1] * B[2] + S[2] * B[3];
  }
}

template <typename T, MatOp opB = MatOp::NORMAL>
A2D_FUNCTION void SMatMatMultCore2x2ScaleAdd(const T scalar, const T S[],
                                             const T B[], T C[]) {
  if constexpr (opB == MatOp::NORMAL) {
    C[0] += scalar * (S[0] * B[0] + S[1] * B[2]);
    C[1] += scalar * (S[0] * B[1] + S[1] * B[3]);
    C[2] += scalar * (S[1] * B[0] + S[2] * B[2]);
    C[3] += scalar * (S[1] * B[1] + S[2] * B[3]);
  } else if constexpr (opB == MatOp::TRANSPOSE) {
    C[0] += scalar * (S[0] * B[0] + S[1] * B[1]);
    C[1] += scalar * (S[0] * B[2] + S[1] * B[3]);
    C[2] += scalar * (S[1] * B[0] + S[2] * B[1]);
    C[3] += scalar * (S[1] * B[2] + S[2] * B[3]);
  }
}

template <typename T, MatOp opB = MatOp::NORMAL>
A2D_FUNCTION void SMatMatMultCore3x3(const T S[], const T B[], T C[]) {
  if constexpr (opB == MatOp::NORMAL) {
    C[0] = S[0] * B[0] + S[1] * B[3] + S[3] * B[6];
    C[1] = S[0] * B[1] + S[1] * B[4] + S[3] * B[7];
    C[2] = S[0] * B[2] + S[1] * B[5] + S[3] * B[8];
    C[3] = S[1] * B[0] + S[2] * B[3] + S[4] * B[6];
    C[4] = S[1] * B[1] + S[2] * B[4] + S[4] * B[7];
    C[5] = S[1] * B[2] + S[2] * B[5] + S[4] * B[8];
    C[6] = S[3] * B[0] + S[4] * B[3] + S[5] * B[6];
    C[7] = S[3] * B[1] + S[4] * B[4] + S[5] * B[7];
    C[8] = S[3] * B[2] + S[4] * B[5] + S[5] * B[8];
  } else if constexpr (opB == MatOp::TRANSPOSE) {
    C[0] = S[0] * B[0] + S[1] * B[1] + S[3] * B[2];
    C[1] = S[0] * B[3] + S[1] * B[4] + S[3] * B[5];
    C[2] = S[0] * B[6] + S[1] * B[7] + S[3] * B[8];
    C[3] = S[1] * B[0] + S[2] * B[1] + S[4] * B[2];
    C[4] = S[1] * B[3] + S[2] * B[4] + S[4] * B[5];
    C[5] = S[1] * B[6] + S[2] * B[7] + S[4] * B[8];
    C[6] = S[3] * B[0] + S[4] * B[1] + S[5] * B[2];
    C[7] = S[3] * B[3] + S[4] * B[4] + S[5] * B[5];
    C[8] = S[3] * B[6] + S[4] * B[7] + S[5] * B[8];
  }
}

template <typename T, MatOp opB = MatOp::NORMAL>
A2D_FUNCTION void SMatMatMultCore3x3Scale(const T scalar, const T S[],
                                          const T B[], T C[]) {
  if constexpr (opB == MatOp::NORMAL) {
    C[0] = scalar * (S[0] * B[0] + S[1] * B[3] + S[3] * B[6]);
    C[1] = scalar * (S[0] * B[1] + S[1] * B[4] + S[3] * B[7]);
    C[2] = scalar * (S[0] * B[2] + S[1] * B[5] + S[3] * B[8]);
    C[3] = scalar * (S[1] * B[0] + S[2] * B[3] + S[4] * B[6]);
    C[4] = scalar * (S[1] * B[1] + S[2] * B[4] + S[4] * B[7]);
    C[5] = scalar * (S[1] * B[2] + S[2] * B[5] + S[4] * B[8]);
    C[6] = scalar * (S[3] * B[0] + S[4] * B[3] + S[5] * B[6]);
    C[7] = scalar * (S[3] * B[1] + S[4] * B[4] + S[5] * B[7]);
    C[8] = scalar * (S[3] * B[2] + S[4] * B[5] + S[5] * B[8]);
  } else if constexpr (opB == MatOp::TRANSPOSE) {
    C[0] = scalar * (S[0] * B[0] + S[1] * B[1] + S[3] * B[2]);
    C[1] = scalar * (S[0] * B[3] + S[1] * B[4] + S[3] * B[5]);
    C[2] = scalar * (S[0] * B[6] + S[1] * B[7] + S[3] * B[8]);
    C[3] = scalar * (S[1] * B[0] + S[2] * B[1] + S[4] * B[2]);
    C[4] = scalar * (S[1] * B[3] + S[2] * B[4] + S[4] * B[5]);
    C[5] = scalar * (S[1] * B[6] + S[2] * B[7] + S[4] * B[8]);
    C[6] = scalar * (S[3] * B[0] + S[4] * B[1] + S[5] * B[2]);
    C[7] = scalar * (S[3] * B[3] + S[4] * B[4] + S[5] * B[5]);
    C[8] = scalar * (S[3] * B[6] + S[4] * B[7] + S[5] * B[8]);
  }
}

template <typename T, MatOp opB = MatOp::NORMAL>
A2D_FUNCTION void SMatMatMultCore3x3Add(const T S[], const T B[], T C[]) {
  if constexpr (opB == MatOp::NORMAL) {
    C[0] += S[0] * B[0] + S[1] * B[3] + S[3] * B[6];
    C[1] += S[0] * B[1] + S[1] * B[4] + S[3] * B[7];
    C[2] += S[0] * B[2] + S[1] * B[5] + S[3] * B[8];
    C[3] += S[1] * B[0] + S[2] * B[3] + S[4] * B[6];
    C[4] += S[1] * B[1] + S[2] * B[4] + S[4] * B[7];
    C[5] += S[1] * B[2] + S[2] * B[5] + S[4] * B[8];
    C[6] += S[3] * B[0] + S[4] * B[3] + S[5] * B[6];
    C[7] += S[3] * B[1] + S[4] * B[4] + S[5] * B[7];
    C[8] += S[3] * B[2] + S[4] * B[5] + S[5] * B[8];
  } else if constexpr (opB == MatOp::TRANSPOSE) {
    C[0] += S[0] * B[0] + S[1] * B[1] + S[3] * B[2];
    C[1] += S[0] * B[3] + S[1] * B[4] + S[3] * B[5];
    C[2] += S[0] * B[6] + S[1] * B[7] + S[3] * B[8];
    C[3] += S[1] * B[0] + S[2] * B[1] + S[4] * B[2];
    C[4] += S[1] * B[3] + S[2] * B[4] + S[4] * B[5];
    C[5] += S[1] * B[6] + S[2] * B[7] + S[4] * B[8];
    C[6] += S[3] * B[0] + S[4] * B[1] + S[5] * B[2];
    C[7] += S[3] * B[3] + S[4] * B[4] + S[5] * B[5];
    C[8] += S[3] * B[6] + S[4] * B[7] + S[5] * B[8];
  }
}

template <typename T, MatOp opB = MatOp::NORMAL>
A2D_FUNCTION void SMatMatMultCore3x3ScaleAdd(const T scalar, const T S[],
                                             const T B[], T C[]) {
  if constexpr (opB == MatOp::NORMAL) {
    C[0] += scalar * (S[0] * B[0] + S[1] * B[3] + S[3] * B[6]);
    C[1] += scalar * (S[0] * B[1] + S[1] * B[4] + S[3] * B[7]);
    C[2] += scalar * (S[0] * B[2] + S[1] * B[5] + S[3] * B[8]);
    C[3] += scalar * (S[1] * B[0] + S[2] * B[3] + S[4] * B[6]);
    C[4] += scalar * (S[1] * B[1] + S[2] * B[4] + S[4] * B[7]);
    C[5] += scalar * (S[1] * B[2] + S[2] * B[5] + S[4] * B[8]);
    C[6] += scalar * (S[3] * B[0] + S[4] * B[3] + S[5] * B[6]);
    C[7] += scalar * (S[3] * B[1] + S[4] * B[4] + S[5] * B[7]);
    C[8] += scalar * (S[3] * B[2] + S[4] * B[5] + S[5] * B[8]);
  } else if constexpr (opB == MatOp::TRANSPOSE) {
    C[0] += scalar * (S[0] * B[0] + S[1] * B[1] + S[3] * B[2]);
    C[1] += scalar * (S[0] * B[3] + S[1] * B[4] + S[3] * B[5]);
    C[2] += scalar * (S[0] * B[6] + S[1] * B[7] + S[3] * B[8]);
    C[3] += scalar * (S[1] * B[0] + S[2] * B[1] + S[4] * B[2]);
    C[4] += scalar * (S[1] * B[3] + S[2] * B[4] + S[4] * B[5]);
    C[5] += scalar * (S[1] * B[6] + S[2] * B[7] + S[4] * B[8]);
    C[6] += scalar * (S[3] * B[0] + S[4] * B[1] + S[5] * B[2]);
    C[7] += scalar * (S[3] * B[3] + S[4] * B[4] + S[5] * B[5]);
    C[8] += scalar * (S[3] * B[6] + S[4] * B[7] + S[5] * B[8]);
  }
}

template <typename T, int Anrows, int Bnrows, int Bncols,
          MatOp opB = MatOp::NORMAL, bool additive = false>
A2D_FUNCTION void SMatMatMultCoreGeneral(const T SA[], const T B[], T C[]) {
  int idim = Anrows;
  int jdim = Anrows;
  int kdim = opB == MatOp::NORMAL ? Bncols : Bnrows;

  if constexpr (opB == MatOp::NORMAL) {
    for (int i = 0; i < idim; i++) {
      for (int k = 0; k < kdim; k++) {
        T value = T(0.0);
        for (int j = 0; j < jdim; j++) {
          value += SA[i >= j ? j + i * (i + 1) / 2 : i + j * (j + 1) / 2] *
                   B[j * Bncols + k];
        }
        if (additive) {
          C[i * Bncols + k] += value;  // C: Anrows-by-Bncols
        } else {
          C[i * Bncols + k] = value;  // C: Anrows-by-Bncols
        }
      }
    }
  } else {
    for (int i = 0; i < idim; i++) {
      for (int k = 0; k < kdim; k++) {
        T value = T(0.0);
        for (int j = 0; j < jdim; j++) {
          value += SA[i >= j ? j + i * (i + 1) / 2 : i + j * (j + 1) / 2] *
                   B[k * Bncols + j];
        }
        if (additive) {
          C[i * Bnrows + k] += value;  // C: Anrows-by-Bnrows
        } else {
          C[i * Bnrows + k] = value;  // C: Anrows-by-Bnrows
        }
      }
    }
  }
}

template <typename T, int Anrows, int Bnrows, int Bncols,
          MatOp opB = MatOp::NORMAL, bool additive = false>
A2D_FUNCTION void SMatMatMultScaleCoreGeneral(const T alpha, const T SA[],
                                              const T B[], T C[]) {
  int idim = Anrows;
  int jdim = Anrows;
  int kdim = opB == MatOp::NORMAL ? Bncols : Bnrows;

  if constexpr (opB == MatOp::NORMAL) {
    for (int i = 0; i < idim; i++) {
      for (int k = 0; k < kdim; k++) {
        T value = T(0.0);
        for (int j = 0; j < jdim; j++) {
          value += SA[i >= j ? j + i * (i + 1) / 2 : i + j * (j + 1) / 2] *
                   B[j * Bncols + k];
        }
        if (additive) {
          C[i * Bncols + k] += alpha * value;  // C: Anrows-by-Bncols
        } else {
          C[i * Bncols + k] = alpha * value;  // C: Anrows-by-Bncols
        }
      }
    }
  } else {
    for (int i = 0; i < idim; i++) {
      for (int k = 0; k < kdim; k++) {
        T value = T(0.0);
        for (int j = 0; j < jdim; j++) {
          value += SA[i >= j ? j + i * (i + 1) / 2 : i + j * (j + 1) / 2] *
                   B[k * Bncols + j];
        }
        if (additive) {
          C[i * Bnrows + k] += alpha * value;  // C: Anrows-by-Bnrows
        } else {
          C[i * Bnrows + k] = alpha * value;  // C: Anrows-by-Bnrows
        }
      }
    }
  }
}

template <typename T, int Anrows, int Bnrows, int Bncols, int Cnrows,
          int Cncols, MatOp opB = MatOp::NORMAL, bool additive = false>
A2D_FUNCTION void SMatMatMultCore(const T S[], const T B[], T C[]) {
  // Check if shapes are consistent
  if constexpr (opB == MatOp::TRANSPOSE) {
    static_assert(Anrows == Bncols && Anrows == Cnrows && Bnrows == Cncols,
                  "Matrix dimensions must agree.");
  } else {
    static_assert(Anrows == Bnrows && Anrows == Cnrows && Bncols == Cncols,
                  "Matrix dimensions must agree.");
  }

  if constexpr (Anrows == 2 && Bnrows == 2 && Bncols == 2) {
    if constexpr (additive) {
      SMatMatMultCore2x2Add<T, opB>(S, B, C);
    } else {
      SMatMatMultCore2x2<T, opB>(S, B, C);
    }

  } else if constexpr (Anrows == 3 && Bnrows == 3 && Bncols == 3) {
    if constexpr (additive) {
      SMatMatMultCore3x3Add<T, opB>(S, B, C);
    } else {
      SMatMatMultCore3x3<T, opB>(S, B, C);
    }
  } else {  // The general fallback implmentation
    SMatMatMultCoreGeneral<T, Anrows, Bnrows, Bncols, opB, additive>(S, B, C);
  }
}

template <typename T, int Anrows, int Bnrows, int Bncols, int Cnrows,
          int Cncols, MatOp opB = MatOp::NORMAL, bool additive = false>
A2D_FUNCTION void SMatMatMultScaleCore(const T alpha, const T S[], const T B[],
                                       T C[]) {
  // Check if shapes are consistent
  if constexpr (opB == MatOp::TRANSPOSE) {
    static_assert(Anrows == Bncols && Anrows == Cnrows && Bnrows == Cncols,
                  "Matrix dimensions must agree.");
  } else {
    static_assert(Anrows == Bnrows && Anrows == Cnrows && Bncols == Cncols,
                  "Matrix dimensions must agree.");
  }

  if constexpr (Anrows == 2 && Bnrows == 2 && Bncols == 2) {
    if constexpr (additive) {
      SMatMatMultCore2x2ScaleAdd<T, opB>(alpha, S, B, C);
    } else {
      SMatMatMultCore2x2Scale<T, opB>(alpha, S, B, C);
    }
  } else if constexpr (Anrows == 3 && Bnrows == 3 && Bncols == 3) {
    if constexpr (additive) {
      SMatMatMultCore3x3ScaleAdd<T, opB>(alpha, S, B, C);
    } else {
      SMatMatMultCore3x3Scale<T, opB>(alpha, S, B, C);
    }
  } else {  // The general fallback implmentation
    SMatMatMultScaleCoreGeneral<T, Anrows, Bnrows, Bncols, opB, additive>(
        alpha, S, B, C);
  }
}

template <typename T, MatOp opA = MatOp::NORMAL>
A2D_FUNCTION void MatSMatMultCore2x2(const T A[], const T S[], T C[]) {
  if constexpr (opA == MatOp::NORMAL) {
    C[0] = A[0] * S[0] + A[1] * S[1];
    C[1] = A[0] * S[1] + A[1] * S[2];
    C[2] = A[2] * S[0] + A[3] * S[1];
    C[3] = A[2] * S[1] + A[3] * S[2];
  } else if constexpr (opA == MatOp::TRANSPOSE) {
    C[0] = A[0] * S[0] + A[2] * S[1];
    C[1] = A[0] * S[1] + A[2] * S[2];
    C[2] = A[1] * S[0] + A[3] * S[1];
    C[3] = A[1] * S[1] + A[3] * S[2];
  }
}

template <typename T, MatOp opA = MatOp::NORMAL>
A2D_FUNCTION void MatSMatMultCore2x2Scale(const T scalar, const T A[],
                                          const T S[], T C[]) {
  if constexpr (opA == MatOp::NORMAL) {
    C[0] = scalar * (A[0] * S[0] + A[1] * S[1]);
    C[1] = scalar * (A[0] * S[1] + A[1] * S[2]);
    C[2] = scalar * (A[2] * S[0] + A[3] * S[1]);
    C[3] = scalar * (A[2] * S[1] + A[3] * S[2]);
  } else if constexpr (opA == MatOp::TRANSPOSE) {
    C[0] = scalar * (A[0] * S[0] + A[2] * S[1]);
    C[1] = scalar * (A[0] * S[1] + A[2] * S[2]);
    C[2] = scalar * (A[1] * S[0] + A[3] * S[1]);
    C[3] = scalar * (A[1] * S[1] + A[3] * S[2]);
  }
}

template <typename T, MatOp opA = MatOp::NORMAL>
A2D_FUNCTION void MatSMatMultCore2x2Add(const T A[], const T S[], T C[]) {
  if constexpr (opA == MatOp::NORMAL) {
    C[0] += A[0] * S[0] + A[1] * S[1];
    C[1] += A[0] * S[1] + A[1] * S[2];
    C[2] += A[2] * S[0] + A[3] * S[1];
    C[3] += A[2] * S[1] + A[3] * S[2];
  } else if constexpr (opA == MatOp::TRANSPOSE) {
    C[0] += A[0] * S[0] + A[2] * S[1];
    C[1] += A[0] * S[1] + A[2] * S[2];
    C[2] += A[1] * S[0] + A[3] * S[1];
    C[3] += A[1] * S[1] + A[3] * S[2];
  }
}

template <typename T, MatOp opA = MatOp::NORMAL>
A2D_FUNCTION void MatSMatMultCore2x2ScaleAdd(const T scalar, const T A[],
                                             const T S[], T C[]) {
  if constexpr (opA == MatOp::NORMAL) {
    C[0] += scalar * (A[0] * S[0] + A[1] * S[1]);
    C[1] += scalar * (A[0] * S[1] + A[1] * S[2]);
    C[2] += scalar * (A[2] * S[0] + A[3] * S[1]);
    C[3] += scalar * (A[2] * S[1] + A[3] * S[2]);
  } else if constexpr (opA == MatOp::TRANSPOSE) {
    C[0] += scalar * (A[0] * S[0] + A[2] * S[1]);
    C[1] += scalar * (A[0] * S[1] + A[2] * S[2]);
    C[2] += scalar * (A[1] * S[0] + A[3] * S[1]);
    C[3] += scalar * (A[1] * S[1] + A[3] * S[2]);
  }
}

template <typename T, MatOp opA = MatOp::NORMAL>
A2D_FUNCTION void MatSMatMultCore3x3(const T A[], const T S[], T C[]) {
  if constexpr (opA == MatOp::NORMAL) {
    C[0] = A[0] * S[0] + A[1] * S[1] + A[2] * S[3];
    C[1] = A[0] * S[1] + A[1] * S[2] + A[2] * S[4];
    C[2] = A[0] * S[3] + A[1] * S[4] + A[2] * S[5];
    C[3] = A[3] * S[0] + A[4] * S[1] + A[5] * S[3];
    C[4] = A[3] * S[1] + A[4] * S[2] + A[5] * S[4];
    C[5] = A[3] * S[3] + A[4] * S[4] + A[5] * S[5];
    C[6] = A[6] * S[0] + A[7] * S[1] + A[8] * S[3];
    C[7] = A[6] * S[1] + A[7] * S[2] + A[8] * S[4];
    C[8] = A[6] * S[3] + A[7] * S[4] + A[8] * S[5];
  } else if constexpr (opA == MatOp::TRANSPOSE) {
    C[0] = A[0] * S[0] + A[3] * S[1] + A[6] * S[3];
    C[1] = A[0] * S[1] + A[3] * S[2] + A[6] * S[4];
    C[2] = A[0] * S[3] + A[3] * S[4] + A[6] * S[5];
    C[3] = A[1] * S[0] + A[4] * S[1] + A[7] * S[3];
    C[4] = A[1] * S[1] + A[4] * S[2] + A[7] * S[4];
    C[5] = A[1] * S[3] + A[4] * S[4] + A[7] * S[5];
    C[6] = A[2] * S[0] + A[5] * S[1] + A[8] * S[3];
    C[7] = A[2] * S[1] + A[5] * S[2] + A[8] * S[4];
    C[8] = A[2] * S[3] + A[5] * S[4] + A[8] * S[5];
  }
}

template <typename T, MatOp opA = MatOp::NORMAL>
A2D_FUNCTION void MatSMatMultCore3x3Scale(const T scalar, const T A[],
                                          const T S[], T C[]) {
  if constexpr (opA == MatOp::NORMAL) {
    C[0] = scalar * (A[0] * S[0] + A[1] * S[1] + A[2] * S[3]);
    C[1] = scalar * (A[0] * S[1] + A[1] * S[2] + A[2] * S[4]);
    C[2] = scalar * (A[0] * S[3] + A[1] * S[4] + A[2] * S[5]);
    C[3] = scalar * (A[3] * S[0] + A[4] * S[1] + A[5] * S[3]);
    C[4] = scalar * (A[3] * S[1] + A[4] * S[2] + A[5] * S[4]);
    C[5] = scalar * (A[3] * S[3] + A[4] * S[4] + A[5] * S[5]);
    C[6] = scalar * (A[6] * S[0] + A[7] * S[1] + A[8] * S[3]);
    C[7] = scalar * (A[6] * S[1] + A[7] * S[2] + A[8] * S[4]);
    C[8] = scalar * (A[6] * S[3] + A[7] * S[4] + A[8] * S[5]);
  } else if constexpr (opA == MatOp::TRANSPOSE) {
    C[0] = scalar * (A[0] * S[0] + A[3] * S[1] + A[6] * S[3]);
    C[1] = scalar * (A[0] * S[1] + A[3] * S[2] + A[6] * S[4]);
    C[2] = scalar * (A[0] * S[3] + A[3] * S[4] + A[6] * S[5]);
    C[3] = scalar * (A[1] * S[0] + A[4] * S[1] + A[7] * S[3]);
    C[4] = scalar * (A[1] * S[1] + A[4] * S[2] + A[7] * S[4]);
    C[5] = scalar * (A[1] * S[3] + A[4] * S[4] + A[7] * S[5]);
    C[6] = scalar * (A[2] * S[0] + A[5] * S[1] + A[8] * S[3]);
    C[7] = scalar * (A[2] * S[1] + A[5] * S[2] + A[8] * S[4]);
    C[8] = scalar * (A[2] * S[3] + A[5] * S[4] + A[8] * S[5]);
  }
}

template <typename T, MatOp opA = MatOp::NORMAL>
A2D_FUNCTION void MatSMatMultCore3x3Add(const T A[], const T S[], T C[]) {
  if constexpr (opA == MatOp::NORMAL) {
    C[0] += A[0] * S[0] + A[1] * S[1] + A[2] * S[3];
    C[1] += A[0] * S[1] + A[1] * S[2] + A[2] * S[4];
    C[2] += A[0] * S[3] + A[1] * S[4] + A[2] * S[5];
    C[3] += A[3] * S[0] + A[4] * S[1] + A[5] * S[3];
    C[4] += A[3] * S[1] + A[4] * S[2] + A[5] * S[4];
    C[5] += A[3] * S[3] + A[4] * S[4] + A[5] * S[5];
    C[6] += A[6] * S[0] + A[7] * S[1] + A[8] * S[3];
    C[7] += A[6] * S[1] + A[7] * S[2] + A[8] * S[4];
    C[8] += A[6] * S[3] + A[7] * S[4] + A[8] * S[5];
  } else if constexpr (opA == MatOp::TRANSPOSE) {
    C[0] += A[0] * S[0] + A[3] * S[1] + A[6] * S[3];
    C[1] += A[0] * S[1] + A[3] * S[2] + A[6] * S[4];
    C[2] += A[0] * S[3] + A[3] * S[4] + A[6] * S[5];
    C[3] += A[1] * S[0] + A[4] * S[1] + A[7] * S[3];
    C[4] += A[1] * S[1] + A[4] * S[2] + A[7] * S[4];
    C[5] += A[1] * S[3] + A[4] * S[4] + A[7] * S[5];
    C[6] += A[2] * S[0] + A[5] * S[1] + A[8] * S[3];
    C[7] += A[2] * S[1] + A[5] * S[2] + A[8] * S[4];
    C[8] += A[2] * S[3] + A[5] * S[4] + A[8] * S[5];
  }
}

template <typename T, MatOp opA = MatOp::NORMAL>
A2D_FUNCTION void MatSMatMultCore3x3ScaleAdd(const T scalar, const T A[],
                                             const T S[], T C[]) {
  if constexpr (opA == MatOp::NORMAL) {
    C[0] += scalar * (A[0] * S[0] + A[1] * S[1] + A[2] * S[3]);
    C[1] += scalar * (A[0] * S[1] + A[1] * S[2] + A[2] * S[4]);
    C[2] += scalar * (A[0] * S[3] + A[1] * S[4] + A[2] * S[5]);
    C[3] += scalar * (A[3] * S[0] + A[4] * S[1] + A[5] * S[3]);
    C[4] += scalar * (A[3] * S[1] + A[4] * S[2] + A[5] * S[4]);
    C[5] += scalar * (A[3] * S[3] + A[4] * S[4] + A[5] * S[5]);
    C[6] += scalar * (A[6] * S[0] + A[7] * S[1] + A[8] * S[3]);
    C[7] += scalar * (A[6] * S[1] + A[7] * S[2] + A[8] * S[4]);
    C[8] += scalar * (A[6] * S[3] + A[7] * S[4] + A[8] * S[5]);
  } else if constexpr (opA == MatOp::TRANSPOSE) {
    C[0] += scalar * (A[0] * S[0] + A[3] * S[1] + A[6] * S[3]);
    C[1] += scalar * (A[0] * S[1] + A[3] * S[2] + A[6] * S[4]);
    C[2] += scalar * (A[0] * S[3] + A[3] * S[4] + A[6] * S[5]);
    C[3] += scalar * (A[1] * S[0] + A[4] * S[1] + A[7] * S[3]);
    C[4] += scalar * (A[1] * S[1] + A[4] * S[2] + A[7] * S[4]);
    C[5] += scalar * (A[1] * S[3] + A[4] * S[4] + A[7] * S[5]);
    C[6] += scalar * (A[2] * S[0] + A[5] * S[1] + A[8] * S[3]);
    C[7] += scalar * (A[2] * S[1] + A[5] * S[2] + A[8] * S[4]);
    C[8] += scalar * (A[2] * S[3] + A[5] * S[4] + A[8] * S[5]);
  }
}

template <typename T, int Anrows, int Ancols, int Bncols,
          MatOp opA = MatOp::NORMAL, bool additive = false>
A2D_FUNCTION void MatSMatMultCoreGeneral(const T A[], const T SB[], T C[]) {
  int idim = opA == MatOp::NORMAL ? Anrows : Ancols;
  int jdim = opA == MatOp::NORMAL ? Ancols : Anrows;
  int kdim = Bncols;

  if constexpr (opA == MatOp::NORMAL) {
    for (int i = 0; i < idim; i++) {
      for (int k = 0; k < kdim; k++) {
        T value = T(0.0);
        for (int j = 0; j < jdim; j++) {
          value += A[i * Ancols + j] *
                   SB[j >= k ? k + j * (j + 1) / 2 : j + k * (k + 1) / 2];
        }
        if (additive) {
          C[i * Bncols + k] += value;  // C: Anrows-by-Bncols
        } else {
          C[i * Bncols + k] = value;  // C: Anrows-by-Bncols
        }
      }
    }
  } else {
    for (int i = 0; i < idim; i++) {
      for (int k = 0; k < kdim; k++) {
        T value = T(0.0);
        for (int j = 0; j < jdim; j++) {
          value += A[j * Ancols + i] *
                   SB[j >= k ? k + j * (j + 1) / 2 : j + k * (k + 1) / 2];
        }
        if (additive) {
          C[i * Bncols + k] += value;  // C: Ancols-by-Bncols
        } else {
          C[i * Bncols + k] = value;  // C: Ancols-by-Bncols
        }
      }
    }
  }
}

template <typename T, int Anrows, int Ancols, int Bncols,
          MatOp opA = MatOp::NORMAL, bool additive = false>
A2D_FUNCTION void MatSMatMultScaleCoreGeneral(const T alpha, const T A[],
                                              const T SB[], T C[]) {
  int idim = opA == MatOp::NORMAL ? Anrows : Ancols;
  int jdim = opA == MatOp::NORMAL ? Ancols : Anrows;
  int kdim = Bncols;

  if constexpr (opA == MatOp::NORMAL) {
    for (int i = 0; i < idim; i++) {
      for (int k = 0; k < kdim; k++) {
        T value = T(0.0);
        for (int j = 0; j < jdim; j++) {
          value += A[i * Ancols + j] *
                   SB[j >= k ? k + j * (j + 1) / 2 : j + k * (k + 1) / 2];
        }
        if (additive) {
          C[i * Bncols + k] += alpha * value;  // C: Anrows-by-Bncols
        } else {
          C[i * Bncols + k] = alpha * value;  // C: Anrows-by-Bncols
        }
      }
    }
  } else {
    for (int i = 0; i < idim; i++) {
      for (int k = 0; k < kdim; k++) {
        T value = T(0.0);
        for (int j = 0; j < jdim; j++) {
          value += A[j * Ancols + i] *
                   SB[j >= k ? k + j * (j + 1) / 2 : j + k * (k + 1) / 2];
        }
        if (additive) {
          C[i * Bncols + k] += alpha * value;  // C: Ancols-by-Bncols
        } else {
          C[i * Bncols + k] = alpha * value;  // C: Ancols-by-Bncols
        }
      }
    }
  }
}

template <typename T, int Anrows, int Ancols, int Bnrows, int Cnrows,
          int Cncols, MatOp opA = MatOp::NORMAL, bool additive = false>
A2D_FUNCTION void MatSMatMultCore(const T A[], const T S[], T C[]) {
  // Check if shapes are consistent
  if constexpr (opA == MatOp::TRANSPOSE) {
    static_assert(Anrows == Bnrows && Ancols == Cnrows && Bnrows == Cncols,
                  "Matrix dimensions must agree.");
  } else {
    static_assert(Ancols == Bnrows && Anrows == Cnrows && Bnrows == Cncols,
                  "Matrix dimensions must agree.");
  }

  if constexpr (Anrows == 2 && Ancols == 2 && Bnrows == 2) {
    if constexpr (additive) {
      MatSMatMultCore2x2Add<T, opA>(A, S, C);
    } else {
      MatSMatMultCore2x2<T, opA>(A, S, C);
    }

  } else if constexpr (Anrows == 3 && Ancols == 3 && Bnrows == 3) {
    if constexpr (additive) {
      MatSMatMultCore3x3Add<T, opA>(A, S, C);
    } else {
      MatSMatMultCore3x3<T, opA>(A, S, C);
    }
  } else {  // The general fallback implmentation
    MatSMatMultCoreGeneral<T, Anrows, Ancols, Bnrows, opA, additive>(A, S, C);
  }
}

template <typename T, int Anrows, int Ancols, int Bnrows, int Cnrows,
          int Cncols, MatOp opA = MatOp::NORMAL, bool additive = false>
A2D_FUNCTION void MatSMatMultScaleCore(const T alpha, const T A[], const T S[],
                                       T C[]) {
  // Check if shapes are consistent
  if constexpr (opA == MatOp::TRANSPOSE) {
    static_assert(Anrows == Bnrows && Ancols == Cnrows && Bnrows == Cncols,
                  "Matrix dimensions must agree.");
  } else {
    static_assert(Ancols == Bnrows && Anrows == Cnrows && Bnrows == Cncols,
                  "Matrix dimensions must agree.");
  }

  if constexpr (Anrows == 2 && Ancols == 2 && Bnrows == 2) {
    if constexpr (additive) {
      MatSMatMultCore2x2ScaleAdd<T, opA>(alpha, A, S, C);
    } else {
      MatSMatMultCore2x2Scale<T, opA>(alpha, A, S, C);
    }

  } else if constexpr (Anrows == 3 && Ancols == 3 && Bnrows == 3) {
    if constexpr (additive) {
      MatSMatMultCore3x3ScaleAdd<T, opA>(alpha, A, S, C);
    } else {
      MatSMatMultCore3x3Scale<T, opA>(alpha, A, S, C);
    }
  } else {  // The general fallback implmentation
    MatSMatMultScaleCoreGeneral<T, Anrows, Ancols, Bnrows, opA, additive>(
        alpha, A, S, C);
  }
}

}  // namespace A2D
#endif  // A2D_GEMMCORE_H
