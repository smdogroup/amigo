#ifndef A2D_MAT_INV_CORE_H
#define A2D_MAT_INV_CORE_H

#include "../../a2ddefs.h"

namespace A2D {

template <typename T, int N>
A2D_FUNCTION void MatInvCore(const T A[], T Ainv[]) {
  static_assert((N >= 1 && N <= 3), "MatInvCore not implemented for N >= 4");

  if constexpr (N == 1) {
    Ainv[0] = 1.0 / A[0];
  } else if constexpr (N == 2) {
    T det = A[0] * A[3] - A[1] * A[2];
    T detinv = 1.0 / det;

    Ainv[0] = A[3] * detinv;
    Ainv[1] = -A[1] * detinv;
    Ainv[2] = -A[2] * detinv;
    Ainv[3] = A[0] * detinv;
  } else {  // N == 3
    T det = (A[8] * (A[0] * A[4] - A[3] * A[1]) -
             A[7] * (A[0] * A[5] - A[3] * A[2]) +
             A[6] * (A[1] * A[5] - A[2] * A[4]));
    T detinv = 1.0 / det;

    Ainv[0] = (A[4] * A[8] - A[5] * A[7]) * detinv;
    Ainv[1] = -1.0 * (A[1] * A[8] - A[2] * A[7]) * detinv;
    Ainv[2] = (A[1] * A[5] - A[2] * A[4]) * detinv;

    Ainv[3] = -1.0 * (A[3] * A[8] - A[5] * A[6]) * detinv;
    Ainv[4] = (A[0] * A[8] - A[2] * A[6]) * detinv;
    Ainv[5] = -1.0 * (A[0] * A[5] - A[2] * A[3]) * detinv;

    Ainv[6] = (A[3] * A[7] - A[4] * A[6]) * detinv;
    Ainv[7] = -1.0 * (A[0] * A[7] - A[1] * A[6]) * detinv;
    Ainv[8] = (A[0] * A[4] - A[1] * A[3]) * detinv;
  }
}

template <typename T, int N>
A2D_FUNCTION void SymMatInvCore(const T S[], T Sinv[]) {
  static_assert((N >= 1 && N <= 3), "MatInvCore not implemented for N >= 4");

  if constexpr (N == 1) {
    Sinv[0] = 1.0 / S[0];
  } else if constexpr (N == 2) {
    T det = S[0] * S[2] - S[1] * S[1];
    T detinv = 1.0 / det;

    Sinv[0] = S[2] * detinv;
    Sinv[1] = -S[1] * detinv;
    Sinv[2] = S[0] * detinv;
  } else {  // N == 3
    T det = (S[5] * (S[0] * S[2] - S[1] * S[1]) -
             S[4] * (S[0] * S[4] - S[1] * S[3]) +
             S[3] * (S[1] * S[4] - S[3] * S[2]));
    T detinv = 1.0 / det;

    Sinv[0] = (S[2] * S[5] - S[4] * S[4]) * detinv;
    Sinv[1] = -(S[1] * S[5] - S[3] * S[4]) * detinv;
    Sinv[3] = (S[1] * S[4] - S[3] * S[2]) * detinv;

    Sinv[2] = (S[0] * S[5] - S[3] * S[3]) * detinv;
    Sinv[4] = -(S[0] * S[4] - S[3] * S[1]) * detinv;

    Sinv[5] = (S[0] * S[2] - S[1] * S[1]) * detinv;
  }
}

}  // namespace A2D

#endif  // A2D_MAT_INV_CORE_H
