#ifndef A2D_GREEN_STRAIN_CORE_H
#define A2D_GREEN_STRAIN_CORE_H

#include "../../a2ddefs.h"

namespace A2D {

template <typename T, int N>
A2D_FUNCTION void LinearGreenStrainCore(const T Ux[], T E[]) {
  static_assert(N == 2 || N == 3,
                "LinearGreenStrainCore must use N == 2 or N == 3");
  // E = 0.5 * (Ux + Ux^{T})
  if constexpr (N == 2) {
    E[0] = Ux[0];

    E[1] = 0.5 * (Ux[1] + Ux[2]);
    E[2] = Ux[3];
  } else {
    E[0] = Ux[0];

    E[1] = 0.5 * (Ux[1] + Ux[3]);
    E[2] = Ux[4];

    E[3] = 0.5 * (Ux[2] + Ux[6]);
    E[4] = 0.5 * (Ux[5] + Ux[7]);
    E[5] = Ux[8];
  }
}

template <typename T, int N>
A2D_FUNCTION void NonlinearGreenStrainCore(const T Ux[], T E[]) {
  static_assert(N == 2 || N == 3,
                "NonlinearGreenStrainCore must use N == 2 or N == 3");

  // E = 0.5 * (Ux + Ux^{T} + Ux^{T} * Ux)
  if constexpr (N == 2) {
    E[0] = Ux[0] + 0.5 * (Ux[0] * Ux[0] + Ux[2] * Ux[2]);

    E[1] = 0.5 * (Ux[1] + Ux[2] + Ux[0] * Ux[1] + Ux[2] * Ux[3]);
    E[2] = Ux[3] + 0.5 * (Ux[1] * Ux[1] + Ux[3] * Ux[3]);
  } else {
    E[0] = Ux[0] + 0.5 * (Ux[0] * Ux[0] + Ux[3] * Ux[3] + Ux[6] * Ux[6]);

    E[1] =
        0.5 * (Ux[1] + Ux[3] + Ux[0] * Ux[1] + Ux[3] * Ux[4] + Ux[6] * Ux[7]);
    E[2] = Ux[4] + 0.5 * (Ux[1] * Ux[1] + Ux[4] * Ux[4] + Ux[7] * Ux[7]);

    E[3] =
        0.5 * (Ux[2] + Ux[6] + Ux[0] * Ux[2] + Ux[3] * Ux[5] + Ux[6] * Ux[8]);
    E[4] =
        0.5 * (Ux[5] + Ux[7] + Ux[1] * Ux[2] + Ux[4] * Ux[5] + Ux[7] * Ux[8]);
    E[5] = Ux[8] + 0.5 * (Ux[2] * Ux[2] + Ux[5] * Ux[5] + Ux[8] * Ux[8]);
  }
}

template <typename T, int N>
A2D_FUNCTION void LinearGreenStrainForwardCore(const T Ud[], T E[]) {
  static_assert(N == 2 || N == 3,
                "MatGreenStrainForwardCore must use N == 2 or N == 3");

  // E = 0.5 * (Ux + Ux^{T})
  if constexpr (N == 2) {
    E[0] = Ud[0];

    E[1] = 0.5 * (Ud[1] + Ud[2]);
    E[2] = Ud[3];
  } else {
    E[0] = Ud[0];

    E[1] = 0.5 * (Ud[1] + Ud[3]);
    E[2] = Ud[4];

    E[3] = 0.5 * (Ud[2] + Ud[6]);
    E[4] = 0.5 * (Ud[5] + Ud[7]);
    E[5] = Ud[8];
  }
}

template <typename T, int N>
A2D_FUNCTION void NonlinearGreenStrainForwardCore(const T Ux[], const T Ud[],
                                                  T E[]) {
  static_assert(N == 2 || N == 3,
                "NonlinearGreenStrainForwardCore must use N == 2 or N == 3");

  // E = 0.5 * (Ux + Ux^{T} + Ux^{T} * Ux)
  if constexpr (N == 2) {
    E[0] = Ud[0] + Ux[0] * Ud[0] + Ux[2] * Ud[2];

    E[1] = 0.5 * (Ud[1] + Ud[2] + Ux[0] * Ud[1] + Ux[2] * Ud[3] +
                  Ud[0] * Ux[1] + Ud[2] * Ux[3]);
    E[2] = Ud[3] + Ux[1] * Ud[1] + Ux[3] * Ud[3];

  } else {
    E[0] = Ud[0] + Ux[0] * Ud[0] + Ux[3] * Ud[3] + Ux[6] * Ud[6];

    E[1] =
        0.5 * (Ud[1] + Ud[3] + Ux[0] * Ud[1] + Ux[3] * Ud[4] + Ux[6] * Ud[7] +
               Ud[0] * Ux[1] + Ud[3] * Ux[4] + Ud[6] * Ux[7]);
    E[2] = Ud[4] + Ux[1] * Ud[1] + Ux[4] * Ud[4] + Ux[7] * Ud[7];

    E[3] =
        0.5 * (Ud[2] + Ud[6] + Ux[0] * Ud[2] + Ux[3] * Ud[5] + Ux[6] * Ud[8] +
               Ud[0] * Ux[2] + Ud[3] * Ux[5] + Ud[6] * Ux[8]);
    E[4] =
        0.5 * (Ud[5] + Ud[7] + Ux[1] * Ud[2] + Ux[4] * Ud[5] + Ux[7] * Ud[8] +
               Ud[1] * Ux[2] + Ud[4] * Ux[5] + Ud[7] * Ux[8]);
    E[5] = Ud[8] + Ux[2] * Ud[2] + Ux[5] * Ud[5] + Ux[8] * Ud[8];
  }
}

template <typename T, int N>
A2D_FUNCTION void LinearGreenStrainReverseCore(const T Eb[], T Ub[]) {
  static_assert(N == 2 || N == 3,
                "LinearGreenStrainReverseCore must use N == 2 or N == 3");

  // E = 0.5 * (Ux + Ux^{T})
  if constexpr (N == 2) {
    Ub[0] += Eb[0];
    Ub[1] += 0.5 * Eb[1];

    Ub[2] += 0.5 * Eb[1];
    Ub[3] += Eb[2];
  } else {
    // Uxb = Eb
    Ub[0] += Eb[0];
    Ub[1] += 0.5 * Eb[1];
    Ub[2] += 0.5 * Eb[3];

    Ub[3] += 0.5 * Eb[1];
    Ub[4] += Eb[2];
    Ub[5] += 0.5 * Eb[4];

    Ub[6] += 0.5 * Eb[3];
    Ub[7] += 0.5 * Eb[4];
    Ub[8] += Eb[5];
  }
}

template <typename T, int N>
A2D_FUNCTION void NonlinearGreenStrainReverseCore(const T Ux[], const T Eb[],
                                                  T Ub[]) {
  static_assert(N == 2 || N == 3,
                "NonlinearGreenStrainReverseCore must use N == 2 or N == 3");

  // Uxb = (I + Ux) * Eb
  if constexpr (N == 2) {
    // Uxb = (I + Ux) * Eb
    Ub[0] += (Ux[0] + 1.0) * Eb[0] + 0.5 * Ux[1] * Eb[1];
    Ub[1] += 0.5 * (Ux[0] + 1.0) * Eb[1] + Ux[1] * Eb[2];

    Ub[2] += Ux[2] * Eb[0] + 0.5 * (Ux[3] + 1.0) * Eb[1];
    Ub[3] += 0.5 * Ux[2] * Eb[1] + (Ux[3] + 1.0) * Eb[2];
  } else {
    Ub[0] += (Ux[0] + 1.0) * Eb[0] + 0.5 * Ux[1] * Eb[1] + 0.5 * Ux[2] * Eb[3];
    Ub[1] += 0.5 * (Ux[0] + 1.0) * Eb[1] + Ux[1] * Eb[2] + 0.5 * Ux[2] * Eb[4];
    Ub[2] += 0.5 * (Ux[0] + 1.0) * Eb[3] + 0.5 * Ux[1] * Eb[4] + Ux[2] * Eb[5];

    Ub[3] += Ux[3] * Eb[0] + 0.5 * (Ux[4] + 1.0) * Eb[1] + 0.5 * Ux[5] * Eb[3];
    Ub[4] += 0.5 * Ux[3] * Eb[1] + (Ux[4] + 1.0) * Eb[2] + 0.5 * Ux[5] * Eb[4];
    Ub[5] += 0.5 * Ux[3] * Eb[3] + 0.5 * (Ux[4] + 1.0) * Eb[4] + Ux[5] * Eb[5];

    Ub[6] += Ux[6] * Eb[0] + 0.5 * Ux[7] * Eb[1] + 0.5 * (Ux[8] + 1.0) * Eb[3];
    Ub[7] += 0.5 * Ux[6] * Eb[1] + Ux[7] * Eb[2] + 0.5 * (Ux[8] + 1.0) * Eb[4];
    Ub[8] += 0.5 * Ux[6] * Eb[3] + 0.5 * Ux[7] * Eb[4] + (Ux[8] + 1.0) * Eb[5];
  }
}

template <typename T, int N>
A2D_FUNCTION void LinearGreenStrainHReverseCore(const T Eh[], T Uh[]) {
  static_assert(N == 2 || N == 3,
                "LinearGreenStrainHReverseCore must use N == 2 or N == 3");

  if constexpr (N == 2) {
    Uh[0] += Eh[0];
    Uh[1] += 0.5 * Eh[1];

    Uh[2] += 0.5 * Eh[1];
    Uh[3] += Eh[2];
  } else {
    Uh[0] += Eh[0];
    Uh[1] += 0.5 * Eh[1];
    Uh[2] += 0.5 * Eh[3];

    Uh[3] += 0.5 * Eh[1];
    Uh[4] += Eh[2];
    Uh[5] += 0.5 * Eh[4];

    Uh[6] += 0.5 * Eh[3];
    Uh[7] += 0.5 * Eh[4];
    Uh[8] += Eh[5];
  }
}

template <typename T, int N>
A2D_FUNCTION void NonlinearGreenStrainHReverseCore(const T Ux[], const T Up[],
                                                   const T Eb[], const T Eh[],
                                                   T Uh[]) {
  static_assert(N == 2 || N == 3,
                "NonlinearGreenStrainHReverseCore must use N == 2 or N == 3");

  if constexpr (N == 2) {
    Uh[0] += Up[0] * Eb[0] + 0.5 * Up[1] * Eb[1];
    Uh[1] += 0.5 * Up[0] * Eb[1] + Up[1] * Eb[2];
    Uh[2] += Up[2] * Eb[0] + 0.5 * Up[3] * Eb[1];
    Uh[3] += 0.5 * Up[2] * Eb[1] + Up[3] * Eb[2];

    Uh[0] += (Ux[0] + 1.0) * Eh[0] + 0.5 * Ux[1] * Eh[1];
    Uh[1] += 0.5 * (Ux[0] + 1.0) * Eh[1] + Ux[1] * Eh[2];
    Uh[2] += Ux[2] * Eh[0] + 0.5 * (Ux[3] + 1.0) * Eh[1];
    Uh[3] += 0.5 * Ux[2] * Eh[1] + (Ux[3] + 1.0) * Eh[2];
  } else {
    Uh[0] += Up[0] * Eb[0] + 0.5 * Up[1] * Eb[1] + 0.5 * Up[2] * Eb[3];
    Uh[1] += 0.5 * Up[0] * Eb[1] + Up[1] * Eb[2] + 0.5 * Up[2] * Eb[4];
    Uh[2] += 0.5 * Up[0] * Eb[3] + 0.5 * Up[1] * Eb[4] + Up[2] * Eb[5];

    Uh[3] += Up[3] * Eb[0] + 0.5 * Up[4] * Eb[1] + 0.5 * Up[5] * Eb[3];
    Uh[4] += 0.5 * Up[3] * Eb[1] + Up[4] * Eb[2] + 0.5 * Up[5] * Eb[4];
    Uh[5] += 0.5 * Up[3] * Eb[3] + 0.5 * Up[4] * Eb[4] + Up[5] * Eb[5];

    Uh[6] += Up[6] * Eb[0] + 0.5 * Up[7] * Eb[1] + 0.5 * Up[8] * Eb[3];
    Uh[7] += 0.5 * Up[6] * Eb[1] + Up[7] * Eb[2] + 0.5 * Up[8] * Eb[4];
    Uh[8] += 0.5 * Up[6] * Eb[3] + 0.5 * Up[7] * Eb[4] + Up[8] * Eb[5];

    Uh[0] += (Ux[0] + 1.0) * Eh[0] + 0.5 * Ux[1] * Eh[1] + 0.5 * Ux[2] * Eh[3];
    Uh[1] += 0.5 * (Ux[0] + 1.0) * Eh[1] + Ux[1] * Eh[2] + 0.5 * Ux[2] * Eh[4];
    Uh[2] += 0.5 * (Ux[0] + 1.0) * Eh[3] + 0.5 * Ux[1] * Eh[4] + Ux[2] * Eh[5];

    Uh[3] += Ux[3] * Eh[0] + 0.5 * (Ux[4] + 1.0) * Eh[1] + 0.5 * Ux[5] * Eh[3];
    Uh[4] += 0.5 * Ux[3] * Eh[1] + (Ux[4] + 1.0) * Eh[2] + 0.5 * Ux[5] * Eh[4];
    Uh[5] += 0.5 * Ux[3] * Eh[3] + 0.5 * (Ux[4] + 1.0) * Eh[4] + Ux[5] * Eh[5];

    Uh[6] += Ux[6] * Eh[0] + 0.5 * Ux[7] * Eh[1] + 0.5 * (Ux[8] + 1.0) * Eh[3];
    Uh[7] += 0.5 * Ux[6] * Eh[1] + Ux[7] * Eh[2] + 0.5 * (Ux[8] + 1.0) * Eh[4];
    Uh[8] += 0.5 * Ux[6] * Eh[3] + 0.5 * Ux[7] * Eh[4] + (Ux[8] + 1.0) * Eh[5];
  }
}

}  // namespace A2D

#endif  // A2D_GREEN_STRAIN_CORE_H