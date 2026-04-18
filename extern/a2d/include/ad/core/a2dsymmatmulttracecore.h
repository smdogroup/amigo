#ifndef A2D_SYM_TRACE_CORE_H
#define A2D_SYM_TRACE_CORE_H

#include "../../a2ddefs.h"

namespace A2D {

template <typename T, int N>
A2D_FUNCTION T SymMatMultTraceCore(const T S[], const T E[]) {
  if constexpr (N == 1) {
    return S[0] * E[0];
  } else if constexpr (N == 2) {
    return S[0] * E[0] + S[2] * E[2] + 2.0 * S[1] * E[1];
  } else if constexpr (N == 3) {
    return S[0] * E[0] + S[2] * E[2] + S[5] * E[5] +
           2.0 * (S[1] * E[1] + S[3] * E[3] + S[4] * E[4]);
  } else {
    T trace = 0.0;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < i; j++, S++, E++) {
        trace += 2.0 * S[0] * E[0];
      }
      trace += S[0] * E[0];
      S++, E++;
    }
    return trace;
  }
}

template <typename T, int N>
A2D_FUNCTION void SymMatMultTraceReverseCore(const T scale, const T S[],
                                             T E[]) {
  if constexpr (N == 1) {
    E[0] += scale * S[0];
  } else if constexpr (N == 2) {
    E[0] += scale * S[0];
    E[1] += 2.0 * scale * S[1];
    E[2] += scale * S[2];
  } else if constexpr (N == 3) {
    E[0] += scale * S[0];
    E[1] += 2.0 * scale * S[1];
    E[2] += scale * S[2];
    E[3] += 2.0 * scale * S[3];
    E[4] += 2.0 * scale * S[4];
    E[5] += scale * S[5];
  } else {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < i; j++, S++, E++) {
        E[0] += 2.0 * scale * S[0];
      }
      E[0] += scale * S[0];
      S++, E++;
    }
  }
}

}  // namespace A2D

#endif  // A2D_SYM_TRACE_CORE_H