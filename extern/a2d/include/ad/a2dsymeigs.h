#ifndef A2D_SYM_MAT_EIGS_H
#define A2D_SYM_MAT_EIGS_H

#include "../a2ddefs.h"

namespace A2D {

template <typename T>
A2D_FUNCTION void SymEigs2x2(const T* A, T* eigs, T* Q = nullptr) {
  T tr = A[0] + A[2];
  T diff = A[0] - A[2];
  T discrm = sqrt(diff * diff + 4.0 * A[1] * A[1]);
  T det = A[0] * A[2] - A[1] * A[1];

  // Compute the eigenvalues such that eigs[0] <= eigs[1]
  if (RealPart(tr) > 0.0) {
    eigs[1] = 0.5 * (tr + discrm);
    eigs[0] = det / eigs[1];
  } else if (RealPart(tr) < 0.0) {
    eigs[0] = 0.5 * (tr - discrm);
    eigs[1] = det / eigs[0];
  } else {
    eigs[0] = -0.5 * discrm;
    eigs[1] = 0.5 * discrm;
  }

  if (Q != nullptr) {
    // Compute the eigenvector components
    T u = 1.0, v = 0.0;
    if (RealPart(A[1]) != 0.0) {
      if (RealPart(diff) > 0.0) {
        T a = 0.5 * (diff + discrm);
        T inv = 1.0 / sqrt(a * a + A[1] * A[1]);
        u = inv * A[1];
        v = -inv * a;
      } else {
        T a = 0.5 * (diff - discrm);
        T inv = 1.0 / sqrt(a * a + A[1] * A[1]);
        u = inv * a;
        v = inv * A[1];
      }
    }

    // Set the eigenvector components
    Q[0] = u;
    Q[1] = -v;
    Q[2] = v;
    Q[3] = u;
  }
}

/**
 * @brief Reduce a symmetric matrix to tridiagonal form
 *
 * @tparam T Scalar type
 * @tparam N Dimension of the symmetric matrix
 * @param A Input matrix - entries are destroyed
 * @param alpha Array of length N
 * @param beta Array of length N
 * @param w Temporary array of length N
 * @param P Identity on input, transform on output (optional)
 */
template <typename T, int N>
A2D_FUNCTION void SymMatTriReduce(T* A, T* alpha, T* beta, T* w,
                                  T* P = nullptr) {
  for (int j = N - 1; j > 0; j--) {
    // Set the column vector for the Householder transform
    T* aj = &A[j * (j + 1) / 2];

    // Set the A matrix entries
    beta[j - 1] = aj[j - 1];
    alpha[j] = aj[j];

    // Complete the computation for the Householder vector
    T sigma = 0.0;
    for (int i = 0; i < j; i++) {
      sigma += aj[i] * aj[i];
    }
    sigma = sqrt(sigma);

    // Set sigma to reduce round-off error
    if (RealPart(aj[j - 1]) < 0.0) {
      sigma *= -1.0;
    }

    // Comupte h = 1/2 *  u^{T} u
    T h = 0.0;
    for (int i = 0; i < j - 1; i++) {
      h += aj[i] * aj[i];
    }
    h += (aj[j - 1] + sigma) * (aj[j - 1] + sigma);
    T hinv = 0.0;
    if (RealPart(h) != 0.0) {
      hinv = 2.0 / h;
    }

    // Compute the matrix-vector product w = hinv * A * u
    const T* ap = A;
    for (int i = 0; i <= j; i++) {
      w[i] = 0.0;
    }
    for (int i = 0; i <= j; i++) {
      T ui = aj[i];
      if (i == j - 1) {
        ui += sigma;
      } else if (i == j) {
        ui = 0.0;
      }

      for (int k = 0; k < i; k++) {
        T uk = aj[k];
        if (k == j - 1) {
          uk += sigma;
        } else if (k == j) {
          uk = 0.0;
        }

        w[k] += ap[0] * ui;
        w[i] += ap[0] * uk;
        ap++;
      }
      w[i] += ap[0] * ui;
      ap++;
    }
    for (int i = 0; i <= j; i++) {
      w[i] *= hinv;
    }

    // Now adjust the aj column so that it stores u
    T* u = aj;
    u[j - 1] += sigma;
    u[j] = 0.0;

    // kappa = 1/2 * hinv * u^{T} * w
    T kappa = 0.0;
    for (int i = 0; i < j; i++) {
      kappa += u[i] * w[i];
    }
    kappa *= 0.5 * hinv;

    // Apply the update to the remaining parts of the matrix
    T* a = A;
    for (int i = 0; i < j; i++) {
      // q = w - kappa * u
      T qi = w[i] - kappa * u[i];
      for (int k = 0; k < i; k++) {
        T qk = w[k] - kappa * u[k];
        a[0] -= qi * u[k] + qk * u[i];
        a++;
      }
      a[0] -= 2.0 * qi * u[i];
      a++;
    }

    T qj = w[j] - kappa * u[j];
    T qk = w[j - 1] - kappa * u[j - 1];
    beta[j - 1] -= qk * u[j] + qj * u[j - 1];
    alpha[j] -= 2.0 * qj * u[j];

    // Store hinv for later
    u[j] = hinv;
  }

  // Set the last diagonal
  alpha[0] = A[0];

  if (P) {
    for (int j = N - 1; j > 0; j--) {
      const T* u = &A[j * (j + 1) / 2];
      const T hinv = u[j];

      // Compute the full
      for (int i = 0; i < N; i++) {
        w[i] = 0.0;
        for (int k = 0; k < j; k++) {
          // w[i] += P[i, k] * u[k]
          w[i] += P[k + i * N] * u[k];
        }
      }

      // P = P - hinv * w * u^{T}
      for (int i = 0; i < N; i++) {
        for (int k = 0; k < j; k++) {
          P[k + i * N] -= hinv * w[i] * u[k];
        }
      }
    }
  }
}

/**
 * @brief Compute the eigenvalues and optionally eigenvectors of a tridiagonal
 * matrix
 *
 * @tparam T Scalar type
 * @tparam N Dimension of the symmetric matrix
 * @param alpha Diagonal entries of length N (eigenvalue outputs)
 * @param beta Off-diagonal entries - length N (note this is not N-1)
 * @param Q Eigenvectors
 */
template <typename T, int N>
void TriSymEigs(T* alpha, T* beta, T* Q) {
  for (int j = 0; j < N; j++) {
    for (int iter = 0; iter < 30; iter++) {
      int m = j;
      for (; m < N - 1; m++) {
        double tr =
            (std::fabs(RealPart(alpha[m + 1])) + std::fabs(RealPart(alpha[m])));
        if (tr + std::fabs(RealPart(beta[m])) == tr) {
          break;
        }
      }

      //  m is the index we were looking for
      if (m != j) {
        // Compute whether the shift should be positive or negative
        T g = (alpha[j + 1] - alpha[j]) / (2.0 * beta[j]);
        T r = sqrt(1.0 + g * g);

        // Compute the shift using the expression with less roundoff error
        if (RealPart(g) >= 0.0) {
          g = alpha[m] - alpha[j] + beta[j] / (g + r);
        } else {
          g = alpha[m] - alpha[j] + beta[j] / (g - r);
        }

        // Apply the transformation
        T c = 1.0, s = 1.0, p = 0.0;
        int i = m - 1;
        for (; i >= j; i--) {
          T f = s * beta[i];
          T b = c * beta[i];
          r = sqrt(f * f + g * g);
          beta[i + 1] = r;

          if (RealPart(r) == 0.0) {
            alpha[i + 1] -= p;
            beta[m] = 0.0;
            break;
          }

          s = f / r;
          c = g / r;
          g = alpha[i + 1] - p;
          r = (alpha[i] - g) * s + 2.0 * c * b;
          p = s * r;
          alpha[i + 1] = g + p;
          g = c * r - b;

          if (Q) {
            // Apply plane rotations to Q
            for (int k = 0; k < N; k++) {
              T q = Q[i + 1 + k * N];
              Q[i + 1 + k * N] = s * Q[i + k * N] + c * q;
              Q[i + k * N] = c * Q[i + k * N] - s * q;
            }
          }
        }
        if (RealPart(r) == 0.0 && i >= j) {
          continue;
        }
        alpha[j] -= p;
        beta[j] = g;
        beta[m] = 0.0;
      }
    }
  }
}

template <typename T, int N>
A2D_FUNCTION void SymEigsGeneral(const T* A, T* eigs, T* Q = nullptr) {
  T Acopy[N * (N + 1) / 2], work[2 * N];
  for (int i = 0; i < N * (N + 1) / 2; i++) {
    Acopy[i] = A[i];
  }

  // Initialize Q
  if (Q) {
    for (int i = 0; i < N * N; i++) {
      Q[i] = 0.0;
    }
    for (int i = 0; i < N; i++) {
      Q[i * (N + 1)] = 1.0;
    }
  }
  SymMatTriReduce<T, N>(Acopy, eigs, &work[0], &work[N], Q);
  TriSymEigs<T, N>(eigs, &work[0], Q);
}

template <typename T, int N>
A2D_FUNCTION void SymEigsForward(const T* eigs, const T* Q, const T* Ad,
                                 T* eigsd) {
  // eigsd[k] = Q[j, k]^{T} * Ad[j, i] * Q[i, k]
  for (int k = 0; k < N; k++) {
    T value = 0.0;
    for (int j = 0; j < N; j++) {
      const T* ad = &Ad[j * (j + 1) / 2];

      int i = 0;
      for (; i < j; i++) {
        value += ad[0] * Q[k + i * N] * Q[k + j * N];
        ad++;
      }
      for (; i < N; i++) {
        value += ad[0] * Q[k + i * N] * Q[k + j * N];
        ad += i + 1;
      }
    }
    eigsd[k] += value;
  }
}

template <typename T, int N>
A2D_FUNCTION void SymEigsReverse(const T* eigs, const T* Q, const T* beigs,
                                 T* bA) {
  // bA[i, j] = Q[i, k] * beigs[k] * Q[j, k]
  for (int j = 0; j < N; j++) {
    for (int i = 0; i <= j; i++) {
      T value = 0.0;
      for (int k = 0; k < N; k++) {
        value += beigs[k] * Q[k + i * N] * Q[k + j * N];
      }

      if (i == j) {
        bA[0] += value;
      } else {
        bA[0] += 2.0 * value;
      }
      bA++;
    }
  }
}

// Compute the reverse mode for the second derivative of the eigenvalues
template <typename T, int N>
A2D_FUNCTION void SymEigsHReverse(const T* eigs, const T* Q, const T* beigs,
                                  const T* Ap, T* Ah) {
  // Compute Bp = Q^{T} * Ap * Q
  // Bp[i, j] = Q[k, i] * Ap[k, l] * Q[l, j]
  T Bp[N * N];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      T value = 0.0;
      for (int l = 0; l < N; l++) {
        const T* ap = &Ap[l * (l + 1) / 2];

        int k = 0;
        for (; k < l; k++) {
          value += ap[0] * Q[i + k * N] * Q[j + l * N];
          ap++;
        }
        for (; k < N; k++) {
          value += ap[0] * Q[i + k * N] * Q[j + l * N];
          ap += k + 1;
        }
      }

      Bp[j + i * N] = value;
    }
  }

  // Multiply by the F-matrix
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      // Bp[i, j] = 2.0 * beigs[i] / (eigs[i] - eigs[j])
      if (i == j || eigs[i] == eigs[j]) {
        Bp[j + i * N] = T(0.0);
      } else {
        Bp[j + i * N] *= 2.0 * beigs[i] / (eigs[i] - eigs[j]);
      }
    }
  }

  // Compute Ah += Q * Bp * Q^{T}
  // Ah[i, j] += Q[i, k] * Bp[k, l] * Q[j, l]
  for (int j = 0; j < N; j++) {
    for (int i = 0; i <= j; i++) {
      T value = 0.0;
      for (int k = 0; k < N; k++) {
        for (int l = 0; l < N; l++) {
          value += Bp[l + k * N] * Q[k + i * N] * Q[l + j * N];
        }
      }
      if (i != j) {
        for (int k = 0; k < N; k++) {
          for (int l = 0; l < N; l++) {
            value += Bp[l + k * N] * Q[k + j * N] * Q[l + i * N];
          }
        }
      }

      Ah[0] += value;
      Ah++;
    }
  }
}

/**
 * Compute the eigenvalues and eigenvectors of a symmetric eigenvalue problem
 */
template <typename T, int N>
A2D_FUNCTION void SymEigs(const SymMat<T, N>& S, Vec<T, N>& eigs) {
  if constexpr (N == 2) {
    SymEigs2x2(get_data(S), get_data(eigs));
  } else {
    SymEigsGeneral<T, N>(get_data(S), get_data(eigs));
  }
}

template <class Stype, class etype>
class SymEigsExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<etype>::type T;

  // Extract the dimensions of the underlying matrices
  static constexpr int N = get_symmatrix_size<Stype>::size;

  // Make sure we have the correct size
  static constexpr int K = get_vec_size<etype>::size;
  static_assert(K == N, "Vector of eigenvalues must be correct size");

  A2D_FUNCTION SymEigsExpr(Stype& S, etype& eigs) : S(S), eigs(eigs) {}

  A2D_FUNCTION void eval() {
    if constexpr (N == 2) {
      SymEigs2x2(get_data(S), get_data(eigs), get_data(Q));
    } else {
      SymEigsGeneral<T, N>(get_data(S), get_data(eigs), get_data(Q));
    }
  }

  A2D_FUNCTION void bzero() { eigs.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;
    SymEigsForward<T, N>(get_data(eigs), get_data(Q),
                         GetSeed<seed>::get_data(S),
                         GetSeed<seed>::get_data(eigs));
  }

  A2D_FUNCTION void reverse() {
    constexpr ADseed seed = ADseed::b;
    SymEigsReverse<T, N>(get_data(eigs), get_data(Q),
                         GetSeed<seed>::get_data(eigs),
                         GetSeed<seed>::get_data(S));
  }

  A2D_FUNCTION void hzero() { eigs.hzero(); }

  A2D_FUNCTION void hreverse() {
    SymEigsReverse<T, N>(get_data(eigs), get_data(Q),
                         GetSeed<ADseed::h>::get_data(eigs),
                         GetSeed<ADseed::h>::get_data(S));
    SymEigsHReverse<T, N>(
        get_data(eigs), get_data(Q), GetSeed<ADseed::b>::get_data(eigs),
        GetSeed<ADseed::p>::get_data(S), GetSeed<ADseed::h>::get_data(S));
  }

 private:
  Stype& S;
  etype& eigs;
  Mat<T, N, N> Q;
};

template <class Stype, class etype>
A2D_FUNCTION auto SymEigs(ADObj<Stype>& S, ADObj<etype>& eigs) {
  return SymEigsExpr<ADObj<Stype>, ADObj<etype>>(S, eigs);
}

template <class Stype, class etype>
A2D_FUNCTION auto SymEigs(A2DObj<Stype>& S, A2DObj<etype>& eigs) {
  return SymEigsExpr<A2DObj<Stype>, A2DObj<etype>>(S, eigs);
}

namespace Test {
template <typename T, int N>
class SymEigsTest : public A2DTest<T, Vec<T, N>, SymMat<T, N>> {
 public:
  using Input = VarTuple<T, SymMat<T, N>>;
  using Output = VarTuple<T, Vec<T, N>>;

  // Assemble a string to describe the test
  std::string name() {
    std::stringstream s;
    s << "SymEigsTest<" << N << ">";
    return s.str();
  }

  // Evaluate the matrix-matrix product
  Output eval(const Input& x) {
    SymMat<T, N> S;
    Vec<T, N> eigs;
    x.get_values(S);
    SymEigs(S, eigs);
    return MakeVarTuple<T>(eigs);
  }

  // Compute the derivative
  void deriv(const Output& seed, const Input& x, Input& g) {
    ADObj<SymMat<T, N>> S;
    ADObj<Vec<T, N>> eigs;
    x.get_values(S.value());
    auto stack = MakeStack(SymEigs(S, eigs));
    seed.get_values(eigs.bvalue());
    stack.reverse();
    g.set_values(S.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output& seed, const Output& hval, const Input& x,
             const Input& p, Input& h) {
    A2DObj<SymMat<T, N>> S;
    A2DObj<Vec<T, N>> eigs;
    x.get_values(S.value());
    p.get_values(S.pvalue());
    auto stack = MakeStack(SymEigs(S, eigs));
    seed.get_values(eigs.bvalue());
    hval.get_values(eigs.hvalue());
    stack.hproduct();
    h.set_values(S.hvalue());
  }
};

inline bool SymEigsTestAll(bool component = false, bool write_output = true) {
  using Tc = A2D_complex_t<double>;

  bool passed = true;
  for (int i = 0; i < 10; i++) {
    SymEigsTest<Tc, 2> test1;
    passed = passed && Run(test1, component, write_output);
  }

  for (int i = 0; i < 10; i++) {
    SymEigsTest<Tc, 3> test1;
    passed = passed && Run(test1, component, write_output);
  }

  for (int i = 0; i < 10; i++) {
    SymEigsTest<Tc, 10> test1;
    passed = passed && Run(test1, component, write_output);
  }

  return passed;
}

}  // namespace Test

}  // namespace A2D

#endif  // A2D_SYM_MAT_EIGS_H
