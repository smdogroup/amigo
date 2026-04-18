#ifndef A2D_SYM_MAT_SUM_H
#define A2D_SYM_MAT_SUM_H

#include <type_traits>

#include "../a2ddefs.h"
#include "a2dmat.h"
#include "a2dtest.h"
#include "core/a2dgemmcore.h"

namespace A2D {

template <typename T, int N, bool additive = false>
A2D_FUNCTION void SymMatSumCore(const T A[], T S[]) {
  if constexpr (additive) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j <= i; j++) {
        S[0] += (A[N * i + j] + A[N * j + i]);
        S++;
      }
    }
  } else {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j <= i; j++) {
        S[0] = (A[N * i + j] + A[N * j + i]);
        S++;
      }
    }
  }
}

template <typename T, int N, bool additive = false>
A2D_FUNCTION void SymMatSumCore(const T alpha, const T A[], T S[]) {
  if constexpr (additive) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j <= i; j++) {
        S[0] += alpha * (A[N * i + j] + A[N * j + i]);
        S++;
      }
    }
  } else {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j <= i; j++) {
        S[0] = alpha * (A[N * i + j] + A[N * j + i]);
        S++;
      }
    }
  }
}

template <typename T, int N>
A2D_FUNCTION void SymMatSumCoreReverse(const T alpha, const T Sb[], T Ab[]) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j <= i; j++) {
      Ab[N * i + j] += alpha * Sb[0];
      Ab[N * j + i] += alpha * Sb[0];
      Sb++;
    }
  }
}

template <typename T, int N>
A2D_FUNCTION T SymMatSumCoreReverse(const T A[], const T Sb[]) {
  T val = 0.0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j <= i; j++) {
      val += Sb[0] * (A[N * i + j] + A[N * j + i]);
      Sb++;
    }
  }
  return val;
}

template <typename T, int N>
A2D_FUNCTION void SymMatSum(const Mat<T, N, N> &A, SymMat<T, N> &S) {
  SymMatSumCore<T, N>(get_data(A), get_data(S));
}

template <typename T, int N>
A2D_FUNCTION void SymMatSum(const T alpha, const Mat<T, N, N> &A,
                            SymMat<T, N> &S) {
  SymMatSumCore<T, N>(alpha, get_data(A), get_data(S));
}

template <class atype, class Atype, class Stype>
class SymMatSumExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<Stype>::type T;

  // Extract the dimensions of the underlying matrix
  static constexpr int M = get_matrix_rows<Atype>::size;
  static constexpr int K = get_matrix_columns<Atype>::size;
  static constexpr int N = get_symmatrix_size<Stype>::size;

  // Get the differentiation order from the output
  static constexpr ADorder order = get_diff_order<Stype>::order;

  // Get the types of the matrices
  static constexpr ADiffType ada = get_diff_type<atype>::diff_type;
  static constexpr ADiffType adA = get_diff_type<Atype>::diff_type;

  // Make sure the matrix dimensions are consistent
  static_assert((N == K && N == M), "Matrix dimensions must agree");

  A2D_FUNCTION SymMatSumExpr(atype alpha, Atype &A, Stype &S)
      : alpha(alpha), A(A), S(S) {}

  A2D_FUNCTION void eval() {
    SymMatSumCore<T, N>(get_data(alpha), get_data(A), get_data(S));
  }

  A2D_FUNCTION void bzero() { S.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;
    if constexpr (ada == ADiffType::ACTIVE && adA == ADiffType::ACTIVE) {
      SymMatSumCore<T, N>(GetSeed<seed>::get_data(alpha), get_data(A),
                          GetSeed<seed>::get_data(S));
      SymMatSumCore<T, N, true>(get_data(alpha), GetSeed<seed>::get_data(A),
                                GetSeed<seed>::get_data(S));
    } else if constexpr (ada == ADiffType::ACTIVE) {
      SymMatSumCore<T, N>(GetSeed<seed>::get_data(alpha), get_data(A),
                          GetSeed<seed>::get_data(S));
    } else if constexpr (adA == ADiffType::ACTIVE) {
      SymMatSumCore<T, N>(get_data(alpha), GetSeed<seed>::get_data(A),
                          GetSeed<seed>::get_data(S));
    }
  }

  A2D_FUNCTION void reverse() {
    constexpr ADseed seed = ADseed::b;
    if constexpr (ada == ADiffType::ACTIVE) {
      GetSeed<seed>::get_data(alpha) +=
          SymMatSumCoreReverse<T, N>(get_data(A), GetSeed<seed>::get_data(S));
    }
    if constexpr (adA == ADiffType::ACTIVE) {
      SymMatSumCoreReverse<T, N>(get_data(alpha), GetSeed<seed>::get_data(S),
                                 GetSeed<seed>::get_data(A));
    }
  }

  A2D_FUNCTION void hzero() { S.hzero(); }

  A2D_FUNCTION void hreverse() {
    constexpr ADseed seed = ADseed::h;
    if constexpr (ada == ADiffType::ACTIVE) {
      GetSeed<seed>::get_data(alpha) +=
          SymMatSumCoreReverse<T, N>(get_data(A), GetSeed<seed>::get_data(S));
    }
    if constexpr (adA == ADiffType::ACTIVE) {
      SymMatSumCoreReverse<T, N>(get_data(alpha), GetSeed<seed>::get_data(S),
                                 GetSeed<seed>::get_data(A));
    }
    if constexpr (ada == ADiffType::ACTIVE && adA == ADiffType::ACTIVE) {
      GetSeed<seed>::get_data(alpha) += SymMatSumCoreReverse<T, N>(
          GetSeed<ADseed::p>::get_data(A), GetSeed<ADseed::b>::get_data(S));
      SymMatSumCoreReverse<T, N>(GetSeed<ADseed::p>::get_data(alpha),
                                 GetSeed<ADseed::b>::get_data(S),
                                 GetSeed<seed>::get_data(A));
    }
  }

  atype alpha;
  Atype &A;
  Stype &S;
};

// Scalar constant alpha = 1.0 case
template <class Atype, class Stype>
A2D_FUNCTION auto SymMatSum(ADObj<Atype> &A, ADObj<Stype> &S) {
  using T = typename get_object_numeric_type<Stype>::type;
  return SymMatSumExpr<const T, ADObj<Atype>, ADObj<Stype>>(T(1.0), A, S);
}

template <class Atype, class Stype>
A2D_FUNCTION auto SymMatSum(A2DObj<Atype> &A, A2DObj<Stype> &S) {
  using T = typename get_object_numeric_type<Stype>::type;
  return SymMatSumExpr<const T, A2DObj<Atype>, A2DObj<Stype>>(T(1.0), A, S);
}

// Cases with alpha
template <class atype, class Atype, class Stype>
A2D_FUNCTION auto SymMatSum(ADObj<atype> &alpha, ADObj<Atype> &A,
                            ADObj<Stype> &S) {
  return SymMatSumExpr<ADObj<atype> &, ADObj<Atype>, ADObj<Stype>>(alpha, A, S);
}

template <class atype, class Atype, class Stype>
A2D_FUNCTION auto SymMatSum(A2DObj<atype> &alpha, A2DObj<Atype> &A,
                            A2DObj<Stype> &S) {
  return SymMatSumExpr<A2DObj<atype> &, A2DObj<Atype>, A2DObj<Stype>>(alpha, A,
                                                                      S);
}

// Cases with passive variables
template <class atype, class Atype, class Stype>
A2D_FUNCTION auto SymMatSum(const atype alpha, ADObj<Atype> &A,
                            ADObj<Stype> &S) {
  return SymMatSumExpr<const atype, ADObj<Atype>, ADObj<Stype>>(alpha, A, S);
}

template <class atype, class Atype, class Stype>
A2D_FUNCTION auto SymMatSum(const atype alpha, A2DObj<Atype> &A,
                            A2DObj<Stype> &S) {
  return SymMatSumExpr<const atype, A2DObj<Atype>, A2DObj<Stype>>(alpha, A, S);
}

template <class atype, class Atype, class Stype>
A2D_FUNCTION auto SymMatSum(ADObj<atype> &alpha, const Atype &A,
                            ADObj<Stype> &S) {
  return SymMatSumExpr<ADObj<atype> &, const Atype, ADObj<Stype>>(alpha, A, S);
}

template <class atype, class Atype, class Stype>
A2D_FUNCTION auto SymMatSum(A2DObj<atype> &alpha, Atype &A, A2DObj<Stype> &S) {
  return SymMatSumExpr<A2DObj<atype> &, const Atype, A2DObj<Stype>>(alpha, A,
                                                                    S);
}

namespace Test {

template <typename T, int N>
class SymMatSumTest : public A2DTest<T, SymMat<T, N>, Mat<T, N, N>> {
 public:
  using Input = VarTuple<T, Mat<T, N, N>>;
  using Output = VarTuple<T, SymMat<T, N>>;

  // Assemble a string to describe the test
  std::string name() {
    std::stringstream s;
    s << "SymMatSum<" << N << ">";
    return s.str();
  }

  // Evaluate the matrix-matrix product
  Output eval(const Input &x) {
    Mat<T, N, N> A;
    SymMat<T, N> S;
    x.get_values(A);
    SymMatSum(A, S);
    return MakeVarTuple<T>(S);
  }

  // Compute the derivative
  void deriv(const Output &seed, const Input &x, Input &g) {
    ADObj<Mat<T, N, N>> A;
    ADObj<SymMat<T, N>> S;
    x.get_values(A.value());
    auto stack = MakeStack(SymMatSum(A, S));
    seed.get_values(S.bvalue());
    stack.reverse();
    g.set_values(A.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output &seed, const Output &hval, const Input &x,
             const Input &p, Input &h) {
    A2DObj<SymMat<T, N>> S;
    A2DObj<Mat<T, N, N>> A;
    x.get_values(A.value());
    p.get_values(A.pvalue());
    auto stack = MakeStack(SymMatSum(A, S));
    seed.get_values(S.bvalue());
    hval.get_values(S.hvalue());
    stack.hproduct();
    h.set_values(A.hvalue());
  }
};

template <typename T, int N>
class SymMatSumScaleTest : public A2DTest<T, SymMat<T, N>, T, Mat<T, N, N>> {
 public:
  using Input = VarTuple<T, T, Mat<T, N, N>>;
  using Output = VarTuple<T, SymMat<T, N>>;

  // Assemble a string to describe the test
  std::string name() {
    std::stringstream s;
    s << "SymMatSum<" << N << ">";
    return s.str();
  }

  // Evaluate the matrix-matrix product
  Output eval(const Input &x) {
    T alpha;
    Mat<T, N, N> A;
    SymMat<T, N> S;
    x.get_values(alpha, A);
    SymMatSum(alpha, A, S);
    return MakeVarTuple<T>(S);
  }

  // Compute the derivative
  void deriv(const Output &seed, const Input &x, Input &g) {
    ADObj<T> alpha;
    ADObj<Mat<T, N, N>> A;
    ADObj<SymMat<T, N>> S;
    x.get_values(alpha.value(), A.value());
    auto stack = MakeStack(SymMatSum(alpha, A, S));
    seed.get_values(S.bvalue());
    stack.reverse();
    g.set_values(alpha.bvalue(), A.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output &seed, const Output &hval, const Input &x,
             const Input &p, Input &h) {
    A2DObj<T> alpha;
    A2DObj<SymMat<T, N>> S;
    A2DObj<Mat<T, N, N>> A;
    x.get_values(alpha.value(), A.value());
    p.get_values(alpha.pvalue(), A.pvalue());
    auto stack = MakeStack(SymMatSum(alpha, A, S));
    seed.get_values(S.bvalue());
    hval.get_values(S.hvalue());
    stack.hproduct();
    h.set_values(alpha.hvalue(), A.hvalue());
  }
};

inline bool SymMatSumTestAll(bool component = false, bool write_output = true) {
  using Tc = A2D_complex_t<double>;

  bool passed = true;
  SymMatSumTest<Tc, 4> test1;
  passed = passed && Run(test1, component, write_output);

  SymMatSumScaleTest<Tc, 3> test2;
  passed = passed && Run(test2, component, write_output);

  return passed;
}

}  // namespace Test

}  // namespace A2D

#endif  //  A2D_SYM_MAT_SUM_H
