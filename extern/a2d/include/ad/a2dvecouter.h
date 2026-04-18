#ifndef A2D_VEC_OUTER_H
#define A2D_VEC_OUTER_H

#include "../a2ddefs.h"
#include "a2dmat.h"
#include "a2dvec.h"
#include "core/a2dmatveccore.h"
#include "core/a2dveccore.h"

namespace A2D {

template <typename T, int M, int N>
A2D_FUNCTION void VecOuter(const Vec<T, M>& x, const Vec<T, N>& y,
                           Mat<T, M, N>& A) {
  VecOuterCore<T, M, N>(get_data(x), get_data(y), get_data(A));
}

template <typename T, int M, int N>
A2D_FUNCTION void VecOuter(const T alpha, const Vec<T, M>& x,
                           const Vec<T, N>& y, Mat<T, M, N>& A) {
  VecOuterCore<T, M, N>(alpha, get_data(x), get_data(y), get_data(A));
}

/*
  Compute the vector outer product

  A = alpha * x * y^{T}

  where alpha is a numeric constant scalar parameter.

  The forward mode derivative is

  dot{A} = alpha * (dot{x} * y^{T} + x * dot{y}^T)

  Using the trace identity the reverse mode derivatives are

  bar{x} = alpha * bar{A} * y
  bar{y} = alpha * bar{A}^{T} * x
*/
template <class xtype, class ytype, class Atype>
// typename T, int M, int N, ADorder order, ADiffType adx, ADiffType ady>
class VecOuterExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<Atype>::type T;

  // Extract the dimensions of the underlying vectors
  static constexpr int M = get_vec_size<xtype>::size;
  static constexpr int N = get_vec_size<ytype>::size;

  // Extract the underlying sizes of the matrix
  static constexpr int K = get_matrix_rows<Atype>::size;
  static constexpr int L = get_matrix_columns<Atype>::size;

  // Get the types of the matrices
  static constexpr ADiffType adx = get_diff_type<xtype>::diff_type;
  static constexpr ADiffType ady = get_diff_type<ytype>::diff_type;

  static_assert((M == K && N == L), "Matrix and vector dimensions must agree");

  A2D_FUNCTION VecOuterExpr(const T alpha, xtype& x, ytype& y, Atype& A)
      : alpha(alpha), x(x), y(y), A(A) {}

  A2D_FUNCTION void eval() {
    VecOuterCore<T, M, N>(alpha, get_data(x), get_data(y), get_data(A));
  }

  A2D_FUNCTION void bzero() { A.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;

    if constexpr (adx == ADiffType::ACTIVE && ady == ADiffType::ACTIVE) {
      constexpr bool additive = true;
      VecOuterCore<T, M, N>(alpha, GetSeed<seed>::get_data(x), get_data(y),
                            GetSeed<seed>::get_data(A));
      VecOuterCore<T, M, N, additive>(alpha, get_data(x),
                                      GetSeed<seed>::get_data(y),
                                      GetSeed<seed>::get_data(A));
    } else if constexpr (adx == ADiffType::ACTIVE) {
      VecOuterCore<T, M, N>(alpha, GetSeed<seed>::get_data(x), get_data(y),
                            GetSeed<seed>::get_data(A));
    } else if constexpr (ady == ADiffType::ACTIVE) {
      VecOuterCore<T, M, N>(alpha, get_data(x), GetSeed<seed>::get_data(y),
                            GetSeed<seed>::get_data(A));
    }
  }

  A2D_FUNCTION void reverse() {
    constexpr ADseed seed = ADseed::b;
    if constexpr (adx == ADiffType::ACTIVE) {
      constexpr bool additive = true;
      MatVecCoreScale<T, M, N, MatOp::NORMAL, additive>(
          alpha, GetSeed<seed>::get_data(A), get_data(y),
          GetSeed<seed>::get_data(x));
    }
    if constexpr (ady == ADiffType::ACTIVE) {
      constexpr bool additive = true;
      MatVecCoreScale<T, M, N, MatOp::TRANSPOSE, additive>(
          alpha, GetSeed<seed>::get_data(A), get_data(x),
          GetSeed<seed>::get_data(y));
    }
  }

  A2D_FUNCTION void hzero() { A.hzero(); }

  A2D_FUNCTION void hreverse() {
    constexpr bool additive = true;
    if constexpr (adx == ADiffType::ACTIVE) {
      MatVecCoreScale<T, M, N, MatOp::NORMAL, additive>(
          alpha, GetSeed<ADseed::h>::get_data(A), get_data(y),
          GetSeed<ADseed::h>::get_data(x));
    }
    if constexpr (ady == ADiffType::ACTIVE) {
      MatVecCoreScale<T, M, N, MatOp::TRANSPOSE, additive>(
          alpha, GetSeed<ADseed::h>::get_data(A), get_data(x),
          GetSeed<ADseed::h>::get_data(y));
    }
    if constexpr (adx == ADiffType::ACTIVE && ady == ADiffType::ACTIVE) {
      MatVecCoreScale<T, M, N, MatOp::NORMAL, additive>(
          alpha, GetSeed<ADseed::b>::get_data(A),
          GetSeed<ADseed::p>::get_data(y), GetSeed<ADseed::h>::get_data(x));
      MatVecCoreScale<T, M, N, MatOp::TRANSPOSE, additive>(
          alpha, GetSeed<ADseed::b>::get_data(A),
          GetSeed<ADseed::p>::get_data(x), GetSeed<ADseed::h>::get_data(y));
    }
  }

 private:
  const T alpha;
  xtype& x;
  ytype& y;
  Atype& A;
};

template <class xtype, class ytype, class Atype>
A2D_FUNCTION auto VecOuter(ADObj<xtype>& x, ADObj<ytype>& y, ADObj<Atype>& A) {
  using T = typename get_object_numeric_type<Atype>::type;
  return VecOuterExpr<ADObj<xtype>, ADObj<ytype>, ADObj<Atype>>(T(1.0), x, y,
                                                                A);
}

template <typename T, class xtype, class ytype, class Atype>
A2D_FUNCTION auto VecOuter(const T alpha, ADObj<xtype>& x, ADObj<ytype>& y,
                           ADObj<Atype>& A) {
  return VecOuterExpr<ADObj<xtype>, ADObj<ytype>, ADObj<Atype>>(alpha, x, y, A);
}

template <class xtype, class ytype, class Atype>
A2D_FUNCTION auto VecOuter(A2DObj<xtype>& x, A2DObj<ytype>& y,
                           A2DObj<Atype>& A) {
  using T = typename get_object_numeric_type<Atype>::type;
  return VecOuterExpr<A2DObj<xtype>, A2DObj<ytype>, A2DObj<Atype>>(T(1.0), x, y,
                                                                   A);
}

template <typename T, class xtype, class ytype, class Atype>
A2D_FUNCTION auto VecOuter(const T alpha, A2DObj<xtype>& x, A2DObj<ytype>& y,
                           A2DObj<Atype>& A) {
  return VecOuterExpr<A2DObj<xtype>, A2DObj<ytype>, A2DObj<Atype>>(alpha, x, y,
                                                                   A);
}

namespace Test {

template <typename T, int N, int M>
class VecOuterTest : public A2DTest<T, Mat<T, N, M>, Vec<T, N>, Vec<T, M>> {
 public:
  using Input = VarTuple<T, Vec<T, N>, Vec<T, M>>;
  using Output = VarTuple<T, Mat<T, N, M>>;

  // Assemble a string to describe the test
  std::string name() { return "VecOuter"; }

  // Evaluate the matrix-matrix product
  Output eval(const Input& X) {
    T alpha(0.3157);
    Vec<T, N> x;
    Vec<T, M> y;
    Mat<T, N, M> A;
    X.get_values(x, y);
    VecOuter(alpha, x, y, A);
    return MakeVarTuple<T>(A);
  }

  // Compute the derivative
  void deriv(const Output& seed, const Input& X, Input& g) {
    T alpha(0.3157);
    ADObj<Vec<T, N>> x;
    ADObj<Vec<T, M>> y;
    ADObj<Mat<T, N, M>> A;
    X.get_values(x.value(), y.value());
    auto stack = MakeStack(VecOuter(alpha, x, y, A));
    seed.get_values(A.bvalue());
    stack.reverse();
    g.set_values(x.bvalue(), y.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output& seed, const Output& hval, const Input& X,
             const Input& p, Input& h) {
    T alpha(0.3157);
    A2DObj<Vec<T, N>> x;
    A2DObj<Vec<T, M>> y;
    A2DObj<Mat<T, N, M>> A;
    X.get_values(x.value(), y.value());
    p.get_values(x.pvalue(), y.pvalue());
    auto stack = MakeStack(VecOuter(alpha, x, y, A));
    seed.get_values(A.bvalue());
    hval.get_values(A.hvalue());
    stack.hproduct();
    h.set_values(x.hvalue(), y.hvalue());
  }
};

inline bool VecOuterTestAll(bool component = false, bool write_output = true) {
  using Tc = A2D_complex_t<double>;

  bool passed = true;
  VecOuterTest<Tc, 3, 5> test1;
  passed = passed && Run(test1, component, write_output);
  VecOuterTest<Tc, 6, 4> test2;
  passed = passed && Run(test2, component, write_output);

  return passed;
}

}  // namespace Test

}  // namespace A2D

#endif  // A2D_VEC_OUTER_H
