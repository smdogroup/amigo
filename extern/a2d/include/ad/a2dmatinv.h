#ifndef A2D_MAT_INV_H
#define A2D_MAT_INV_H

#include "../a2ddefs.h"
#include "a2dmat.h"
#include "a2dobj.h"
#include "a2dtest.h"
#include "core/a2dgemmcore.h"
#include "core/a2dmatinvcore.h"
#include "core/a2dveccore.h"

namespace A2D {

/*
  Compute Ainv = A^{-1} for small matrices

  dot{Ainv} = - A^{-1} * dot{A} * A^{-1}

  hA = A^{-T} * Ap^{T} * A^{-T} * Ainvb * A^{-T} +
      A^{-T} * Ainvb * A^{-T} * Ap^{T} * A^{-T} =
    = - (A^{-T} * Ap^{T} * Ab + Ab * Ap^{T} * A^{-T})
*/

template <typename T, int N>
A2D_FUNCTION void MatInv(const Mat<T, N, N> &A, Mat<T, N, N> &Ainv) {
  MatInvCore<T, N>(get_data(A), get_data(Ainv));
}
template <typename T, int N>
A2D_FUNCTION void MatInv(const SymMat<T, N> &S, SymMat<T, N> &Sinv) {
  SymMatInvCore<T, N>(get_data(S), get_data(Sinv));
}

template <class Atype, class Btype>
class MatInvExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<Btype>::type T;

  // Extract the dimensions of the matrix
  static constexpr int N = get_matrix_rows<Atype>::size;
  static constexpr int M = get_matrix_columns<Atype>::size;
  static constexpr int K = get_matrix_rows<Btype>::size;
  static constexpr int L = get_matrix_columns<Btype>::size;

  // Assert that the matrix is square
  static_assert(N == M, "Matrix must be square");
  static_assert(N == K && M == L, "B matrix dimensions must match");

  // Get the differentiation order from the output
  static constexpr ADorder order = get_diff_order<Btype>::order;

  // Make sure that the order is correct
  static_assert(get_diff_order<Atype>::order == order,
                "ADorder does not match");

  static constexpr MatOp NORMAL = MatOp::NORMAL;
  static constexpr MatOp TRANSPOSE = MatOp::TRANSPOSE;

  A2D_FUNCTION MatInvExpr(Atype &A, Btype &Ainv) : A(A), Ainv(Ainv) {}

  A2D_FUNCTION void eval() { MatInvCore<T, N>(get_data(A), get_data(Ainv)); }

  A2D_FUNCTION void bzero() { Ainv.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    static_assert(
        !(order == ADorder::FIRST && forder == ADorder::SECOND),
        "Can't perform second order forward with first order objects");
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;

    T temp[N * N];
    MatMatMultCore<T, N, N, N, N, N, N, NORMAL, NORMAL>(
        get_data(Ainv), GetSeed<seed>::get_data(A), temp);
    MatMatMultScaleCore<T, N, N, N, N, N, N, NORMAL, NORMAL>(
        T(-1.0), temp, get_data(Ainv), GetSeed<seed>::get_data(Ainv));
  }

  A2D_FUNCTION void reverse() {
    T temp[N * N];
    const bool additive = true;
    MatMatMultCore<T, N, N, N, N, N, N, TRANSPOSE, NORMAL>(
        get_data(Ainv), GetSeed<ADseed::b>::get_data(Ainv), temp);
    MatMatMultScaleCore<T, N, N, N, N, N, N, NORMAL, TRANSPOSE, additive>(
        T(-1.0), temp, get_data(Ainv), GetSeed<ADseed::b>::get_data(A));
  }

  A2D_FUNCTION void hzero() { Ainv.hzero(); }

  A2D_FUNCTION void hreverse() {
    static_assert(order == ADorder::SECOND,
                  "hreverse() can be called for only second order objects.");

    T Ab[N * N], temp[N * N];
    const bool additive = true;

    // Compute the derivative contribution
    MatMatMultCore<T, N, N, N, N, N, N, TRANSPOSE, NORMAL>(
        get_data(Ainv), GetSeed<ADseed::b>::get_data(Ainv), temp);
    MatMatMultScaleCore<T, N, N, N, N, N, N, NORMAL, TRANSPOSE, false>(
        T(-1.0), temp, get_data(Ainv), Ab);

    // - A^{-T} * Ap^{T} * Ab
    MatMatMultCore<T, N, N, N, N, N, N, TRANSPOSE, TRANSPOSE>(
        get_data(Ainv), GetSeed<ADseed::p>::get_data(A), temp);
    MatMatMultScaleCore<T, N, N, N, N, N, N, NORMAL, NORMAL, additive>(
        T(-1.0), temp, Ab, GetSeed<ADseed::h>::get_data(A));

    // - Ab * Ap^{T} * A^{-T}
    MatMatMultCore<T, N, N, N, N, N, N, NORMAL, TRANSPOSE>(
        Ab, GetSeed<ADseed::p>::get_data(A), temp);
    MatMatMultScaleCore<T, N, N, N, N, N, N, NORMAL, TRANSPOSE, additive>(
        T(-1.0), temp, get_data(Ainv), GetSeed<ADseed::h>::get_data(A));

    // - A^{-T} * Ainvh * A^{-T}
    MatMatMultCore<T, N, N, N, N, N, N, TRANSPOSE, NORMAL>(
        get_data(Ainv), GetSeed<ADseed::h>::get_data(Ainv), temp);
    MatMatMultScaleCore<T, N, N, N, N, N, N, NORMAL, TRANSPOSE, additive>(
        T(-1.0), temp, get_data(Ainv), GetSeed<ADseed::h>::get_data(A));
  }

  Atype &A;
  Btype &Ainv;
};

template <class Atype, class Btype>
A2D_FUNCTION auto MatInv(ADObj<Atype> &A, ADObj<Btype> &Ainv) {
  return MatInvExpr<ADObj<Atype>, ADObj<Btype>>(A, Ainv);
}

template <class Atype, class Btype>
A2D_FUNCTION auto MatInv(A2DObj<Atype> &A, A2DObj<Btype> &Ainv) {
  return MatInvExpr<A2DObj<Atype>, A2DObj<Btype>>(A, Ainv);
}

namespace Test {

template <typename T, int N>
class MatInvTest : public A2DTest<T, Mat<T, N, N>, Mat<T, N, N>> {
 public:
  using Input = VarTuple<T, Mat<T, N, N>>;
  using Output = VarTuple<T, Mat<T, N, N>>;

  // Assemble a string to describe the test
  std::string name() {
    std::stringstream s;
    s << "MatInv<" << N << "," << N << ">";
    return s.str();
  }

  // Evaluate the matrix-matrix product
  Output eval(const Input &x) {
    Mat<T, N, N> A;
    Mat<T, N, N> B;
    x.get_values(A);
    MatInv(A, B);
    return MakeVarTuple<T>(B);
  }

  // Compute the derivative
  void deriv(const Output &seed, const Input &x, Input &g) {
    ADObj<Mat<T, N, N>> A;
    ADObj<Mat<T, N, N>> B;

    x.get_values(A.value());
    auto stack = MakeStack(MatInv(A, B));
    seed.get_values(B.bvalue());
    stack.reverse();
    g.set_values(A.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output &seed, const Output &hval, const Input &x,
             const Input &p, Input &h) {
    A2DObj<Mat<T, N, N>> A;
    A2DObj<Mat<T, N, N>> B;

    x.get_values(A.value());
    p.get_values(A.pvalue());
    auto stack = MakeStack(MatInv(A, B));
    seed.get_values(B.bvalue());
    hval.get_values(B.hvalue());
    stack.hproduct();
    h.set_values(A.hvalue());
  }
};

inline bool MatInvTestAll(bool component = false, bool write_output = true) {
  using Tc = A2D_complex_t<double>;

  bool passed = true;
  MatInvTest<Tc, 2> test1;
  passed = passed && Run(test1, component, write_output);
  MatInvTest<Tc, 3> test2;
  passed = passed && Run(test2, component, write_output);

  return passed;
}

}  // namespace Test

}  // namespace A2D

#endif  // A2D_MAT_INV_H
