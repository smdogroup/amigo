#ifndef A2D_SYMMAT_RK_H
#define A2D_SYMMAT_RK_H

#include <type_traits>

#include "../a2ddefs.h"
#include "a2dmat.h"
#include "a2dtest.h"
#include "core/a2dgemmcore.h"
#include "core/a2dsymrkcore.h"

namespace A2D {

template <typename T, int N, int K, int P>
A2D_FUNCTION void SymMatRK(const Mat<T, N, K>& A, SymMat<T, P>& S) {
  static_assert(P == N, "SymMatRK matrix dimensions must agree");
  SymMatRKCore<T, N, K, MatOp::NORMAL>(get_data(A), get_data(S));
}

template <typename T, int N, int K, int P>
A2D_FUNCTION void SymMatRK(const T alpha, const Mat<T, N, K>& A,
                           SymMat<T, P>& S) {
  static_assert(P == N, "SymMatRK matrix dimensions must agree");
  SymMatRKCoreScale<T, N, K, MatOp::NORMAL>(get_data(alpha), get_data(A),
                                            get_data(S));
}

template <MatOp op, typename T, int N, int K, int P>
A2D_FUNCTION void SymMatRK(const Mat<T, N, K>& A, SymMat<T, P>& S) {
  static_assert(
      (op == MatOp::NORMAL && P == N) || (op == MatOp::TRANSPOSE && K == P),
      "SymMatRK matrix dimensions must agree");
  SymMatRKCore<T, N, K, op>(get_data(A), get_data(S));
}

template <MatOp op, typename T, int N, int K, int P>
A2D_FUNCTION void SymMatRK(const T alpha, const Mat<T, N, K>& A,
                           SymMat<T, P>& S) {
  static_assert(
      (op == MatOp::NORMAL && P == N) || (op == MatOp::TRANSPOSE && K == P),
      "SymMatRK matrix dimensions must agree");
  SymMatRKCoreScale<T, N, K, op>(get_data(alpha), get_data(A), get_data(S));
}

template <class Atype, class Stype, MatOp op = MatOp::NORMAL>
class SymMatRKExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<Stype>::type T;

  // Extract the dimensions of the underlying matrix
  static constexpr int N = get_matrix_rows<Atype>::size;
  static constexpr int K = get_matrix_columns<Atype>::size;
  static constexpr int P = get_symmatrix_size<Stype>::size;

  A2D_FUNCTION SymMatRKExpr(Atype& A, Stype& S) : A(A), S(S) {
    static_assert(
        (op == MatOp::NORMAL && P == N) || (op == MatOp::TRANSPOSE && K == P),
        "SymMatRK matrix dimensions must agree");
  }

  A2D_FUNCTION void eval() {
    SymMatRKCore<T, N, K, op>(get_data(A), get_data(S));
  }

  A2D_FUNCTION void bzero() { S.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;
    SymMatR2KCore<T, N, K, op>(get_data(A), GetSeed<seed>::get_data(A),
                               GetSeed<seed>::get_data(S));
  }

  A2D_FUNCTION void reverse() {
    constexpr ADseed seed = ADseed::b;
    SymMatRKCoreReverse<T, N, K, op>(get_data(A), GetSeed<seed>::get_data(S),
                                     GetSeed<seed>::get_data(A));
  }

  A2D_FUNCTION void hzero() { S.hzero(); }

  A2D_FUNCTION void hreverse() {
    SymMatRKCoreReverse<T, N, K, op>(get_data(A),
                                     GetSeed<ADseed::h>::get_data(S),
                                     GetSeed<ADseed::h>::get_data(A));

    SymMatRKCoreReverse<T, N, K, op>(GetSeed<ADseed::p>::get_data(A),
                                     GetSeed<ADseed::b>::get_data(S),
                                     GetSeed<ADseed::h>::get_data(A));
  }

  Atype& A;
  Stype& S;
};

template <class Atype, class Stype>
A2D_FUNCTION auto SymMatRK(ADObj<Atype>& A, ADObj<Stype>& S) {
  return SymMatRKExpr<ADObj<Atype>, ADObj<Stype>>(A, S);
}

template <MatOp op, class Atype, class Stype>
A2D_FUNCTION auto SymMatRK(ADObj<Atype>& A, ADObj<Stype>& S) {
  return SymMatRKExpr<ADObj<Atype>, ADObj<Stype>, op>(A, S);
}

template <class Atype, class Stype>
A2D_FUNCTION auto SymMatRK(A2DObj<Atype>& A, A2DObj<Stype>& S) {
  return SymMatRKExpr<A2DObj<Atype>, A2DObj<Stype>>(A, S);
}

template <MatOp op, class Atype, class Stype>
A2D_FUNCTION auto SymMatRK(A2DObj<Atype>& A, A2DObj<Stype>& S) {
  return SymMatRKExpr<A2DObj<Atype>, A2DObj<Stype>, op>(A, S);
}

template <class Atype, class Stype, MatOp op = MatOp::NORMAL>
class SymMatRKScaleExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<Stype>::type T;

  // Extract the dimensions of the underlying matrix
  static constexpr int N = get_matrix_rows<Atype>::size;
  static constexpr int K = get_matrix_columns<Atype>::size;
  static constexpr int P = get_symmatrix_size<Stype>::size;

  A2D_FUNCTION SymMatRKScaleExpr(const T alpha, Atype& A, Stype& S)
      : alpha(alpha), A(A), S(S) {
    static_assert(
        (op == MatOp::NORMAL && P == N) || (op == MatOp::TRANSPOSE && K == P),
        "SymMatRK matrix dimensions must agree");
  }

  A2D_FUNCTION void eval() {
    SymMatRKCoreScale<T, N, K, op>(alpha, get_data(A), get_data(S));
  }

  A2D_FUNCTION void bzero() { S.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;
    SymMatR2KCoreScale<T, N, K, op>(alpha, get_data(A),
                                    GetSeed<seed>::get_data(A),
                                    GetSeed<seed>::get_data(S));
  }

  A2D_FUNCTION void reverse() {
    constexpr ADseed seed = ADseed::b;

    SymMatRKCoreReverseScale<T, N, K, op>(alpha, get_data(A),
                                          GetSeed<seed>::get_data(S),
                                          GetSeed<seed>::get_data(A));
  }

  A2D_FUNCTION void hzero() { S.hzero(); }

  A2D_FUNCTION void hreverse() {
    SymMatRKCoreReverseScale<T, N, K, op>(alpha, get_data(A),
                                          GetSeed<ADseed::h>::get_data(S),
                                          GetSeed<ADseed::h>::get_data(A));
    SymMatRKCoreReverseScale<T, N, K, op>(
        alpha, GetSeed<ADseed::p>::get_data(A), GetSeed<ADseed::b>::get_data(S),
        GetSeed<ADseed::h>::get_data(A));
  }

  const T alpha;
  Atype& A;
  Stype& S;
};

template <typename T, class Atype, class Stype>
A2D_FUNCTION auto SymMatRK(const T alpha, ADObj<Atype>& A, ADObj<Stype>& S) {
  return SymMatRKScaleExpr<ADObj<Atype>, ADObj<Stype>>(alpha, A, S);
}

template <MatOp op, typename T, class Atype, class Stype>
A2D_FUNCTION auto SymMatRK(const T alpha, ADObj<Atype>& A, ADObj<Stype>& S) {
  return SymMatRKScaleExpr<ADObj<Atype>, ADObj<Stype>, op>(alpha, A, S);
}

template <typename T, class Atype, class Stype>
A2D_FUNCTION auto SymMatRK(const T alpha, A2DObj<Atype>& A, A2DObj<Stype>& S) {
  return SymMatRKScaleExpr<A2DObj<Atype>, A2DObj<Stype>>(alpha, A, S);
}

template <MatOp op, typename T, class Atype, class Stype>
A2D_FUNCTION auto SymMatRK(const T alpha, A2DObj<Atype>& A, A2DObj<Stype>& S) {
  return SymMatRKScaleExpr<A2DObj<Atype>, A2DObj<Stype>, op>(alpha, A, S);
}

namespace Test {

template <MatOp op, typename T, int N, int M, int P>
class SymMatRKTest : public A2DTest<T, SymMat<T, P>, Mat<T, N, M>> {
 public:
  using Input = VarTuple<T, Mat<T, N, M>>;
  using Output = VarTuple<T, SymMat<T, P>>;

  // Assemble a string to describe the test
  std::string name() {
    std::stringstream s;
    s << "SymMatRKTest<";
    if (op == MatOp::NORMAL) {
      s << "N,";
    } else {
      s << "T,";
    }
    s << N << "," << M << "," << P << ">";
    return s.str();
  }

  // Evaluate the matrix-matrix product
  Output eval(const Input& x) {
    Mat<T, N, M> A;
    SymMat<T, P> S;
    x.get_values(A);
    SymMatRK<op>(A, S);
    return MakeVarTuple<T>(S);
  }

  // Compute the derivative
  void deriv(const Output& seed, const Input& x, Input& g) {
    ADObj<Mat<T, N, M>> A;
    ADObj<SymMat<T, P>> S;

    x.get_values(A.value());
    auto stack = MakeStack(SymMatRK<op>(A, S));
    seed.get_values(S.bvalue());
    stack.reverse();
    g.set_values(A.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output& seed, const Output& hval, const Input& x,
             const Input& p, Input& h) {
    A2DObj<Mat<T, N, M>> A;
    A2DObj<SymMat<T, P>> S;

    x.get_values(A.value());
    p.get_values(A.pvalue());
    auto stack = MakeStack(SymMatRK<op>(A, S));
    seed.get_values(S.bvalue());
    hval.get_values(S.hvalue());
    stack.hproduct();
    h.set_values(A.hvalue());
  }
};

template <MatOp op, typename T, int N, int M, int P>
class SymMatRKScaleTest : public A2DTest<T, SymMat<T, P>, Mat<T, N, M>> {
 public:
  using Input = VarTuple<T, Mat<T, N, M>>;
  using Output = VarTuple<T, SymMat<T, P>>;

  // Assemble a string to describe the test
  std::string name() {
    std::stringstream s;
    s << "SymMatRKScaleTest<";
    if (op == MatOp::NORMAL) {
      s << "N,";
    } else {
      s << "T,";
    }
    s << N << "," << M << "," << P << ">";
    return s.str();
  }

  // Evaluate the matrix-matrix product
  Output eval(const Input& x) {
    const T alpha = 0.25;
    Mat<T, N, M> A;
    SymMat<T, P> S;
    x.get_values(A);
    SymMatRK<op>(alpha, A, S);
    return MakeVarTuple<T>(S);
  }

  // Compute the derivative
  void deriv(const Output& seed, const Input& x, Input& g) {
    const T alpha = 0.25;
    ADObj<Mat<T, N, M>> A;
    ADObj<SymMat<T, P>> S;

    x.get_values(A.value());
    auto stack = MakeStack(SymMatRK<op>(alpha, A, S));
    seed.get_values(S.bvalue());
    stack.reverse();
    g.set_values(A.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output& seed, const Output& hval, const Input& x,
             const Input& p, Input& h) {
    const T alpha = 0.25;
    A2DObj<Mat<T, N, M>> A;
    A2DObj<SymMat<T, P>> S;

    x.get_values(A.value());
    p.get_values(A.pvalue());
    auto stack = MakeStack(SymMatRK<op>(alpha, A, S));
    seed.get_values(S.bvalue());
    hval.get_values(S.hvalue());
    stack.hproduct();
    h.set_values(A.hvalue());
  }
};

template <typename T, int N, int K>
bool SymMatRKTestHelper(bool component = false, bool write_output = true) {
  const MatOp NORMAL = MatOp::NORMAL;
  const MatOp TRANSPOSE = MatOp::TRANSPOSE;
  using Tc = A2D_complex_t<T>;

  bool passed = true;
  SymMatRKTest<NORMAL, Tc, N, K, N> test1;
  passed = passed && Run(test1, component, write_output);
  SymMatRKTest<TRANSPOSE, Tc, N, K, K> test2;
  passed = passed && Run(test2, component, write_output);

  SymMatRKScaleTest<NORMAL, Tc, N, K, N> test3;
  passed = passed && Run(test3, component, write_output);
  SymMatRKScaleTest<TRANSPOSE, Tc, N, K, K> test4;
  passed = passed && Run(test4, component, write_output);

  return passed;
}

inline bool SymMatRKTestAll(bool component = false, bool write_output = true) {
  bool passed = true;
  passed = passed && SymMatRKTestHelper<double, 3, 3>(component, write_output);
  passed = passed && SymMatRKTestHelper<double, 2, 4>(component, write_output);
  passed = passed && SymMatRKTestHelper<double, 5, 2>(component, write_output);
  passed = passed && SymMatRKTestHelper<double, 7, 5>(component, write_output);

  return passed;
}

}  // namespace Test

}  // namespace A2D

#endif  // A2D_SYMMAT_RK_H
