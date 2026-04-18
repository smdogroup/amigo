#ifndef A2D_GEMM_H
#define A2D_GEMM_H

#include <type_traits>

#include "../a2ddefs.h"
#include "a2dmat.h"
#include "a2dstack.h"
#include "a2dtest.h"
#include "core/a2dgemmcore.h"

namespace A2D {

/*
  Compute C = opA(A) * opB(B)

  Forward mode-derivative:

  dot{C} =  opA(dot{A}) * opB(B) + opA(A) * opB(dot{B})

  Derivation of the reverse-mode derivative:

  tr(bar{C}^{T} * dot{C})
  = tr(bar{C}^{T} * opA(dot{A}) * opB(B)) +
    tr(bar{C}^{T} * opA(A) * opB(dot{B}))
  = tr(opB(B) * bar{C}^{T} * opA(dot{A})) +
    tr(bar{C}^{T} * opA(A) * opB(dot{B}))

  In general:

  if opA == NORMAL then
    tr(bar{B}^{T} * dot{A})
    ==> bar{A} = bar{B}
  if opA == TRANSPOSE
    tr(bar{B}^{T} * dot{A^{T}}) = tr(dot{A} * bar{B}) = tr(bar{B} * dot{A})
    ==> bar{A} = bar{B}^{T}

  Therefore:

  if opA == NORMAL then:
    bar{A} = (opB(B) * bar{C}^{T})^{T} = bar{C} * not_opB(B)
  if opA == TRANSPOSE then:
    bar{A} = opB(B) * bar{C}^{T}

  if opB == NORMAL
    bar{B} = (bar{C}^{T} * opA(A))^{T} = not_opA(A) * bar{C}
  if opB == TRANSPOSE then:
    bar{B} = bar{C}^{T} * opA(A)
*/

// compute C = op(A) * op(B) and returns nothing, where A and B are all
// passive variables
template <typename T, int N, int M, int K, int L, int P, int Q>
A2D_FUNCTION void MatMatMult(const Mat<T, N, M>& A, const Mat<T, K, L>& B,
                             Mat<T, P, Q>& C) {
  MatMatMultCore<T, N, M, K, L, P, Q, MatOp::NORMAL, MatOp::NORMAL>(
      get_data(A), get_data(B), get_data(C));
}
template <MatOp opA, MatOp opB, typename T, int N, int M, int K, int L, int P,
          int Q>
A2D_FUNCTION void MatMatMult(const Mat<T, N, M>& A, const Mat<T, K, L>& B,
                             Mat<T, P, Q>& C) {
  MatMatMultCore<T, N, M, K, L, P, Q, opA, opB>(get_data(A), get_data(B),
                                                get_data(C));
}

// Note: we probably don't want these unless they're really inevitable, because
// This opens up a whole bunch of combinations of AD/A2D functions to
// implement...
//
// template <typename T, int N>
// A2D_FUNCTION void MatMatMult(const SymMat<T, N>& SA, const SymMat<T, N>& SB,
//                              Mat<T, N, N>& C) {
//   SMatSMatMultCore<T, N>(get_data(SA), get_data(SB), get_data(C));
// }

template <MatOp opA, MatOp opB, class Atype, class Btype, class Ctype>
class MatMatMultExpr {
 public:
  static constexpr MatOp not_opA =
      conditional_value<MatOp, opA == MatOp::NORMAL, MatOp::TRANSPOSE,
                        MatOp::NORMAL>::value;
  static constexpr MatOp not_opB =
      conditional_value<MatOp, opB == MatOp::NORMAL, MatOp::TRANSPOSE,
                        MatOp::NORMAL>::value;

  // Extract the numeric type to use
  typedef typename get_object_numeric_type<Ctype>::type T;

  // Extract the dimensions of the matrices
  static constexpr int N = get_matrix_rows<Atype>::size;
  static constexpr int M = get_matrix_columns<Atype>::size;
  static constexpr int K = get_matrix_rows<Btype>::size;
  static constexpr int L = get_matrix_columns<Btype>::size;
  static constexpr int P = get_matrix_rows<Ctype>::size;
  static constexpr int Q = get_matrix_columns<Ctype>::size;

  // Get the types of the matrices
  static constexpr ADiffType adA = get_diff_type<Atype>::diff_type;
  static constexpr ADiffType adB = get_diff_type<Btype>::diff_type;

  // Get the differentiation order from the output
  static constexpr ADorder order = get_diff_order<Ctype>::order;

  A2D_FUNCTION MatMatMultExpr(Atype& A, Btype& B, Ctype& C)
      : A(A), B(B), C(C) {}

  A2D_FUNCTION void eval() {
    MatMatMultCore<T, N, M, K, L, P, Q, opA, opB>(get_data(A), get_data(B),
                                                  get_data(C));
  }

  A2D_FUNCTION void bzero() { C.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    static_assert(
        !(order == ADorder::FIRST and forder == ADorder::SECOND),
        "Can't perform second order forward with first order objects");
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;
    if constexpr (adA == ADiffType::ACTIVE && adB == ADiffType::ACTIVE) {
      constexpr bool additive = true;
      MatMatMultCore<T, N, M, K, L, P, Q, opA, opB>(
          GetSeed<seed>::get_data(A), get_data(B), GetSeed<seed>::get_data(C));
      MatMatMultCore<T, N, M, K, L, P, Q, opA, opB, additive>(
          get_data(A), GetSeed<seed>::get_data(B), GetSeed<seed>::get_data(C));
    } else if constexpr (adA == ADiffType::ACTIVE) {
      MatMatMultCore<T, N, M, K, L, P, Q, opA, opB>(
          GetSeed<seed>::get_data(A), get_data(B), GetSeed<seed>::get_data(C));
    } else if constexpr (adB == ADiffType::ACTIVE) {
      MatMatMultCore<T, N, M, K, L, P, Q, opA, opB>(
          get_data(A), GetSeed<seed>::get_data(B), GetSeed<seed>::get_data(C));
    }
  }

  A2D_FUNCTION void reverse() {
    if constexpr (adA == ADiffType::ACTIVE) {
      if constexpr (opA == MatOp::NORMAL) {
        // bar{A} += bar{C} * not_opB(B)
        MatMatMultCore<T, P, Q, K, L, N, M, MatOp::NORMAL, not_opB, true>(
            GetSeed<ADseed::b>::get_data(C), get_data(B),
            GetSeed<ADseed::b>::get_data(A));
      } else {
        // bar{A} += opB(B) * bar{C}^{T}
        MatMatMultCore<T, K, L, P, Q, N, M, opB, MatOp::TRANSPOSE, true>(
            get_data(B), GetSeed<ADseed::b>::get_data(C),
            GetSeed<ADseed::b>::get_data(A));
      }
    }
    if constexpr (adB == ADiffType::ACTIVE) {
      if constexpr (opB == MatOp::NORMAL) {
        // bar{B} += not_opA(A) * bar{C}
        MatMatMultCore<T, N, M, P, Q, K, L, not_opA, MatOp::NORMAL, true>(
            get_data(A), GetSeed<ADseed::b>::get_data(C),
            GetSeed<ADseed::b>::get_data(B));
      } else {
        // bar{B} += bar{C}^{T} * opA(A)
        MatMatMultCore<T, P, Q, N, M, K, L, MatOp::TRANSPOSE, opA, true>(
            GetSeed<ADseed::b>::get_data(C), get_data(A),
            GetSeed<ADseed::b>::get_data(B));
      }
    }
  }

  A2D_FUNCTION void hzero() { C.hzero(); }

  A2D_FUNCTION void hreverse() {
    static_assert(order == ADorder::SECOND,
                  "hreverse() can be called for only second order objects.");

    if constexpr (adA == ADiffType::ACTIVE) {
      if constexpr (opA == MatOp::NORMAL) {
        MatMatMultCore<T, P, Q, K, L, N, M, MatOp::NORMAL, not_opB, true>(
            GetSeed<ADseed::h>::get_data(C), get_data(B),
            GetSeed<ADseed::h>::get_data(A));
      } else {
        MatMatMultCore<T, K, L, P, Q, N, M, opB, MatOp::TRANSPOSE, true>(
            get_data(B), GetSeed<ADseed::h>::get_data(C),
            GetSeed<ADseed::h>::get_data(A));
      }
    }
    if constexpr (adB == ADiffType::ACTIVE) {
      if constexpr (opB == MatOp::NORMAL) {
        MatMatMultCore<T, N, M, P, Q, K, L, not_opA, MatOp::NORMAL, true>(
            get_data(A), GetSeed<ADseed::h>::get_data(C),
            GetSeed<ADseed::h>::get_data(B));
      } else {
        MatMatMultCore<T, P, Q, N, M, K, L, MatOp::TRANSPOSE, opA, true>(
            GetSeed<ADseed::h>::get_data(C), get_data(A),
            GetSeed<ADseed::h>::get_data(B));
      }
    }
    if constexpr (adA == ADiffType::ACTIVE and adB == ADiffType::ACTIVE) {
      if constexpr (opA == MatOp::NORMAL) {
        MatMatMultCore<T, P, Q, K, L, N, M, MatOp::NORMAL, not_opB, true>(
            GetSeed<ADseed::b>::get_data(C), GetSeed<ADseed::p>::get_data(B),
            GetSeed<ADseed::h>::get_data(A));

      } else {
        MatMatMultCore<T, K, L, P, Q, N, M, opB, MatOp::TRANSPOSE, true>(
            GetSeed<ADseed::p>::get_data(B), GetSeed<ADseed::b>::get_data(C),
            GetSeed<ADseed::h>::get_data(A));
      }

      if constexpr (opB == MatOp::NORMAL) {
        MatMatMultCore<T, N, M, P, Q, K, L, not_opA, MatOp::NORMAL, true>(
            GetSeed<ADseed::p>::get_data(A), GetSeed<ADseed::b>::get_data(C),
            GetSeed<ADseed::h>::get_data(B));

      } else {
        MatMatMultCore<T, P, Q, N, M, K, L, MatOp::TRANSPOSE, opA, true>(
            GetSeed<ADseed::b>::get_data(C), GetSeed<ADseed::p>::get_data(A),
            GetSeed<ADseed::h>::get_data(B));
      }
    }
  }

 private:
  Atype& A;
  Btype& B;
  Ctype& C;
};

// compute C = op(A) * op(B) and return an expression, where A and B are all
// active variables
template <class Atype, class Btype, class Ctype>
A2D_FUNCTION auto MatMatMult(ADObj<Atype>& A, ADObj<Btype>& B,
                             ADObj<Ctype>& C) {
  return MatMatMultExpr<MatOp::NORMAL, MatOp::NORMAL, ADObj<Atype>,
                        ADObj<Btype>, ADObj<Ctype>>(A, B, C);
}
template <class Atype, class Btype, class Ctype>
A2D_FUNCTION auto MatMatMult(A2DObj<Atype>& A, A2DObj<Btype>& B,
                             A2DObj<Ctype>& C) {
  return MatMatMultExpr<MatOp::NORMAL, MatOp::NORMAL, A2DObj<Atype>,
                        A2DObj<Btype>, A2DObj<Ctype>>(A, B, C);
}
template <MatOp opA, MatOp opB, class Atype, class Btype, class Ctype>
A2D_FUNCTION auto MatMatMult(ADObj<Atype>& A, ADObj<Btype>& B,
                             ADObj<Ctype>& C) {
  return MatMatMultExpr<opA, opB, ADObj<Atype>, ADObj<Btype>, ADObj<Ctype>>(
      A, B, C);
}
template <MatOp opA, MatOp opB, class Atype, class Btype, class Ctype>
A2D_FUNCTION auto MatMatMult(A2DObj<Atype>& A, A2DObj<Btype>& B,
                             A2DObj<Ctype>& C) {
  return MatMatMultExpr<opA, opB, A2DObj<Atype>, A2DObj<Btype>, A2DObj<Ctype>>(
      A, B, C);
}

// compute C = op(A) * op(B) and return an expression, where A is passive, B is
// active variables
template <class Atype, class Btype, class Ctype>
A2D_FUNCTION auto MatMatMult(const Atype& A, ADObj<Btype>& B, ADObj<Ctype>& C) {
  return MatMatMultExpr<MatOp::NORMAL, MatOp::NORMAL, const Atype, ADObj<Btype>,
                        ADObj<Ctype>>(A, B, C);
}
template <class Atype, class Btype, class Ctype>
A2D_FUNCTION auto MatMatMult(const Atype& A, A2DObj<Btype>& B,
                             A2DObj<Ctype>& C) {
  return MatMatMultExpr<MatOp::NORMAL, MatOp::NORMAL, const Atype,
                        A2DObj<Btype>, A2DObj<Ctype>>(A, B, C);
}
template <MatOp opA, MatOp opB, class Atype, class Btype, class Ctype>
A2D_FUNCTION auto MatMatMult(const Atype& A, ADObj<Btype>& B, ADObj<Ctype>& C) {
  return MatMatMultExpr<opA, opB, const Atype, ADObj<Btype>, ADObj<Ctype>>(A, B,
                                                                           C);
}
template <MatOp opA, MatOp opB, class Atype, class Btype, class Ctype>
A2D_FUNCTION auto MatMatMult(const Atype& A, A2DObj<Btype>& B,
                             A2DObj<Ctype>& C) {
  return MatMatMultExpr<opA, opB, const Atype, A2DObj<Btype>, A2DObj<Ctype>>(
      A, B, C);
}

// compute C = op(A) * op(B) and return an expression, where A is active, B
// is passive variables

template <class Atype, class Btype, class Ctype>
A2D_FUNCTION auto MatMatMult(ADObj<Atype>& A, const Btype& B, ADObj<Ctype>& C) {
  return MatMatMultExpr<MatOp::NORMAL, MatOp::NORMAL, ADObj<Atype>, const Btype,
                        ADObj<Ctype>>(A, B, C);
}
template <class Atype, class Btype, class Ctype>
A2D_FUNCTION auto MatMatMult(A2DObj<Atype>& A, const Btype& B,
                             A2DObj<Ctype>& C) {
  return MatMatMultExpr<MatOp::NORMAL, MatOp::NORMAL, A2DObj<Atype>,
                        const Btype, A2DObj<Ctype>>(A, B, C);
}
template <MatOp opA, MatOp opB, class Atype, class Btype, class Ctype>
A2D_FUNCTION auto MatMatMult(ADObj<Atype>& A, const Btype& B, ADObj<Ctype>& C) {
  return MatMatMultExpr<opA, opB, ADObj<Atype>, const Btype, ADObj<Ctype>>(A, B,
                                                                           C);
}
template <MatOp opA, MatOp opB, class Atype, class Btype, class Ctype>
A2D_FUNCTION auto MatMatMult(A2DObj<Atype>& A, const Btype& B,
                             A2DObj<Ctype>& C) {
  return MatMatMultExpr<opA, opB, A2DObj<Atype>, const Btype, A2DObj<Ctype>>(
      A, B, C);
}

namespace Test {

template <MatOp opA, MatOp opB, typename T, int N, int M, int K, int L, int P,
          int Q>
class MatMatMultTest
    : public A2DTest<T, Mat<T, P, Q>, Mat<T, N, M>, Mat<T, K, L>> {
 public:
  using Input = VarTuple<T, Mat<T, N, M>, Mat<T, K, L>>;
  using Output = VarTuple<T, Mat<T, P, Q>>;

  // Assemble a string to describe the test
  std::string name() {
    std::stringstream s;
    s << "MatMatMult<";
    if (opA == MatOp::NORMAL) {
      s << "N,";
    } else {
      s << "T,";
    }
    if (opB == MatOp::NORMAL) {
      s << "N,";
    } else {
      s << "T,";
    }
    s << N << "," << M << "," << K << "," << L << "," << P << "," << Q << ">";

    return s.str();
  }

  // Evaluate the matrix-matrix product
  Output eval(const Input& x) {
    Mat<T, N, M> A;
    Mat<T, K, L> B;
    Mat<T, P, Q> C;

    x.get_values(A, B);
    MatMatMult<opA, opB>(A, B, C);
    return MakeVarTuple<T>(C);
  }

  // Compute the derivative
  void deriv(const Output& seed, const Input& x, Input& g) {
    ADObj<Mat<T, N, M>> A;
    ADObj<Mat<T, K, L>> B;
    ADObj<Mat<T, P, Q>> C;

    x.get_values(A.value(), B.value());
    auto stack = MakeStack(MatMatMult<opA, opB>(A, B, C));
    seed.get_values(C.bvalue());
    stack.reverse();
    g.set_values(A.bvalue(), B.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output& seed, const Output& hval, const Input& x,
             const Input& p, Input& h) {
    A2DObj<Mat<T, N, M>> A;
    A2DObj<Mat<T, K, L>> B;
    A2DObj<Mat<T, P, Q>> C;

    x.get_values(A.value(), B.value());
    p.get_values(A.pvalue(), B.pvalue());
    auto stack = MakeStack(MatMatMult<opA, opB>(A, B, C));
    seed.get_values(C.bvalue());
    hval.get_values(C.hvalue());
    stack.hproduct();
    h.set_values(A.hvalue(), B.hvalue());
  }
};

template <typename T, int N, int M, int K>
bool MatMatMultTestHelper(bool component = false, bool write_output = true) {
  const MatOp NORMAL = MatOp::NORMAL;
  const MatOp TRANSPOSE = MatOp::TRANSPOSE;
  using Tc = A2D_complex_t<T>;

  bool passed = true;
  MatMatMultTest<NORMAL, NORMAL, Tc, N, M, M, K, N, K> test1;
  passed = passed && Run(test1, component, write_output);
  MatMatMultTest<NORMAL, TRANSPOSE, Tc, N, M, K, M, N, K> test2;
  passed = passed && Run(test2, component, write_output);
  MatMatMultTest<TRANSPOSE, NORMAL, Tc, N, M, N, K, M, K> test3;
  passed = passed && Run(test3, component, write_output);
  MatMatMultTest<TRANSPOSE, TRANSPOSE, Tc, N, M, K, N, M, K> test4;
  passed = passed && Run(test4, component, write_output);

  return passed;
}

inline bool MatMatMultTestAll(bool component = false,
                              bool write_output = true) {
  bool passed = true;
  passed =
      passed && MatMatMultTestHelper<double, 3, 3, 3>(component, write_output);
  passed =
      passed && MatMatMultTestHelper<double, 2, 3, 4>(component, write_output);
  passed =
      passed && MatMatMultTestHelper<double, 5, 4, 2>(component, write_output);

  return passed;
}

}  // namespace Test

}  // namespace A2D

#endif  // A2D_GEMM_H
