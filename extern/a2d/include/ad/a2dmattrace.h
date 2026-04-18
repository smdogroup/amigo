#ifndef A2D_MAT_TRACE_H
#define A2D_MAT_TRACE_H

#include <type_traits>

#include "../a2ddefs.h"
#include "a2dmat.h"
#include "a2dobj.h"
#include "a2dstack.h"
#include "a2dtest.h"

namespace A2D {

template <typename T, int M>
A2D_FUNCTION T MatTraceCore(const T* A) {
  T trace = T(0.0);
  for (int i = 0; i < M; i++) {
    trace += A[0];
    A += M + 1;
  }
  return trace;
}

template <typename T, int M>
A2D_FUNCTION T SymMatTraceCore(const T* S) {
  T trace = T(0.0);
  for (int i = 0; i < M; i++, S++) {
    S += i;
    trace += S[0];
  }
  return trace;
}

template <typename T, int M>
A2D_FUNCTION void MatAddDiagCore(const T diag, T* A) {
  for (int i = 0; i < M; i++) {
    A[0] += diag;
    A += M + 1;
  }
}

template <typename T, int M>
A2D_FUNCTION void SymMatAddDiagCore(const T diag, T* S) {
  for (int i = 0; i < M; i++, S++) {
    S += i;
    S[0] += diag;
  }
}

template <typename T, int M>
A2D_FUNCTION void MatTrace(Mat<T, M, M>& A, T& trace) {
  trace = MatTraceCore<T, M>(get_data(A));
}

template <typename T, int M>
A2D_FUNCTION void MatTrace(SymMat<T, M>& S, T& trace) {
  trace = SymMatTraceCore<T, M>(get_data(S));
}

template <class Atype, class dtype>
class MatTraceExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<dtype>::type T;

  // Extract the dimensions of the underlying matrix
  static constexpr int N = get_matrix_rows<Atype>::size;
  static constexpr int M = get_matrix_columns<Atype>::size;

  // Get the differentiation order from the output
  static constexpr ADorder order = get_diff_order<dtype>::order;

  // Make sure the matrix dimensions are consistent
  static_assert((N == M), "Matrix must be square");

  // Make sure that the order matches
  static_assert(get_diff_order<Atype>::order == order,
                "ADorder does not match");

  A2D_FUNCTION MatTraceExpr(Atype& A, dtype& tr) : A(A), tr(tr) {}

  A2D_FUNCTION void eval() { get_data(tr) = MatTraceCore<T, M>(get_data(A)); }

  A2D_FUNCTION void bzero() { tr.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    static_assert(
        !(order == ADorder::FIRST and forder == ADorder::SECOND),
        "Can't perform second order forward with first order objects");
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;
    GetSeed<seed>::get_data(tr) =
        MatTraceCore<T, M>(GetSeed<seed>::get_data(A));
  }

  A2D_FUNCTION void reverse() {
    MatAddDiagCore<T, M>(GetSeed<ADseed::b>::get_data(tr),
                         GetSeed<ADseed::b>::get_data(A));
  }

  A2D_FUNCTION void hzero() { tr.hzero(); }

  A2D_FUNCTION void hreverse() {
    MatAddDiagCore<T, M>(GetSeed<ADseed::h>::get_data(tr),
                         GetSeed<ADseed::h>::get_data(A));
  }

 private:
  Atype& A;
  dtype& tr;
};

template <
    class Atype, class dtype,
    std::enable_if_t<get_a2d_object_type<Atype>::value == ADObjType::MATRIX,
                     bool> = true>
A2D_FUNCTION auto MatTrace(ADObj<Atype>& A, ADObj<dtype>& tr) {
  return MatTraceExpr<ADObj<Atype>, ADObj<dtype>>(A, tr);
}

template <
    class Atype, class dtype,
    std::enable_if_t<get_a2d_object_type<Atype>::value == ADObjType::MATRIX,
                     bool> = true>
A2D_FUNCTION auto MatTrace(A2DObj<Atype>& A, A2DObj<dtype>& tr) {
  return MatTraceExpr<A2DObj<Atype>, A2DObj<dtype>>(A, tr);
}

template <class Stype, class dtype>
class SymTraceExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<dtype>::type T;

  // Extract the dimensions of the underlying matrix
  static constexpr int M = get_symmatrix_size<Stype>::size;

  // Get the differentiation order from the output
  static constexpr ADorder order = get_diff_order<dtype>::order;

  // Make sure that the order matches
  static_assert(get_diff_order<Stype>::order == order,
                "ADorder does not match");

  A2D_FUNCTION SymTraceExpr(Stype& S, dtype& tr) : S(S), tr(tr) {}

  A2D_FUNCTION void eval() {
    get_data(tr) = SymMatTraceCore<T, M>(get_data(S));
  }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    static_assert(
        !(order == ADorder::FIRST and forder == ADorder::SECOND),
        "Can't perform second order forward with first order objects");
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;
    GetSeed<seed>::get_data(tr) =
        SymMatTraceCore<T, M>(GetSeed<seed>::get_data(S));
  }

  A2D_FUNCTION void reverse() {
    SymMatAddDiagCore<T, M>(GetSeed<ADseed::b>::get_data(tr),
                            GetSeed<ADseed::b>::get_data(S));
  }

  A2D_FUNCTION void hreverse() {
    SymMatAddDiagCore<T, M>(GetSeed<ADseed::h>::get_data(tr),
                            GetSeed<ADseed::h>::get_data(S));
  }

 private:
  Stype& S;
  dtype& tr;
};

template <
    class Stype, class dtype,
    std::enable_if_t<get_a2d_object_type<Stype>::value == ADObjType::SYMMAT,
                     bool> = true>
A2D_FUNCTION auto MatTrace(ADObj<Stype>& S, ADObj<dtype>& tr) {
  return SymTraceExpr<ADObj<Stype>, ADObj<dtype>>(S, tr);
}

template <
    class Stype, class dtype,
    std::enable_if_t<get_a2d_object_type<Stype>::value == ADObjType::SYMMAT,
                     bool> = true>
A2D_FUNCTION auto MatTrace(A2DObj<Stype>& S, A2DObj<dtype>& tr) {
  return SymTraceExpr<A2DObj<Stype>, A2DObj<dtype>>(S, tr);
}

namespace Test {

template <typename T, int N>
class MatTraceTest : public A2DTest<T, T, Mat<T, N, N>> {
 public:
  using Input = VarTuple<T, Mat<T, N, N>>;
  using Output = VarTuple<T, T>;

  // Assemble a string to describe the test
  std::string name() {
    std::stringstream s;
    s << "MatTrace<" << N << "," << N << ">";
    return s.str();
  }

  // Evaluate the matrix-matrix product
  Output eval(const Input& x) {
    T trace;
    Mat<T, N, N> A;
    x.get_values(A);
    MatTrace(A, trace);
    return MakeVarTuple<T>(trace);
  }

  // Compute the derivative
  void deriv(const Output& seed, const Input& x, Input& g) {
    ADObj<T> trace;
    ADObj<Mat<T, N, N>> A;

    x.get_values(A.value());
    auto stack = MakeStack(MatTrace(A, trace));
    seed.get_values(trace.bvalue());
    stack.reverse();
    g.set_values(A.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output& seed, const Output& hval, const Input& x,
             const Input& p, Input& h) {
    A2DObj<T> trace;
    A2DObj<Mat<T, N, N>> A;
    x.get_values(A.value());
    p.get_values(A.pvalue());
    auto stack = MakeStack(MatTrace(A, trace));
    seed.get_values(trace.bvalue());
    hval.get_values(trace.hvalue());
    stack.hproduct();
    h.set_values(A.hvalue());
  }
};

template <typename T, int N>
class SymTraceTest : public A2DTest<T, T, SymMat<T, N>> {
 public:
  using Input = VarTuple<T, SymMat<T, N>>;
  using Output = VarTuple<T, T>;

  // Assemble a string to describe the test
  std::string name() {
    std::stringstream s;
    s << "MatTrace<" << N << ">";
    return s.str();
  }

  // Evaluate the matrix-matrix product
  Output eval(const Input& x) {
    T trace;
    SymMat<T, N> A;
    x.get_values(A);
    MatTrace(A, trace);
    return MakeVarTuple<T>(trace);
  }

  // Compute the derivative
  void deriv(const Output& seed, const Input& x, Input& g) {
    ADObj<T> trace;
    ADObj<SymMat<T, N>> A;

    x.get_values(A.value());
    auto stack = MakeStack(MatTrace(A, trace));
    seed.get_values(trace.bvalue());
    stack.reverse();
    g.set_values(A.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output& seed, const Output& hval, const Input& x,
             const Input& p, Input& h) {
    A2DObj<T> trace;
    A2DObj<SymMat<T, N>> A;

    x.get_values(A.value());
    p.get_values(A.pvalue());
    auto stack = MakeStack(MatTrace(A, trace));
    seed.get_values(trace.bvalue());
    hval.get_values(trace.hvalue());
    stack.hproduct();
    h.set_values(A.hvalue());
  }
};

inline bool MatTraceTestAll(bool component = false, bool write_output = true) {
  using Tc = A2D_complex_t<double>;

  bool passed = true;
  MatTraceTest<Tc, 2> test1;
  passed = passed && Run(test1, component, write_output);
  MatTraceTest<Tc, 4> test2;
  passed = passed && Run(test2, component, write_output);

  SymTraceTest<Tc, 4> test3;
  passed = passed && Run(test3, component, write_output);

  return passed;
}

}  // namespace Test

}  // namespace A2D

#endif  // A2D_MAT_TRACE_H
