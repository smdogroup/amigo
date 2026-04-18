#ifndef A2D_SYM_MAT_TRACE_H
#define A2D_SYM_MAT_TRACE_H

#include <type_traits>

#include "../a2ddefs.h"
#include "a2dmat.h"
#include "a2dstack.h"
#include "a2dtest.h"
#include "core/a2dsymmatmulttracecore.h"

namespace A2D {

template <typename T, int N>
A2D_FUNCTION void SymMatMultTrace(const SymMat<T, N>& S, const SymMat<T, N>& E,
                                  T& output) {
  output = SymMatMultTraceCore<T, N>(get_data(S), get_data(E));
}

template <class Stype, class Etype, class dtype>
class SymMatMultTraceExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<Etype>::type T;

  // Extract the dimensions of the underlying matrices
  static constexpr int N = get_symmatrix_size<Etype>::size;
  static constexpr int M = get_symmatrix_size<Stype>::size;

  // Get the differentiation order from the output
  static constexpr ADorder order = get_diff_order<dtype>::order;

  // Make sure the matrix dimensions are consistent
  static_assert((N == M), "Matrix dimensions must agree");

  // Make sure that the order matches
  static_assert(get_diff_order<Stype>::order == order,
                "ADorder does not match");
  static_assert(get_diff_order<Etype>::order == order,
                "ADorder does not match");

  A2D_FUNCTION SymMatMultTraceExpr(Stype& S, Stype& E, dtype& out)
      : S(S), E(E), out(out) {}

  A2D_FUNCTION void eval() {
    get_data(out) = SymMatMultTraceCore<T, N>(get_data(S), get_data(E));
  }

  A2D_FUNCTION void bzero() { out.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    static_assert(
        !(order == ADorder::FIRST and forder == ADorder::SECOND),
        "Can't perform second order forward with first order objects");
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;

    GetSeed<seed>::get_data(out) =
        SymMatMultTraceCore<T, N>(GetSeed<seed>::get_data(S), get_data(E)) +
        SymMatMultTraceCore<T, N>(get_data(S), GetSeed<seed>::get_data(E));
  }

  A2D_FUNCTION void reverse() {
    SymMatMultTraceReverseCore<T, N>(out.bvalue(), get_data(S),
                                     GetSeed<ADseed::b>::get_data(E));
    SymMatMultTraceReverseCore<T, N>(out.bvalue(), get_data(E),
                                     GetSeed<ADseed::b>::get_data(S));
  }

  A2D_FUNCTION void hzero() { out.hzero(); }

  A2D_FUNCTION void hreverse() {
    SymMatMultTraceReverseCore<T, N>(out.bvalue(),
                                     GetSeed<ADseed::p>::get_data(S),
                                     GetSeed<ADseed::h>::get_data(E));
    SymMatMultTraceReverseCore<T, N>(out.hvalue(), get_data(S),
                                     GetSeed<ADseed::h>::get_data(E));

    SymMatMultTraceReverseCore<T, N>(out.bvalue(),
                                     GetSeed<ADseed::p>::get_data(E),
                                     GetSeed<ADseed::h>::get_data(S));
    SymMatMultTraceReverseCore<T, N>(out.hvalue(), get_data(E),
                                     GetSeed<ADseed::h>::get_data(S));
  }

  Stype &S, &E;
  dtype& out;
};

template <class Stype, class Etype, class dtype>
A2D_FUNCTION auto SymMatMultTrace(ADObj<Stype>& S, ADObj<Etype>& E,
                                  ADObj<dtype>& output) {
  return SymMatMultTraceExpr<ADObj<Stype>, ADObj<Etype>, ADObj<dtype>>(S, E,
                                                                       output);
}

template <class Stype, class Etype, class dtype>
A2D_FUNCTION auto SymMatMultTrace(A2DObj<Stype>& S, A2DObj<Etype>& E,
                                  A2DObj<dtype>& output) {
  return SymMatMultTraceExpr<A2DObj<Stype>, A2DObj<Etype>, A2DObj<dtype>>(
      S, E, output);
}

namespace Test {

template <typename T, int N>
class SymMatMultTraceTest : public A2DTest<T, T, SymMat<T, N>, SymMat<T, N>> {
 public:
  using Input = VarTuple<T, SymMat<T, N>, SymMat<T, N>>;
  using Output = VarTuple<T, T>;

  // Assemble a string to describe the test
  std::string name() {
    std::stringstream s;
    s << "SymMatMultTrace<" << N << ">";
    return s.str();
  }

  // Evaluate the matrix-matrix product
  Output eval(const Input& x) {
    T output;
    SymMat<T, N> S, E;
    x.get_values(S, E);
    SymMatMultTrace(S, E, output);
    return MakeVarTuple<T>(output);
  }

  // Compute the derivative
  void deriv(const Output& seed, const Input& x, Input& g) {
    ADObj<T> output;
    ADObj<SymMat<T, N>> S, E;

    x.get_values(S.value(), E.value());
    auto stack = MakeStack(SymMatMultTrace(S, E, output));
    seed.get_values(output.bvalue());
    stack.reverse();
    g.set_values(S.bvalue(), E.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output& seed, const Output& hval, const Input& x,
             const Input& p, Input& h) {
    A2DObj<T> output;
    A2DObj<SymMat<T, N>> S, E;

    x.get_values(S.value(), E.value());
    p.get_values(S.pvalue(), E.pvalue());
    auto stack = MakeStack(SymMatMultTrace(S, E, output));
    seed.get_values(output.bvalue());
    hval.get_values(output.hvalue());
    stack.hproduct();
    h.set_values(S.hvalue(), E.hvalue());
  }
};

inline bool SymMatMultTraceTestAll(bool component = false,
                                   bool write_output = true) {
  using Tc = A2D_complex_t<double>;

  bool passed = true;
  SymMatMultTraceTest<Tc, 2> test1;
  passed = passed && Run(test1, component, write_output);
  SymMatMultTraceTest<Tc, 3> test2;
  passed = passed && Run(test2, component, write_output);
  SymMatMultTraceTest<Tc, 4> test3;
  passed = passed && Run(test3, component, write_output);

  return passed;
}

}  // namespace Test

}  // namespace A2D

#endif  // A2D_SYM_MAT_TRACE_H
