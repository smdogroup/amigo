#ifndef A2D_GREEN_STRAIN_H
#define A2D_GREEN_STRAIN_H

#include <type_traits>

#include "../a2ddefs.h"
#include "a2dmat.h"
#include "a2dstack.h"
#include "a2dtest.h"
#include "core/a2dgreenstraincore.h"

namespace A2D {

enum class GreenStrainType { LINEAR, NONLINEAR };

template <GreenStrainType etype, typename T, int N>
A2D_FUNCTION void MatGreenStrain(const Mat<T, N, N>& Ux, SymMat<T, N>& E) {
  if constexpr (etype == GreenStrainType::LINEAR) {
    LinearGreenStrainCore<T, N>(get_data(Ux), get_data(E));
  } else {
    NonlinearGreenStrainCore<T, N>(get_data(Ux), get_data(E));
  }
}

template <GreenStrainType etype, class Utype, class Etype>
class MatGreenStrainExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<Etype>::type T;

  // Extract the dimensions of the underlying matrix
  static constexpr int M = get_matrix_rows<Utype>::size;
  static constexpr int K = get_matrix_columns<Utype>::size;
  static constexpr int N = get_symmatrix_size<Etype>::size;

  // Get the differentiation order from the output
  static constexpr ADorder order = get_diff_order<Etype>::order;

  // Make sure the matrix dimensions are consistent
  static_assert((N == K && N == M), "Matrix dimensions must agree");

  // Make sure that the order matches
  static_assert(get_diff_order<Utype>::order == order,
                "ADorder does not match");

  A2D_FUNCTION MatGreenStrainExpr(Utype& Ux, Etype& E) : Ux(Ux), E(E) {
    // printf("made mat green strain expression\n");
  }

  A2D_FUNCTION void eval() {
    if constexpr (etype == GreenStrainType::LINEAR) {
      LinearGreenStrainCore<T, N>(get_data(Ux), get_data(E));
    } else {
      NonlinearGreenStrainCore<T, N>(get_data(Ux), get_data(E));
    }
  }

  A2D_FUNCTION void bzero() { E.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    static_assert(
        !(order == ADorder::FIRST and forder == ADorder::SECOND),
        "Can't perform second order forward with first order objects");
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;

    if constexpr (etype == GreenStrainType::LINEAR) {
      LinearGreenStrainForwardCore<T, N>(GetSeed<seed>::get_data(Ux),
                                         GetSeed<seed>::get_data(E));
    } else {
      NonlinearGreenStrainForwardCore<T, N>(get_data(Ux),
                                            GetSeed<seed>::get_data(Ux),
                                            GetSeed<seed>::get_data(E));
    }
  }

  A2D_FUNCTION void reverse() {
    if constexpr (etype == GreenStrainType::LINEAR) {
      LinearGreenStrainReverseCore<T, N>(GetSeed<ADseed::b>::get_data(E),
                                         GetSeed<ADseed::b>::get_data(Ux));

    } else {
      NonlinearGreenStrainReverseCore<T, N>(get_data(Ux),
                                            GetSeed<ADseed::b>::get_data(E),
                                            GetSeed<ADseed::b>::get_data(Ux));
    }
  }

  A2D_FUNCTION void hzero() { E.hzero(); }

  A2D_FUNCTION void hreverse() {
    if constexpr (etype == GreenStrainType::LINEAR) {
      LinearGreenStrainHReverseCore<T, N>(GetSeed<ADseed::h>::get_data(E),
                                          GetSeed<ADseed::h>::get_data(Ux));

    } else {
      NonlinearGreenStrainHReverseCore<T, N>(
          get_data(Ux), GetSeed<ADseed::p>::get_data(Ux),
          GetSeed<ADseed::b>::get_data(E), GetSeed<ADseed::h>::get_data(E),
          GetSeed<ADseed::h>::get_data(Ux));
    }
  }

  Utype& Ux;
  Etype& E;
};

template <GreenStrainType etype, class UxMat, class EMat>
A2D_FUNCTION auto MatGreenStrain(ADObj<UxMat>& Ux, ADObj<EMat>& E) {
  return MatGreenStrainExpr<etype, ADObj<UxMat>, ADObj<EMat>>(Ux, E);
}

template <GreenStrainType etype, class UxMat, class EMat>
A2D_FUNCTION auto MatGreenStrain(A2DObj<UxMat>& Ux, A2DObj<EMat>& E) {
  return MatGreenStrainExpr<etype, A2DObj<UxMat>, A2DObj<EMat>>(Ux, E);
}

namespace Test {

template <GreenStrainType etype, typename T, int N>
class MatGreenStrainTest : public A2DTest<T, SymMat<T, N>, Mat<T, N, N>> {
 public:
  using Input = VarTuple<T, Mat<T, N, N>>;
  using Output = VarTuple<T, SymMat<T, N>>;

  // Assemble a string to describe the test
  std::string name() {
    std::stringstream s;
    s << "MatGreenStrain<";
    if (etype == GreenStrainType::LINEAR) {
      s << "LINEAR," << N << ">";
    } else {
      s << "NONLINEAR," << N << ">";
    }

    return s.str();
  }

  // Evaluate the matrix-matrix product
  Output eval(const Input& x) {
    Mat<T, N, N> Ux;
    SymMat<T, N> E;
    x.get_values(Ux);
    MatGreenStrain<etype>(Ux, E);
    return MakeVarTuple<T>(E);
  }

  // Compute the derivative
  void deriv(const Output& seed, const Input& x, Input& g) {
    ADObj<Mat<T, N, N>> Ux;
    ADObj<SymMat<T, N>> E;

    x.get_values(Ux.value());
    auto stack = MakeStack(MatGreenStrain<etype>(Ux, E));
    seed.get_values(E.bvalue());
    stack.reverse();
    g.set_values(Ux.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output& seed, const Output& hval, const Input& x,
             const Input& p, Input& h) {
    A2DObj<Mat<T, N, N>> Ux;
    A2DObj<SymMat<T, N>> E;

    x.get_values(Ux.value());
    p.get_values(Ux.pvalue());
    auto stack = MakeStack(MatGreenStrain<etype>(Ux, E));
    seed.get_values(E.bvalue());
    hval.get_values(E.hvalue());
    stack.hproduct();
    h.set_values(Ux.hvalue());
  }
};

inline bool MatGreenStrainTestAll(bool component = false,
                                  bool write_output = true) {
  using Tc = A2D_complex_t<double>;

  bool passed = true;
  MatGreenStrainTest<GreenStrainType::LINEAR, Tc, 2> test1;
  passed = passed && Run(test1, component, write_output);
  MatGreenStrainTest<GreenStrainType::NONLINEAR, Tc, 2> test2;
  passed = passed && Run(test2, component, write_output);

  MatGreenStrainTest<GreenStrainType::LINEAR, Tc, 3> test3;
  passed = passed && Run(test3, component, write_output);
  MatGreenStrainTest<GreenStrainType::NONLINEAR, Tc, 3> test4;
  passed = passed && Run(test4, component, write_output);

  return passed;
}

}  // namespace Test

}  // namespace A2D

#endif  // A2D_GREEN_STRAIN_H
