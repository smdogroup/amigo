#ifndef A2D_HADAMARD_H
#define A2D_HADAMARD_H

#include "../a2ddefs.h"
#include "a2dstack.h"
#include "a2dtest.h"
#include "a2dvec.h"
#include "core/a2dveccore.h"

namespace A2D {

template <typename T, int N>
A2D_FUNCTION void VecHadamard(const Vec<T, N> &x, const Vec<T, N> &y,
                              Vec<T, N> &z) {
  VecHadamardCore<T, N>(get_data(x), get_data(y), get_data(z));
}

template <class xtype, class ytype, class ztype>
class VecHadamardExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<ztype>::type T;

  // Extract the dimensions of the underlying vectors
  static constexpr int N = get_vec_size<xtype>::size;
  static constexpr int M = get_vec_size<ytype>::size;
  static constexpr int K = get_vec_size<ztype>::size;

  // Get the types of the vectors
  static constexpr ADiffType adx = get_diff_type<xtype>::diff_type;
  static constexpr ADiffType ady = get_diff_type<ytype>::diff_type;

  // Make sure the vector dimensions are consistent
  static_assert((N == M && M == K), "Vector sizes must agree");

  A2D_FUNCTION
  VecHadamardExpr(xtype &x, ytype &y, ztype &z) : x(x), y(y), z(z) {}

  A2D_FUNCTION void eval() {
    VecHadamardCore<T, N>(get_data(x), get_data(y), get_data(z));
  }

  A2D_FUNCTION void bzero() { z.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;
    if constexpr (adx == ADiffType::ACTIVE && ady == ADiffType::ACTIVE) {
      VecHadamardDoubleCore<T, N>(get_data(x), GetSeed<seed>::get_data(x),
                                  get_data(y), GetSeed<seed>::get_data(y),
                                  GetSeed<seed>::get_data(z));
    } else if constexpr (adx == ADiffType::ACTIVE) {
      VecHadamardSingleCore<T, N>(get_data(y), GetSeed<seed>::get_data(x),
                                  GetSeed<seed>::get_data(z));
    } else if constexpr (ady == ADiffType::ACTIVE) {
      VecHadamardSingleCore<T, N>(get_data(x), GetSeed<seed>::get_data(y),
                                  GetSeed<seed>::get_data(z));
    }
  }

  A2D_FUNCTION void reverse() {
    constexpr ADseed seed = ADseed::b;
    if constexpr (adx == ADiffType::ACTIVE) {
      VecHadamardAddCore<T, N>(get_data(y), GetSeed<seed>::get_data(z),
                               GetSeed<seed>::get_data(x));
    }
    if constexpr (ady == ADiffType::ACTIVE) {
      VecHadamardAddCore<T, N>(get_data(x), GetSeed<seed>::get_data(z),
                               GetSeed<seed>::get_data(y));
    }
  }

  A2D_FUNCTION void hzero() { z.hzero(); }

  A2D_FUNCTION void hreverse() {
    constexpr ADseed seed = ADseed::h;
    if constexpr (adx == ADiffType::ACTIVE) {
      VecHadamardAddCore<T, N>(get_data(y), GetSeed<seed>::get_data(z),
                               GetSeed<seed>::get_data(x));
    }
    if constexpr (ady == ADiffType::ACTIVE) {
      VecHadamardAddCore<T, N>(get_data(x), GetSeed<seed>::get_data(z),
                               GetSeed<seed>::get_data(y));
    }
    if constexpr (adx == ADiffType::ACTIVE && ady == ADiffType::ACTIVE) {
      VecHadamardAddCore<T, N>(GetSeed<ADseed::p>::get_data(y),
                               GetSeed<ADseed::b>::get_data(z),
                               GetSeed<ADseed::h>::get_data(x));
      VecHadamardAddCore<T, N>(GetSeed<ADseed::b>::get_data(z),
                               GetSeed<ADseed::p>::get_data(x),
                               GetSeed<ADseed::h>::get_data(y));
    }
  }

  xtype &x;
  ytype &y;
  ztype &z;
};

template <class xtype, class ytype, class ztype>
A2D_FUNCTION auto VecHadamard(ADObj<xtype> &x, ADObj<ytype> &y,
                              ADObj<ztype> &z) {
  return VecHadamardExpr<ADObj<xtype>, ADObj<ytype>, ADObj<ztype>>(x, y, z);
}

template <class xtype, class ytype, class ztype>
A2D_FUNCTION auto VecHadamard(A2DObj<xtype> &x, A2DObj<ytype> &y,
                              A2DObj<ztype> &z) {
  return VecHadamardExpr<A2DObj<xtype>, A2DObj<ytype>, A2DObj<ztype>>(x, y, z);
}

template <class xtype, class ytype, class ztype>
A2D_FUNCTION auto VecHadamard(const xtype &x, ADObj<ytype> &y,
                              ADObj<ztype> &z) {
  return VecHadamardExpr<const xtype, ADObj<ytype>, ADObj<ztype>>(x, y, z);
}

template <class xtype, class ytype, class ztype>
A2D_FUNCTION auto VecHadamard(const xtype &x, A2DObj<ytype> &y,
                              A2DObj<ztype> &z) {
  return VecHadamardExpr<const xtype, A2DObj<ytype>, A2DObj<ztype>>(x, y, z);
}

template <class xtype, class ytype, class ztype>
A2D_FUNCTION auto VecHadamard(ADObj<xtype> &x, const ytype &y,
                              ADObj<ztype> &z) {
  return VecHadamardExpr<ADObj<xtype>, const ytype, ADObj<ztype>>(x, y, z);
}

template <class xtype, class ytype, class ztype>
A2D_FUNCTION auto VecHadamard(A2DObj<xtype> &x, const ytype &y,
                              A2DObj<ztype> &z) {
  return VecHadamardExpr<A2DObj<xtype>, const ytype, A2DObj<ztype>>(x, y, z);
}

namespace Test {

template <typename T, int N>
class VecHadamardTest : public A2DTest<T, Vec<T, N>, Vec<T, N>, Vec<T, N>> {
 public:
  using Input = VarTuple<T, Vec<T, N>, Vec<T, N>>;
  using Output = VarTuple<T, Vec<T, N>>;

  // Assemble a string to describe the test
  std::string name() {
    std::stringstream s;
    s << "VecHadamard<" << N << ">";
    return s.str();
  }

  // Evaluate the operation
  Output eval(const Input &x) {
    Vec<T, N> A, B, C;
    x.get_values(A, B);
    VecHadamard(A, B, C);
    return MakeVarTuple<T>(C);
  }

  // Compute the derivative
  void deriv(const Output &seed, const Input &x, Input &g) {
    ADObj<Vec<T, N>> A, B, C;
    x.get_values(A.value(), B.value());
    auto stack = MakeStack(VecHadamard(A, B, C));
    seed.get_values(C.bvalue());
    stack.reverse();
    g.set_values(A.bvalue(), B.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output &seed, const Output &hval, const Input &x,
             const Input &p, Input &h) {
    A2DObj<Vec<T, N>> A, B, C;
    x.get_values(A.value(), B.value());
    p.get_values(A.pvalue(), B.pvalue());
    auto stack = MakeStack(VecHadamard(A, B, C));
    seed.get_values(C.bvalue());
    hval.get_values(C.hvalue());
    stack.hproduct();
    h.set_values(A.hvalue(), B.hvalue());
  }
};

inline bool VecHadamardTestAll(bool component = false,
                               bool write_output = true) {
  using Tc = A2D_complex_t<double>;

  bool passed = true;
  VecHadamardTest<Tc, 3> test1;
  passed = passed && Run(test1, component, write_output);

  return passed;
}

}  // namespace Test

}  // namespace A2D

#endif  // A2D_HADAMARD_H
