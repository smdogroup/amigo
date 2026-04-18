#ifndef A2D_VEC_SUM_H
#define A2D_VEC_SUM_H

#include "../a2ddefs.h"
#include "a2dstack.h"
#include "a2dtest.h"
#include "a2dvec.h"
#include "core/a2dveccore.h"

namespace A2D {

template <typename T, int N>
A2D_FUNCTION void VecSum(const Vec<T, N> &x, const Vec<T, N> &y, Vec<T, N> &z) {
  VecSumCore<T, N>(get_data(x), get_data(y), get_data(z));
}

template <class xtype, class ytype, class ztype>
class VecSumExpr {
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

  // Make sure the matrix dimensions are consistent
  static_assert((N == M && M == K), "Vector sizes must agree");

  A2D_FUNCTION
  VecSumExpr(xtype &x, ytype &y, ztype &z) : x(x), y(y), z(z) {}

  A2D_FUNCTION void eval() {
    VecSumCore<T, N>(get_data(x), get_data(y), get_data(z));
  }

  A2D_FUNCTION void bzero() { z.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;
    if constexpr (adx == ADiffType::ACTIVE && ady == ADiffType::ACTIVE) {
      VecSumCore<T, N>(GetSeed<seed>::get_data(x), GetSeed<seed>::get_data(y),
                       GetSeed<seed>::get_data(z));
    } else if constexpr (adx == ADiffType::ACTIVE) {
      VecCopyCore<T, N>(GetSeed<seed>::get_data(x), GetSeed<seed>::get_data(z));
    } else if constexpr (ady == ADiffType::ACTIVE) {
      VecCopyCore<T, N>(GetSeed<seed>::get_data(y), GetSeed<seed>::get_data(z));
    }
  }

  A2D_FUNCTION void reverse() {
    constexpr ADseed seed = ADseed::b;
    if constexpr (adx == ADiffType::ACTIVE) {
      VecAddCore<T, N>(GetSeed<seed>::get_data(z), GetSeed<seed>::get_data(x));
    }
    if constexpr (ady == ADiffType::ACTIVE) {
      VecAddCore<T, N>(GetSeed<seed>::get_data(z), GetSeed<seed>::get_data(y));
    }
  }

  A2D_FUNCTION void hzero() { z.hzero(); }

  A2D_FUNCTION void hreverse() {
    constexpr ADseed seed = ADseed::h;
    if constexpr (adx == ADiffType::ACTIVE) {
      VecAddCore<T, N>(GetSeed<seed>::get_data(z), GetSeed<seed>::get_data(x));
    }
    if constexpr (ady == ADiffType::ACTIVE) {
      VecAddCore<T, N>(GetSeed<seed>::get_data(z), GetSeed<seed>::get_data(y));
    }
  }

  xtype &x;
  ytype &y;
  ztype &z;
};

template <class xtype, class ytype, class ztype>
A2D_FUNCTION auto VecSum(ADObj<xtype> &x, ADObj<ytype> &y, ADObj<ztype> &z) {
  return VecSumExpr<ADObj<xtype>, ADObj<ytype>, ADObj<ztype>>(x, y, z);
}

template <class xtype, class ytype, class ztype>
A2D_FUNCTION auto VecSum(A2DObj<xtype> &x, A2DObj<ytype> &y, A2DObj<ztype> &z) {
  return VecSumExpr<A2DObj<xtype>, A2DObj<ytype>, A2DObj<ztype>>(x, y, z);
}

template <class xtype, class ytype, class ztype>
A2D_FUNCTION auto VecSum(const xtype &x, ADObj<ytype> &y, ADObj<ztype> &z) {
  return VecSumExpr<const xtype, ADObj<ytype>, ADObj<ztype>>(x, y, z);
}

template <class xtype, class ytype, class ztype>
A2D_FUNCTION auto VecSum(const xtype &x, A2DObj<ytype> &y, A2DObj<ztype> &z) {
  return VecSumExpr<const xtype, A2DObj<ytype>, A2DObj<ztype>>(x, y, z);
}

template <class xtype, class ytype, class ztype>
A2D_FUNCTION auto VecSum(ADObj<xtype> &x, const ytype &y, ADObj<ztype> &z) {
  return VecSumExpr<ADObj<xtype>, const ytype, ADObj<ztype>>(x, y, z);
}

template <class xtype, class ytype, class ztype>
A2D_FUNCTION auto VecSum(A2DObj<xtype> &x, const ytype &y, A2DObj<ztype> &z) {
  return VecSumExpr<A2DObj<xtype>, const ytype, A2DObj<ztype>>(x, y, z);
}

template <typename T, int N>
A2D_FUNCTION void VecSum(const T alpha, const Vec<T, N> &x, const T beta,
                         const Vec<T, N> &y, Vec<T, N> &z) {
  VecSumCore<T, N>(alpha, get_data(x), beta, get_data(y), get_data(z));
}

template <class atype, class xtype, class btype, class ytype, class ztype>
class VecSumScaleExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<ztype>::type T;

  // Extract the dimensions of the underlying vectors
  static constexpr int N = get_vec_size<xtype>::size;
  static constexpr int M = get_vec_size<ytype>::size;
  static constexpr int K = get_vec_size<ztype>::size;

  // Get the types of the vectors
  static constexpr ADiffType ada = get_diff_type<atype>::diff_type;
  static constexpr ADiffType adb = get_diff_type<btype>::diff_type;
  static constexpr ADiffType adx = get_diff_type<xtype>::diff_type;
  static constexpr ADiffType ady = get_diff_type<ytype>::diff_type;

  // Make sure the matrix dimensions are consistent
  static_assert((N == M && M == K), "Vector sizes must agree");

  A2D_FUNCTION
  VecSumScaleExpr(atype alpha, xtype &x, btype beta, ytype &y, ztype &z)
      : alpha(alpha), x(x), beta(beta), y(y), z(z) {}

  A2D_FUNCTION void eval() {
    VecSumCore<T, N>(get_data(alpha), get_data(x), get_data(beta), get_data(y),
                     get_data(z));
  }

  A2D_FUNCTION void bzero() { z.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;

    if constexpr (ada == ADiffType::ACTIVE && adx == ADiffType::ACTIVE &&
                  adb == ADiffType::ACTIVE && ady == ADiffType::ACTIVE) {
      VecSumCore<T, N>(get_data(alpha), GetSeed<seed>::get_data(x),
                       get_data(beta), GetSeed<seed>::get_data(y),
                       GetSeed<seed>::get_data(z));
      VecSumCoreAdd<T, N>(GetSeed<seed>::get_data(alpha), get_data(x),
                          GetSeed<seed>::get_data(beta), get_data(y),
                          GetSeed<seed>::get_data(z));
    } else {
      VecZeroCore<T, N>(GetSeed<seed>::get_data(z));
      if constexpr (adx == ADiffType::ACTIVE) {
        VecAddCore<T, N>(get_data(alpha), GetSeed<seed>::get_data(x),
                         GetSeed<seed>::get_data(z));
      }
      if constexpr (ady == ADiffType::ACTIVE) {
        VecAddCore<T, N>(get_data(beta), GetSeed<seed>::get_data(y),
                         GetSeed<seed>::get_data(z));
      }
      if constexpr (ada == ADiffType::ACTIVE) {
        VecAddCore<T, N>(GetSeed<seed>::get_data(alpha), get_data(x),
                         GetSeed<seed>::get_data(z));
      }
      if constexpr (adb == ADiffType::ACTIVE) {
        VecAddCore<T, N>(GetSeed<seed>::get_data(beta), get_data(y),
                         GetSeed<seed>::get_data(z));
      }
    }
  }

  A2D_FUNCTION void reverse() {
    constexpr ADseed seed = ADseed::b;
    if constexpr (adx == ADiffType::ACTIVE) {
      VecAddCore<T, N>(get_data(alpha), GetSeed<seed>::get_data(z),
                       GetSeed<seed>::get_data(x));
    }
    if constexpr (ady == ADiffType::ACTIVE) {
      VecAddCore<T, N>(get_data(beta), GetSeed<seed>::get_data(z),
                       GetSeed<seed>::get_data(y));
    }
    if constexpr (ada == ADiffType::ACTIVE) {
      GetSeed<seed>::get_data(alpha) +=
          VecDotCore<T, N>(GetSeed<seed>::get_data(z), get_data(x));
    }
    if constexpr (adb == ADiffType::ACTIVE) {
      GetSeed<seed>::get_data(beta) +=
          VecDotCore<T, N>(GetSeed<seed>::get_data(z), get_data(y));
    }
  }

  A2D_FUNCTION void hzero() { z.hzero(); }

  A2D_FUNCTION void hreverse() {
    constexpr ADseed seed = ADseed::h;
    if constexpr (adx == ADiffType::ACTIVE) {
      VecAddCore<T, N>(get_data(alpha), GetSeed<seed>::get_data(z),
                       GetSeed<seed>::get_data(x));
    }
    if constexpr (ady == ADiffType::ACTIVE) {
      VecAddCore<T, N>(get_data(beta), GetSeed<seed>::get_data(z),
                       GetSeed<seed>::get_data(y));
    }
    if constexpr (ada == ADiffType::ACTIVE) {
      GetSeed<seed>::get_data(alpha) +=
          VecDotCore<T, N>(GetSeed<seed>::get_data(z), get_data(x));
    }
    if constexpr (adb == ADiffType::ACTIVE) {
      GetSeed<seed>::get_data(beta) +=
          VecDotCore<T, N>(GetSeed<seed>::get_data(z), get_data(y));
    }
    if constexpr (adx == ADiffType::ACTIVE && ada == ADiffType::ACTIVE) {
      GetSeed<seed>::get_data(alpha) += VecDotCore<T, N>(
          GetSeed<ADseed::b>::get_data(z), GetSeed<ADseed::p>::get_data(x));
      VecAddCore<T, N>(GetSeed<ADseed::p>::get_data(alpha),
                       GetSeed<ADseed::b>::get_data(z),
                       GetSeed<seed>::get_data(x));
    }
    if constexpr (ady == ADiffType::ACTIVE && adb == ADiffType::ACTIVE) {
      GetSeed<seed>::get_data(beta) += VecDotCore<T, N>(
          GetSeed<ADseed::b>::get_data(z), GetSeed<ADseed::p>::get_data(y));
      VecAddCore<T, N>(GetSeed<ADseed::p>::get_data(beta),
                       GetSeed<ADseed::b>::get_data(z),
                       GetSeed<seed>::get_data(y));
    }
  }

  atype alpha;
  xtype &x;
  btype beta;
  ytype &y;
  ztype &z;
};

// Full AD
template <class atype, class xtype, class btype, class ytype, class ztype>
A2D_FUNCTION auto VecSum(ADObj<atype> &alpha, ADObj<xtype> &x,
                         ADObj<btype> &beta, ADObj<ytype> &y, ADObj<ztype> &z) {
  return VecSumScaleExpr<ADObj<atype> &, ADObj<xtype>, ADObj<btype> &,
                         ADObj<ytype>, ADObj<ztype>>(alpha, x, beta, y, z);
}

template <class atype, class xtype, class btype, class ytype, class ztype>
A2D_FUNCTION auto VecSum(A2DObj<atype> &alpha, A2DObj<xtype> &x,
                         A2DObj<btype> &beta, A2DObj<ytype> &y,
                         A2DObj<ztype> &z) {
  return VecSumScaleExpr<A2DObj<atype> &, A2DObj<xtype>, A2DObj<btype> &,
                         A2DObj<ytype>, A2DObj<ztype>>(alpha, x, beta, y, z);
}

template <class atype, class xtype, class btype, class ytype, class ztype>
A2D_FUNCTION auto VecSum(const atype alpha, ADObj<xtype> &x, const btype beta,
                         ADObj<ytype> &y, ADObj<ztype> &z) {
  return VecSumScaleExpr<const atype, ADObj<xtype>, const btype, ADObj<ytype>,
                         ADObj<ztype>>(alpha, x, beta, y, z);
}

template <class atype, class xtype, class btype, class ytype, class ztype>
A2D_FUNCTION auto VecSum(const atype alpha, A2DObj<xtype> &x, const btype beta,
                         A2DObj<ytype> &y, A2DObj<ztype> &z) {
  return VecSumScaleExpr<const atype, A2DObj<xtype>, const btype, A2DObj<ytype>,
                         A2DObj<ztype>>(alpha, x, beta, y, z);
}

template <class atype, class xtype, class btype, class ytype, class ztype>
A2D_FUNCTION auto VecSum(const atype alpha, const xtype &x, const btype beta,
                         ADObj<ytype> &y, ADObj<ztype> &z) {
  return VecSumScaleExpr<const atype, const xtype, const btype, ADObj<ytype>,
                         ADObj<ztype>>(alpha, x, beta, y, z);
}

template <class atype, class xtype, class btype, class ytype, class ztype>
A2D_FUNCTION auto VecSum(const atype alpha, const xtype &x, const btype beta,
                         A2DObj<ytype> &y, A2DObj<ztype> &z) {
  return VecSumScaleExpr<const atype, const xtype, const btype, A2DObj<ytype>,
                         A2DObj<ztype>>(alpha, x, beta, y, z);
}

template <class atype, class xtype, class btype, class ytype, class ztype>
A2D_FUNCTION auto VecSum(const atype alpha, ADObj<xtype> &x, const btype beta,
                         const ytype &y, ADObj<ztype> &z) {
  return VecSumScaleExpr<const atype, ADObj<xtype>, const btype, const ytype,
                         ADObj<ztype>>(alpha, x, beta, y, z);
}

template <class atype, class xtype, class btype, class ytype, class ztype>
A2D_FUNCTION auto VecSum(const atype alpha, A2DObj<xtype> &x, const btype beta,
                         const ytype &y, A2DObj<ztype> &z) {
  return VecSumScaleExpr<const atype, A2DObj<xtype>, const btype, const ytype,
                         A2DObj<ztype>>(alpha, x, beta, y, z);
}

template <class atype, class xtype, class btype, class ytype, class ztype>
A2D_FUNCTION auto VecSum(ADObj<atype> &alpha, const xtype &x,
                         ADObj<btype> &beta, const ytype &y, ADObj<ztype> &z) {
  return VecSumScaleExpr<ADObj<atype> &, const xtype, ADObj<btype> &,
                         const ytype, ADObj<ztype>>(alpha, x, beta, y, z);
}

template <class atype, class xtype, class btype, class ytype, class ztype>
A2D_FUNCTION auto VecSum(A2DObj<atype> &alpha, const xtype &x,
                         A2DObj<btype> &beta, const ytype &y,
                         A2DObj<ztype> &z) {
  return VecSumScaleExpr<A2DObj<atype> &, const xtype, A2DObj<btype> &,
                         const ytype, A2DObj<ztype>>(alpha, x, beta, y, z);
}

namespace Test {

template <typename T, int N>
class VecSumTest : public A2DTest<T, Vec<T, N>, Vec<T, N>, Vec<T, N>> {
 public:
  using Input = VarTuple<T, Vec<T, N>, Vec<T, N>>;
  using Output = VarTuple<T, Vec<T, N>>;

  // Assemble a string to describe the test
  std::string name() {
    std::stringstream s;
    s << "VecSum<" << N << ">";
    return s.str();
  }

  // Evaluate the operation
  Output eval(const Input &x) {
    Vec<T, N> A, B, C;
    x.get_values(A, B);
    VecSum(A, B, C);
    return MakeVarTuple<T>(C);
  }

  // Compute the derivative
  void deriv(const Output &seed, const Input &x, Input &g) {
    ADObj<Vec<T, N>> A, B, C;
    x.get_values(A.value(), B.value());
    auto stack = MakeStack(VecSum(A, B, C));
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
    auto stack = MakeStack(VecSum(A, B, C));
    seed.get_values(C.bvalue());
    hval.get_values(C.hvalue());
    stack.hproduct();
    h.set_values(A.hvalue(), B.hvalue());
  }
};

template <typename T, int N>
class VecSumScaleTest
    : public A2DTest<T, Vec<T, N>, T, Vec<T, N>, T, Vec<T, N>> {
 public:
  using Input = VarTuple<T, T, Vec<T, N>, T, Vec<T, N>>;
  using Output = VarTuple<T, Vec<T, N>>;

  // Assemble a string to describe the test
  std::string name() {
    std::stringstream s;
    s << "VecSum<" << N << ">";
    return s.str();
  }

  // Evaluate the Vecrix-Vecrix product
  Output eval(const Input &x) {
    T alpha, beta;
    Vec<T, N> A, B, C;
    x.get_values(alpha, A, beta, B);
    VecSum(alpha, A, beta, B, C);
    return MakeVarTuple<T>(C);
  }

  // Compute the derivative
  void deriv(const Output &seed, const Input &x, Input &g) {
    ADObj<T> alpha, beta;
    ADObj<Vec<T, N>> A, B, C;
    x.get_values(alpha.value(), A.value(), beta.value(), B.value());
    auto stack = MakeStack(VecSum(alpha, A, beta, B, C));
    seed.get_values(C.bvalue());
    stack.reverse();
    g.set_values(alpha.bvalue(), A.bvalue(), beta.bvalue(), B.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output &seed, const Output &hval, const Input &x,
             const Input &p, Input &h) {
    A2DObj<T> alpha, beta;
    A2DObj<Vec<T, N>> A, B, C;
    x.get_values(alpha.value(), A.value(), beta.value(), B.value());
    p.get_values(alpha.pvalue(), A.pvalue(), beta.pvalue(), B.pvalue());
    auto stack = MakeStack(VecSum(alpha, A, beta, B, C));
    seed.get_values(C.bvalue());
    hval.get_values(C.hvalue());
    stack.hproduct();
    h.set_values(alpha.hvalue(), A.hvalue(), beta.hvalue(), B.hvalue());
  }
};

inline bool VecSumTestAll(bool component = false, bool write_output = true) {
  using Tc = A2D_complex_t<double>;

  bool passed = true;
  VecSumTest<Tc, 3> test1;
  passed = passed && Run(test1, component, write_output);
  VecSumTest<Tc, 5> test2;
  passed = passed && Run(test2, component, write_output);

  VecSumScaleTest<Tc, 4> test3;
  passed = passed && Run(test3, component, write_output);

  return passed;
}

}  // namespace Test

}  // namespace A2D

#endif  // A2D_VEC_SUM_H
