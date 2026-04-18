#ifndef A2D_VEC_NORM_H
#define A2D_VEC_NORM_H

#include "../a2ddefs.h"
#include "a2dstack.h"
#include "a2dtest.h"
#include "a2dvec.h"
#include "core/a2dveccore.h"

namespace A2D {

template <typename T, int N>
A2D_FUNCTION void VecNorm(const Vec<T, N> &x, T &alpha) {
  alpha = sqrt(VecDotCore<T, N>(get_data(x), get_data(x)));
}

template <class vtype, class dtype>
class VecNormExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<dtype>::type T;

  // Extract the dimensions of the underlying vectors
  static constexpr int N = get_vec_size<vtype>::size;

  // Get the differentiation order from the output
  static constexpr ADorder order = get_diff_order<dtype>::order;

  A2D_FUNCTION VecNormExpr(vtype &x, dtype &alpha) : x(x), alpha(alpha) {}

  A2D_FUNCTION void eval() {
    get_data(alpha) = sqrt(VecDotCore<T, N>(get_data(x), get_data(x)));
    inv = 1.0 / get_data(alpha);
  }
  A2D_FUNCTION void bzero() { alpha.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;
    GetSeed<seed>::get_data(alpha) =
        inv * VecDotCore<T, N>(GetSeed<seed>::get_data(x), get_data(x));
  }
  A2D_FUNCTION void reverse() {
    constexpr ADseed seed = ADseed::b;
    VecAddCore<T, N>(inv * GetSeed<seed>::get_data(alpha), get_data(x),
                     GetSeed<seed>::get_data(x));
  }

  A2D_FUNCTION void hzero() { alpha.hzero(); }

  A2D_FUNCTION void hreverse() {
    VecAddCore<T, N>(inv * GetSeed<ADseed::h>::get_data(alpha), get_data(x),
                     GetSeed<ADseed::h>::get_data(x));

    VecAddCore<T, N>(inv * GetSeed<ADseed::b>::get_data(alpha),
                     GetSeed<ADseed::p>::get_data(x),
                     GetSeed<ADseed::h>::get_data(x));

    T scale = -inv * inv * inv * GetSeed<ADseed::b>::get_data(alpha) *
              VecDotCore<T, N>(GetSeed<ADseed::p>::get_data(x), get_data(x));
    VecAddCore<T, N>(scale, get_data(x), GetSeed<ADseed::h>::get_data(x));
  }

  vtype &x;
  dtype &alpha;
  T inv;
};

template <class vtype, class dtype>
A2D_FUNCTION auto VecNorm(ADObj<vtype> &x, ADObj<dtype> &alpha) {
  return VecNormExpr<ADObj<vtype>, ADObj<dtype>>(x, alpha);
}

template <class vtype, class dtype>
A2D_FUNCTION auto VecNorm(A2DObj<vtype> &x, A2DObj<dtype> &alpha) {
  return VecNormExpr<A2DObj<vtype>, A2DObj<dtype>>(x, alpha);
}

template <typename T, int N>
A2D_FUNCTION void VecNormalize(const Vec<T, N> &x, Vec<T, N> &y) {
  T alpha = sqrt(VecDotCore<T, N>(get_data(x), get_data(x)));
  VecScaleCore<T, N>(1.0 / alpha, get_data(x), get_data(y));
}

template <class xtype, class ytype>
class VecNormalizeExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<ytype>::type T;

  // Extract the dimensions of the underlying vectors
  static constexpr int N = get_vec_size<xtype>::size;
  static constexpr int M = get_vec_size<ytype>::size;

  // Get the differentiation order from the output
  static constexpr ADorder order = get_diff_order<ytype>::order;

  // Make sure the vector dimensions are consistent
  static_assert((N == M), "Matrix dimensions must agree");

  // Make sure that the order matches
  static_assert(get_diff_order<xtype>::order == order,
                "ADorder does not match");

  A2D_FUNCTION VecNormalizeExpr(xtype &x, ytype &y) : x(x), y(y) {}

  A2D_FUNCTION void eval() {
    T alpha = sqrt(VecDotCore<T, N>(get_data(x), get_data(x)));
    inv = 1.0 / alpha;
    VecScaleCore<T, N>(inv, get_data(x), get_data(y));
  }

  A2D_FUNCTION void bzero() { y.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;
    // yp = inv * ( - (xp, y) * y + xp )
    VecScaleCore<T, N>(inv, GetSeed<seed>::get_data(x),
                       GetSeed<seed>::get_data(y));

    T scale = -inv * VecDotCore<T, N>(GetSeed<seed>::get_data(x), get_data(y));
    VecAddCore<T, N>(scale, get_data(y), GetSeed<seed>::get_data(y));
  }

  A2D_FUNCTION void reverse() {
    constexpr ADseed seed = ADseed::b;
    // xb = inv * yb - inv * inv * (yb, x) * y
    VecAddCore<T, N>(inv, GetSeed<seed>::get_data(y),
                     GetSeed<seed>::get_data(x));
    T scale =
        -inv * inv * VecDotCore<T, N>(GetSeed<seed>::get_data(y), get_data(x));
    VecAddCore<T, N>(scale, get_data(y), GetSeed<seed>::get_data(x));
  }

  A2D_FUNCTION void hzero() { y.hzero(); }

  A2D_FUNCTION void hreverse() {
    T tmp1 = VecDotCore<T, N>(GetSeed<ADseed::b>::get_data(y), get_data(y));
    T tmp2 = VecDotCore<T, N>(GetSeed<ADseed::p>::get_data(x), get_data(y));
    T tmp3 = VecDotCore<T, N>(GetSeed<ADseed::h>::get_data(y), get_data(x));
    T tmp4 = VecDotCore<T, N>(GetSeed<ADseed::b>::get_data(y),
                              GetSeed<ADseed::p>::get_data(x));

    VecAddCore<T, N>(inv, GetSeed<ADseed::h>::get_data(y),
                     GetSeed<ADseed::h>::get_data(x));

    T inv2 = inv * inv;
    VecAddCore<T, N>(-inv2 * tmp1, GetSeed<ADseed::p>::get_data(x),
                     GetSeed<ADseed::h>::get_data(x));
    VecAddCore<T, N>(-inv2 * tmp2, GetSeed<ADseed::b>::get_data(y),
                     GetSeed<ADseed::h>::get_data(x));

    T scale = inv2 * (3.0 * tmp1 * tmp2 - tmp3 - tmp4);
    VecAddCore<T, N>(scale, get_data(y), GetSeed<ADseed::h>::get_data(x));
  }

  T inv;
  xtype &x;
  ytype &y;
};

template <class xtype, class ytype>
A2D_FUNCTION auto VecNormalize(ADObj<xtype> &x, ADObj<ytype> &y) {
  return VecNormalizeExpr<ADObj<xtype>, ADObj<ytype>>(x, y);
}

template <class xtype, class ytype>
A2D_FUNCTION auto VecNormalize(A2DObj<xtype> &x, A2DObj<ytype> &y) {
  return VecNormalizeExpr<A2DObj<xtype>, A2DObj<ytype>>(x, y);
}

template <typename T, int N>
A2D_FUNCTION void VecScale(const T alpha, const Vec<T, N> &x, Vec<T, N> &y) {
  VecScaleCore<T, N>(alpha, get_data(x), get_data(y));
}

template <class dtype, class xtype, class ytype>
class VecScaleExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<dtype>::type T;

  // Extract the dimensions of the underlying vectors
  static constexpr int N = get_vec_size<xtype>::size;
  static constexpr int M = get_vec_size<ytype>::size;

  // Get the differentiation order from the output
  static constexpr ADorder order = get_diff_order<ytype>::order;

  // Get the types of the matrices
  static constexpr ADiffType ada = get_diff_type<dtype>::diff_type;
  static constexpr ADiffType adx = get_diff_type<xtype>::diff_type;

  // Make sure the matrix dimensions are consistent
  static_assert((N == M), "Vector sizes must agree");

  A2D_FUNCTION VecScaleExpr(dtype alpha, xtype &x, ytype &y)
      : alpha(alpha), x(x), y(y) {}

  A2D_FUNCTION void eval() {
    VecScaleCore<T, N>(get_data(alpha), get_data(x), get_data(y));
  }

  A2D_FUNCTION void bzero() { y.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;

    if constexpr (ada == ADiffType::ACTIVE && adx == ADiffType::ACTIVE) {
      VecScaleCore<T, N>(GetSeed<seed>::get_data(alpha), get_data(x),
                         GetSeed<seed>::get_data(y));
      VecAddCore<T, N>(get_data(alpha), GetSeed<seed>::get_data(x),
                       GetSeed<seed>::get_data(y));
    } else if constexpr (ada == ADiffType::ACTIVE) {
      VecScaleCore<T, N>(GetSeed<seed>::get_data(alpha), get_data(x),
                         GetSeed<seed>::get_data(y));
    } else if constexpr (adx == ADiffType::ACTIVE) {
      VecScaleCore<T, N>(get_data(alpha), GetSeed<seed>::get_data(x),
                         GetSeed<seed>::get_data(y));
    }
  }
  A2D_FUNCTION void reverse() {
    constexpr ADseed seed = ADseed::b;
    if constexpr (ada == ADiffType::ACTIVE) {
      GetSeed<seed>::get_data(alpha) +=
          VecDotCore<T, N>(GetSeed<seed>::get_data(y), get_data(x));
    }
    if constexpr (adx == ADiffType::ACTIVE) {
      VecAddCore<T, N>(get_data(alpha), GetSeed<seed>::get_data(y),
                       GetSeed<seed>::get_data(x));
    }
  }

  A2D_FUNCTION void hzero() { y.hzero(); }

  A2D_FUNCTION void hreverse() {
    if constexpr (ada == ADiffType::ACTIVE) {
      GetSeed<ADseed::h>::get_data(alpha) +=
          VecDotCore<T, N>(GetSeed<ADseed::h>::get_data(y), get_data(x));
    }
    if constexpr (adx == ADiffType::ACTIVE) {
      VecAddCore<T, N>(get_data(alpha), GetSeed<ADseed::h>::get_data(y),
                       GetSeed<ADseed::h>::get_data(x));
    }
    if constexpr (ada == ADiffType::ACTIVE && adx == ADiffType::ACTIVE) {
      GetSeed<ADseed::h>::get_data(alpha) += VecDotCore<T, N>(
          GetSeed<ADseed::b>::get_data(y), GetSeed<ADseed::p>::get_data(x));
      VecAddCore<T, N>(GetSeed<ADseed::p>::get_data(alpha),
                       GetSeed<ADseed::b>::get_data(y),
                       GetSeed<ADseed::h>::get_data(x));
    }
  }

  dtype alpha;
  xtype &x;
  ytype &y;
};

template <class T, class xtype, class ytype>
A2D_FUNCTION auto VecScale(ADObj<T> &alpha, ADObj<xtype> &x, ADObj<ytype> &y) {
  return VecScaleExpr<ADObj<T> &, ADObj<xtype>, ADObj<xtype>>(alpha, x, y);
}

template <class T, class xtype, class ytype>
A2D_FUNCTION auto VecScale(const T alpha, ADObj<xtype> &x, ADObj<ytype> &y) {
  return VecScaleExpr<const T, ADObj<xtype>, ADObj<xtype>>(alpha, x, y);
}

template <class T, class xtype, class ytype>
A2D_FUNCTION auto VecScale(ADObj<T> &alpha, const xtype &x, ADObj<ytype> &y) {
  return VecScaleExpr<ADObj<T> &, const xtype, ADObj<xtype>>(alpha, x, y);
}

template <class T, class xtype, class ytype>
A2D_FUNCTION auto VecScale(A2DObj<T> &alpha, A2DObj<xtype> &x,
                           A2DObj<ytype> &y) {
  return VecScaleExpr<A2DObj<T> &, A2DObj<xtype>, A2DObj<xtype>>(alpha, x, y);
}

template <class T, class xtype, class ytype>
A2D_FUNCTION auto VecScale(const T alpha, A2DObj<xtype> &x, A2DObj<ytype> &y) {
  return VecScaleExpr<const T, A2DObj<xtype>, A2DObj<xtype>>(alpha, x, y);
}

template <class T, class xtype, class ytype>
A2D_FUNCTION auto VecScale(A2DObj<T> &alpha, const xtype &x, A2DObj<ytype> &y) {
  return VecScaleExpr<A2DObj<T> &, const xtype, A2DObj<xtype>>(alpha, x, y);
}

template <typename T, int N>
A2D_FUNCTION void VecDot(const Vec<T, N> &x, const Vec<T, N> &y, T &alpha) {
  alpha = VecDotCore<T, N>(get_data(x), get_data(y));
}

template <class xtype, class ytype, class dtype>
class VecDotExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<dtype>::type T;

  // Extract the dimensions of the underlying vectors
  static constexpr int N = get_vec_size<xtype>::size;
  static constexpr int M = get_vec_size<ytype>::size;

  // Get the differentiation order from the output
  static constexpr ADorder order = get_diff_order<dtype>::order;

  // Get the types of the matrices
  static constexpr ADiffType adx = get_diff_type<xtype>::diff_type;
  static constexpr ADiffType ady = get_diff_type<ytype>::diff_type;

  // Make sure the matrix dimensions are consistent
  static_assert((N == M), "Vector dimensions must agree");

  A2D_FUNCTION VecDotExpr(xtype &x, ytype &y, dtype &alpha)
      : x(x), y(y), alpha(alpha) {}

  A2D_FUNCTION void eval() {
    get_data(alpha) = VecDotCore<T, N>(get_data(x), get_data(y));
  }

  A2D_FUNCTION void bzero() { alpha.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;

    if constexpr (adx == ADiffType::ACTIVE && ady == ADiffType::ACTIVE) {
      GetSeed<seed>::get_data(alpha) =
          VecDotCore<T, N>(GetSeed<seed>::get_data(x), get_data(y)) +
          VecDotCore<T, N>(get_data(x), GetSeed<seed>::get_data(y));

    } else if constexpr (adx == ADiffType::ACTIVE) {
      GetSeed<seed>::get_data(alpha) =
          VecDotCore<T, N>(GetSeed<seed>::get_data(x), get_data(y));
    } else if constexpr (ady == ADiffType::ACTIVE) {
      GetSeed<seed>::get_data(alpha) =
          VecDotCore<T, N>(get_data(x), GetSeed<seed>::get_data(y));
    }
  }
  A2D_FUNCTION void reverse() {
    constexpr ADseed seed = ADseed::b;
    if constexpr (adx == ADiffType::ACTIVE) {
      VecAddCore<T, N>(GetSeed<seed>::get_data(alpha), get_data(y),
                       GetSeed<seed>::get_data(x));
    }
    if constexpr (ady == ADiffType::ACTIVE) {
      VecAddCore<T, N>(GetSeed<seed>::get_data(alpha), get_data(x),
                       GetSeed<seed>::get_data(y));
    }
  }

  A2D_FUNCTION void hzero() { alpha.hzero(); }

  A2D_FUNCTION void hreverse() {
    if constexpr (adx == ADiffType::ACTIVE) {
      VecAddCore<T, N>(GetSeed<ADseed::h>::get_data(alpha), get_data(y),
                       GetSeed<ADseed::h>::get_data(x));
    }
    if constexpr (ady == ADiffType::ACTIVE) {
      VecAddCore<T, N>(GetSeed<ADseed::h>::get_data(alpha), get_data(x),
                       GetSeed<ADseed::h>::get_data(y));
    }
    if constexpr (adx == ADiffType::ACTIVE && ady == ADiffType::ACTIVE) {
      VecAddCore<T, N>(GetSeed<ADseed::b>::get_data(alpha),
                       GetSeed<ADseed::p>::get_data(y),
                       GetSeed<ADseed::h>::get_data(x));
      VecAddCore<T, N>(GetSeed<ADseed::b>::get_data(alpha),
                       GetSeed<ADseed::p>::get_data(x),
                       GetSeed<ADseed::h>::get_data(y));
    }
  }

  xtype &x;
  ytype &y;
  dtype &alpha;
};

template <class xtype, class ytype, class dtype>
A2D_FUNCTION auto VecDot(ADObj<xtype> &x, ADObj<ytype> &y,
                         ADObj<dtype> &alpha) {
  return VecDotExpr<ADObj<xtype>, ADObj<ytype>, ADObj<dtype>>(x, y, alpha);
}

template <class xtype, class ytype, class dtype>
A2D_FUNCTION auto VecDot(const xtype &x, ADObj<ytype> &y, ADObj<dtype> &alpha) {
  return VecDotExpr<const xtype, ADObj<ytype>, ADObj<dtype>>(x, y, alpha);
}

template <class xtype, class ytype, class dtype>
A2D_FUNCTION auto VecDot(ADObj<xtype> &x, const ytype &y, ADObj<dtype> &alpha) {
  return VecDotExpr<ADObj<xtype>, const ytype, ADObj<dtype>>(x, y, alpha);
}

template <class xtype, class ytype, class dtype>
A2D_FUNCTION auto VecDot(A2DObj<xtype> &x, A2DObj<ytype> &y,
                         A2DObj<dtype> &alpha) {
  return VecDotExpr<A2DObj<xtype>, A2DObj<ytype>, A2DObj<dtype>>(x, y, alpha);
}

template <class xtype, class ytype, class dtype>
A2D_FUNCTION auto VecDot(const xtype &x, A2DObj<ytype> &y,
                         A2DObj<dtype> &alpha) {
  return VecDotExpr<const xtype, A2DObj<ytype>, A2DObj<dtype>>(x, y, alpha);
}

template <class xtype, class ytype, class dtype>
A2D_FUNCTION auto VecDot(A2DObj<xtype> &x, const ytype &y,
                         A2DObj<dtype> &alpha) {
  return VecDotExpr<A2DObj<xtype>, const ytype, A2DObj<dtype>>(x, y, alpha);
}

namespace Test {

template <typename T, int N>
class VecNormTest : public A2DTest<T, T, Vec<T, N>> {
 public:
  using Input = VarTuple<T, Vec<T, N>>;
  using Output = VarTuple<T, T>;

  // Assemble a string to describe the test
  std::string name() { return "VecNorm"; }

  // Evaluate the matrix-matrix product
  Output eval(const Input &X) {
    T alpha;
    Vec<T, N> x;
    X.get_values(x);
    VecNorm(x, alpha);
    return MakeVarTuple<T>(alpha);
  }

  // Compute the derivative
  void deriv(const Output &seed, const Input &X, Input &g) {
    ADObj<T> alpha;
    ADObj<Vec<T, N>> x;
    X.get_values(x.value());
    auto stack = MakeStack(VecNorm(x, alpha));
    seed.get_values(alpha.bvalue());
    stack.reverse();
    g.set_values(x.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output &seed, const Output &hval, const Input &X,
             const Input &p, Input &h) {
    A2DObj<T> alpha;
    A2DObj<Vec<T, N>> x;
    X.get_values(x.value());
    p.get_values(x.pvalue());
    auto stack = MakeStack(VecNorm(x, alpha));
    seed.get_values(alpha.bvalue());
    hval.get_values(alpha.hvalue());
    stack.hproduct();
    h.set_values(x.hvalue());
  }
};

inline bool VecNormTestAll(bool component = false, bool write_output = true) {
  using Tc = A2D_complex_t<double>;

  bool passed = true;
  VecNormTest<Tc, 3> test1;
  passed = passed && Run(test1, component, write_output);
  VecNormTest<Tc, 6> test2;
  passed = passed && Run(test2, component, write_output);

  return passed;
}

template <typename T, int N>
class VecScaleTest : public A2DTest<T, Vec<T, N>, T, Vec<T, N>> {
 public:
  using Input = VarTuple<T, T, Vec<T, N>>;
  using Output = VarTuple<T, Vec<T, N>>;

  // Assemble a string to describe the test
  std::string name() { return "VecScale"; }

  // Evaluate the matrix-matrix product
  Output eval(const Input &X) {
    T alpha;
    Vec<T, N> x, y;
    X.get_values(alpha, x);
    VecScale(alpha, x, y);
    return MakeVarTuple<T>(y);
  }

  // Compute the derivative
  void deriv(const Output &seed, const Input &X, Input &g) {
    ADObj<T> alpha;
    ADObj<Vec<T, N>> x, y;
    X.get_values(alpha.value(), x.value());
    auto stack = MakeStack(VecScale(alpha, x, y));
    seed.get_values(y.bvalue());
    stack.reverse();
    g.set_values(alpha.bvalue(), x.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output &seed, const Output &hval, const Input &X,
             const Input &p, Input &h) {
    A2DObj<T> alpha;
    A2DObj<Vec<T, N>> x, y;
    X.get_values(alpha.value(), x.value());
    p.get_values(alpha.pvalue(), x.pvalue());
    auto stack = MakeStack(VecScale(alpha, x, y));
    seed.get_values(y.bvalue());
    hval.get_values(y.hvalue());
    stack.hproduct();
    h.set_values(alpha.hvalue(), x.hvalue());
  }
};

inline bool VecScaleTestAll(bool component = false, bool write_output = true) {
  using Tc = A2D_complex_t<double>;

  bool passed = true;
  VecScaleTest<Tc, 3> test1;
  passed = passed && Run(test1, component, write_output);
  VecScaleTest<Tc, 6> test2;
  passed = passed && Run(test2, component, write_output);

  return passed;
}

template <typename T, int N>
class VecNormalizeTest : public A2DTest<T, Vec<T, N>, Vec<T, N>> {
 public:
  using Input = VarTuple<T, Vec<T, N>>;
  using Output = VarTuple<T, Vec<T, N>>;

  // Assemble a string to describe the test
  std::string name() { return "VecNormalize"; }

  // Evaluate the matrix-matrix product
  Output eval(const Input &X) {
    Vec<T, N> x, y;
    X.get_values(x);
    VecNormalize(x, y);
    return MakeVarTuple<T>(y);
  }

  // Compute the derivative
  void deriv(const Output &seed, const Input &X, Input &g) {
    ADObj<Vec<T, N>> x, y;
    X.get_values(x.value());
    auto stack = MakeStack(VecNormalize(x, y));
    seed.get_values(y.bvalue());
    stack.reverse();
    g.set_values(x.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output &seed, const Output &hval, const Input &X,
             const Input &p, Input &h) {
    A2DObj<Vec<T, N>> x, y;
    X.get_values(x.value());
    p.get_values(x.pvalue());
    auto stack = MakeStack(VecNormalize(x, y));
    seed.get_values(y.bvalue());
    hval.get_values(y.hvalue());
    stack.hproduct();
    h.set_values(x.hvalue());
  }
};

inline bool VecNormalizeTestAll(bool component = false,
                                bool write_output = true) {
  using Tc = A2D_complex_t<double>;

  bool passed = true;
  VecNormalizeTest<Tc, 3> test1;
  passed = passed && Run(test1, component, write_output);
  VecNormalizeTest<Tc, 6> test2;
  passed = passed && Run(test2, component, write_output);

  return passed;
}

template <typename T, int N>
class VecDotTest : public A2DTest<T, T, Vec<T, N>, Vec<T, N>> {
 public:
  using Input = VarTuple<T, Vec<T, N>, Vec<T, N>>;
  using Output = VarTuple<T, T>;

  // Assemble a string to describe the test
  std::string name() { return "VecDot"; }

  // Evaluate the matrix-matrix product
  Output eval(const Input &X) {
    T alpha;
    Vec<T, N> x, y;
    X.get_values(x, y);
    VecDot(x, y, alpha);
    return MakeVarTuple<T>(alpha);
  }

  // Compute the derivative
  void deriv(const Output &seed, const Input &X, Input &g) {
    ADObj<T> alpha;
    ADObj<Vec<T, N>> x, y;
    X.get_values(x.value(), y.value());
    auto stack = MakeStack(VecDot(x, y, alpha));
    seed.get_values(alpha.bvalue());
    stack.reverse();
    g.set_values(x.bvalue(), y.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output &seed, const Output &hval, const Input &X,
             const Input &p, Input &h) {
    A2DObj<T> alpha;
    A2DObj<Vec<T, N>> x, y;
    X.get_values(x.value(), y.value());
    p.get_values(x.pvalue(), y.pvalue());
    auto stack = MakeStack(VecDot(x, y, alpha));
    seed.get_values(alpha.bvalue());
    hval.get_values(alpha.hvalue());
    stack.hproduct();
    h.set_values(x.hvalue(), y.hvalue());
  }
};

inline bool VecDotTestAll(bool component = false, bool write_output = true) {
  using Tc = A2D_complex_t<double>;

  bool passed = true;
  VecDotTest<Tc, 3> test1;
  passed = passed && Run(test1, component, write_output);
  VecDotTest<Tc, 6> test2;
  passed = passed && Run(test2, component, write_output);

  return passed;
}

}  // namespace Test

}  // namespace A2D

#endif  //  A2D_VEC_NORM_H
