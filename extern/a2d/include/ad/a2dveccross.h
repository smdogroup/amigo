#ifndef A2D_VEC_CROSS_H
#define A2D_VEC_CROSS_H

#include "../a2ddefs.h"
#include "a2dstack.h"
#include "a2dtest.h"
#include "a2dvec.h"
#include "core/a2dveccore.h"

namespace A2D {

// z = x cross y
template <typename T>
A2D_FUNCTION void VecCross(const Vec<T, 3> &x, const Vec<T, 3> &y,
                           Vec<T, 3> &z) {
  VecCrossCore<T>(get_data(x), get_data(y), get_data(z));
}

// 2D variants
// Use vec(x) = x * hat(k)
template <typename Tx, typename T>
A2D_FUNCTION void VecCross(const Tx x, const Vec<T, 2> &y, Vec<T, 2> &z) {
  z(0) = -x * y(1);
  z(1) = x * y(0);
}

// Use vec(y) = y * hat(k)
template <typename T, typename Ty>
A2D_FUNCTION void VecCross(const Vec<T, 2> &x, const Ty y, Vec<T, 2> &z) {
  z(0) = y * x(1);
  z(1) = -y * x(0);
}

template <class xtype, class ytype, class ztype>
class VecCrossExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<ztype>::type T;

  // Extract the dimensions of the underlying vectors
  static constexpr int N = get_vec_size<xtype>::size;
  static constexpr int M = get_vec_size<ytype>::size;
  static constexpr int K = get_vec_size<ztype>::size;

  // Get the types of the matrices
  static constexpr ADiffType adx = get_diff_type<xtype>::diff_type;
  static constexpr ADiffType ady = get_diff_type<ytype>::diff_type;

  // Make sure the matrix dimensions are consistent
  static_assert((N == M && M == K && K == 3), "Vector dimensions must agree");

  A2D_FUNCTION VecCrossExpr(xtype &x, ytype &y, ztype &z) : x(x), y(y), z(z) {}

  A2D_FUNCTION void eval() {
    VecCrossCore<T>(get_data(x), get_data(y), get_data(z));
  }

  A2D_FUNCTION void bzero() { z.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;

    if constexpr (adx == ADiffType::ACTIVE && ady == ADiffType::ACTIVE) {
      VecCrossCore<T>(GetSeed<seed>::get_data(x), get_data(y),
                      GetSeed<seed>::get_data(z));
      VecCrossCoreAdd<T>(get_data(x), GetSeed<seed>::get_data(y),
                         GetSeed<seed>::get_data(z));
    } else if constexpr (adx == ADiffType::ACTIVE) {
      VecCrossCore<T>(GetSeed<seed>::get_data(x), get_data(y),
                      GetSeed<seed>::get_data(z));
    } else if constexpr (ady == ADiffType::ACTIVE) {
      VecCrossCore<T>(get_data(x), GetSeed<seed>::get_data(y),
                      GetSeed<seed>::get_data(z));
    }
  }
  A2D_FUNCTION void reverse() {
    constexpr ADseed seed = ADseed::b;

    if constexpr (adx == ADiffType::ACTIVE) {
      VecCrossCoreAdd<T>(get_data(y), GetSeed<seed>::get_data(z),
                         GetSeed<seed>::get_data(x));
    }
    if constexpr (ady == ADiffType::ACTIVE) {
      VecCrossCoreAdd<T>(GetSeed<seed>::get_data(z), get_data(x),
                         GetSeed<seed>::get_data(y));
    }
  }

  A2D_FUNCTION void hzero() { z.hzero(); }

  A2D_FUNCTION void hreverse() {
    if constexpr (adx == ADiffType::ACTIVE) {
      VecCrossCoreAdd<T>(get_data(y), GetSeed<ADseed::h>::get_data(z),
                         GetSeed<ADseed::h>::get_data(x));
    }
    if constexpr (ady == ADiffType::ACTIVE) {
      VecCrossCoreAdd<T>(GetSeed<ADseed::h>::get_data(z), get_data(x),
                         GetSeed<ADseed::h>::get_data(y));
    }
    if constexpr (adx == ADiffType::ACTIVE && ady == ADiffType::ACTIVE) {
      VecCrossCoreAdd<T>(GetSeed<ADseed::p>::get_data(y),
                         GetSeed<ADseed::b>::get_data(z),
                         GetSeed<ADseed::h>::get_data(x));
      VecCrossCoreAdd<T>(GetSeed<ADseed::b>::get_data(z),
                         GetSeed<ADseed::p>::get_data(x),
                         GetSeed<ADseed::h>::get_data(y));
    }
  }

  xtype &x;
  ytype &y;
  ztype &z;
};

template <class xtype, class ytype, class ztype,
          std::enable_if_t<get_vec_size<ztype>::size == 3, bool> = true>
A2D_FUNCTION auto VecCross(ADObj<xtype> &x, ADObj<ytype> &y, ADObj<ztype> &z) {
  return VecCrossExpr<ADObj<xtype>, ADObj<ytype>, ADObj<ztype>>(x, y, z);
}

template <class xtype, class ytype, class ztype,
          std::enable_if_t<get_vec_size<ztype>::size == 3, bool> = true>
A2D_FUNCTION auto VecCross(const xtype &x, ADObj<ytype> &y, ADObj<ztype> &z) {
  return VecCrossExpr<const xtype, ADObj<ytype>, ADObj<ztype>>(x, y, z);
}

template <class xtype, class ytype, class ztype,
          std::enable_if_t<get_vec_size<ztype>::size == 3, bool> = true>
A2D_FUNCTION auto VecCross(ADObj<xtype> &x, const ytype &y, ADObj<ztype> &z) {
  return VecCrossExpr<ADObj<xtype>, const ytype, ADObj<ztype>>(x, y, z);
}

template <class xtype, class ytype, class ztype,
          std::enable_if_t<get_vec_size<ztype>::size == 3, bool> = true>
A2D_FUNCTION auto VecCross(A2DObj<xtype> &x, A2DObj<ytype> &y,
                           A2DObj<ztype> &z) {
  return VecCrossExpr<A2DObj<xtype>, A2DObj<ytype>, A2DObj<ztype>>(x, y, z);
}

template <class xtype, class ytype, class ztype,
          std::enable_if_t<get_vec_size<ztype>::size == 3, bool> = true>
A2D_FUNCTION auto VecCross(const xtype &x, A2DObj<ytype> &y, A2DObj<ztype> &z) {
  return VecCrossExpr<const xtype, A2DObj<ytype>, A2DObj<ztype>>(x, y, z);
}

template <class xtype, class ytype, class ztype,
          std::enable_if_t<get_vec_size<ztype>::size == 3, bool> = true>
A2D_FUNCTION auto VecCross(A2DObj<xtype> &x, const ytype &y, A2DObj<ztype> &z) {
  return VecCrossExpr<A2DObj<xtype>, const ytype, A2DObj<ztype>>(x, y, z);
}

template <class xtype, class ytype, class ztype>
class VecCross2DExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<ztype>::type T;

  // Extract the diff type
  static constexpr ADiffType adx = get_diff_type<xtype>::diff_type;
  static constexpr ADiffType ady = get_diff_type<ytype>::diff_type;

  // Check if x is a scalar
  const static bool is_x_scalar =
      is_scalar_type<typename remove_a2dobj<xtype>::type>::value;

  A2D_FUNCTION VecCross2DExpr(xtype x, ytype y, ztype &z) : x(x), y(y), z(z) {}

  A2D_FUNCTION void eval() {
    auto x0 = get_data(x);
    auto y0 = get_data(y);
    auto z0 = get_data(z);
    if constexpr (is_x_scalar) {
      z0[0] = -x0 * y0[1];
      z0[1] = x0 * y0[0];
    } else if constexpr (!is_x_scalar) {
      z0[0] = y0 * x0[1];
      z0[1] = -y0 * x0[0];
    }
  }

  A2D_FUNCTION void bzero() { z.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;
    auto x0 = get_data(x);
    auto y0 = get_data(y);
    auto zp = GetSeed<seed>::get_data(z);

    if constexpr (adx == ADiffType::ACTIVE && ady == ADiffType::ACTIVE) {
      auto xp = GetSeed<seed>::get_data(x);
      auto yp = GetSeed<seed>::get_data(y);
      if constexpr (is_x_scalar) {
        zp[0] = -xp * y0[1] - x0 * yp[1];
        zp[1] = xp * y0[0] + x0 * yp[0];
      } else if constexpr (!is_x_scalar) {
        zp[0] = yp * x0[1] + y0 * xp[1];
        zp[1] = -yp * x0[0] - y0 * xp[0];
      }
    } else if constexpr (adx == ADiffType::ACTIVE) {
      auto xp = GetSeed<seed>::get_data(x);
      if constexpr (is_x_scalar) {
        zp[0] = -xp * y0[1];
        zp[1] = xp * y0[0];
      } else if constexpr (!is_x_scalar) {
        zp[0] = y0 * xp[1];
        zp[1] = -y0 * xp[0];
      }
    } else if constexpr (ady == ADiffType::ACTIVE) {
      auto yp = GetSeed<seed>::get_data(y);
      if constexpr (is_x_scalar) {
        zp[0] = -x0 * yp[1];
        zp[1] = x0 * yp[0];
      } else if constexpr (!is_x_scalar) {
        zp[0] = yp * x0[1];
        zp[1] = -yp * x0[0];
      }
    }
  }
  A2D_FUNCTION void reverse() {
    constexpr ADseed seed = ADseed::b;
    auto x0 = get_data(x);
    auto y0 = get_data(y);
    auto zb = GetSeed<seed>::get_data(z);

    if constexpr (is_x_scalar) {
      if constexpr (adx == ADiffType::ACTIVE) {
        GetSeed<seed>::get_data(x) += (zb[1] * y0[0] - zb[0] * y0[1]);
      }
      if constexpr (ady == ADiffType::ACTIVE) {
        auto yb = GetSeed<seed>::get_data(y);
        yb[0] += x0 * zb[1];
        yb[1] -= x0 * zb[0];
      }
    } else if constexpr (!is_x_scalar) {
      if constexpr (ady == ADiffType::ACTIVE) {
        GetSeed<seed>::get_data(y) += (zb[0] * x0[1] - zb[1] * x0[0]);
      }
      if constexpr (adx == ADiffType::ACTIVE) {
        auto xb = GetSeed<seed>::get_data(x);
        xb[0] -= y0 * zb[1];
        xb[1] += y0 * zb[0];
      }
    }
  }

  A2D_FUNCTION void hzero() { z.hzero(); }

  A2D_FUNCTION void hreverse() {
    constexpr ADseed seed = ADseed::h;
    auto x0 = get_data(x);
    auto y0 = get_data(y);
    auto zh = GetSeed<seed>::get_data(z);

    if constexpr (is_x_scalar) {
      if constexpr (adx == ADiffType::ACTIVE) {
        GetSeed<seed>::get_data(x) += (zh[1] * y0[0] - zh[0] * y0[1]);
      }
      if constexpr (ady == ADiffType::ACTIVE) {
        auto yh = GetSeed<seed>::get_data(y);
        yh[0] += x0 * zh[1];
        yh[1] -= x0 * zh[0];
      }
      if constexpr (adx == ADiffType::ACTIVE && ady == ADiffType::ACTIVE) {
        auto zb = GetSeed<ADseed::b>::get_data(z);
        auto xp = GetSeed<ADseed::p>::get_data(x);
        auto yp = GetSeed<ADseed::p>::get_data(y);
        auto yh = GetSeed<seed>::get_data(y);
        yh[0] += xp * zb[1];
        yh[1] -= xp * zb[0];
        GetSeed<seed>::get_data(x) += (zb[1] * yp[0] - zb[0] * yp[1]);
      }
    } else if constexpr (!is_x_scalar) {
      if constexpr (ady == ADiffType::ACTIVE) {
        GetSeed<seed>::get_data(y) += (zh[0] * x0[1] - zh[1] * x0[0]);
      }
      if constexpr (adx == ADiffType::ACTIVE) {
        auto xh = GetSeed<seed>::get_data(x);
        xh[0] -= y0 * zh[1];
        xh[1] += y0 * zh[0];
      }
      if constexpr (adx == ADiffType::ACTIVE && ady == ADiffType::ACTIVE) {
        auto zb = GetSeed<ADseed::b>::get_data(z);
        auto xp = GetSeed<ADseed::p>::get_data(x);
        auto yp = GetSeed<ADseed::p>::get_data(y);
        auto xh = GetSeed<seed>::get_data(x);
        xh[0] -= yp * zb[1];
        xh[1] += yp * zb[0];
        GetSeed<seed>::get_data(y) += (zb[0] * xp[1] - zb[1] * xp[0]);
      }
    }
  }

  xtype x;
  ytype y;
  ztype &z;
};

template <class xtype, class ytype, class ztype,
          std::enable_if_t<get_vec_size<ztype>::size == 2, bool> = true>
A2D_FUNCTION auto VecCross(ADObj<xtype> &x, ADObj<ytype> &y, ADObj<ztype> &z) {
  return VecCross2DExpr<ADObj<xtype> &, ADObj<ytype> &, ADObj<ztype>>(x, y, z);
}

template <class xtype, class ytype, class ztype,
          std::enable_if_t<get_vec_size<ztype>::size == 2, bool> = true>
A2D_FUNCTION auto VecCross(const xtype x, ADObj<ytype> &y, ADObj<ztype> &z) {
  return VecCross2DExpr<const xtype, ADObj<ytype> &, ADObj<ztype>>(x, y, z);
}

template <class xtype, class ytype, class ztype,
          std::enable_if_t<get_vec_size<ztype>::size == 2, bool> = true>
A2D_FUNCTION auto VecCross(ADObj<xtype> &x, const ytype &y, ADObj<ztype> &z) {
  return VecCross2DExpr<ADObj<xtype> &, const ytype, ADObj<ztype>>(x, y, z);
}

template <class xtype, class ytype, class ztype,
          std::enable_if_t<get_vec_size<ztype>::size == 2, bool> = true>
A2D_FUNCTION auto VecCross(A2DObj<xtype> &x, A2DObj<ytype> &y,
                           A2DObj<ztype> &z) {
  return VecCross2DExpr<A2DObj<xtype> &, A2DObj<ytype> &, A2DObj<ztype>>(x, y,
                                                                         z);
}

template <class xtype, class ytype, class ztype,
          std::enable_if_t<get_vec_size<ztype>::size == 2, bool> = true>
A2D_FUNCTION auto VecCross(const xtype x, A2DObj<ytype> &y, A2DObj<ztype> &z) {
  return VecCross2DExpr<const xtype, A2DObj<ytype> &, A2DObj<ztype>>(x, y, z);
}

template <class xtype, class ytype, class ztype,
          std::enable_if_t<get_vec_size<ztype>::size == 2, bool> = true>
A2D_FUNCTION auto VecCross(A2DObj<xtype> &x, const ytype y, A2DObj<ztype> &z) {
  return VecCross2DExpr<A2DObj<xtype> &, const ytype, A2DObj<ztype>>(x, y, z);
}

namespace Test {

template <typename T>
class VecCross3DTest : public A2DTest<T, Vec<T, 3>, Vec<T, 3>, Vec<T, 3>> {
 public:
  using Input = VarTuple<T, Vec<T, 3>, Vec<T, 3>>;
  using Output = VarTuple<T, Vec<T, 3>>;

  // Assemble a string to describe the test
  std::string name() { return "VecCross"; }

  // Evaluate the matrix-matrix product
  Output eval(const Input &X) {
    Vec<T, 3> x, y, z;
    X.get_values(x, y);
    VecCross(x, y, z);
    return MakeVarTuple<T>(z);
  }

  // Compute the derivative
  void deriv(const Output &seed, const Input &X, Input &g) {
    ADObj<Vec<T, 3>> x, y, z;
    X.get_values(x.value(), y.value());
    auto stack = MakeStack(VecCross(x, y, z));
    seed.get_values(z.bvalue());
    stack.reverse();
    g.set_values(x.bvalue(), y.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output &seed, const Output &hval, const Input &X,
             const Input &p, Input &h) {
    A2DObj<Vec<T, 3>> x, y, z;
    X.get_values(x.value(), y.value());
    p.get_values(x.pvalue(), y.pvalue());
    auto stack = MakeStack(VecCross(x, y, z));
    seed.get_values(z.bvalue());
    hval.get_values(z.hvalue());
    stack.hproduct();
    h.set_values(x.hvalue(), y.hvalue());
  }
};

template <typename T>
class VecCross2DTest : public A2DTest<T, Vec<T, 2>, Vec<T, 2>, T> {
 public:
  using Input = VarTuple<T, Vec<T, 2>, T>;
  using Output = VarTuple<T, Vec<T, 2>>;

  // Assemble a string to describe the test
  std::string name() { return "VecCross"; }

  // Evaluate the matrix-matrix product
  Output eval(const Input &X) {
    Vec<T, 2> x, z;
    T y;
    X.get_values(x, y);
    VecCross(x, y, z);
    return MakeVarTuple<T>(z);
  }

  // Compute the derivative
  void deriv(const Output &seed, const Input &X, Input &g) {
    ADObj<Vec<T, 2>> x, z;
    ADObj<T> y;
    X.get_values(x.value(), y.value());
    auto stack = MakeStack(VecCross(x, y, z));
    seed.get_values(z.bvalue());
    stack.reverse();
    g.set_values(x.bvalue(), y.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output &seed, const Output &hval, const Input &X,
             const Input &p, Input &h) {
    A2DObj<Vec<T, 2>> x, z;
    A2DObj<T> y;
    X.get_values(x.value(), y.value());
    p.get_values(x.pvalue(), y.pvalue());
    auto stack = MakeStack(VecCross(x, y, z));
    seed.get_values(z.bvalue());
    hval.get_values(z.hvalue());
    stack.hproduct();
    h.set_values(x.hvalue(), y.hvalue());
  }
};

inline bool VecCrossTestAll(bool component = false, bool write_output = true) {
  using Tc = A2D_complex_t<double>;

  bool passed = true;
  VecCross3DTest<Tc> test1;
  passed = passed && Run(test1, component, write_output);

  VecCross2DTest<Tc> test2;
  passed = passed && Run(test2, component, write_output);

  return passed;
}

}  // namespace Test

}  // namespace A2D

#endif  // A2D_VEC_CROSS_H
