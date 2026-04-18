#ifndef A2D_UNARY_OPS_H
#define A2D_UNARY_OPS_H

#include "../a2ddefs.h"

namespace A2D {

/*
  Definitions for memory-less forward and reverse-mode first-order AD

  OBJNAME: Expression template name
  OPERNAME: Name of the operator
  FUNCBODY: Body of the function evaluation
  DERIVBODY: Body of the derivative evaluation
*/
#define A2D_1ST_UNARY_BASIC(OBJNAME, OPERNAME, FUNCBODY, DERIVBODY)         \
                                                                            \
  template <class A, class Ta, class T, bool CA>                            \
  class OBJNAME : public ADExpr<OBJNAME<A, Ta, T, CA>, T> {                 \
   public:                                                                  \
    using expr_t = typename std::conditional<CA, const ADExpr<A, Ta>,       \
                                             ADExpr<A, Ta>>::type;          \
    using A_t = typename std::conditional<CA, A, A &>::type;                \
    A2D_FUNCTION OBJNAME(expr_t &a0) : a(a0.self()), val(0.0), bval(0.0) {} \
    A2D_FUNCTION void eval() {                                              \
      a.eval();                                                             \
      val = (FUNCBODY);                                                     \
    }                                                                       \
    A2D_FUNCTION void forward() {                                           \
      a.forward();                                                          \
      bval = (DERIVBODY)*a.bvalue();                                        \
    }                                                                       \
    A2D_FUNCTION void reverse() {                                           \
      a.bvalue() += (DERIVBODY)*bval;                                       \
      a.reverse();                                                          \
    }                                                                       \
    A2D_FUNCTION void bzero() {                                             \
      bval = T(0.0);                                                        \
      a.bzero();                                                            \
    }                                                                       \
    A2D_FUNCTION T &value() { return val; }                                 \
    A2D_FUNCTION const T &value() const { return val; }                     \
    A2D_FUNCTION T &bvalue() { return bval; }                               \
    A2D_FUNCTION const T &bvalue() const { return bval; }                   \
                                                                            \
   private:                                                                 \
    A_t a;                                                                  \
    T val, bval;                                                            \
  };                                                                        \
  template <class A, class Ta>                                              \
  A2D_FUNCTION auto OPERNAME(const ADExpr<A, Ta> &a) {                      \
    using T = typename remove_const_and_refs<Ta>::type;                     \
    return OBJNAME<A, Ta, T, true>(a);                                      \
  }                                                                         \
  template <class A, class Ta>                                              \
  A2D_FUNCTION auto OPERNAME(ADExpr<A, Ta> &a) {                            \
    using T = typename remove_const_and_refs<Ta>::type;                     \
    return OBJNAME<A, Ta, T, false>(a);                                     \
  }

// A2D_1ST_UNARY_BASIC(OBJNAME, OPERNAME, FUNCBODY, DERIVBODY)
A2D_1ST_UNARY_BASIC(ExpExpr, exp, exp(a.value()), val)
A2D_1ST_UNARY_BASIC(UnaryPos, operator+, a.value(), T(1.0))
A2D_1ST_UNARY_BASIC(UnaryNeg, operator-, -a.value(), -T(1.0))

/*
  Definitions for second-derivative implementations with zero second derivative

  OBJNAME: Expression template name
  OPERNAME: Name of the operator
  FUNCBODY: Body of the function evaluation
  DERIVBODY: Body of the derivative evaluation
*/
#define A2D_2ND_UNARY_BASIC(OBJNAME, OPERNAME, FUNCBODY, DERIVBODY)            \
                                                                               \
  template <class A, class Ta, class T, bool CA>                               \
  class OBJNAME : public A2DExpr<OBJNAME<A, Ta, T, CA>, T> {                   \
   public:                                                                     \
    using expr_t = typename std::conditional<CA, const A2DExpr<A, Ta>,         \
                                             A2DExpr<A, Ta>>::type;            \
    using A_t = typename std::conditional<CA, A, A &>::type;                   \
    A2D_FUNCTION OBJNAME(expr_t &a0)                                           \
        : a(a0.self()), val(0.0), bval(0.0), pval(0.0), hval(0.0), tmp(0.0) {} \
    A2D_FUNCTION void eval() {                                                 \
      a.eval();                                                                \
      val = (FUNCBODY);                                                        \
    }                                                                          \
    A2D_FUNCTION void reverse() {                                              \
      a.bvalue() += (DERIVBODY)*bval;                                          \
      a.reverse();                                                             \
    }                                                                          \
    A2D_FUNCTION void hforward() {                                             \
      a.hforward();                                                            \
      pval = (DERIVBODY)*a.pvalue();                                           \
    }                                                                          \
    A2D_FUNCTION void hreverse() {                                             \
      a.hvalue() += (DERIVBODY)*hval;                                          \
      a.hreverse();                                                            \
    }                                                                          \
    A2D_FUNCTION void bzero() {                                                \
      bval = T(0.0);                                                           \
      a.bzero();                                                               \
    }                                                                          \
    A2D_FUNCTION void hzero() {                                                \
      hval = T(0.0);                                                           \
      a.hzero();                                                               \
    }                                                                          \
    A2D_FUNCTION T &value() { return val; }                                    \
    A2D_FUNCTION const T &value() const { return val; }                        \
    A2D_FUNCTION T &bvalue() { return bval; }                                  \
    A2D_FUNCTION const T &bvalue() const { return bval; }                      \
    A2D_FUNCTION T &pvalue() { return pval; }                                  \
    A2D_FUNCTION const T &pvalue() const { return pval; }                      \
    A2D_FUNCTION T &hvalue() { return hval; }                                  \
    A2D_FUNCTION const T &hvalue() const { return hval; }                      \
                                                                               \
   private:                                                                    \
    A_t a;                                                                     \
    T val, bval, pval, hval, tmp;                                              \
  };                                                                           \
  template <class A, class Ta>                                                 \
  A2D_FUNCTION auto OPERNAME(const A2DExpr<A, Ta> &a) {                        \
    using T = typename remove_const_and_refs<Ta>::type;                        \
    return OBJNAME<A, Ta, T, true>(a);                                         \
  }                                                                            \
  template <class A, class Ta>                                                 \
  A2D_FUNCTION auto OPERNAME(A2DExpr<A, Ta> &a) {                              \
    using T = typename remove_const_and_refs<Ta>::type;                        \
    return OBJNAME<A, Ta, T, false>(a);                                        \
  }

A2D_2ND_UNARY_BASIC(UnaryPos2, operator+, a.value(), T(1.0))
A2D_2ND_UNARY_BASIC(UnaryNeg2, operator-, -a.value(), -T(1.0))

/*
  Definitions for forward and reverse-mode first-order AD with temporary

  OBJNAME: Expression template name
  OPERNAME: Name of the operator
  FUNCBODY: Body of the function evaluation
  TEMPBODY: Body of the temporary variable calculation (computed in eval)
  DERIVBODY: Body of the derivative evaluation
*/
#define A2D_1ST_UNARY(OBJNAME, OPERNAME, FUNCBODY, TEMPBODY, DERIVBODY) \
                                                                        \
  template <class A, class Ta, class T, bool CA>                        \
  class OBJNAME : public ADExpr<OBJNAME<A, Ta, T, CA>, T> {             \
   public:                                                              \
    using expr_t = typename std::conditional<CA, const ADExpr<A, Ta>,   \
                                             ADExpr<A, Ta>>::type;      \
    using A_t = typename std::conditional<CA, A, A &>::type;            \
    A2D_FUNCTION OBJNAME(expr_t &a0)                                    \
        : a(a0.self()), val(0.0), bval(0.0), tmp(0.0) {}                \
    A2D_FUNCTION void eval() {                                          \
      a.eval();                                                         \
      val = (FUNCBODY);                                                 \
      tmp = (TEMPBODY);                                                 \
    }                                                                   \
    A2D_FUNCTION void forward() {                                       \
      a.forward();                                                      \
      bval = (DERIVBODY)*a.bvalue();                                    \
    }                                                                   \
    A2D_FUNCTION void reverse() {                                       \
      a.bvalue() += (DERIVBODY)*bval;                                   \
      a.reverse();                                                      \
    }                                                                   \
    A2D_FUNCTION void bzero() {                                         \
      bval = T(0.0);                                                    \
      a.bzero();                                                        \
    }                                                                   \
    A2D_FUNCTION T &value() { return val; }                             \
    A2D_FUNCTION const T &value() const { return val; }                 \
    A2D_FUNCTION T &bvalue() { return bval; }                           \
    A2D_FUNCTION const T &bvalue() const { return bval; }               \
                                                                        \
   private:                                                             \
    A_t a;                                                              \
    T val, bval, tmp;                                                   \
  };                                                                    \
  template <class A, class Ta>                                          \
  A2D_FUNCTION auto OPERNAME(const ADExpr<A, Ta> &a) {                  \
    using T = typename remove_const_and_refs<Ta>::type;                 \
    return OBJNAME<A, Ta, T, true>(a);                                  \
  }                                                                     \
  template <class A, class Ta>                                          \
  A2D_FUNCTION auto OPERNAME(ADExpr<A, Ta> &a) {                        \
    using T = typename remove_const_and_refs<Ta>::type;                 \
    return OBJNAME<A, Ta, T, false>(a);                                 \
  }

// A2D_1ST_UNARY(OBJNAME, OPERNAME, FUNCBODY, TEMPBODY, DERIVBODY)
A2D_1ST_UNARY(FabsExpr, fabs, (a.value() < 0.0 ? -a.value() : a.value()),
              (a.value() < 0.0 ? -1.0 : 1.0), tmp)
A2D_1ST_UNARY(SinExpr, sin, sin(a.value()), cos(a.value()), tmp)
A2D_1ST_UNARY(CosExpr, cos, cos(a.value()), sin(a.value()), -tmp)
A2D_1ST_UNARY(SqrtExpr, sqrt, sqrt(a.value()), 1.0 / val, 0.5 * tmp)
A2D_1ST_UNARY(LogExpr, log, log(a.value()), 1.0 / a.value(), tmp)
A2D_1ST_UNARY(ACosExpr, acos, acos(a.value()),
              -1.0 / sqrt(1.0 - a.value() * a.value()), tmp)
A2D_1ST_UNARY(ASinExpr, asin, asin(a.value()),
              1.0 / sqrt(1.0 - a.value() * a.value()), tmp)
A2D_1ST_UNARY(ATanExpr, atan, atan(a.value()),
              1.0 / (1.0 + a.value() * a.value()), tmp)

/*
  Definitions for forward and reverse-mode first-order AD with temporary

  OBJNAME: Expression template name
  OPERNAME: Name of the operator
  FUNCBODY: Body of the function evaluation
  TEMPBODY: Body of the temporary variable calculation (computed in eval)
  DERIVBODY: Body of the derivative evaluation
  DERIV2BODY: Body of the second derivative evaluation
*/
#define A2D_2ND_UNARY(OBJNAME, OPERNAME, FUNCBODY, TEMPBODY, DERIVBODY,        \
                      DERIV2BODY)                                              \
                                                                               \
  template <class A, class Ta, class T, bool CA>                               \
  class OBJNAME : public A2DExpr<OBJNAME<A, Ta, T, CA>, T> {                   \
   public:                                                                     \
    using expr_t = typename std::conditional<CA, const A2DExpr<A, Ta>,         \
                                             A2DExpr<A, Ta>>::type;            \
    using A_t = typename std::conditional<CA, A, A &>::type;                   \
    A2D_FUNCTION OBJNAME(expr_t &a0)                                           \
        : a(a0.self()), val(0.0), bval(0.0), pval(0.0), hval(0.0), tmp(0.0) {} \
    A2D_FUNCTION void eval() {                                                 \
      a.eval();                                                                \
      val = (FUNCBODY);                                                        \
      tmp = (TEMPBODY);                                                        \
    }                                                                          \
    A2D_FUNCTION void reverse() {                                              \
      a.bvalue() += (DERIVBODY)*bval;                                          \
      a.reverse();                                                             \
    }                                                                          \
    A2D_FUNCTION void hforward() {                                             \
      a.hforward();                                                            \
      pval = (DERIVBODY)*a.pvalue();                                           \
    }                                                                          \
    A2D_FUNCTION void hreverse() {                                             \
      a.hvalue() += (DERIVBODY)*hval + (DERIV2BODY)*bval * a.pvalue();         \
      a.hreverse();                                                            \
    }                                                                          \
    A2D_FUNCTION void bzero() {                                                \
      bval = T(0.0);                                                           \
      a.bzero();                                                               \
    }                                                                          \
    A2D_FUNCTION void hzero() {                                                \
      hval = T(0.0);                                                           \
      a.hzero();                                                               \
    }                                                                          \
    A2D_FUNCTION T &value() { return val; }                                    \
    A2D_FUNCTION const T &value() const { return val; }                        \
    A2D_FUNCTION T &bvalue() { return bval; }                                  \
    A2D_FUNCTION const T &bvalue() const { return bval; }                      \
    A2D_FUNCTION T &pvalue() { return pval; }                                  \
    A2D_FUNCTION const T &pvalue() const { return pval; }                      \
    A2D_FUNCTION T &hvalue() { return hval; }                                  \
    A2D_FUNCTION const T &hvalue() const { return hval; }                      \
                                                                               \
   private:                                                                    \
    A_t a;                                                                     \
    T val, bval, pval, hval, tmp;                                              \
  };                                                                           \
  template <class A, class Ta>                                                 \
  A2D_FUNCTION auto OPERNAME(const A2DExpr<A, Ta> &a) {                        \
    using T = typename remove_const_and_refs<Ta>::type;                        \
    return OBJNAME<A, Ta, T, true>(a);                                         \
  }                                                                            \
  template <class A, class Ta>                                                 \
  A2D_FUNCTION auto OPERNAME(A2DExpr<A, Ta> &a) {                              \
    using T = typename remove_const_and_refs<Ta>::type;                        \
    return OBJNAME<A, Ta, T, false>(a);                                        \
  }

// A2D_2ST_UNARY(OBJNAME, OPERNAME, FUNCBODY, TEMPBODY, DERIVBODY, DERIV2BODY)
A2D_2ND_UNARY(FabsExpr2, fabs, (a.value() < 0.0 ? -a.value() : a.value()),
              (a.value() < 0.0 ? -1.0 : 1.0), tmp, 0.0)
A2D_2ND_UNARY(ExpExpr2, exp, exp(a.value()), val, tmp, tmp)
A2D_2ND_UNARY(SinExpr2, sin, sin(a.value()), cos(a.value()), tmp, -val)
A2D_2ND_UNARY(CosExpr2, cos, cos(a.value()), sin(a.value()), -tmp, -val)
A2D_2ND_UNARY(SqrtExpr2, sqrt, sqrt(a.value()), 1.0 / val, 0.5 * tmp,
              -0.25 * tmp * tmp * tmp)
A2D_2ND_UNARY(LogExpr2, log, log(a.value()), 1.0 / a.value(), tmp, -tmp *tmp)
A2D_2ND_UNARY(ACosExpr2, acos, acos(a.value()),
              -1.0 / sqrt(1.0 - a.value() * a.value()), tmp,
              -a.value() / pow(1.0 - a.value() * a.value(), 1.5))
A2D_2ND_UNARY(ASinExpr2, asin, asin(a.value()),
              1.0 / sqrt(1.0 - a.value() * a.value()), tmp,
              a.value() / pow(1.0 - a.value() * a.value(), 1.5))
A2D_2ND_UNARY(ATanExpr2, atan, atan(a.value()),
              1.0 / (1.0 + a.value() * a.value()), tmp,
              -2.0 * a.value() * tmp * tmp)

}  // namespace A2D

#endif  // A2D_UNARY_OPS_H
