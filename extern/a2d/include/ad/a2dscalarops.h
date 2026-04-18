#ifndef A2D_SCALAR_OPS_H
#define A2D_SCALAR_OPS_H

#include "../a2ddefs.h"
#include "a2dbinary.h"
#include "a2dtest.h"
#include "a2dunary.h"

namespace A2D {

template <class Expr, class T>
class EvalExpr {
 public:
  A2D_FUNCTION EvalExpr(Expr&& expr, ADObj<T>& out)
      : expr(a2d_forward<Expr>(expr)), out(out) {}

  A2D_FUNCTION void eval() {
    expr.eval();
    out.value() = expr.value();
  }
  A2D_FUNCTION void bzero() {
    out.bzero();
    expr.bzero();
  }
  template <ADorder forder>
  A2D_FUNCTION void forward() {
    static_assert(forder == ADorder::FIRST,
                  "EvalExpr only works for first-order AD");
    expr.forward();
    out.bvalue() = expr.bvalue();
  }
  A2D_FUNCTION void reverse() {
    expr.bvalue() = out.bvalue();
    expr.reverse();
  }

 private:
  Expr expr;
  ADObj<T>& out;
};

template <class Expr, class T>
class EvalExprRef {
 public:
  A2D_FUNCTION EvalExprRef(Expr&& expr, ADObj<T&> out)
      : expr(a2d_forward<Expr>(expr)), out(out) {}

  A2D_FUNCTION void eval() {
    expr.eval();
    out.value() = expr.value();
  }
  A2D_FUNCTION void bzero() {
    out.bzero();
    expr.bzero();
  }
  template <ADorder forder>
  A2D_FUNCTION void forward() {
    static_assert(forder == ADorder::FIRST,
                  "EvalExprRef only works for first-order AD");
    expr.forward();
    out.bvalue() = expr.bvalue();
  }
  A2D_FUNCTION void reverse() {
    expr.bvalue() = out.bvalue();
    expr.reverse();
  }

 private:
  Expr expr;
  ADObj<T&> out;
};

template <class Expr, class T,
          std::enable_if_t<!std::is_reference<T>::value, bool> = true>
A2D_FUNCTION auto Eval(Expr&& expr, ADObj<T>& out) {
  return EvalExpr<Expr, T>(a2d_forward<Expr>(expr), out);
}

template <class Expr, class T,
          std::enable_if_t<!std::is_reference<T>::value, bool> = true>
A2D_FUNCTION auto Eval(Expr&& expr, ADObj<T&> out) {
  return EvalExprRef<Expr, T>(a2d_forward<Expr>(expr), out);
}

template <class Expr, class T>
class EvalExpr2 {
 public:
  A2D_FUNCTION EvalExpr2(Expr&& expr, A2DObj<T>& out)
      : expr(a2d_forward<Expr>(expr)), out(out) {}

  A2D_FUNCTION void eval() {
    expr.eval();
    out.value() = expr.value();
  }
  A2D_FUNCTION void bzero() {
    out.bzero();
    expr.bzero();
  }
  A2D_FUNCTION void reverse() {
    expr.bvalue() += out.bvalue();
    expr.reverse();
  }
  template <ADorder forder>
  A2D_FUNCTION void forward() {
    static_assert(forder == ADorder::SECOND,
                  "EvalExpr2 only works for second-order AD");
    expr.hforward();
    out.pvalue() = expr.pvalue();
  }
  A2D_FUNCTION void hzero() {
    out.hzero();
    expr.hzero();
  }
  A2D_FUNCTION void hreverse() {
    expr.hvalue() += out.hvalue();
    expr.hreverse();
  }

 private:
  Expr expr;
  A2DObj<T>& out;
};

template <class Expr, class T>
class EvalExprRef2 {
 public:
  A2D_FUNCTION EvalExprRef2(Expr&& expr, A2DObj<T&> out)
      : expr(a2d_forward<Expr>(expr)), out(out) {}

  A2D_FUNCTION void eval() {
    expr.eval();
    out.value() = expr.value();
  }
  A2D_FUNCTION void bzero() {
    out.bzero();
    expr.bzero();
  }
  A2D_FUNCTION void reverse() {
    expr.bvalue() += out.bvalue();
    expr.reverse();
  }
  template <ADorder forder>
  A2D_FUNCTION void forward() {
    static_assert(forder == ADorder::SECOND,
                  "EvalExprRef2 only works for second-order AD");
    expr.hforward();
    out.pvalue() = expr.pvalue();
  }
  A2D_FUNCTION void hzero() {
    out.hzero();
    expr.hzero();
  }
  A2D_FUNCTION void hreverse() {
    expr.hvalue() += out.hvalue();
    expr.hreverse();
  }

 private:
  Expr expr;
  A2DObj<T&> out;
};

template <class Expr, class T,
          std::enable_if_t<!std::is_reference<T>::value, bool> = true>
A2D_FUNCTION auto Eval(Expr&& expr, A2DObj<T>& out) {
  return EvalExpr2<Expr, T>(a2d_forward<Expr>(expr), out);
}

template <class Expr, class T,
          std::enable_if_t<!std::is_reference<T>::value, bool> = true>
A2D_FUNCTION auto Eval(Expr&& expr, A2DObj<T&> out) {
  return EvalExprRef2<Expr, T>(a2d_forward<Expr>(expr), out);
}

namespace Test {
template <typename T>
class ScalarTest : public A2DTest<T, T, T, T> {
 public:
  using Input = VarTuple<T, T, T>;
  using Output = VarTuple<T, T>;

  // Assemble a string to describe the test
  std::string name() { return std::string("ScalarOperations"); }

  // Evaluate the matrix-matrix product
  Output eval(const Input& x) {
    T a, b, f;
    x.get_values(a, b);

    f = log(a * a * sqrt(exp(a * sin(a) + 3.0 * a)) + 2.0 * a * a * a * a) +
        max2(a, min2(a * b, b * b)) - 4.0 * a / b + pow(5.0 / (b * b), 2.0) +
        acos(a * 0.1);

    return MakeVarTuple<T>(f);
  }

  // Compute the derivative
  void deriv(const Output& seed, const Input& x, Input& g) {
    T a0, ab, b0, bb;
    ADObj<T&> a(a0, ab), b(b0, bb);
    ADObj<T> f;
    x.get_values(a.value(), b.value());

    auto stack = MakeStack(Eval(
        log(a * a * sqrt(exp(a * sin(a) + 3.0 * a)) + 2.0 * a * a * a * a) +
            max2(a, min2(a * b, b * b)) - 4.0 * a / b +
            pow(5.0 / (b * b), 2.0) + acos(a * 0.1),
        f));

    seed.get_values(f.bvalue());
    stack.reverse();
    g.set_values(a.bvalue(), b.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output& seed, const Output& hval, const Input& x,
             const Input& p, Input& h) {
    T a0, ab, ap, ah, b0, bb, bp, bh;
    A2DObj<T&> a(a0, ab, ap, ah), b(b0, bb, bp, bh);
    A2DObj<T> f;
    x.get_values(a.value(), b.value());
    p.get_values(a.pvalue(), b.pvalue());

    auto stack = MakeStack(Eval(
        log(a * a * sqrt(exp(a * sin(a) + 3.0 * a)) + 2.0 * a * a * a * a) +
            max2(a, min2(a * b, b * b)) - 4.0 * a / b +
            pow(5.0 / (b * b), 2.0) + acos(a * 0.1),
        f));

    seed.get_values(f.bvalue());
    hval.get_values(f.hvalue());
    stack.hproduct();
    h.set_values(a.hvalue(), b.hvalue());
  }
};

inline bool ScalarTestAll(bool component, bool write_output) {
  using Tc = A2D_complex_t<double>;

  bool passed = true;
  ScalarTest<Tc> test1;
  test1.set_step_size(1e-8);  // inverse trigonometric functions may suffer from
                              // subtraction cancellation even for complex step
                              // with certain underlying implementation
  passed = passed && Run(test1, component, write_output);

  return passed;
}

}  // namespace Test

}  // namespace A2D

#endif  // A2D_SCALAR_OPS_H
