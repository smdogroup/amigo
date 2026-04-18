#ifndef A2D_QUATERNION_H
#define A2D_QUATERNION_H

#include <iostream>

#include "../a2ddefs.h"
#include "a2dgemm.h"
#include "a2dmat.h"
#include "a2dvec.h"

namespace A2D {

template <typename T>
void QuaternionMatrixCore(const T q[], T C[]) {
  C[0] = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]);
  C[1] = 2.0 * (q[1] * q[2] + q[3] * q[0]);
  C[2] = 2.0 * (q[1] * q[3] - q[2] * q[0]);

  C[3] = 2.0 * (q[2] * q[1] - q[3] * q[0]);
  C[4] = 1.0 - 2.0 * (q[1] * q[1] + q[3] * q[3]);
  C[5] = 2.0 * (q[2] * q[3] + q[1] * q[0]);

  C[6] = 2.0 * (q[3] * q[1] + q[2] * q[0]);
  C[7] = 2.0 * (q[3] * q[2] - q[1] * q[0]);
  C[8] = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]);
}

template <typename T>
void QuaternionMatrixForwardCore(const T q[], const T qd[], T Cd[]) {
  Cd[0] = -4.0 * (q[2] * qd[2] + q[3] * qd[3]);
  Cd[1] = 2.0 * (q[1] * qd[2] + q[3] * qd[0] + qd[1] * q[2] + qd[3] * q[0]);
  Cd[2] = 2.0 * (q[1] * qd[3] - q[2] * qd[0] + qd[1] * q[3] - qd[2] * q[0]);

  Cd[3] = 2.0 * (q[2] * qd[1] - q[3] * qd[0] + qd[2] * q[1] - qd[3] * q[0]);
  Cd[4] = -4.0 * (q[1] * qd[1] + q[3] * qd[3]);
  Cd[5] = 2.0 * (q[2] * qd[3] + q[1] * qd[0] + qd[2] * q[3] + qd[1] * q[0]);

  Cd[6] = 2.0 * (q[3] * qd[1] + q[2] * qd[0] + qd[3] * q[1] + qd[2] * q[0]);
  Cd[7] = 2.0 * (q[3] * qd[2] - q[1] * qd[0] + qd[3] * q[2] - qd[1] * q[0]);
  Cd[8] = -4.0 * (q[1] * qd[1] + q[2] * qd[2]);
}

template <typename T>
void QuaternionMatrixReverseCore(const T q[], const T dC[], T r[]) {
  r[0] += 2.0 * (q[3] * (dC[1] - dC[3]) + q[2] * (dC[6] - dC[2]) +
                 q[1] * (dC[5] - dC[7]));
  r[1] += 2.0 * (q[0] * (dC[5] - dC[7]) - 2.0 * q[1] * (dC[4] + dC[8]) +
                 q[2] * (dC[1] + dC[3]) + q[3] * (dC[2] + dC[6]));
  r[2] += 2.0 * (q[0] * (dC[6] - dC[2]) + q[1] * (dC[1] + dC[3]) -
                 2.0 * q[2] * (dC[0] + dC[8]) + q[3] * (dC[7] + dC[5]));
  r[3] += 2.0 * (q[0] * (dC[1] - dC[3]) + q[1] * (dC[2] + dC[6]) +
                 q[2] * (dC[5] + dC[7]) - 2.0 * q[3] * (dC[0] + dC[4]));
}

template <typename T>
void QuaternionMatrix(const Vec<T, 4>& q, Mat<T, 3, 3>& C) {
  QuaternionMatrixCore<T>(get_data(q), get_data(C));
}

template <class qtype, class Ctype>
class QuaternionMatrixExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<Ctype>::type T;

  // Extract the underlying sizes of the matrix
  static constexpr int M = get_matrix_rows<Ctype>::size;
  static constexpr int N = get_matrix_columns<Ctype>::size;

  // Extract the dimensions of the underlying vectors
  static constexpr int L = get_vec_size<qtype>::size;

  static_assert(M == N && N == 3, "Rotation matrix dimension must be 3");
  static_assert(L == 4, "Quaternion dimension must be 4");

  A2D_FUNCTION QuaternionMatrixExpr(qtype& q, Ctype& C) : q(q), C(C) {}

  A2D_FUNCTION void eval() {
    QuaternionMatrixCore<T>(get_data(q), get_data(C));
  }

  A2D_FUNCTION void bzero() { C.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;
    QuaternionMatrixForwardCore<T>(get_data(q), GetSeed<seed>::get_data(q),
                                   GetSeed<seed>::get_data(C));
  }

  A2D_FUNCTION void reverse() {
    constexpr ADseed seed = ADseed::b;
    QuaternionMatrixReverseCore<T>(get_data(q), GetSeed<seed>::get_data(C),
                                   GetSeed<seed>::get_data(q));
  }

  A2D_FUNCTION void hzero() { C.hzero(); }

  A2D_FUNCTION void hreverse() {
    QuaternionMatrixReverseCore<T>(get_data(q), GetSeed<ADseed::h>::get_data(C),
                                   GetSeed<ADseed::h>::get_data(q));
    QuaternionMatrixReverseCore<T>(GetSeed<ADseed::p>::get_data(q),
                                   GetSeed<ADseed::b>::get_data(C),
                                   GetSeed<ADseed::h>::get_data(q));
  }

 private:
  qtype& q;
  Ctype& C;
};

template <class qtype, class Ctype>
A2D_FUNCTION auto QuaternionMatrix(ADObj<qtype>& q, ADObj<Ctype>& C) {
  return QuaternionMatrixExpr<ADObj<qtype>, ADObj<Ctype>>(q, C);
}

template <class qtype, class Ctype>
A2D_FUNCTION auto QuaternionMatrix(A2DObj<qtype>& q, A2DObj<Ctype>& C) {
  return QuaternionMatrixExpr<A2DObj<qtype>, A2DObj<Ctype>>(q, C);
}

template <typename T>
void QuaternionAngularVelocityCore(const T q[], const T qdot[], T omega[]) {
  omega[0] =
      2.0 * (q[0] * qdot[1] - qdot[0] * q[1] + q[3] * qdot[2] - q[2] * qdot[3]);
  omega[1] =
      2.0 * (q[0] * qdot[2] - qdot[0] * q[2] + q[1] * qdot[3] - q[3] * qdot[1]);
  omega[2] =
      2.0 * (q[0] * qdot[3] - qdot[0] * q[3] + q[2] * qdot[1] - q[1] * qdot[2]);
}

template <typename T>
void QuaternionAngularVelocityForwardCore(const T q[], const T qdot[],
                                          const T qd[], const T qdotd[],
                                          T omega[]) {
  omega[0] = 2.0 * (qd[0] * qdot[1] - qdotd[0] * q[1] + qd[3] * qdot[2] -
                    qd[2] * qdot[3] + q[0] * qdotd[1] - qdot[0] * qd[1] +
                    q[3] * qdotd[2] - q[2] * qdotd[3]);
  omega[1] = 2.0 * (qd[0] * qdot[2] - qdotd[0] * q[2] + qd[1] * qdot[3] -
                    qd[3] * qdot[1] + q[0] * qdotd[2] - qdot[0] * qd[2] +
                    q[1] * qdotd[3] - q[3] * qdotd[1]);
  omega[2] = 2.0 * (qd[0] * qdot[3] - qdotd[0] * q[3] + qd[2] * qdot[1] -
                    qd[1] * qdot[2] + q[0] * qdotd[3] - qdot[0] * qd[3] +
                    q[2] * qdotd[1] - q[1] * qdotd[2]);
}

template <typename T>
void QuaternionAngularVelocityReverseCore(const T q[], const T qdot[],
                                          const T omegab[], T qb[], T qdotb[]) {
  qb[0] +=
      2.0 * (qdot[1] * omegab[0] + qdot[2] * omegab[1] + qdot[3] * omegab[2]);
  qb[1] +=
      2.0 * (-qdot[0] * omegab[0] + qdot[3] * omegab[1] - qdot[2] * omegab[2]);
  qb[2] +=
      2.0 * (-qdot[3] * omegab[0] - qdot[0] * omegab[1] + qdot[1] * omegab[2]);
  qb[3] +=
      2.0 * (qdot[2] * omegab[0] - qdot[1] * omegab[1] - qdot[0] * omegab[2]);

  qdotb[0] -= 2.0 * (q[1] * omegab[0] + q[2] * omegab[1] + q[3] * omegab[2]);
  qdotb[1] += 2.0 * (q[0] * omegab[0] - q[3] * omegab[1] + q[2] * omegab[2]);
  qdotb[2] += 2.0 * (q[3] * omegab[0] + q[0] * omegab[1] - q[1] * omegab[2]);
  qdotb[3] += 2.0 * (-q[2] * omegab[0] + q[1] * omegab[1] + q[0] * omegab[2]);
}

template <typename T>
void QuaternionAngularVelocity(const Vec<T, 4>& q, const Vec<T, 4>& qdot,
                               Vec<T, 3>& omega) {
  QuaternionAngularVelocityCore<T>(get_data(q), get_data(qdot),
                                   get_data(omega));
}

template <class qtype, class wtype>
class QuaternionAngularVelocityExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<wtype>::type T;

  // Extract the underlying sizes of the matrix
  static constexpr int M = get_vec_size<wtype>::size;

  // Extract the dimensions of the underlying vectors
  static constexpr int L = get_vec_size<qtype>::size;

  static_assert(M == 3, "Rotation matrix dimension must be 3");
  static_assert(L == 4, "Quaternion dimension must be 4");

  A2D_FUNCTION QuaternionAngularVelocityExpr(qtype& q, qtype& qdot,
                                             wtype& omega)
      : q(q), qdot(qdot), omega(omega) {}

  A2D_FUNCTION void eval() {
    QuaternionAngularVelocityCore<T>(get_data(q), get_data(qdot),
                                     get_data(omega));
  }

  A2D_FUNCTION void bzero() { omega.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;
    QuaternionAngularVelocityForwardCore<T>(
        get_data(q), get_data(qdot), GetSeed<seed>::get_data(q),
        GetSeed<seed>::get_data(qdot), GetSeed<seed>::get_data(omega));
  }

  A2D_FUNCTION void reverse() {
    constexpr ADseed seed = ADseed::b;
    QuaternionAngularVelocityReverseCore<T>(
        get_data(q), get_data(qdot), GetSeed<seed>::get_data(omega),
        GetSeed<seed>::get_data(q), GetSeed<seed>::get_data(qdot));
  }

  A2D_FUNCTION void hzero() { omega.hzero(); }

  A2D_FUNCTION void hreverse() {
    QuaternionAngularVelocityReverseCore<T>(
        get_data(q), get_data(qdot), GetSeed<ADseed::h>::get_data(omega),
        GetSeed<ADseed::h>::get_data(q), GetSeed<ADseed::h>::get_data(qdot));

    QuaternionAngularVelocityReverseCore<T>(
        GetSeed<ADseed::p>::get_data(q), GetSeed<ADseed::p>::get_data(qdot),
        GetSeed<ADseed::b>::get_data(omega), GetSeed<ADseed::h>::get_data(q),
        GetSeed<ADseed::h>::get_data(qdot));
  }

 private:
  qtype& q;
  qtype& qdot;
  wtype& omega;
};

template <class qtype, class wtype>
A2D_FUNCTION auto QuaternionAngularVelocity(ADObj<qtype>& q, ADObj<qtype>& qdot,
                                            ADObj<wtype>& omega) {
  return QuaternionAngularVelocityExpr<ADObj<qtype>, ADObj<wtype>>(q, qdot,
                                                                   omega);
}

template <class qtype, class wtype>
A2D_FUNCTION auto QuaternionAngularVelocity(A2DObj<qtype>& q,
                                            A2DObj<qtype>& qdot,
                                            A2DObj<wtype>& omega) {
  return QuaternionAngularVelocityExpr<A2DObj<qtype>, A2DObj<wtype>>(q, qdot,
                                                                     omega);
}

namespace Test {

template <typename T>
class QuaternionMatrixTest : public A2DTest<T, Mat<T, 3, 3>, Vec<T, 4>> {
 public:
  using Input = VarTuple<T, Vec<T, 4>>;
  using Output = VarTuple<T, Mat<T, 3, 3>>;

  // Assemble a string to describe the test
  std::string name() { return "QuaternionMatrix"; }

  // Evaluate the matrix-matrix product
  Output eval(const Input& X) {
    Vec<T, 4> q;
    Mat<T, 3, 3> C;
    X.get_values(q);
    QuaternionMatrix(q, C);
    return MakeVarTuple<T>(C);
  }

  // Compute the derivative
  void deriv(const Output& seed, const Input& X, Input& g) {
    ADObj<Vec<T, 4>> q;
    ADObj<Mat<T, 3, 3>> C;
    X.get_values(q.value());
    auto stack = MakeStack(QuaternionMatrix(q, C));
    seed.get_values(C.bvalue());
    stack.reverse();
    g.set_values(q.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output& seed, const Output& hval, const Input& X,
             const Input& p, Input& h) {
    A2DObj<Vec<T, 4>> q;
    A2DObj<Mat<T, 3, 3>> C;
    X.get_values(q.value());
    p.get_values(q.pvalue());
    auto stack = MakeStack(QuaternionMatrix(q, C));
    seed.get_values(C.bvalue());
    hval.get_values(C.hvalue());
    stack.hproduct();
    h.set_values(q.hvalue());
  }
};

inline bool QuaternionMatrixTestAll(bool component = false,
                                    bool write_output = true) {
  using Tc = A2D_complex_t<double>;

  QuaternionMatrixTest<Tc> test1;
  bool passed = Run(test1, component, write_output);

  return passed;
}

template <typename T>
class QuaternionAngularVelocityTest
    : public A2DTest<T, Vec<T, 3>, Vec<T, 4>, Vec<T, 4>> {
 public:
  using Input = VarTuple<T, Vec<T, 4>, Vec<T, 4>>;
  using Output = VarTuple<T, Vec<T, 3>>;

  // Assemble a string to describe the test
  std::string name() { return "QuaternionMatrix"; }

  // Evaluate the matrix-matrix product
  Output eval(const Input& X) {
    Vec<T, 4> q, qdot;
    Vec<T, 3> omega;
    X.get_values(q, qdot);
    QuaternionAngularVelocity(q, qdot, omega);
    return MakeVarTuple<T>(omega);
  }

  // Compute the derivative
  void deriv(const Output& seed, const Input& X, Input& g) {
    ADObj<Vec<T, 4>> q, qdot;
    ADObj<Vec<T, 3>> omega;
    X.get_values(q.value(), qdot.value());
    auto stack = MakeStack(QuaternionAngularVelocity(q, qdot, omega));
    seed.get_values(omega.bvalue());
    stack.reverse();
    g.set_values(q.bvalue(), qdot.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output& seed, const Output& hval, const Input& X,
             const Input& p, Input& h) {
    A2DObj<Vec<T, 4>> q, qdot;
    A2DObj<Vec<T, 3>> omega;
    X.get_values(q.value(), qdot.value());
    p.get_values(q.pvalue(), qdot.pvalue());
    auto stack = MakeStack(QuaternionAngularVelocity(q, qdot, omega));
    seed.get_values(omega.bvalue());
    hval.get_values(omega.hvalue());
    stack.hproduct();
    h.set_values(q.hvalue(), qdot.hvalue());
  }
};

template <typename T>
bool TestQuaternions(QuaternionAngularVelocityTest<T> test,
                     bool write_output = false, const double dh = 1e-30,
                     const double rtol = 1e-14) {
  using Tc = A2D_complex_t<double>;

  Vec<Tc, 4> q, qdot;
  Mat<Tc, 3, 3> C, Cdot, A;
  Vec<Tc, 3> omega;
  Mat<Tc, 3, 3> OmegaX, negOmegaX;

  Tc qnrm = 0.0;
  for (int i = 0; i < 4; i++) {
    q(i) = -1.0 + 2.0 * (static_cast<double>(rand()) / RAND_MAX);
    qnrm += q(i) * q(i);
  }
  qnrm = sqrt(qnrm);

  for (int i = 0; i < 4; i++) {
    q(i) = q(i) / qnrm;
  }

  // Compute a qdot such that q^{T} * qdot = 0
  for (int i = 1; i < 4; i++) {
    qdot(i) = -1.0 + 2.0 * (static_cast<double>(rand()) / RAND_MAX);
    qdot(0) -= q(i) * qdot(i) / q(0);
  }

  for (int i = 0; i < 4; i++) {
    q(i) = Tc(RealPart(q(i)), dh * RealPart(qdot(i)));
  }

  // Compute the quaternion matrix and the angular rate
  QuaternionMatrix(q, C);
  QuaternionAngularVelocity(q, qdot, omega);

  // Set Cdot
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Cdot(i, j) = ImagPart(C(i, j)) / dh;
    }
  }

  // Set the output matrix
  OmegaX(0, 1) = -omega(2);
  OmegaX(0, 2) = omega(1);

  OmegaX(1, 0) = omega(2);
  OmegaX(1, 2) = -omega(0);

  OmegaX(2, 0) = -omega(1);
  OmegaX(2, 1) = omega(0);

  double err = 0.0;

  // Make sure C^{T} * C = I
  MatMatMult<MatOp::TRANSPOSE, MatOp::NORMAL>(C, C, A);

  // Compute the error for C^{T} * C - I
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j) {
        err += std::fabs(RealPart(A(i, j)) - 1.0);
      } else {
        err += std::fabs(RealPart(A(i, j)));
      }

      if (write_output) {
        if (i == j) {
          test.write_result("C^{T} * C", std::cout, A(i, j), 1.0);
        } else {
          test.write_result("C^{T} * C", std::cout, A(i, j), 0.0);
        }
      }
    }
  }

  // Omega^{x} = - dot{C} * C^{T}
  MatMatMult<MatOp::NORMAL, MatOp::TRANSPOSE>(Cdot, C, negOmegaX);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Tc ans = OmegaX(i, j) + negOmegaX(i, j);
      err += std::fabs(RealPart(ans));

      if (write_output) {
        test.write_result("omega^{x} + dot{C} * C^{T}", std::cout, ans, 0.0);
      }
    }
  }

  bool passed = (std::fabs(err) < rtol);

  return passed;
}

inline bool QuaternionAngularVelocityTestAll(bool component = false,
                                             bool write_output = true) {
  using Tc = A2D_complex_t<double>;

  QuaternionAngularVelocityTest<Tc> test1;
  test1.set_tolerances(1e-10, 1e-14);
  bool passed = TestQuaternions(test1, write_output, 1e-30, 1e-12);
  passed = passed && Run(test1, component, write_output);

  return passed;
}

}  // namespace Test

}  // namespace A2D

#endif  // A2D_QUATERNION_H
