#ifndef A2D_MAT_ADD_H
#define A2D_MAT_ADD_H

#include <type_traits>

#include "../a2ddefs.h"
#include "a2dmat.h"
#include "a2dstack.h"
#include "a2dtest.h"
#include "core/a2dveccore.h"

namespace A2D {

template <typename T, int N, int M>
A2D_FUNCTION void MatSum(const Mat<T, N, M> &A, const Mat<T, N, M> &B,
                         Mat<T, N, M> &C) {
  VecSumCore<T, N * M>(get_data(A), get_data(B), get_data(C));
}

template <typename T, int N, int M>
A2D_FUNCTION void MatSum(const T alpha, const Mat<T, N, M> &A, const T beta,
                         const Mat<T, N, M> &B, Mat<T, N, M> &C) {
  VecSumCore<T, N * M>(alpha, get_data(A), beta, get_data(B), get_data(C));
}

template <typename T, int N>
A2D_FUNCTION void MatSum(const SymMat<T, N> &A, const SymMat<T, N> &B,
                         SymMat<T, N> &C) {
  VecSumCore<T, (N * (N + 1)) / 2>(get_data(A), get_data(B), get_data(C));
}

template <typename T, int N>
A2D_FUNCTION void MatSum(const T alpha, const SymMat<T, N> &A, const T beta,
                         const SymMat<T, N> &B, SymMat<T, N> &C) {
  VecSumCore<T, (N * (N + 1)) / 2>(alpha, get_data(A), beta, get_data(B),
                                   get_data(C));
}

template <class Atype, class Btype, class Ctype>
class MatSumExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<Ctype>::type T;

  // Get the sizes of the matrices
  static const int size = get_num_matrix_entries<Ctype>::size;

  // Get the types of the matrices
  static constexpr ADiffType adA = get_diff_type<Atype>::diff_type;
  static constexpr ADiffType adB = get_diff_type<Btype>::diff_type;

  // Assert that all the matrices are compatible types
  static_assert(((get_a2d_object_type<Atype>::value ==
                  get_a2d_object_type<Btype>::value) &&
                 (get_a2d_object_type<Btype>::value ==
                  get_a2d_object_type<Ctype>::value)),
                "Matrices are not all of the same type");

  static_assert((get_num_matrix_entries<Atype>::size == size &&
                 get_num_matrix_entries<Btype>::size == size),
                "Matrix sizes must agree");

  A2D_FUNCTION
  MatSumExpr(Atype &A, Btype &B, Ctype &C) : A(A), B(B), C(C) {}

  A2D_FUNCTION void eval() {
    VecSumCore<T, size>(get_data(A), get_data(B), get_data(C));
  }

  A2D_FUNCTION void bzero() { C.bzero(); }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;
    if constexpr (adA == ADiffType::ACTIVE && adB == ADiffType::ACTIVE) {
      VecSumCore<T, size>(GetSeed<seed>::get_data(A),
                          GetSeed<seed>::get_data(B),
                          GetSeed<seed>::get_data(C));
    } else if constexpr (adA == ADiffType::ACTIVE) {
      VecCopyCore<T, size>(GetSeed<seed>::get_data(A),
                           GetSeed<seed>::get_data(C));
    } else if constexpr (adB == ADiffType::ACTIVE) {
      VecCopyCore<T, size>(GetSeed<seed>::get_data(B),
                           GetSeed<seed>::get_data(C));
    }
  }

  A2D_FUNCTION void reverse() {
    constexpr ADseed seed = ADseed::b;
    if constexpr (adA == ADiffType::ACTIVE) {
      VecAddCore<T, size>(GetSeed<seed>::get_data(C),
                          GetSeed<seed>::get_data(A));
    }
    if constexpr (adB == ADiffType::ACTIVE) {
      VecAddCore<T, size>(GetSeed<seed>::get_data(C),
                          GetSeed<seed>::get_data(B));
    }
  }

  A2D_FUNCTION void hzero() { C.hzero(); }

  A2D_FUNCTION void hreverse() {
    constexpr ADseed seed = ADseed::h;
    if constexpr (adA == ADiffType::ACTIVE) {
      VecAddCore<T, size>(GetSeed<seed>::get_data(C),
                          GetSeed<seed>::get_data(A));
    }
    if constexpr (adB == ADiffType::ACTIVE) {
      VecAddCore<T, size>(GetSeed<seed>::get_data(C),
                          GetSeed<seed>::get_data(B));
    }
  }

  Atype &A;
  Btype &B;
  Ctype &C;
};

// Full active variants
template <class Atype, class Btype, class Ctype>
A2D_FUNCTION auto MatSum(ADObj<Atype> &A, ADObj<Btype> &B, ADObj<Ctype> &C) {
  return MatSumExpr<ADObj<Atype>, ADObj<Btype>, ADObj<Ctype>>(A, B, C);
}

template <class Atype, class Btype, class Ctype>
A2D_FUNCTION auto MatSum(A2DObj<Atype> &A, A2DObj<Btype> &B, A2DObj<Ctype> &C) {
  return MatSumExpr<A2DObj<Atype>, A2DObj<Btype>, A2DObj<Ctype>>(A, B, C);
}

template <class Atype, class Btype, class Ctype>
A2D_FUNCTION auto MatSum(const Atype &A, ADObj<Btype> &B, ADObj<Ctype> &C) {
  return MatSumExpr<const Atype, ADObj<Btype>, ADObj<Ctype>>(A, B, C);
}

template <class Atype, class Btype, class Ctype>
A2D_FUNCTION auto MatSum(const Atype &A, A2DObj<Btype> &B, A2DObj<Ctype> &C) {
  return MatSumExpr<const Atype, A2DObj<Btype>, A2DObj<Ctype>>(A, B, C);
}

template <class Atype, class Btype, class Ctype>
A2D_FUNCTION auto MatSum(ADObj<Atype> &A, const Btype &B, ADObj<Ctype> &C) {
  return MatSumExpr<ADObj<Atype>, const Btype, ADObj<Ctype>>(A, B, C);
}

template <class Atype, class Btype, class Ctype>
A2D_FUNCTION auto MatSum(A2DObj<Atype> &A, const Btype &B, A2DObj<Ctype> &C) {
  return MatSumExpr<A2DObj<Atype>, const Btype, A2DObj<Ctype>>(A, B, C);
}

template <class atype, class Atype, class btype, class Btype, class Ctype>
class MatSumScaleExpr {
 public:
  // Extract the numeric type to use
  typedef typename get_object_numeric_type<Ctype>::type T;

  // Get the sizes of the matrices
  static const int size = get_num_matrix_entries<Ctype>::size;

  // Get the types of the matrices and scalars
  static constexpr ADiffType ada = get_diff_type<atype>::diff_type;
  static constexpr ADiffType adb = get_diff_type<btype>::diff_type;
  static constexpr ADiffType adA = get_diff_type<Atype>::diff_type;
  static constexpr ADiffType adB = get_diff_type<Btype>::diff_type;

  // Assert that all the matrices are compatible types
  static_assert(((get_a2d_object_type<Atype>::value ==
                  get_a2d_object_type<Btype>::value) &&
                 (get_a2d_object_type<Btype>::value ==
                  get_a2d_object_type<Ctype>::value)),
                "Matrices are not all of the same type");

  static_assert((get_num_matrix_entries<Atype>::size == size &&
                 get_num_matrix_entries<Btype>::size == size),
                "Matrix sizes must agree");

  A2D_FUNCTION
  MatSumScaleExpr(atype alpha, Atype &A, btype beta, Btype &B, Ctype &C)
      : alpha(alpha), A(A), beta(beta), B(B), C(C) {}

  A2D_FUNCTION void eval() {
    VecSumCore<T, size>(get_data(alpha), get_data(A), get_data(beta),
                        get_data(B), get_data(C));
  }

  template <ADorder forder>
  A2D_FUNCTION void forward() {
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;

    if constexpr (ada == ADiffType::ACTIVE && adA == ADiffType::ACTIVE &&
                  adb == ADiffType::ACTIVE && adB == ADiffType::ACTIVE) {
      VecSumCore<T, size>(get_data(alpha), GetSeed<seed>::get_data(A),
                          get_data(beta), GetSeed<seed>::get_data(B),
                          GetSeed<seed>::get_data(C));
      VecSumCoreAdd<T, size>(GetSeed<seed>::get_data(alpha), get_data(A),
                             GetSeed<seed>::get_data(beta), get_data(B),
                             GetSeed<seed>::get_data(C));
    } else {
      VecZeroCore<T, size>(GetSeed<seed>::get_data(C));
      if constexpr (adA == ADiffType::ACTIVE) {
        VecAddCore<T, size>(get_data(alpha), GetSeed<seed>::get_data(A),
                            GetSeed<seed>::get_data(C));
      }
      if constexpr (adB == ADiffType::ACTIVE) {
        VecAddCore<T, size>(get_data(beta), GetSeed<seed>::get_data(B),
                            GetSeed<seed>::get_data(C));
      }
      if constexpr (ada == ADiffType::ACTIVE) {
        VecAddCore<T, size>(GetSeed<seed>::get_data(alpha), get_data(A),
                            GetSeed<seed>::get_data(C));
      }
      if constexpr (adb == ADiffType::ACTIVE) {
        VecAddCore<T, size>(GetSeed<seed>::get_data(beta), get_data(B),
                            GetSeed<seed>::get_data(C));
      }
    }
  }

  A2D_FUNCTION void bzero() { B.bzero(); }

  A2D_FUNCTION void reverse() {
    constexpr ADseed seed = ADseed::b;
    if constexpr (adA == ADiffType::ACTIVE) {
      VecAddCore<T, size>(get_data(alpha), GetSeed<seed>::get_data(C),
                          GetSeed<seed>::get_data(A));
    }
    if constexpr (adB == ADiffType::ACTIVE) {
      VecAddCore<T, size>(get_data(beta), GetSeed<seed>::get_data(C),
                          GetSeed<seed>::get_data(B));
    }
    if constexpr (ada == ADiffType::ACTIVE) {
      GetSeed<seed>::get_data(alpha) +=
          VecDotCore<T, size>(GetSeed<seed>::get_data(C), get_data(A));
    }
    if constexpr (adb == ADiffType::ACTIVE) {
      GetSeed<seed>::get_data(beta) +=
          VecDotCore<T, size>(GetSeed<seed>::get_data(C), get_data(B));
    }
  }

  A2D_FUNCTION void hzero() { C.hzero(); }

  A2D_FUNCTION void hreverse() {
    constexpr ADseed seed = ADseed::h;
    if constexpr (adA == ADiffType::ACTIVE) {
      VecAddCore<T, size>(get_data(alpha), GetSeed<seed>::get_data(C),
                          GetSeed<seed>::get_data(A));
    }
    if constexpr (adB == ADiffType::ACTIVE) {
      VecAddCore<T, size>(get_data(beta), GetSeed<seed>::get_data(C),
                          GetSeed<seed>::get_data(B));
    }
    if constexpr (ada == ADiffType::ACTIVE) {
      GetSeed<seed>::get_data(alpha) +=
          VecDotCore<T, size>(GetSeed<seed>::get_data(C), get_data(A));
    }
    if constexpr (adb == ADiffType::ACTIVE) {
      GetSeed<seed>::get_data(beta) +=
          VecDotCore<T, size>(GetSeed<seed>::get_data(C), get_data(B));
    }
    if constexpr (adA == ADiffType::ACTIVE && ada == ADiffType::ACTIVE) {
      GetSeed<seed>::get_data(alpha) += VecDotCore<T, size>(
          GetSeed<ADseed::b>::get_data(C), GetSeed<ADseed::p>::get_data(A));
      VecAddCore<T, size>(GetSeed<ADseed::p>::get_data(alpha),
                          GetSeed<ADseed::b>::get_data(C),
                          GetSeed<seed>::get_data(A));
    }
    if constexpr (adB == ADiffType::ACTIVE && adb == ADiffType::ACTIVE) {
      GetSeed<seed>::get_data(beta) += VecDotCore<T, size>(
          GetSeed<ADseed::b>::get_data(C), GetSeed<ADseed::p>::get_data(B));
      VecAddCore<T, size>(GetSeed<ADseed::p>::get_data(beta),
                          GetSeed<ADseed::b>::get_data(C),
                          GetSeed<seed>::get_data(B));
    }
  }

  atype alpha;
  Atype &A;
  btype beta;
  Btype &B;
  Ctype &C;
};

// Fully AD
template <class atype, class Atype, class btype, class Btype, class Ctype>
A2D_FUNCTION auto MatSum(ADObj<atype> &alpha, ADObj<Atype> &A,
                         ADObj<btype> &beta, ADObj<Btype> &B, ADObj<Ctype> &C) {
  return MatSumScaleExpr<ADObj<atype> &, ADObj<Atype>, ADObj<btype> &,
                         ADObj<Btype>, ADObj<Ctype>>(alpha, A, beta, B, C);
}

template <class atype, class Atype, class btype, class Btype, class Ctype>
A2D_FUNCTION auto MatSum(A2DObj<atype> &alpha, A2DObj<Atype> &A,
                         A2DObj<btype> &beta, A2DObj<Btype> &B,
                         A2DObj<Ctype> &C) {
  return MatSumScaleExpr<A2DObj<atype> &, A2DObj<Atype>, A2DObj<btype> &,
                         A2DObj<Btype>, A2DObj<Ctype>>(alpha, A, beta, B, C);
}

// Fully AD
template <class atype, class Atype, class btype, class Btype, class Ctype>
A2D_FUNCTION auto MatSum(const atype alpha, ADObj<Atype> &A, const btype beta,
                         ADObj<Btype> &B, ADObj<Ctype> &C) {
  return MatSumScaleExpr<const atype, ADObj<Atype>, const btype, ADObj<Btype>,
                         ADObj<Ctype>>(alpha, A, beta, B, C);
}

template <class atype, class Atype, class btype, class Btype, class Ctype>
A2D_FUNCTION auto MatSum(const atype alpha, A2DObj<Atype> &A, const btype beta,
                         A2DObj<Btype> &B, A2DObj<Ctype> &C) {
  return MatSumScaleExpr<const atype, A2DObj<Atype>, const btype, A2DObj<Btype>,
                         A2DObj<Ctype>>(alpha, A, beta, B, C);
}

namespace Test {

template <typename T, int N, int M>
class MatSumTest : public A2DTest<T, Mat<T, N, M>, Mat<T, N, M>, Mat<T, N, M>> {
 public:
  using Input = VarTuple<T, Mat<T, N, M>, Mat<T, N, M>>;
  using Output = VarTuple<T, Mat<T, N, M>>;

  // Assemble a string to describe the test
  std::string name() {
    std::stringstream s;
    s << "MatSum<" << N << "," << M << ">";
    return s.str();
  }

  // Evaluate the matrix-matrix product
  Output eval(const Input &x) {
    Mat<T, N, M> A, B, C;
    x.get_values(A, B);
    MatSum(A, B, C);
    return MakeVarTuple<T>(C);
  }

  // Compute the derivative
  void deriv(const Output &seed, const Input &x, Input &g) {
    ADObj<Mat<T, N, M>> A, B, C;
    x.get_values(A.value(), B.value());
    auto stack = MakeStack(MatSum(A, B, C));
    seed.get_values(C.bvalue());
    stack.reverse();
    g.set_values(A.bvalue(), B.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output &seed, const Output &hval, const Input &x,
             const Input &p, Input &h) {
    A2DObj<Mat<T, N, M>> A, B, C;
    x.get_values(A.value(), B.value());
    p.get_values(A.pvalue(), B.pvalue());
    auto stack = MakeStack(MatSum(A, B, C));
    seed.get_values(C.bvalue());
    hval.get_values(C.hvalue());
    stack.hproduct();
    h.set_values(A.hvalue(), B.hvalue());
  }
};

template <typename T, int N, int M>
class MatSumScaleTest
    : public A2DTest<T, Mat<T, N, M>, T, Mat<T, N, M>, T, Mat<T, N, M>> {
 public:
  using Input = VarTuple<T, T, Mat<T, N, M>, T, Mat<T, N, M>>;
  using Output = VarTuple<T, Mat<T, N, M>>;

  // Assemble a string to describe the test
  std::string name() {
    std::stringstream s;
    s << "MatSum<" << N << "," << M << ">";
    return s.str();
  }

  // Evaluate the matrix-matrix product
  Output eval(const Input &x) {
    T alpha, beta;
    Mat<T, N, M> A, B, C;
    x.get_values(alpha, A, beta, B);
    MatSum(alpha, A, beta, B, C);
    return MakeVarTuple<T>(C);
  }

  // Compute the derivative
  void deriv(const Output &seed, const Input &x, Input &g) {
    ADObj<T> alpha, beta;
    ADObj<Mat<T, N, M>> A, B, C;
    x.get_values(alpha.value(), A.value(), beta.value(), B.value());
    auto stack = MakeStack(MatSum(alpha, A, beta, B, C));
    seed.get_values(C.bvalue());
    stack.reverse();
    g.set_values(alpha.bvalue(), A.bvalue(), beta.bvalue(), B.bvalue());
  }

  // Compute the second-derivative
  void hprod(const Output &seed, const Output &hval, const Input &x,
             const Input &p, Input &h) {
    A2DObj<T> alpha, beta;
    A2DObj<Mat<T, N, M>> A, B, C;
    x.get_values(alpha.value(), A.value(), beta.value(), B.value());
    p.get_values(alpha.pvalue(), A.pvalue(), beta.pvalue(), B.pvalue());
    auto stack = MakeStack(MatSum(alpha, A, beta, B, C));
    seed.get_values(C.bvalue());
    hval.get_values(C.hvalue());
    stack.hproduct();
    h.set_values(alpha.hvalue(), A.hvalue(), beta.hvalue(), B.hvalue());
  }
};

inline bool MatSumTestAll(bool component = false, bool write_output = true) {
  using Tc = A2D_complex_t<double>;

  bool passed = true;
  MatSumTest<Tc, 3, 4> test1;
  passed = passed && Run(test1, component, write_output);
  MatSumTest<Tc, 5, 3> test2;
  passed = passed && Run(test2, component, write_output);

  MatSumScaleTest<Tc, 3, 4> test3;
  passed = passed && Run(test3, component, write_output);

  return passed;
}

}  // namespace Test

}  // namespace A2D

#endif  // A2D_MAT_ADD_H
