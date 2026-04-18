#ifndef A2D_MAT_H
#define A2D_MAT_H

#include "../a2ddefs.h"

namespace A2D {

template <typename T, int M, int N>
class Mat {
 public:
  typedef T type;
  static const ADObjType obj_type = ADObjType::MATRIX;
  static const index_t ncomp = M * N;
  static const int nrows = M;
  static const int ncols = N;

  A2D_FUNCTION Mat() {
    for (int i = 0; i < M * N; i++) {
      A[i] = 0.0;
    }
  }
  template <typename T2>
  A2D_FUNCTION Mat(const T2* vals) {
    for (int i = 0; i < M * N; i++) {
      A[i] = vals[i];
    }
  }
  template <typename T2>
  A2D_FUNCTION Mat(const Mat<T2, M, N>& src) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        A[N * i + j] = src(i, j);
      }
    }
  }
  A2D_FUNCTION void zero() {
    for (int i = 0; i < M * N; i++) {
      A[i] = 0.0;
    }
  }
  template <typename T2>
  A2D_FUNCTION void copy(const Mat<T2, M, N>& src) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        A[N * i + j] = src(i, j);
      }
    }
  }
  template <typename T2>
  A2D_FUNCTION void get(Mat<T2, M, N>& mat) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        mat(i, j) = A[N * i + j];
      }
    }
  }
  template <class IdxType1, class IdxType2>
  A2D_FUNCTION T& operator()(const IdxType1 i, const IdxType2 j) {
    return A[N * i + j];
  }
  template <class IdxType1, class IdxType2>
  A2D_FUNCTION const T& operator()(const IdxType1 i, const IdxType2 j) const {
    return A[N * i + j];
  }

  A2D_FUNCTION T* get_data() { return A; }
  A2D_FUNCTION const T* get_data() const { return A; }

  template <typename I>
  A2D_FUNCTION T& operator[](const I i) {
    return A[i];
  }

  template <typename I>
  A2D_FUNCTION const T& operator[](const I i) const {
    return A[i];
  }

 private:
  T A[M * N];
};

/*
 * Only lower triangle entries of the SymMat are stored in the following order:
 *
 * S[0]
 * S[1] S[2]
 * S[3] S[4] S[5]
 * S[6] S[7] S[8] S[9]
 * ...
 *
 * */
template <typename T, int N>
class SymMat {
 public:
  typedef T type;
  static const ADObjType obj_type = ADObjType::SYMMAT;
  static const int MAT_SIZE = (N * (N + 1)) / 2;
  static const index_t ncomp = MAT_SIZE;
  static constexpr int nrows = N;
  static constexpr int ncols = N;

  A2D_FUNCTION SymMat() {
    for (int i = 0; i < MAT_SIZE; i++) {
      A[i] = 0.0;
    }
  }
  A2D_FUNCTION SymMat(const T* vals) {
    for (int i = 0; i < MAT_SIZE; i++) {
      A[i] = vals[i];
    }
  }
  template <typename T2>
  A2D_FUNCTION SymMat(const SymMat<T2, N>& src) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j <= i; j++) {
        A[j + i * (i + 1) / 2] = src(i, j);
      }
    }
  }
  A2D_FUNCTION void zero() {
    for (int i = 0; i < MAT_SIZE; i++) {
      A[i] = 0.0;
    }
  }
  template <typename T2>
  A2D_FUNCTION void copy(const SymMat<T2, N>& src) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j <= i; j++) {
        A[j + i * (i + 1) / 2] = src(i, j);
      }
    }
  }
  template <typename T2>
  A2D_FUNCTION void get(SymMat<T2, N>& mat) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j <= i; j++) {
        mat(i, j) = A[j + i * (i + 1) / 2];
      }
    }
  }

  template <class IdxType1, class IdxType2>
  A2D_FUNCTION T& operator()(const IdxType1 i, const IdxType2 j) {
    if (i >= j) {
      return A[j + i * (i + 1) / 2];
    } else {
      return A[i + j * (j + 1) / 2];
    }
  }
  template <class IdxType1, class IdxType2>
  A2D_FUNCTION const T& operator()(const IdxType1 i, const IdxType2 j) const {
    if (i >= j) {
      return A[j + i * (i + 1) / 2];
    } else {
      return A[i + j * (j + 1) / 2];
    }
  }

  A2D_FUNCTION T* get_data() { return A; }
  A2D_FUNCTION const T* get_data() const { return A; }

  template <typename I>
  A2D_FUNCTION T& operator[](const I i) {
    return A[i];
  }
  template <typename I>
  A2D_FUNCTION const T& operator[](const I i) const {
    return A[i];
  }

 private:
  T A[MAT_SIZE];
};

template <typename T>
struct is_a2d_matrix : std::false_type {};

template <typename U, int N, int M>
struct is_a2d_matrix<Mat<U, N, M>> : std::true_type {};

template <typename T>
struct is_a2d_sym_matrix : std::false_type {};

template <typename U, int N>
struct is_a2d_sym_matrix<SymMat<U, N>> : std::true_type {};

}  // namespace A2D

#endif  // A2D_MAT_H
