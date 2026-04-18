#ifndef A2D_OBJECTS_H
#define A2D_OBJECTS_H

#include "../a2ddefs.h"
#include "a2dmat.h"
#include "a2dvec.h"
#include "adscalar.h"

namespace A2D {

/*
  Expression template for first derivatives
*/
template <class A, typename T>
class ADExpr {
 public:
  A2D_FUNCTION A& self() { return static_cast<A&>(*this); }
  A2D_FUNCTION const A& self() const { return static_cast<const A&>(*this); }

  // Evaluation and derivatives
  A2D_FUNCTION void eval() { self().eval(); }
  A2D_FUNCTION void forward() { self().forward(); }
  A2D_FUNCTION void reverse() { self().reverse(); }

  // Access the values and derivatives
  A2D_FUNCTION T& value() { return self().value(); }
  A2D_FUNCTION const T& value() const { return self().value(); }
  A2D_FUNCTION T& bvalue() { return self().bvalue(); }
  A2D_FUNCTION const T& bvalue() const { return self().bvalue(); }
};

/*
  The base first-order AD class
*/
template <class T>
class ADObj : public ADExpr<ADObj<T>, T> {
 public:
  typedef typename get_object_numeric_type<T>::type type;
  static constexpr ADObjType obj_type = get_a2d_object_type<T>::value;

  // If this is a scalar, non-reference type - initialize values to zero
  A2D_FUNCTION ADObj() {
    if constexpr (is_scalar_type<T>::value && !std::is_reference<T>::value) {
      A = Ab = type(0.0);
    }
  }

  // If this is a scalar, non-reference type - initialize values to zero
  A2D_FUNCTION ADObj(const T& A) : A(A) {
    if constexpr (is_scalar_type<T>::value && !std::is_reference<T>::value) {
      Ab = type(0.0);
    }
  }

  // Initialize with both values
  template <typename U = T,
            std::enable_if_t<std::is_reference<U>::value, bool> = true>
  A2D_FUNCTION ADObj(T& A, T& Ab) : A(A), Ab(Ab) {}

  template <typename U = T,
            std::enable_if_t<!std::is_reference<U>::value, bool> = true>
  A2D_FUNCTION ADObj(const T& A, const T& Ab) : A(A), Ab(Ab) {}

  // Evaluation and derivatives
  A2D_FUNCTION void eval() {}
  A2D_FUNCTION void forward() {}
  A2D_FUNCTION void reverse() {}
  A2D_FUNCTION void bzero() {
    if constexpr (obj_type == ADObjType::SCALAR) {
      Ab = type(0.0);
    } else {
      Ab.zero();
    }
  }

  // Extract the data from the underlying objects
  A2D_FUNCTION T& value() { return A; }
  A2D_FUNCTION const T& value() const { return A; }
  A2D_FUNCTION T& bvalue() { return Ab; }
  A2D_FUNCTION const T& bvalue() const { return Ab; }

  template <typename I, typename U = T,
            std::enable_if_t<
                is_a2d_vector<typename remove_const_and_refs<U>::type>::value,
                bool> = true>
  A2D_FUNCTION ADObj<type&> operator[](const I i) {
    return ADObj<type&>(A[i], Ab[i]);
  }

  template <typename I, typename U = T,
            std::enable_if_t<
                is_a2d_vector<typename remove_const_and_refs<U>::type>::value,
                bool> = true>
  A2D_FUNCTION ADObj<type&> operator()(const I i) {
    return ADObj<type&>(A[i], Ab[i]);
  }

  template <typename I, typename U = T,
            std::enable_if_t<
                is_a2d_matrix<typename remove_const_and_refs<U>::type>::value,
                bool> = true>
  A2D_FUNCTION ADObj<type&> operator()(const I i, const I j) {
    return ADObj<type&>(A(i, j), Ab(i, j));
  }

  template <
      typename I, typename U = T,
      std::enable_if_t<
          is_a2d_sym_matrix<typename remove_const_and_refs<U>::type>::value,
          bool> = true>
  A2D_FUNCTION ADObj<type&> operator()(const I i, const I j) {
    return ADObj<type&>(A(i, j), Ab(i, j));
  }

 private:
  T A;   // Object
  T Ab;  // Reverse mode derivative value
};

/*
  Expression template for second derivatives
*/
template <class A, typename T>
class A2DExpr {
 public:
  A2D_FUNCTION A& self() { return static_cast<A&>(*this); }
  A2D_FUNCTION const A& self() const { return static_cast<const A&>(*this); }

  // Evaluation and derivatives
  A2D_FUNCTION void eval() { self().eval(); }
  A2D_FUNCTION void reverse() { self().reverse(); }
  A2D_FUNCTION void hforward() { self().hforward(); }
  A2D_FUNCTION void hreverse() { self().hreverse(); }
  A2D_FUNCTION void bzero() { self().bzero(); }
  A2D_FUNCTION void hzero() { self().hzero(); }

  // Access the values and derivatives
  A2D_FUNCTION T& value() { return self().value(); }
  A2D_FUNCTION const T& value() const { return self().value(); }
  A2D_FUNCTION T& bvalue() { return self().bvalue(); }
  A2D_FUNCTION const T& bvalue() const { return self().bvalue(); }
  A2D_FUNCTION T& pvalue() { return self().pvalue(); }
  A2D_FUNCTION const T& pvalue() const { return self().pvalue(); }
  A2D_FUNCTION T& hvalue() { return self().hvalue(); }
  A2D_FUNCTION const T& hvalue() const { return self().hvalue(); }
};

/*
  The base second-order AD class
*/
template <class T>
class A2DObj : public A2DExpr<A2DObj<T>, T> {
 public:
  typedef typename get_object_numeric_type<T>::type type;
  static constexpr ADObjType obj_type = get_a2d_object_type<T>::value;

  A2D_FUNCTION A2DObj() {
    if constexpr (is_scalar_type<T>::value && !std::is_reference<T>::value) {
      A = Ab = Ap = Ah = type(0.0);
    }
  }
  A2D_FUNCTION A2DObj(const T& A) : A(A) {
    if constexpr (is_scalar_type<T>::value && !std::is_reference<T>::value) {
      Ab = Ap = Ah = type(0.0);
    }
  }
  A2D_FUNCTION A2DObj(const T& A, const T& Ab) : A(A), Ab(Ab) {
    if constexpr (is_scalar_type<T>::value && !std::is_reference<T>::value) {
      Ap = Ah = type(0.0);
    }
  }
  A2D_FUNCTION A2DObj(const T& A, const T& Ab, const T& Ap)
      : A(A), Ab(Ab), Ap(Ap) {
    if constexpr (is_scalar_type<T>::value && !std::is_reference<T>::value) {
      Ah = type(0.0);
    }
  }
  template <typename U = T,
            std::enable_if_t<std::is_reference<U>::value, bool> = true>
  A2D_FUNCTION A2DObj(T& A, T& Ab, T& Ap, T& Ah)
      : A(A), Ab(Ab), Ap(Ap), Ah(Ah) {}

  template <typename U = T,
            std::enable_if_t<!std::is_reference<U>::value, bool> = true>
  A2D_FUNCTION A2DObj(const T& A, const T& Ab, const T& Ap, const T& Ah)
      : A(A), Ab(Ab), Ap(Ap), Ah(Ah) {}

  // Evaluation and derivatives
  A2D_FUNCTION void eval() {}
  A2D_FUNCTION void reverse() {}
  A2D_FUNCTION void hforward() {}
  A2D_FUNCTION void hreverse() {}

  A2D_FUNCTION void bzero() {
    if constexpr (obj_type == ADObjType::SCALAR) {
      Ab = type(0.0);
    } else {
      Ab.zero();
    }
  }
  A2D_FUNCTION void hzero() {
    if constexpr (obj_type == ADObjType::SCALAR) {
      Ah = type(0.0);
    } else {
      Ah.zero();
    }
  }
  A2D_FUNCTION T& value() { return A; }
  A2D_FUNCTION const T& value() const { return A; }
  A2D_FUNCTION T& bvalue() { return Ab; }
  A2D_FUNCTION const T& bvalue() const { return Ab; }
  A2D_FUNCTION T& pvalue() { return Ap; }
  A2D_FUNCTION const T& pvalue() const { return Ap; }
  A2D_FUNCTION T& hvalue() { return Ah; }
  A2D_FUNCTION const T& hvalue() const { return Ah; }

  template <typename I, typename U = T,
            std::enable_if_t<
                is_a2d_vector<typename remove_const_and_refs<U>::type>::value,
                bool> = true>
  A2D_FUNCTION A2DObj<type&> operator[](const I i) {
    return A2DObj<type&>(A[i], Ab[i], Ap[i], Ah[i]);
  }

  template <typename I, typename U = T,
            std::enable_if_t<
                is_a2d_vector<typename remove_const_and_refs<U>::type>::value,
                bool> = true>
  A2D_FUNCTION A2DObj<type&> operator()(const I i) {
    return A2DObj<type&>(A[i], Ab[i], Ap[i], Ah[i]);
  }

  template <typename I, typename U = T,
            std::enable_if_t<
                is_a2d_matrix<typename remove_const_and_refs<U>::type>::value,
                bool> = true>
  A2D_FUNCTION A2DObj<type&> operator()(const I i, const I j) {
    return A2DObj<type&>(A(i, j), Ab(i, j), Ap(i, j), Ah(i, j));
  }

  template <
      typename I, typename U = T,
      std::enable_if_t<
          is_a2d_sym_matrix<typename remove_const_and_refs<U>::type>::value,
          bool> = true>
  A2D_FUNCTION A2DObj<type&> operator()(const I i, const I j) {
    return A2DObj<type&>(A(i, j), Ab(i, j), Ap(i, j), Ah(i, j));
  }

 private:
  T A;   // Object
  T Ab;  // Reverse mode derivative value
  T Ap;  // Projected second derivative value
  T Ah;  // Reverse mode second derivative
};

/*
  Remove the A2D template wrapper remove_a2dobj<T>::type provides the base type
*/
template <class T>
struct __remove_a2dobj {
  typedef T type;
};

template <class T>
struct __remove_a2dobj<ADObj<T>> {
  typedef typename std::remove_reference<T>::type type;
};

template <class T>
struct __remove_a2dobj<A2DObj<T>> {
  typedef typename std::remove_reference<T>::type type;
};

template <class T>
struct remove_a2dobj
    : __remove_a2dobj<typename remove_const_and_refs<T>::type> {};

template <class Ta, class Tb>
struct is_same_type {
  const static bool value =
      std::is_same<typename remove_const_and_refs<Ta>::type,
                   typename remove_const_and_refs<Tb>::type>::value;
};

/*
  Get whether the type is passive or not...
*/
template <class T>
struct get_diff_type {
  static constexpr ADiffType diff_type = ADiffType::PASSIVE;
};

template <class T>
struct get_diff_type<ADObj<T>> {
  static constexpr ADiffType diff_type = ADiffType::ACTIVE;
};

template <class T>
struct get_diff_type<A2DObj<T>> {
  static constexpr ADiffType diff_type = ADiffType::ACTIVE;
};

template <class T>
struct get_diff_type<ADObj<T>&> {
  static constexpr ADiffType diff_type = ADiffType::ACTIVE;
};

template <class T>
struct get_diff_type<A2DObj<T>&> {
  static constexpr ADiffType diff_type = ADiffType::ACTIVE;
};

/*
  Get whether the type is passive or not...
*/
template <class T>
struct get_diff_order {
  static constexpr ADorder order = ADorder::ZERO;
};

template <class T>
struct get_diff_order<ADObj<T>> {
  static constexpr ADorder order = ADorder::FIRST;
};

template <class T>
struct get_diff_order<A2DObj<T>> {
  static constexpr ADorder order = ADorder::SECOND;
};

/*
  Get the vector size
*/
template <class T>
struct __get_vec_size {
  static constexpr int size = 0;
};

// This will pick up vectors too...
template <template <typename, int> class Vec, typename T, int N>
struct __get_vec_size<Vec<T, N>> {
  static constexpr int size = N;
};

template <class T>
struct get_vec_size : __get_vec_size<typename remove_a2dobj<T>::type> {
  static_assert(get_a2d_object_type<T>::value == ADObjType::VECTOR,
                "get_symmatrix_size called on incorrect type");
};

/*
  Get the symmetric matrix size
*/
template <class T>
struct __get_symmatrix_size {
  static constexpr int size = 0;
};

// This will pick up vectors too...
template <template <typename, int> class SymMat, typename T, int N>
struct __get_symmatrix_size<SymMat<T, N>> {
  static constexpr int size = N;
};

template <class T>
struct get_symmatrix_size
    : __get_symmatrix_size<typename remove_a2dobj<T>::type> {
  static_assert(get_a2d_object_type<T>::value == ADObjType::SYMMAT,
                "get_symmatrix_size called on incorrect type");
};

/*
  Get the number of matrix rows
*/
template <class T>
struct __get_matrix_rows {
  static constexpr int size = 0;
};

template <template <typename, int, int> class Mat, typename T, int N, int M>
struct __get_matrix_rows<Mat<T, N, M>> {
  static constexpr int size = N;
};

template <template <typename, int> class SymMat, typename T, int N>
struct __get_matrix_rows<SymMat<T, N>> {
  static constexpr int size = N;
};

template <class T>
struct get_matrix_rows : __get_matrix_rows<typename remove_a2dobj<T>::type> {
  static_assert(get_a2d_object_type<T>::value == ADObjType::MATRIX,
                "get_matrix_rows called on incorrect type");
};

/*
  Get the number of matrix columns
*/
template <class T>
struct __get_matrix_columns {
  static constexpr int size = 0;
};

template <template <typename, int, int> class Mat, typename T, int N, int M>
struct __get_matrix_columns<Mat<T, N, M>> {
  static constexpr int size = M;
};

template <template <typename, int> class SymMat, typename T, int N>
struct __get_matrix_columns<SymMat<T, N>> {
  static constexpr int size = N;
};

template <class T>
struct get_matrix_columns
    : __get_matrix_columns<typename remove_a2dobj<T>::type> {
  static_assert(get_a2d_object_type<T>::value == ADObjType::MATRIX,
                "get_matrix_rows called on incorrect type");
};

/*
  Get the number of entries in the matrix -- symmetric or non-symmetric
*/
template <class T>
struct __get_num_matrix_entries {
  static constexpr int size = 0;
};

template <template <typename, int> class SymMat, typename T, int N>
struct __get_num_matrix_entries<SymMat<T, N>> {
  static constexpr int size = N * (N + 1) / 2;
};

template <template <typename, int, int> class Mat, typename T, int N, int M>
struct __get_num_matrix_entries<Mat<T, N, M>> {
  static constexpr int size = N * M;
};

template <class T>
struct get_num_matrix_entries
    : __get_num_matrix_entries<typename remove_a2dobj<T>::type> {
  static_assert((get_a2d_object_type<T>::value == ADObjType::MATRIX ||
                 get_a2d_object_type<T>::value == ADObjType::SYMMAT),
                "get_num_matrix_entries called on incorrect type");
};

/**
 * @brief Select type based on supplied ADiffType value
 *
 * For example, the following types are equivalent:
 *
 * Obj<...>         ==
 *    ADObjSelect<ADiffType::PASSIVE, *,               Obj<...>>;
 * ADObj<Obj<...>>  ==
 *    ADObjSelect<ADiffType::ACTIVE,  ADorder::FIRST,  Obj<...>>;
 * A2DObj<Obj<...>> ==
 *    ADObjSelect<ADiffType::ACTIVE,  ADorder::SECOND, Obj<...>>;
 *
 * @tparam adiff_type passive or active
 * @tparam order first (AD) or second (A2D)
 * @tparam Obj the numeric type of the matrix
 */
template <ADiffType adiff_type, ADorder order, class Obj>
using ADObjSelect = typename std::conditional<
    adiff_type == ADiffType::ACTIVE,
    typename std::conditional<order == ADorder::FIRST, ADObj<Obj>,
                              A2DObj<Obj>>::type,
    Obj>::type;

/**
 * @brief Select the type of variable to use, depending on the input parameter
 *
 * @tparam wrt Parameter indicating data, geo or state
 * @tparam Data Data variable type
 * @tparam Geo Geo variable type
 * @tparam State State variable type
 */
template <FEVarType wrt, class Data, class Geo, class State>
using FEVarSelect = typename std::conditional<
    wrt == FEVarType::DATA, Data,
    typename std::conditional<wrt == FEVarType::GEOMETRY, Geo,
                              State>::type>::type;

/**
 * @brief Given the variable type, store the number of components
 *
 * @tparam wrt The variable type
 * @tparam ndata number of data components
 * @tparam ngeo number of geometry components
 * @tparam nstate number of state components
 */
template <index_t ndata, index_t ngeo, index_t nstate, FEVarType wrt>
struct get_var_dim {
  static const index_t value = 0;
};

template <index_t ndata, index_t ngeo, index_t nstate>
struct get_var_dim<ndata, ngeo, nstate, FEVarType::DATA> {
  static const index_t value = ndata;
};

template <index_t ndata, index_t ngeo, index_t nstate>
struct get_var_dim<ndata, ngeo, nstate, FEVarType::GEOMETRY> {
  static const index_t value = ngeo;
};

template <index_t ndata, index_t ngeo, index_t nstate>
struct get_var_dim<ndata, ngeo, nstate, FEVarType::STATE> {
  static const index_t value = nstate;
};

/**
 * @brief Select the type of matrix to use - symmetric if of = wrt
 *
 * @tparam of Parameter indicating the residual type
 * @tparam wrt Parameter indicating the derivative type
 * @tparam T typename of the scalar value
 * @tparam ndata number of data components
 * @tparam ngeo number of geometry components
 * @tparam nstate number of state components
 */
template <FEVarType of, FEVarType wrt, typename T, index_t ndata, index_t ngeo,
          index_t nstate>
using FESymMatSelect = typename std::conditional<
    of == wrt, SymMat<T, get_var_dim<ndata, ngeo, nstate, of>::value>,
    Mat<T, get_var_dim<ndata, ngeo, nstate, of>::value,
        get_var_dim<ndata, ngeo, nstate, wrt>::value>>::type;

/**
 * @brief Select the type of matrix to use - always non-symmetric
 *
 * @tparam of Parameter indicating the residual type
 * @tparam wrt Parameter indicating the derivative type
 * @tparam T typename of the scalar value
 * @tparam ndata number of data components
 * @tparam ngeo number of geometry components
 * @tparam nstate number of state components
 */
template <FEVarType of, FEVarType wrt, typename T, index_t ndata, index_t ngeo,
          index_t nstate>
using FEMatSelect = Mat<T, get_var_dim<ndata, ngeo, nstate, of>::value,
                        get_var_dim<ndata, ngeo, nstate, wrt>::value>;

/**
 * @brief Get objects and pointers to seed data (bvalue(), pvalue(), hvalue)
 */
template <ADseed seed>
class GetSeed {
 public:
  template <typename T>
  static A2D_FUNCTION T& get_obj(ADObj<T>& value) {
    static_assert(seed == ADseed::b, "Incompatible seed type for ADObj");
    return value.bvalue();
  }

  template <typename T>
  static A2D_FUNCTION T& get_obj(A2DObj<T>& value) {
    static_assert(seed == ADseed::b or seed == ADseed::p or seed == ADseed::h,
                  "Incompatible seed type for A2DObj");
    if constexpr (seed == ADseed::b) {
      return value.bvalue();
    } else if constexpr (seed == ADseed::p) {
      return value.pvalue();
    } else {  // seed == ADseed::h
      return value.hvalue();
    }
  }

  template <typename T,
            std::enable_if_t<is_numeric_type<T>::value, bool> = true>
  static A2D_FUNCTION T& get_data(ADObj<T>& value) {
    static_assert(seed == ADseed::b, "Incompatible seed type for ADObj");
    return value.bvalue();
  }

  template <typename T,
            std::enable_if_t<is_numeric_type<T>::value, bool> = true>
  static A2D_FUNCTION T& get_data(A2DObj<T>& value) {
    static_assert(seed == ADseed::b or seed == ADseed::p or seed == ADseed::h,
                  "Incompatible seed type for A2DObj");
    if constexpr (seed == ADseed::b) {
      return value.bvalue();
    } else if constexpr (seed == ADseed::p) {
      return value.pvalue();
    } else {  // seed == ADseed::h
      return value.hvalue();
    }
  }

  template <typename T, int N>
  static A2D_FUNCTION ADScalar<T, N>& get_data(ADObj<ADScalar<T, N>>& value) {
    static_assert(seed == ADseed::b, "Incompatible seed type for ADObj");
    return value.bvalue();
  }

  template <typename T, int N>
  static A2D_FUNCTION ADScalar<T, N>& get_data(A2DObj<ADScalar<T, N>>& value) {
    static_assert(seed == ADseed::b or seed == ADseed::p or seed == ADseed::h,
                  "Incompatible seed type for A2DObj");
    if constexpr (seed == ADseed::b) {
      return value.bvalue();
    } else if constexpr (seed == ADseed::p) {
      return value.pvalue();
    } else {  // seed == ADseed::h
      return value.hvalue();
    }
  }

  template <typename T, int N>
  static A2D_FUNCTION T* get_data(ADObj<Vec<T, N>>& value) {
    static_assert(seed == ADseed::b, "Incompatible seed type for ADObj");
    return value.bvalue().get_data();
  }

  template <typename T, int N>
  static A2D_FUNCTION T* get_data(A2DObj<Vec<T, N>>& value) {
    static_assert(seed == ADseed::b or seed == ADseed::p or seed == ADseed::h,
                  "Incompatible seed type for A2DObj");
    if constexpr (seed == ADseed::b) {
      return value.bvalue().get_data();
    } else if constexpr (seed == ADseed::p) {
      return value.pvalue().get_data();
    } else {  // seed == ADseed::h
      return value.hvalue().get_data();
    }
  }

  template <typename T, int m, int n>
  static A2D_FUNCTION T* get_data(ADObj<Mat<T, m, n>>& mat) {
    static_assert(seed == ADseed::b, "Incompatible seed type for ADObj");
    return mat.bvalue().get_data();
  }

  template <typename T, int m, int n>
  static A2D_FUNCTION T* get_data(A2DObj<Mat<T, m, n>>& mat) {
    static_assert(seed == ADseed::b or seed == ADseed::p or seed == ADseed::h,
                  "Incompatible seed type for A2DObj");
    if constexpr (seed == ADseed::b) {
      return mat.bvalue().get_data();
    } else if constexpr (seed == ADseed::p) {
      return mat.pvalue().get_data();
    } else {  // seed == ADseed::h
      return mat.hvalue().get_data();
    }
  }

  template <typename T, int m>
  static A2D_FUNCTION T* get_data(ADObj<SymMat<T, m>>& mat) {
    static_assert(seed == ADseed::b, "Incompatible seed type for ADObj");
    return mat.bvalue().get_data();
  }

  template <typename T, int m>
  static A2D_FUNCTION T* get_data(A2DObj<SymMat<T, m>>& mat) {
    static_assert(seed == ADseed::b or seed == ADseed::p or seed == ADseed::h,
                  "Incompatible seed type for A2DObj");
    if constexpr (seed == ADseed::b) {
      return mat.bvalue().get_data();
    } else if constexpr (seed == ADseed::p) {
      return mat.pvalue().get_data();
    } else {  // seed == ADseed::h
      return mat.hvalue().get_data();
    }
  }

  template <typename T, int N>
  static A2D_FUNCTION T* get_data(ADObj<Vec<T, N>&>& value) {
    static_assert(seed == ADseed::b, "Incompatible seed type for ADObj");
    return value.bvalue().get_data();
  }

  template <typename T, int N>
  static A2D_FUNCTION T* get_data(A2DObj<Vec<T, N>&>& value) {
    static_assert(seed == ADseed::b or seed == ADseed::p or seed == ADseed::h,
                  "Incompatible seed type for A2DObj");
    if constexpr (seed == ADseed::b) {
      return value.bvalue().get_data();
    } else if constexpr (seed == ADseed::p) {
      return value.pvalue().get_data();
    } else {  // seed == ADseed::h
      return value.hvalue().get_data();
    }
  }

  template <typename T, int m, int n>
  static A2D_FUNCTION T* get_data(ADObj<Mat<T, m, n>&>& mat) {
    static_assert(seed == ADseed::b, "Incompatible seed type for ADObj");
    return mat.bvalue().get_data();
  }

  template <typename T, int m, int n>
  static A2D_FUNCTION T* get_data(A2DObj<Mat<T, m, n>&>& mat) {
    static_assert(seed == ADseed::b or seed == ADseed::p or seed == ADseed::h,
                  "Incompatible seed type for A2DObj");
    if constexpr (seed == ADseed::b) {
      return mat.bvalue().get_data();
    } else if constexpr (seed == ADseed::p) {
      return mat.pvalue().get_data();
    } else {  // seed == ADseed::h
      return mat.hvalue().get_data();
    }
  }

  template <typename T, int m>
  static A2D_FUNCTION T* get_data(ADObj<SymMat<T, m>&>& mat) {
    static_assert(seed == ADseed::b, "Incompatible seed type for ADObj");
    return mat.bvalue().get_data();
  }

  template <typename T, int m>
  static A2D_FUNCTION T* get_data(A2DObj<SymMat<T, m>&>& mat) {
    static_assert(seed == ADseed::b or seed == ADseed::p or seed == ADseed::h,
                  "Incompatible seed type for A2DObj");
    if constexpr (seed == ADseed::b) {
      return mat.bvalue().get_data();
    } else if constexpr (seed == ADseed::p) {
      return mat.pvalue().get_data();
    } else {  // seed == ADseed::h
      return mat.hvalue().get_data();
    }
  }
};

template <typename T, std::enable_if_t<is_numeric_type<T>::value, bool> = true>
A2D_FUNCTION T& get_data(T& value) {
  return value;
}

template <typename T, std::enable_if_t<is_numeric_type<T>::value, bool> = true>
A2D_FUNCTION const T& get_data(const T& value) {
  return value;
}

template <typename T, std::enable_if_t<is_numeric_type<T>::value, bool> = true>
A2D_FUNCTION T& get_data(ADObj<T>& value) {
  return value.value();
}

template <typename T, std::enable_if_t<is_numeric_type<T>::value, bool> = true>
A2D_FUNCTION const T& get_data(const ADObj<T>& value) {
  return value.value();
}

template <typename T, std::enable_if_t<is_numeric_type<T>::value, bool> = true>
A2D_FUNCTION T& get_data(A2DObj<T>& value) {
  return value.value();
}

template <typename T, std::enable_if_t<is_numeric_type<T>::value, bool> = true>
A2D_FUNCTION const T& get_data(const A2DObj<T>& value) {
  return value.value();
}

/**
 * @brief Get data pointers from objects
 */
template <typename T, int m, int n>
A2D_FUNCTION T* get_data(Mat<T, m, n>& mat) {
  return mat.get_data();
}

template <typename T, int m, int n>
A2D_FUNCTION const T* get_data(const Mat<T, m, n>& mat) {
  return mat.get_data();
}

template <typename T, int m, int n>
A2D_FUNCTION T* get_data(ADObj<Mat<T, m, n>>& mat) {
  return mat.value().get_data();
}

template <typename T, int m, int n>
A2D_FUNCTION T* get_data(A2DObj<Mat<T, m, n>>& mat) {
  return mat.value().get_data();
}

template <typename T, int m, int n>
A2D_FUNCTION T* get_data(ADObj<Mat<T, m, n>&>& mat) {
  return mat.value().get_data();
}

template <typename T, int m, int n>
A2D_FUNCTION T* get_data(A2DObj<Mat<T, m, n>&>& mat) {
  return mat.value().get_data();
}

template <typename T, int m>
A2D_FUNCTION T* get_data(SymMat<T, m>& mat) {
  return mat.get_data();
}

template <typename T, int m>
A2D_FUNCTION const T* get_data(const SymMat<T, m>& mat) {
  return mat.get_data();
}

template <typename T, int m>
A2D_FUNCTION T* get_data(ADObj<SymMat<T, m>>& mat) {
  return mat.value().get_data();
}

template <typename T, int m>
A2D_FUNCTION T* get_data(A2DObj<SymMat<T, m>>& mat) {
  return mat.value().get_data();
}

template <typename T, int m>
A2D_FUNCTION T* get_data(ADObj<SymMat<T, m>&>& mat) {
  return mat.value().get_data();
}

template <typename T, int m>
A2D_FUNCTION T* get_data(A2DObj<SymMat<T, m>&>& mat) {
  return mat.value().get_data();
}

template <typename T, int n>
A2D_FUNCTION T* get_data(Vec<T, n>& vec) {
  return vec.get_data();
}

template <typename T, int n>
A2D_FUNCTION const T* get_data(const Vec<T, n>& vec) {
  return vec.get_data();
}

template <typename T, int n>
A2D_FUNCTION T* get_data(ADObj<Vec<T, n>>& vec) {
  return vec.value().get_data();
}

template <typename T, int n>
A2D_FUNCTION T* get_data(A2DObj<Vec<T, n>>& vec) {
  return vec.value().get_data();
}

template <typename T, int n>
A2D_FUNCTION T* get_data(ADObj<Vec<T, n>&>& vec) {
  return vec.value().get_data();
}

template <typename T, int n>
A2D_FUNCTION T* get_data(A2DObj<Vec<T, n>&>& vec) {
  return vec.value().get_data();
}

// new ADScalar get_data  (SPE)
template <class T, int N>
struct __is_numeric_type<ADScalar<T, N>> : std::is_floating_point<T> {};

template <class T, int N>
struct __is_numeric_type<ADScalar<A2D_complex_t<T>, N>>
    : std::is_floating_point<T> {};

template <int N>
struct __get_object_numeric_type<ADScalar<double, N>> {
  using type = ADScalar<double, N>;
};

template <int N>
struct __get_object_numeric_type<ADScalar<A2D_complex_t<double>, N>> {
  using type = ADScalar<A2D_complex_t<double>, N>;
};

template <typename T, int N>
ADScalar<T, N>& get_data(ADScalar<T, N>& value) {
  return value;
}

template <typename T, int N>
const ADScalar<T, N>& get_data(const ADScalar<T, N>& value) {
  return value;
}

template <typename T, int N>
A2D_FUNCTION ADScalar<T, N>& get_data(ADObj<ADScalar<T, N>>& value) {
  return value.value();
}

template <typename T, int N>
A2D_FUNCTION const ADScalar<T, N>& get_data(
    const ADObj<ADScalar<T, N>>& value) {
  return value.value();
}

template <typename T, int N>
A2D_FUNCTION ADScalar<T, N>& get_data(A2DObj<ADScalar<T, N>>& value) {
  return value.value();
}

template <typename T, int N>
A2D_FUNCTION const ADScalar<T, N>& get_data(
    const A2DObj<ADScalar<T, N>>& value) {
  return value.value();
}

/**
 * @brief Get data pointers from objects
 */
template <typename T, int N, int m, int n>
A2D_FUNCTION ADScalar<T, N>* get_data(Mat<ADScalar<T, N>, m, n>& mat) {
  return mat.get_data();
}

template <typename T, int N, int m, int n>
A2D_FUNCTION const ADScalar<T, N>* get_data(
    const Mat<ADScalar<T, N>, m, n>& mat) {
  return mat.get_data();
}

template <typename T, int N, int m, int n>
A2D_FUNCTION ADScalar<T, N>* get_data(ADObj<Mat<ADScalar<T, N>, m, n>>& mat) {
  return mat.value().get_data();
}

template <typename T, int N, int m, int n>
A2D_FUNCTION ADScalar<T, N>* get_data(A2DObj<Mat<ADScalar<T, N>, m, n>>& mat) {
  return mat.value().get_data();
}

template <typename T, int N, int m, int n>
A2D_FUNCTION ADScalar<T, N>* get_data(ADObj<Mat<ADScalar<T, N>, m, n>&>& mat) {
  return mat.value().get_data();
}

template <typename T, int N, int m, int n>
A2D_FUNCTION ADScalar<T, N>* get_data(A2DObj<Mat<ADScalar<T, N>, m, n>&>& mat) {
  return mat.value().get_data();
}

template <typename T, int N, int m>
A2D_FUNCTION ADScalar<T, N>* get_data(SymMat<ADScalar<T, N>, m>& mat) {
  return mat.get_data();
}

template <typename T, int N, int m>
A2D_FUNCTION const ADScalar<T, N>* get_data(
    const SymMat<ADScalar<T, N>, m>& mat) {
  return mat.get_data();
}

template <typename T, int N, int m>
A2D_FUNCTION ADScalar<T, N>* get_data(ADObj<SymMat<ADScalar<T, N>, m>>& mat) {
  return mat.value().get_data();
}

template <typename T, int N, int m>
A2D_FUNCTION ADScalar<T, N>* get_data(A2DObj<SymMat<ADScalar<T, N>, m>>& mat) {
  return mat.value().get_data();
}

template <typename T, int N, int m>
A2D_FUNCTION ADScalar<T, N>* get_data(ADObj<SymMat<ADScalar<T, N>, m>&>& mat) {
  return mat.value().get_data();
}

template <typename T, int N, int m>
A2D_FUNCTION ADScalar<T, N>* get_data(A2DObj<SymMat<ADScalar<T, N>, m>&>& mat) {
  return mat.value().get_data();
}

template <typename T, int N, int n>
A2D_FUNCTION ADScalar<T, N>* get_data(Vec<ADScalar<T, N>, n>& vec) {
  return vec.get_data();
}

template <typename T, int N, int n>
A2D_FUNCTION const ADScalar<T, N>* get_data(const Vec<ADScalar<T, N>, n>& vec) {
  return vec.get_data();
}

template <typename T, int N, int n>
A2D_FUNCTION ADScalar<T, N>* get_data(ADObj<Vec<ADScalar<T, N>, n>>& vec) {
  return vec.value().get_data();
}

template <typename T, int N, int n>
A2D_FUNCTION ADScalar<T, N>* get_data(A2DObj<Vec<ADScalar<T, N>, n>>& vec) {
  return vec.value().get_data();
}

template <typename T, int N, int n>
A2D_FUNCTION ADScalar<T, N>* get_data(ADObj<Vec<ADScalar<T, N>, n>&>& vec) {
  return vec.value().get_data();
}

template <typename T, int N, int n>
A2D_FUNCTION ADScalar<T, N>* get_data(A2DObj<Vec<ADScalar<T, N>, n>&>& vec) {
  return vec.value().get_data();
}

}  // namespace A2D

#endif  // A2D_OBJECTS_H
