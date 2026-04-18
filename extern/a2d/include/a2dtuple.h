#ifndef A2D_TUPLE_H
#define A2D_TUPLE_H

#include <type_traits>
#include <utility>

#include "a2ddefs.h"

namespace A2D {

template <class _Tp>
A2D_FUNCTION inline constexpr _Tp &&a2d_forward(
    typename std::remove_reference<_Tp>::type &__t) noexcept {
  return static_cast<_Tp &&>(__t);
}

/*
 * The implementation of the tuple below is adopted from the following article:
 * https://medium.com/@mortificador/implementing-std-tuple-in-c-17-3cc5c6da7277
 *
 * This custom tuple is useful when using a2d on CUDA becasue std::tuple is not
 * supported
 * */
// Actual implementation for a type
template <std::size_t _index, typename T>
class _tuple_impl {
  using value_type =
      typename std::remove_const<typename std::remove_reference<T>::type>::type;

 public:
  A2D_FUNCTION _tuple_impl(value_type &v) : val(v) {}

  A2D_FUNCTION _tuple_impl(value_type &&v) : val(std::move(v)) {}

  A2D_FUNCTION _tuple_impl() : val(T{}) {}

  A2D_FUNCTION value_type &a2d_get() { return val; }
  A2D_FUNCTION const value_type &a2d_get() const { return val; }

 private:
  T val;
};

// general template, will be used only when there is no arguments
template <std::size_t _index, typename... types>
class _tuple_recurr_base {};

// This is a partial specialization, so as long as there is at least one
// argument this specialization is preferred to the
// _tuple_recurr_base<std::size_t, typename ...types>
template <std::size_t _index, typename L, typename... types>
class _tuple_recurr_base<_index, L, types...>
    // : public _tuple_impl<_index, typename std::remove_reference<L>::type>,
    : public _tuple_impl<_index, L>,
      public _tuple_recurr_base<_index + 1, types...> {
 public:
  // Default Constructor that takes in no objects
  A2D_FUNCTION _tuple_recurr_base()
      // : _tuple_impl<_index, typename std::remove_reference<L>::type>(),
      : _tuple_impl<_index, L>(), _tuple_recurr_base<_index + 1, types...>() {}

  template <typename CL, typename... CArgs>
  A2D_FUNCTION _tuple_recurr_base(CL &&arg, CArgs &&...args)
      // : _tuple_impl<_index, typename std::remove_reference<CL>::type>(
      : _tuple_impl<_index,
                    typename std::conditional<
                        std::is_reference<L>::value, CL,
                        typename std::remove_reference<CL>::type>::type>(arg),
        _tuple_recurr_base<_index + 1, types...>(args...) {}
};

template <typename L, typename... types>
class a2d_tuple : public _tuple_recurr_base<0, L, types...> {
 public:
  // Default Constructor that takes in no objects
  A2D_FUNCTION a2d_tuple() : _tuple_recurr_base<0, L, types...>() {}

  // The constructor uses the same recursion as the inheritance
  template <typename... CArgs>
  A2D_FUNCTION a2d_tuple(CArgs &&...args)
      : _tuple_recurr_base<0, L, types...>(a2d_forward<CArgs>(args)...) {}
  //
};

// template deduction guideline
template <typename... CArgs>
a2d_tuple(CArgs... args) -> a2d_tuple<CArgs...>;

// extract_type_at is a class that, given a list of types and an index, defines
// a type member with the type of the index given from the list (zero based
// index). E.g. extract<1, int, double, float>::type == double For this we
// define ::type recursively, until we hit index zero, at that point there is a
// specialization that defines the member ::type, and stops the recursion
template <std::size_t index, typename L, typename... Args>
struct extract_type_at {
  using type = typename extract_type_at<index - 1, Args...>::type;
};

// This is the stop type. If the index is zero, we define the member type to be
// the correspondent type
template <typename L, typename... Args>
struct extract_type_at<0, L, Args...> {
  using type = L;
};

// Method to get the value of a tuple, given an index
// We cast the tuple to the base class that corresponds to the index
// and type for that index
template <std::size_t index, typename... Args>
A2D_FUNCTION auto &a2d_get(a2d_tuple<Args...> &t) {
  return (static_cast<_tuple_impl<
              index, typename extract_type_at<index, Args...>::type> &>(t))
      .a2d_get();
}
template <std::size_t index, typename... Args>
A2D_FUNCTION const auto &a2d_get(const a2d_tuple<Args...> &t) {
  return (static_cast<const _tuple_impl<
              index, typename extract_type_at<index, Args...>::type> &>(t))
      .a2d_get();
}

}  // namespace A2D

#endif  //  A2D_TUPLE_H
