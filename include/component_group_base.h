#ifndef AMIGO_COMPONENT_GROUP_BASE_H
#define AMIGO_COMPONENT_GROUP_BASE_H

#include "csr_matrix.h"
#include "node_owners.h"
#include "vector.h"

namespace amigo {

template <typename T>
class ComponentGroupBase {
 public:
  virtual ~ComponentGroupBase() {}

  virtual std::shared_ptr<ComponentGroupBase<T>> clone(
      int num_elements, std::shared_ptr<Vector<int>> data_idx,
      std::shared_ptr<Vector<int>> layout_idx,
      std::shared_ptr<Vector<int>> output_idx) const = 0;

  // Group analysis functions
  virtual T lagrangian(const Vector<T>& data, const Vector<T>& x) const {
    return T(0.0);
  }
  virtual void add_gradient(const Vector<T>& data, const Vector<T>& x,
                            Vector<T>& g) const {}
  virtual void add_hessian_product(const Vector<T>& data, const Vector<T>& x,
                                   const Vector<T>& p, Vector<T>& h) const {}
  virtual void add_hessian(const Vector<T>& data, const Vector<T>& x,
                           const NodeOwners& owners, CSRMat<T>& mat) const {}

  // Output functions
  virtual void add_output(const Vector<T>& data, const Vector<T>& vec,
                          Vector<T>& output) const {}
  virtual void add_input_jacobian(const Vector<T>& data, const Vector<T>& vec,
                                  CSRMat<T>& jac) const {}
  virtual void add_data_jacobian(const Vector<T>& data, const Vector<T>& vec,
                                 CSRMat<T>& jac) const {}

  // Ordering information
  virtual void get_data_layout_data(int* num_elements, int* nodes_per_elem,
                                    const int** array) const {}
  virtual void get_layout_data(int* num_elements, int* nodes_per_elem,
                               const int** array) const {}
  virtual void get_output_layout_data(int* num_elements, int* outputs_per_elem,
                                      const int** array) const {}
};

template <typename R, class... Ts>
struct __get_collection_input_type;

template <typename R, class T>
struct __get_collection_input_type<R, T> {
  using Input = typename T::template Input<R>;
};

template <typename R, class T, class... Ts>
struct __get_collection_input_type<R, T, Ts...> {
  using Input = typename __get_collection_input_type<R, Ts...>::Input;
};

template <class... Ts>
struct __get_collection_ncomp;

template <class T>
struct __get_collection_ncomp<T> {
  static constexpr int value = T::ncomp;
};

template <class T, class... Ts>
struct __get_collection_ncomp<T, Ts...> {
  static constexpr int value = __get_collection_ncomp<Ts...>::value;
};

template <typename R, class... Ts>
struct __get_collection_data_type;

template <typename R, class T>
struct __get_collection_data_type<R, T> {
  using Data = typename T::template Data<R>;
};

template <typename R, class T, class... Ts>
struct __get_collection_data_type<R, T, Ts...> {
  using Data = typename __get_collection_data_type<R, Ts...>::Data;
};

template <class... Ts>
struct __get_collection_ndata;

template <class T>
struct __get_collection_ndata<T> {
  static constexpr int value = T::ndata;
};

template <class T, class... Ts>
struct __get_collection_ndata<T, Ts...> {
  static constexpr int value = __get_collection_ndata<Ts...>::value;
};

template <typename R, class... Ts>
struct __get_collection_output_type;

template <typename R, class T>
struct __get_collection_output_type<R, T> {
  using Output = typename T::template Output<R>;
};

template <typename R, class T, class... Ts>
struct __get_collection_output_type<R, T, Ts...> {
  using Output = typename __get_collection_output_type<R, Ts...>::Output;
};

template <class... Ts>
struct __get_collection_noutputs;

template <class T>
struct __get_collection_noutputs<T> {
  static constexpr int value = T::noutputs;
};

template <class T, class... Ts>
struct __get_collection_noutputs<T, Ts...> {
  static constexpr int value = __get_collection_noutputs<Ts...>::value;
};
}  // namespace amigo

#endif  // AMIGO_COMPONENT_GROUP_BASE_H
