#ifndef AMIGO_COMPONENT_GROUP_BASE_H
#define AMIGO_COMPONENT_GROUP_BASE_H

#include "csr_matrix.h"
#include "node_owners.h"
#include "vector.h"

namespace amigo {

template <typename T, ExecPolicy policy>
class ComponentGroupBase {
 public:
  virtual ~ComponentGroupBase() {}

  virtual std::shared_ptr<ComponentGroupBase<T, policy>> clone(
      int num_elements, std::shared_ptr<Vector<int>> data_idx,
      std::shared_ptr<Vector<int>> layout_idx,
      std::shared_ptr<Vector<int>> output_idx) const = 0;

  // Update any externally stored values
  virtual void update(const Vector<T>& x) {}

  // Group analysis functions
  virtual T lagrangian(T alpha, const Vector<T>& data,
                       const Vector<T>& x) const {
    return T(0.0);
  }
  virtual void add_gradient(T alpha, const Vector<T>& data, const Vector<T>& x,
                            Vector<T>& g) const {}
  virtual void add_hessian_product(T alpha, const Vector<T>& data,
                                   const Vector<T>& x, const Vector<T>& p,
                                   Vector<T>& h) const {}
  virtual void initialize_hessian_pattern(const NodeOwners& owners,
                                          CSRMat<T>& mat) {}
  virtual void add_hessian(T alpha, const Vector<T>& data, const Vector<T>& x,
                           const NodeOwners& owners, CSRMat<T>& mat) const {}

  // Gradient coupling function
  virtual void add_grad_jac_product_wrt_data(const Vector<T>& data,
                                             const Vector<T>& x,
                                             const Vector<T>& pdata,
                                             Vector<T>& g) const {}
  virtual void add_grad_jac_tproduct_wrt_data(const Vector<T>& data,
                                              const Vector<T>& x,
                                              const Vector<T>& px,
                                              Vector<T>& dx) const {}
  virtual void add_grad_jac_wrt_data(const Vector<T>& data_vec,
                                     const Vector<T>& vec,
                                     const NodeOwners& owners,
                                     CSRMat<T>& jact) const {}

  // Compute the output
  virtual void add_output(const Vector<T>& data, const Vector<T>& x,
                          Vector<T>& output) const {}

  // Functions needed for the post-optimality direct and adjoint methods
  virtual void add_output_jac_wrt_input(const Vector<T>& data,
                                        const Vector<T>& x,
                                        CSRMat<T>& jac) const {}
  virtual void add_output_jac_wrt_data(const Vector<T>& data,
                                       const Vector<T>& x,
                                       CSRMat<T>& jac) const {}

  // Get the information about the non-zero pattern
  virtual void get_csr_data(int* nvars, const int* vars[], int* ncon,
                            const int* cons[], const int* jac_rowp[],
                            const int* jac_cols[], const int* hess_rowp[],
                            const int* hess_cols[]) const {
    if (nvars) {
      *nvars = 0;
    }
    if (vars) {
      *vars = nullptr;
    }
    if (ncon) {
      *ncon = 0;
    }
    if (cons) {
      *cons = nullptr;
    }
    if (jac_rowp) {
      *jac_rowp = nullptr;
    }
    if (jac_cols) {
      *jac_cols = nullptr;
    }
    if (hess_rowp) {
      *hess_rowp = nullptr;
    }
    if (hess_cols) {
      *hess_cols = nullptr;
    }
  }
  virtual void get_data_layout_data(int* num_elements, int* nodes_per_elem,
                                    const int** array) const {
    if (num_elements) {
      *num_elements = 0;
    }
    if (nodes_per_elem) {
      *nodes_per_elem = 0;
    }
    if (array) {
      *array = nullptr;
    }
  }
  virtual void get_layout_data(int* num_elements, int* nodes_per_elem,
                               const int** array) const {
    if (num_elements) {
      *num_elements = 0;
    }
    if (nodes_per_elem) {
      *nodes_per_elem = 0;
    }
    if (array) {
      *array = nullptr;
    }
  }
  virtual void get_output_layout_data(int* num_elements, int* outputs_per_elem,
                                      const int** array) const {
    if (num_elements) {
      *num_elements = 0;
    }
    if (outputs_per_elem) {
      *outputs_per_elem = 0;
    }
    if (array) {
      *array = nullptr;
    }
  }
};

template <typename R, class... Ts>
struct __get_collection_input_type;

template <typename R, class T>
struct __get_collection_input_type<R, T> {
  using type = typename T::template Input<R>;
};

template <typename R, class T, class... Ts>
struct __get_collection_input_type<R, T, Ts...> {
  using type = typename __get_collection_input_type<R, Ts...>::type;
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
  using type = typename T::template Data<R>;
};

template <typename R, class T, class... Ts>
struct __get_collection_data_type<R, T, Ts...> {
  using type = typename __get_collection_data_type<R, Ts...>::type;
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
  using type = typename T::template Output<R>;
};

template <typename R, class T, class... Ts>
struct __get_collection_output_type<R, T, Ts...> {
  using type = typename __get_collection_output_type<R, Ts...>::type;
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
