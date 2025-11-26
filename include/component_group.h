#ifndef AMIGO_COMPONENT_GROUP_H
#define AMIGO_COMPONENT_GROUP_H

#include "a2dcore.h"
#include "component_group_base.h"
#include "csr_matrix.h"
#include "layout.h"
#include "node_owners.h"
#include "ordering_utils.h"
#include "vector.h"

#ifdef AMIGO_USE_CUDA
#include "cuda/component_group_backend.cuh"
#endif

namespace amigo {

namespace detail {

template <typename T, class Data, class Input, class Component, class... Remain>
T compute_lagrangian(T alpha, Data& data, Input& input) {
  T value = 0.0;

  if constexpr (!Component::is_compute_empty) {
    value = Component::lagrange(alpha, data, input);
  }

  if constexpr (sizeof...(Remain) > 0) {
    return value +
           compute_lagrangian<T, Data, Input, Remain...>(alpha, data, input);
  } else {
    return value;
  }
}

template <typename T, class Data, class Input, class Component, class... Remain>
void compute_gradient(T alpha, Data& data, Input& input, Input& grad) {
  if constexpr (!Component::is_compute_empty) {
    Component::gradient(alpha, data, input, grad);
  }

  if constexpr (sizeof...(Remain) > 0) {
    compute_gradient<T, Data, Input, Remain...>(alpha, data, input, grad);
  }
}

template <typename T, class Data, class Input, class Component, class... Remain>
void compute_hessian(T alpha, Data& data, Input& input, Input& dir, Input& grad,
                     Input& h) {
  if constexpr (!Component::is_compute_empty) {
    Component::hessian(alpha, data, input, dir, grad, h);
  }

  if constexpr (sizeof...(Remain) > 0) {
    compute_hessian<T, Data, Input, Remain...>(alpha, data, input, dir, grad,
                                               h);
  }
}

template <typename T, int ncomp, class Input, int ndata, class Data,
          class... Components>
class SerialGroupBackend {
 public:
  template <int noutputs>
  SerialGroupBackend(IndexLayout<ndata>& data_layout,
                     IndexLayout<ncomp>& layout,
                     IndexLayout<noutputs>& outputs) {}

  T lagrangian_kernel(T alpha, const IndexLayout<ndata>& data_layout,
                      const IndexLayout<ncomp>& layout,
                      const Vector<T>& data_vec, const Vector<T>& vec) const {
    T value = 0.0;
    if constexpr (ncomp > 0) {
      int num_elems = layout.get_num_elements();
      for (int i = 0; i < num_elems; i++) {
        Data data;
        Input input;
        data_layout.get_values(i, data_vec, data);
        layout.get_values(i, vec, input);
        value += compute_lagrangian<T, Data, Input, Components...>(alpha, data,
                                                                   input);
      }
    }
    return value;
  }

  void add_gradient_kernel(T alpha, const IndexLayout<ndata>& data_layout,
                           const IndexLayout<ncomp>& layout,
                           const Vector<T>& data_vec, const Vector<T>& vec,
                           Vector<T>& res) const {
    if constexpr (ncomp > 0) {
      int num_elems = layout.get_num_elements();
      for (int i = 0; i < num_elems; i++) {
        Data data;
        Input input, gradient;
        data_layout.get_values(i, data_vec, data);
        layout.get_values(i, vec, input);
        gradient.zero();
        compute_gradient<T, Data, Input, Components...>(alpha, data, input,
                                                        gradient);
        layout.add_values(i, gradient, res);
      }
    }
  }

  void add_hessian_product_kernel(T alpha,
                                  const IndexLayout<ndata>& data_layout,
                                  const IndexLayout<ncomp>& layout,
                                  const Vector<T>& data_vec,
                                  const Vector<T>& vec, const Vector<T>& dir,
                                  Vector<T>& res) const {
    if constexpr (ncomp > 0) {
      int num_elems = layout.get_num_elements();
      for (int i = 0; i < num_elems; i++) {
        Data data;
        Input input, gradient, direction, result;
        data_layout.get_values(i, data_vec, data);
        layout.get_values(i, vec, input);
        layout.get_values(i, dir, direction);
        gradient.zero();
        result.zero();
        compute_hessian<T, Data, Input, Components...>(
            alpha, data, input, direction, gradient, result);
        layout.add_values(i, result, res);
      }
    }
  }

  void initialize_hessian_pattern(const IndexLayout<ncomp>& layout,
                                  const NodeOwners& owners,
                                  CSRMat<T>& mat) const {}

  void add_hessian_kernel(T alpha, const IndexLayout<ndata>& data_layout,
                          const IndexLayout<ncomp>& layout,
                          const Vector<T>& data_vec, const Vector<T>& vec,
                          const NodeOwners& owners, CSRMat<T>& jac) const {
    if constexpr (ncomp > 0) {
      int num_elems = layout.get_num_elements();
      Data data;
      Input input, gradient, direction, result;
      for (int i = 0; i < num_elems; i++) {
        int index[ncomp], index_global[ncomp];
        layout.get_indices(i, index);
        owners.local_to_global(ncomp, index, index_global);

        data_layout.get_values(i, data_vec, data);
        layout.get_values(i, vec, input);

        for (int j = 0; j < ncomp; j++) {
          direction.zero();
          gradient.zero();
          result.zero();

          direction[j] = 1.0;

          compute_hessian<T, Data, Input, Components...>(
              alpha, data, input, direction, gradient, result);

          jac.add_row(index[j], ncomp, index_global, result);
        }
      }
    }
  }

  void add_grad_jac_product_wrt_data_kernel(
      const IndexLayout<ndata>& data_layout, const IndexLayout<ncomp>& layout,
      const Vector<T>& data_vec, const Vector<T>& vec, const Vector<T>& dir,
      Vector<T>& res) const {}

  void add_grad_jac_tproduct_wrt_data_kernel(
      const IndexLayout<ndata>& data_layout, const IndexLayout<ncomp>& layout,
      const Vector<T>& data_vec, const Vector<T>& vec, const Vector<T>& dir,
      Vector<T>& res) const {}

  void add_grad_jac_wrt_data_kernel(const IndexLayout<ndata>& data_layout,
                                    const IndexLayout<ncomp>& layout,
                                    const Vector<T>& data_vec,
                                    const Vector<T>& vec,
                                    const NodeOwners& owners,
                                    CSRMat<T>& jac) const {
    using ADtype = A2D::ADScalar<T, 1>;
    using DataAD =
        typename __get_collection_data_type<ADtype, Components...>::type;
    using InputAD =
        typename __get_collection_input_type<ADtype, Components...>::type;

    if constexpr (ncomp > 0 && ndata > 0) {
      int num_elems = layout.get_num_elements();

      for (int i = 0; i < num_elems; i++) {
        InputAD input;
        layout.get_values(i, vec, input);

        T jac_elem[ncomp * ndata];
        for (int j = 0; j < ndata; j++) {
          DataAD data;
          data_layout.get_values(i, data_vec, data);
          data[j].deriv[0] = 1.0;

          InputAD gradient;
          gradient.zero();

          ADtype alpha(1.0);
          compute_gradient<ADtype, DataAD, InputAD, Components...>(
              alpha, data, input, gradient);

          for (int k = 0; k < ncomp; k++) {
            jac_elem[ndata * k + j] = gradient[k].deriv[0];
          }
        }

        int input_index[ncomp];
        layout.get_indices(i, input_index);

        int data_indices[ndata], data_indices_global[ndata];
        data_layout.get_indices(i, data_indices);
        owners.local_to_global(ndata, data_indices, data_indices_global);

        for (int j = 0; j < ncomp; j++) {
          jac.add_row(input_index[j], ndata, data_indices_global,
                      &jac_elem[ndata * j]);
        }
      }
    }
  }
};

template <typename T, int ncomp, int ndata, int noutputs, class... Components>
class SerialOutputBackend {
 public:
  SerialOutputBackend() {}

  void add_output_kernel(const IndexLayout<ndata>& data_layout,
                         const IndexLayout<ncomp>& layout,
                         const IndexLayout<noutputs>& output_layout,
                         const Vector<T>& data_vec, const Vector<T>& vec,
                         Vector<T>& out) const {
    add_output_kernel<Components...>(data_layout, layout, output_layout,
                                     data_vec, vec, out);
  }

  template <class Component, class... Remain>
  void add_output_kernel(const IndexLayout<ndata>& data_layout,
                         const IndexLayout<ncomp>& layout,
                         const IndexLayout<noutputs>& output_layout,
                         const Vector<T>& data_vec, const Vector<T>& vec,
                         Vector<T>& out) const {
    if constexpr (Component::noutputs > 0) {
      typename Component::template Data<T> data;
      typename Component::template Input<T> input;
      typename Component::template Output<T> output;
      int num_elems = layout.get_num_elements();

      for (int i = 0; i < num_elems; i++) {
        data_layout.get_values(i, data_vec, data);
        layout.get_values(i, vec, input);
        Component::compute_output(data, input, output);
        output_layout.add_values(i, output, out);
      }
    }

    if constexpr (sizeof...(Remain) > 0) {
      add_output_kernel<Remain...>(data_layout, layout, output_layout, data_vec,
                                   vec, out);
    }
  }

  void add_output_jac_wrt_input_kernel(
      const IndexLayout<ndata>& data_layout, const IndexLayout<ncomp>& layout,
      const IndexLayout<noutputs>& output_layout, const Vector<T>& data_vec,
      const Vector<T>& vec, CSRMat<T>& jac) const {
    add_output_jac_wrt_input_kernel<Components...>(
        data_layout, layout, output_layout, data_vec, vec, jac);
  }

  template <class Component, class... Remain>
  void add_output_jac_wrt_input_kernel(
      const IndexLayout<ndata>& data_layout, const IndexLayout<ncomp>& layout,
      const IndexLayout<noutputs>& output_layout, const Vector<T>& data_vec,
      const Vector<T>& vec, CSRMat<T>& jac) const {
    if constexpr (Component::noutputs > 0 && Component::ncomp > 0) {
      typename Component::template Data<A2D::ADScalar<T, 1>> data;
      typename Component::template Input<A2D::ADScalar<T, 1>> input;
      typename Component::template Output<A2D::ADScalar<T, 1>> output;
      int num_elems = layout.get_num_elements();

      for (int i = 0; i < num_elems; i++) {
        data_layout.get_values(i, data_vec, data);

        T jac_elem[noutputs * ncomp];
        for (int j = 0; j < ncomp; j++) {
          layout.get_values(i, vec, input);
          input[j].deriv[0] = 1.0;
          Component::compute_output(data, input, output);

          for (int k = 0; k < noutputs; k++) {
            jac_elem[ncomp * k + j] = output[k].deriv[0];
          }
        }

        int input_index[ncomp];
        layout.get_indices(i, input_index);

        int output_index[noutputs];
        output_layout.get_indices(i, output_index);

        for (int j = 0; j < noutputs; j++) {
          jac.add_row(output_index[j], ncomp, input_index,
                      &jac_elem[ncomp * j]);
        }
      }
    }

    if constexpr (sizeof...(Remain) > 0) {
      add_output_jac_wrt_input_kernel<Remain...>(
          data_layout, layout, output_layout, data_vec, vec, jac);
    }
  }

  void add_output_jac_wrt_data_kernel(
      const IndexLayout<ndata>& data_layout, const IndexLayout<ncomp>& layout,
      const IndexLayout<noutputs>& output_layout, const Vector<T>& data_vec,
      const Vector<T>& vec, CSRMat<T>& jac) const {
    add_output_jac_wrt_data_kernel<Components...>(
        data_layout, layout, output_layout, data_vec, vec, jac);
  }

  template <class Component, class... Remain>
  void add_output_jac_wrt_data_kernel(
      const IndexLayout<ndata>& data_layout, const IndexLayout<ncomp>& layout,
      const IndexLayout<noutputs>& output_layout, const Vector<T>& data_vec,
      const Vector<T>& vec, CSRMat<T>& jac) const {
    if constexpr (Component::noutputs > 0 && Component::ndata > 0) {
      typename Component::template Data<A2D::ADScalar<T, 1>> data;
      typename Component::template Input<A2D::ADScalar<T, 1>> input;
      typename Component::template Output<A2D::ADScalar<T, 1>> output;
      int num_elems = layout.get_num_elements();

      for (int i = 0; i < num_elems; i++) {
        layout.get_values(i, vec, input);

        T jac_elem[noutputs * ndata];
        for (int j = 0; j < ndata; j++) {
          data_layout.get_values(i, data_vec, data);
          data[j].deriv[0] = 1.0;
          Component::compute_output(data, input, output);

          for (int k = 0; k < noutputs; k++) {
            jac_elem[ncomp * k + j] = output[k].deriv[0];
          }
        }

        int data_index[ndata];
        data_layout.get_indices(i, data_index);

        int output_index[noutputs];
        output_layout.get_indices(i, output_index);

        for (int j = 0; j < noutputs; j++) {
          jac.add_row(output_index[j], ndata, data_index, &jac_elem[ndata * j]);
        }
      }
    }

    if constexpr (sizeof...(Remain) > 0) {
      add_output_jac_wrt_data_kernel<Remain...>(
          data_layout, layout, output_layout, data_vec, vec, jac);
    }
  }
};

template <typename T, int ncomp, int ndata, int noutput, class... Components>
using DefaultOutputBackend =
    SerialOutputBackend<T, ncomp, ndata, noutput, Components...>;

template <typename T, int ncomp, class Input, int ndata, class Data,
          class... Components>
class OmpGroupBackend {
 public:
  template <int noutputs>
  OmpGroupBackend(IndexLayout<ndata>& data_layout, IndexLayout<ncomp>& layout,
                  IndexLayout<noutputs>& output_layout)
      : num_colors(0), elem_per_color(nullptr) {
    int num_elems, ncomps;
    const int* array;
    layout.get_data(&num_elems, &ncomps, &array);

    // Create a coloring for the layout
    int* elem_by_color_ptr;
    int* elem_by_color;
    OrderingUtils::color_elements(num_elems, ncomp, array, &num_colors,
                                  &elem_by_color_ptr, &elem_by_color);

    // Re-order the data and the variable layouts
    data_layout.reorder(elem_by_color);
    layout.reorder(elem_by_color);
    output_layout.reorder(elem_by_color);

    elem_per_color = new int[num_colors];
    for (int i = 0; i < num_colors; i++) {
      elem_per_color[i] = elem_by_color_ptr[i + 1] - elem_by_color_ptr[i];
    }

    delete[] elem_by_color_ptr;
    delete[] elem_by_color;
  }
  ~OmpGroupBackend() { delete[] elem_per_color; }

  T lagrangian_kernel(T alpha, const IndexLayout<ndata>& data_layout,
                      const IndexLayout<ncomp>& layout,
                      const Vector<T>& data_vec, const Vector<T>& vec) const {
    T value = 0.0;
    if constexpr (ncomp > 0) {
      int num_elems = layout.get_num_elements();

#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for reduction(+ : value)
#endif
      for (int i = 0; i < num_elems; i++) {
        Data data;
        Input input;
        data_layout.get_values(i, data_vec, data);
        layout.get_values(i, vec, input);
        value += compute_lagrangian<T, Data, Input, Components...>(alpha, data,
                                                                   input);
      }
    }

    return value;
  }

  void add_gradient_kernel(T alpha, const IndexLayout<ndata>& data_layout,
                           const IndexLayout<ncomp>& layout,
                           const Vector<T>& data_vec, const Vector<T>& vec,
                           Vector<T>& res) const {
    if constexpr (ncomp > 0) {
      int end = 0;
      for (int j = 0; j < num_colors; j++) {
        int start = end;
        end = start + elem_per_color[j];
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif
        for (int elem = start; elem < end; elem++) {
          Data data;
          Input input, gradient;
          data_layout.get_values(elem, data_vec, data);
          layout.get_values(elem, vec, input);
          gradient.zero();
          compute_gradient<T, Data, Input, Components...>(alpha, data, input,
                                                          gradient);
          layout.add_values(elem, gradient, res);
        }
      }
    }
  }

  void add_hessian_product_kernel(T alpha,
                                  const IndexLayout<ndata>& data_layout,
                                  const IndexLayout<ncomp>& layout,
                                  const Vector<T>& data_vec,
                                  const Vector<T>& vec, const Vector<T>& dir,
                                  Vector<T>& res) const {
    if constexpr (ncomp > 0) {
      int end = 0;
      for (int j = 0; j < num_colors; j++) {
        int start = end;
        end = start + elem_per_color[j];
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif
        for (int elem = start; elem < end; elem++) {
          Data data;
          Input input, gradient, direction, result;
          data_layout.get_values(elem, data_vec, data);
          layout.get_values(elem, vec, input);
          layout.get_values(elem, dir, direction);
          gradient.zero();
          result.zero();
          compute_hessian<T, Data, Input, Components...>(
              alpha, data, input, direction, gradient, result);
          layout.add_values(elem, result, res);
        }
      }
    }
  }

  void initialize_hessian_pattern(const IndexLayout<ncomp>& layout,
                                  const NodeOwners& owners, CSRMat<T>& mat) {}

  void add_hessian_kernel(T alpha, const IndexLayout<ndata>& data_layout,
                          const IndexLayout<ncomp>& layout,
                          const Vector<T>& data_vec, const Vector<T>& vec,
                          const NodeOwners& owners, CSRMat<T>& jac) const {
    if constexpr (ncomp > 0) {
      int end = 0;
      for (int j = 0; j < num_colors; j++) {
        int start = end;
        end = start + elem_per_color[j];
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif
        for (int elem = start; elem < end; elem++) {
          Data data;
          Input input, gradient, direction, result;
          int index[ncomp], index_global[ncomp];
          layout.get_indices(elem, index);
          owners.local_to_global(ncomp, index, index_global);

          data_layout.get_values(elem, data_vec, data);
          layout.get_values(elem, vec, input);

          for (int k = 0; k < ncomp; k++) {
            direction.zero();
            gradient.zero();
            result.zero();

            direction[k] = 1.0;

            compute_hessian<T, Data, Input, Components...>(
                alpha, data, input, direction, gradient, result);

            jac.add_row(index[k], ncomp, index_global, result);
          }
        }
      }
    }
  }

  void add_grad_jac_product_wrt_data_kernel(
      const IndexLayout<ndata>& data_layout, const IndexLayout<ncomp>& layout,
      const Vector<T>& data_vec, const Vector<T>& vec, const Vector<T>& dir,
      Vector<T>& res) const {}

  void add_grad_jac_tproduct_wrt_data_kernel(
      const IndexLayout<ndata>& data_layout, const IndexLayout<ncomp>& layout,
      const Vector<T>& data_vec, const Vector<T>& vec, const Vector<T>& dir,
      Vector<T>& res) const {}

  void add_grad_jac_wrt_data_kernel(const IndexLayout<ndata>& data_layout,
                                    const IndexLayout<ncomp>& layout,
                                    const Vector<T>& data_vec,
                                    const Vector<T>& vec,
                                    const NodeOwners& owners,
                                    CSRMat<T>& jac) const {
    using ADtype = A2D::ADScalar<T, 1>;
    using DataAD =
        typename __get_collection_data_type<ADtype, Components...>::type;
    using InputAD =
        typename __get_collection_input_type<ADtype, Components...>::type;

    if constexpr (ncomp > 0 && ndata > 0) {
      int end = 0;
      for (int j = 0; j < num_colors; j++) {
        int start = end;
        end = start + elem_per_color[j];
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif
        for (int elem = start; elem < end; elem++) {
          InputAD input;
          layout.get_values(elem, vec, input);

          T jac_elem[ncomp * ndata];
          for (int i = 0; i < ndata; i++) {
            DataAD data;
            data_layout.get_values(elem, data_vec, data);
            data[i].deriv[0] = 1.0;

            InputAD gradient;
            gradient.zero();

            ADtype alpha(1.0);
            compute_gradient<ADtype, DataAD, InputAD, Components...>(
                alpha, data, input, gradient);

            for (int k = 0; k < ncomp; k++) {
              jac_elem[ndata * k + i] = gradient[k].deriv[0];
            }
          }

          int input_index[ncomp];
          layout.get_indices(elem, input_index);

          int data_indices[ndata], data_indices_global[ndata];
          data_layout.get_indices(elem, data_indices);
          owners.local_to_global(ndata, data_indices, data_indices_global);

          for (int i = 0; i < ncomp; i++) {
            jac.add_row(input_index[i], ndata, data_indices_global,
                        &jac_elem[ndata * i]);
          }
        }
      }
    }
  }

 private:
  int num_colors;
  int* elem_per_color;
};

// Set up a struct to select the backend type depending on policy
template <ExecPolicy policy, typename T, int ncomp, typename Input, int ndata,
          typename Data, typename... Components>
struct GroupBackendSelector;

// SERIAL specialization
template <typename T, int ncomp, typename Input, int ndata, typename Data,
          typename... Components>
struct GroupBackendSelector<ExecPolicy::SERIAL, T, ncomp, Input, ndata, Data,
                            Components...> {
  using type = SerialGroupBackend<T, ncomp, Input, ndata, Data, Components...>;
};

// OPENMP specialization
template <typename T, int ncomp, typename Input, int ndata, typename Data,
          typename... Components>
struct GroupBackendSelector<ExecPolicy::OPENMP, T, ncomp, Input, ndata, Data,
                            Components...> {
  using type = OmpGroupBackend<T, ncomp, Input, ndata, Data, Components...>;
};

#ifdef AMIGO_USE_CUDA
// CUDA specialization only when enabled
template <typename T, int ncomp, typename Input, int ndata, typename Data,
          typename... Components>
struct GroupBackendSelector<ExecPolicy::CUDA, T, ncomp, Input, ndata, Data,
                            Components...> {
  using type = CudaGroupBackend<T, ncomp, Input, ndata, Data, Components...>;
};
#endif

}  // namespace detail

template <typename T, ExecPolicy policy, class... Components>
class ComponentGroup : public ComponentGroupBase<T, policy> {
 public:
  static constexpr int num_components = sizeof...(Components);

  using Input = typename __get_collection_input_type<T, Components...>::type;
  using Data = typename __get_collection_data_type<T, Components...>::type;

  static constexpr int ncomp = __get_collection_ncomp<Components...>::value;
  static constexpr int ndata = __get_collection_ndata<Components...>::value;
  static constexpr int noutputs =
      __get_collection_noutputs<Components...>::value;

  // Use whatever class is defined as the default backend
  using Backend =
      typename detail::GroupBackendSelector<policy, T, ncomp, Input, ndata,
                                            Data, Components...>::type;

  using OutputBackend =
      detail::DefaultOutputBackend<T, ncomp, ndata, noutputs, Components...>;

  ComponentGroup(int num_elements, std::shared_ptr<Vector<int>> data_indices,
                 std::shared_ptr<Vector<int>> indices,
                 std::shared_ptr<Vector<int>> output_indices)
      : data_layout(num_elements, data_indices),
        layout(num_elements, indices),
        output_layout(num_elements, output_indices),
        backend(data_layout, layout, output_layout) {}

  std::shared_ptr<ComponentGroupBase<T, policy>> clone(
      int num_elements, std::shared_ptr<Vector<int>> data_idx,
      std::shared_ptr<Vector<int>> layout_idx,
      std::shared_ptr<Vector<int>> output_idx) const {
    return std::make_shared<ComponentGroup<T, policy, Components...>>(
        num_elements, data_idx, layout_idx, output_idx);
  }

  // Group compute functions
  T lagrangian(T alpha, const Vector<T>& data_vec, const Vector<T>& vec) const {
    return backend.lagrangian_kernel(alpha, data_layout, layout, data_vec, vec);
  }
  void add_gradient(T alpha, const Vector<T>& data_vec, const Vector<T>& vec,
                    Vector<T>& res) const {
    backend.add_gradient_kernel(alpha, data_layout, layout, data_vec, vec, res);
  }
  void add_hessian_product(T alpha, const Vector<T>& data_vec,
                           const Vector<T>& vec, const Vector<T>& dir,
                           Vector<T>& res) const {
    backend.add_hessian_product_kernel(alpha, data_layout, layout, data_vec,
                                       vec, dir, res);
  }
  void initialize_hessian_pattern(const NodeOwners& owners, CSRMat<T>& mat) {
    backend.initialize_hessian_pattern(layout, owners, mat);
  }
  void add_hessian(T alpha, const Vector<T>& data_vec, const Vector<T>& vec,
                   const NodeOwners& owners, CSRMat<T>& jac) const {
    backend.add_hessian_kernel(alpha, data_layout, layout, data_vec, vec,
                               owners, jac);
  }

  // Gradient coupling function
  void add_grad_jac_product_wrt_data(const Vector<T>& data_vec,
                                     const Vector<T>& vec,
                                     const Vector<T>& pdata,
                                     Vector<T>& grad) const {
    backend.add_grad_jac_product_wrt_data_kernel(data_layout, layout, data_vec,
                                                 vec, pdata, grad);
  }
  void add_grad_jac_tproduct_wrt_data(const Vector<T>& data_vec,
                                      const Vector<T>& vec,
                                      const Vector<T>& pvec,
                                      Vector<T>& pdata) const {
    backend.add_grad_jac_tproduct_wrt_data_kernel(data_layout, layout, data_vec,
                                                  vec, pvec, pdata);
  }
  void add_grad_jac_wrt_data(const Vector<T>& data_vec, const Vector<T>& vec,
                             const NodeOwners& owners, CSRMat<T>& jac) const {
    backend.add_grad_jac_wrt_data_kernel(data_layout, layout, data_vec, vec,
                                         owners, jac);
  }

  // Compute the output as a function of the input and data
  void add_output(const Vector<T>& data_vec, const Vector<T>& vec,
                  Vector<T>& output) const {
    output_backend.add_output_kernel(data_layout, layout, output_layout,
                                     data_vec, vec, output);
  }
  void add_output_jac_wrt_input(const Vector<T>& data_vec, const Vector<T>& vec,
                                CSRMat<T>& jac) const {
    output_backend.add_output_jac_wrt_input_kernel(
        data_layout, layout, output_layout, data_vec, vec, jac);
  }
  void add_output_jac_wrt_data(const Vector<T>& data_vec, const Vector<T>& vec,
                               CSRMat<T>& jac) const {
    output_backend.add_output_jac_wrt_data_kernel(
        data_layout, layout, output_layout, data_vec, vec, jac);
  }

  // Get the ordering information
  void get_data_layout_data(int* num_elements, int* nodes_per_elem,
                            const int** array) const {
    data_layout.get_data(num_elements, nodes_per_elem, array);
  }
  void get_layout_data(int* num_elements, int* nodes_per_elem,
                       const int** array) const {
    layout.get_data(num_elements, nodes_per_elem, array);
  }
  void get_output_layout_data(int* num_elements, int* outputs_per_elem,
                              const int** array) const {
    output_layout.get_data(num_elements, outputs_per_elem, array);
  }

 private:
  IndexLayout<ndata> data_layout;
  IndexLayout<ncomp> layout;
  IndexLayout<noutputs> output_layout;

  Backend backend;
  OutputBackend output_backend;
};

}  // namespace amigo

#endif  // AMIGO_COMPONENT_GROUP_H
