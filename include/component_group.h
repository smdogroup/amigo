#ifndef AMIGO_COMPONENT_GROUP_H
#define AMIGO_COMPONENT_GROUP_H

#include "a2dcore.h"
#include "component_group_base.h"
#include "csr_matrix.h"
#include "layout.h"
#include "node_owners.h"
#include "ordering_utils.h"
#include "vector.h"

namespace amigo {

template <typename T, int ncomp, class Input, int ndata, class Data,
          class... Components>
class SerialGroupBackend {
 public:
  template <int noutputs>
  SerialGroupBackend(IndexLayout<ndata> &data_layout,
                     IndexLayout<ncomp> &layout,
                     IndexLayout<noutputs> &outputs) {}

  T lagrangian_kernel(const IndexLayout<ndata> &data_layout,
                      const IndexLayout<ncomp> &layout,
                      const Vector<T> &data_vec, const Vector<T> &vec) const {
    return lagrangian_kernel<Components...>(data_layout, layout, data_vec, vec);
  }

  template <class Component, class... Remain>
  T lagrangian_kernel(const IndexLayout<ndata> &data_layout,
                      const IndexLayout<ncomp> &layout,
                      const Vector<T> &data_vec, const Vector<T> &vec) const {
    Data data;
    Input input;
    T value = 0.0;
    int num_elems = layout.get_num_elements();

    for (int i = 0; i < num_elems; i++) {
      data_layout.get_values(i, data_vec, data);
      layout.get_values(i, vec, input);
      value += Component::lagrange(data, input);
    }

    if constexpr (sizeof...(Remain) > 0) {
      return value +
             lagrangian_kernel<Remain...>(data_layout, layout, data_vec, vec);
    } else {
      return value;
    }
  }

  void add_gradient_kernel(const IndexLayout<ndata> &data_layout,
                           const IndexLayout<ncomp> &layout,
                           const Vector<T> &data_vec, const Vector<T> &vec,
                           Vector<T> &res) const {
    add_gradient_kernel<Components...>(data_layout, layout, data_vec, vec, res);
  }

  template <class Component, class... Remain>
  void add_gradient_kernel(const IndexLayout<ndata> &data_layout,
                           const IndexLayout<ncomp> &layout,
                           const Vector<T> &data_vec, const Vector<T> &vec,
                           Vector<T> &res) const {
    Data data;
    Input input, gradient;
    for (int i = 0; i < layout.get_num_elements(); i++) {
      data_layout.get_values(i, data_vec, data);
      gradient.zero();
      layout.get_values(i, vec, input);
      Component::gradient(data, input, gradient);
      layout.add_values(i, gradient, res);
    }

    if constexpr (sizeof...(Remain) > 0) {
      add_gradient_kernel<Remain...>(data_layout, layout, data_vec, vec, res);
    }
  }

  void add_hessian_product_kernel(const IndexLayout<ndata> &data_layout,
                                  const IndexLayout<ncomp> &layout,
                                  const Vector<T> &data_vec,
                                  const Vector<T> &vec, const Vector<T> &dir,
                                  Vector<T> &res) const {
    add_hessian_product_kernel<Components...>(data_layout, layout, data_vec,
                                              vec, dir, res);
  }

  template <class Component, class... Remain>
  void add_hessian_product_kernel(const IndexLayout<ndata> &data_layout,
                                  const IndexLayout<ncomp> &layout,
                                  const Vector<T> &data_vec,
                                  const Vector<T> &vec, const Vector<T> &dir,
                                  Vector<T> &res) const {
    Data data;
    Input input, gradient, direction, result;
    for (int i = 0; i < layout.get_num_elements(); i++) {
      data_layout.get_values(i, data_vec, data);
      gradient.zero();
      result.zero();
      layout.get_values(i, vec, input);
      layout.get_values(i, dir, direction);
      Component::hessian(data, input, direction, gradient, result);
      layout.add_values(i, result, res);
    }

    if constexpr (sizeof...(Remain) > 0) {
      add_hessian_product_kernel<Remain...>(data_layout, layout, data_vec, vec,
                                            dir, res);
    }
  }

  void add_hessian_kernel(const IndexLayout<ndata> &data_layout,
                          const IndexLayout<ncomp> &layout,
                          const Vector<T> &data_vec, const Vector<T> &vec,
                          const NodeOwners &owners, CSRMat<T> &jac) const {
    add_hessian_kernel<Components...>(data_layout, layout, data_vec, vec,
                                      owners, jac);
  }

  template <class Component, class... Remain>
  void add_hessian_kernel(const IndexLayout<ndata> &data_layout,
                          const IndexLayout<ncomp> &layout,
                          const Vector<T> &data_vec, const Vector<T> &vec,
                          const NodeOwners &owners, CSRMat<T> &jac) const {
    Data data;
    Input input, gradient, direction, result;
    for (int i = 0; i < layout.get_num_elements(); i++) {
      int index[ncomp], index_global[ncomp];
      layout.get_indices(i, index);
      owners.local_to_global(ncomp, index, index_global);

      data_layout.get_values(i, data_vec, data);
      layout.get_values(i, vec, input);

      for (int j = 0; j < Component::ncomp; j++) {
        direction.zero();
        gradient.zero();
        result.zero();

        direction[j] = 1.0;

        Component::hessian(data, input, direction, gradient, result);

        jac.add_row(index[j], Component::ncomp, index_global, result);
      }
    }

    if constexpr (sizeof...(Remain) > 0) {
      add_hessian_kernel<Remain...>(data_layout, layout, data_vec, vec, owners,
                                    jac);
    }
  }
};

template <typename T, int ncomp, int ndata, int noutputs, class... Components>
class SerialOutputBackend {
 public:
  SerialOutputBackend() {}

  void add_output_kernel(const IndexLayout<ndata> &data_layout,
                         const IndexLayout<ncomp> &layout,
                         const IndexLayout<noutputs> &output_layout,
                         const Vector<T> &data_vec, const Vector<T> &vec,
                         Vector<T> &out) const {
    add_output_kernel<Components...>(data_layout, layout, output_layout,
                                     data_vec, vec, out);
  }

  template <class Component, class... Remain>
  void add_output_kernel(const IndexLayout<ndata> &data_layout,
                         const IndexLayout<ncomp> &layout,
                         const IndexLayout<noutputs> &output_layout,
                         const Vector<T> &data_vec, const Vector<T> &vec,
                         Vector<T> &out) const {
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

  void add_input_jacobian_kernel(const IndexLayout<ndata> &data_layout,
                                 const IndexLayout<ncomp> &layout,
                                 const IndexLayout<noutputs> &output_layout,
                                 const Vector<T> &data_vec,
                                 const Vector<T> &vec, CSRMat<T> &jac) const {
    add_input_jacobian_kernel<Components...>(data_layout, layout, output_layout,
                                             data_vec, vec, jac);
  }

  template <class Component, class... Remain>
  void add_input_jacobian_kernel(const IndexLayout<ndata> &data_layout,
                                 const IndexLayout<ncomp> &layout,
                                 const IndexLayout<noutputs> &output_layout,
                                 const Vector<T> &data_vec,
                                 const Vector<T> &vec, CSRMat<T> &jac) const {
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
            jac_elem[ncomp * k + i] = output[k].deriv[0];
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
      add_input_jacobian_kernel<Remain...>(data_layout, layout, output_layout,
                                           data_vec, vec, jac);
    }
  }

  void add_data_jacobian_kernel(const IndexLayout<ndata> &data_layout,
                                const IndexLayout<ncomp> &layout,
                                const IndexLayout<noutputs> &output_layout,
                                const Vector<T> &data_vec, const Vector<T> &vec,
                                CSRMat<T> &jac) const {
    add_data_jacobian_kernel<Components...>(data_layout, layout, output_layout,
                                            data_vec, vec, jac);
  }

  template <class Component, class... Remain>
  void add_data_jacobian_kernel(const IndexLayout<ndata> &data_layout,
                                const IndexLayout<ncomp> &layout,
                                const IndexLayout<noutputs> &output_layout,
                                const Vector<T> &data_vec, const Vector<T> &vec,
                                CSRMat<T> &jac) const {
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
            jac_elem[ncomp * k + i] = output[k].deriv[0];
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
      add_data_jacobian_kernel<Remain...>(data_layout, layout, output_layout,
                                          data_vec, vec, jac);
    }
  }
};

template <typename T, int ncomp, int ndata, int noutput, class... Components>
using DefaultOutputBackend =
    SerialOutputBackend<T, ncomp, ndata, noutput, Components...>;

#ifdef AMIGO_USE_OPENMP

template <typename T, int ncomp, class Input, int ndata, class Data,
          class... Components>
class OmpGroupBackend {
 public:
  OmpGroupBackend(IndexLayout<ndata> &data_layout, IndexLayout<ncomp> &layout,
                  IndexLayout<noutputs> &output_layout)
      : num_colors(0), elem_per_color(nullptr) {
    int num_elems, ncomps;
    const int *array;
    layout.get_data(&num_elems, &ncomps, &array);

    // Create a coloring for the layout
    int *elem_by_color_ptr;
    int *elem_by_color;
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

  T lagrangian_kernel(const IndexLayout<ndata> &data_layout,
                      const IndexLayout<ncomp> &layout,
                      const Vector<T> &data_vec, const Vector<T> &vec) const {
    return lagrangian_kernel<Components...>(data_layout, layout, data_vec, vec);
  }

  template <class Component, class... Remain>
  T lagrangian_kernel(const IndexLayout<ndata> &data_layout,
                      const IndexLayout<ncomp> &layout,
                      const Vector<T> &data_vec, const Vector<T> &vec) const {
    T value = 0.0;
    int num_elems = layout.get_num_elements();

#pragma omp parallel for reduction(+ : value)
    for (int i = 0; i < num_elems; i++) {
      Data data;
      Input input;
      data_layout.get_values(i, data_vec, data);
      layout.get_values(i, vec, input);
      value += Component::lagrange(data, input);
    }

    if constexpr (sizeof...(Remain) > 0) {
      return value +
             lagrangian_kernel<Remain...>(data_layout, layout, data_vec, vec);
    } else {
      return value;
    }
  }

  void add_gradient_kernel(const IndexLayout<ndata> &data_layout,
                           const IndexLayout<ncomp> &layout,
                           const Vector<T> &data_vec, const Vector<T> &vec,
                           Vector<T> &res) const {
    add_gradient_kernel<Components...>(data_layout, layout, data_vec, vec, res);
  }

  template <class Component, class... Remain>
  void add_gradient_kernel(const IndexLayout<ndata> &data_layout,
                           const IndexLayout<ncomp> &layout,
                           const Vector<T> &data_vec, const Vector<T> &vec,
                           Vector<T> &res) const {
    int end = 0;
    for (int j = 0; j < num_colors; j++) {
      int start = end;
      end = start + elem_per_color[j];
#pragma omp parallel for
      for (int elem = start; elem < end; elem++) {
        Data data;
        Input input, gradient;

        data_layout.get_values(elem, data_vec, data);
        gradient.zero();
        layout.get_values(elem, vec, input);
        Component::gradient(data, input, gradient);
        layout.add_values(elem, gradient, res);
      }
    }

    if constexpr (sizeof...(Remain) > 0) {
      add_gradient_kernel<Remain...>(data_layout, layout, data_vec, vec, res);
    }
  }

  void add_hessian_product_kernel(const IndexLayout<ndata> &data_layout,
                                  const IndexLayout<ncomp> &layout,
                                  const Vector<T> &data_vec,
                                  const Vector<T> &vec, const Vector<T> &dir,
                                  Vector<T> &res) const {
    add_hessian_product_kernel<Components...>(data_layout, layout, data_vec,
                                              vec, dir, res);
  }

  template <class Component, class... Remain>
  void add_hessian_product_kernel(const IndexLayout<ndata> &data_layout,
                                  const IndexLayout<ncomp> &layout,
                                  const Vector<T> &data_vec,
                                  const Vector<T> &vec, const Vector<T> &dir,
                                  Vector<T> &res) const {
    int end = 0;
    for (int j = 0; j < num_colors; j++) {
      int start = end;
      end = start + elem_per_color[j];
#pragma omp parallel for
      for (int elem = start; elem < end; elem++) {
        Data data;
        Input input, gradient, direction, result;

        data_layout.get_values(elem, data_vec, data);
        gradient.zero();
        result.zero();
        layout.get_values(elem, vec, input);
        layout.get_values(elem, dir, direction);
        Component::hessian(data, input, direction, gradient, result);
        layout.add_values(elem, result, res);
      }
    }

    if constexpr (sizeof...(Remain) > 0) {
      add_hessian_product_kernel<Remain...>(data_layout, layout, data_vec, vec,
                                            dir, res);
    }
  }

  void add_hessian_kernel(const IndexLayout<ndata> &data_layout,
                          const IndexLayout<ncomp> &layout,
                          const Vector<T> &data_vec, const Vector<T> &vec,
                          const NodeOwners &owners, CSRMat<T> &jac) const {
    add_hessian_kernel<Components...>(data_layout, layout, data_vec, vec,
                                      owners, jac);
  }

  template <class Component, class... Remain>
  void add_hessian_kernel(const IndexLayout<ndata> &data_layout,
                          const IndexLayout<ncomp> &layout,
                          const Vector<T> &data_vec, const Vector<T> &vec,
                          const NodeOwners &owners, CSRMat<T> &jac) const {
    int end = 0;
    for (int j = 0; j < num_colors; j++) {
      int start = end;
      end = start + elem_per_color[j];
#pragma omp parallel for
      for (int elem = start; elem < end; elem++) {
        Data data;
        Input input, gradient, direction, result;
        int index[ncomp], index_global[ncomp];
        layout.get_indices(elem, index);
        owners.local_to_global(ncomp, index, index_global);

        data_layout.get_values(elem, data_vec, data);
        layout.get_values(elem, vec, input);

        for (int k = 0; k < Component::ncomp; k++) {
          direction.zero();
          gradient.zero();
          result.zero();

          direction[k] = 1.0;

          Component::hessian(data, input, direction, gradient, result);

          jac.add_row(index[k], Component::ncomp, index_global, result);
        }
      }
    }

    if constexpr (sizeof...(Remain) > 0) {
      add_hessian_kernel<Remain...>(data_layout, layout, data_vec, vec, owners,
                                    jac);
    }
  }

 private:
  int num_colors;
  int *elem_per_color;
};

template <typename T, int ncomp, class Input, int ndata, class Data,
          class... Components>
using DefaultGroupBackend =
    OmpGroupBackend<T, ncomp, Input, ndata, Data, Components...>;

#elif defined(AMIGO_USE_CUDA)

template <typename T, class Component>
AMIGO_KERNEL T lagrange_kernel() {}

template <typename T, class Component>
AMIGO_KERNEL void add_gradient_kernel() {}

template <typename T, class Component>
AMIGO_KERNEL void add_hessian_product_kernel() {}

template <typename T, class Component>
AMIGO_KERNEL void add_hessian_kernel() {}

template <typename T, class Component>
class CudaGroupBackend {
 public:
  static constexpr int ncomp = Component::ncomp;
  static constexpr int ndata = Component::ncomp;
  using Input = typename Component::Input;
  using Data = typename Component::Data;

  CudaGroupBackend(IndexLayout<ndata> &data_layout,
                   IndexLayout<ncomp> &layout) {
    data_layout.copy_host_to_device();
    layout.copy_host_to_device();
  }
  T lagrangian_kernel(const IndexLayout<ndata> &data_layout,
                      const IndexLayout<ncomp> &layout,
                      const Vector<T> &data_vec, const Vector<T> &vec) const {}

  void add_gradient_kernel(const IndexLayout<ndata> &data_layout,
                           const IndexLayout<ncomp> &layout,
                           const Vector<T> &data_vec, const Vector<T> &vec,
                           Vector<T> &res) const {}
  void add_hessian_product_kernel(const IndexLayout<ndata> &data_layout,
                                  const IndexLayout<ncomp> &layout,
                                  const Vector<T> &data_vec,
                                  const Vector<T> &vec, const Vector<T> &dir,
                                  Vector<T> &res) const {}
  void add_hessian_kernel(const IndexLayout<ndata> &data_layout,
                          const IndexLayout<ncomp> &layout,
                          const Vector<T> &data_vec, const Vector<T> &vec,
                          CSRMat<T> &jac) const {}
};

#else  // Default to serial implementation
template <typename T, int ncomp, class Input, int ndata, class Data,
          class... Components>
using DefaultGroupBackend =
    SerialGroupBackend<T, ncomp, Input, ndata, Data, Components...>;
#endif

template <typename T, class... Components>
class ComponentGroup : public ComponentGroupBase<T> {
 public:
  static constexpr int num_components = sizeof...(Components);

  using Input = typename __get_collection_input_type<T, Components...>::Input;
  using Data = typename __get_collection_data_type<T, Components...>::Data;

  static constexpr int ncomp = __get_collection_ncomp<Components...>::value;
  static constexpr int ndata = __get_collection_ndata<Components...>::value;
  static constexpr int noutputs =
      __get_collection_noutputs<Components...>::value;

  // Use whatever class is defined as the default backend
  using Backend =
      DefaultGroupBackend<T, ncomp, Input, ndata, Data, Components...>;

  using OutputBackend =
      DefaultOutputBackend<T, ncomp, ndata, noutputs, Components...>;

  ComponentGroup(int num_elements, std::shared_ptr<Vector<int>> data_indices,
                 std::shared_ptr<Vector<int>> indices,
                 std::shared_ptr<Vector<int>> output_indices)
      : data_layout(num_elements, data_indices),
        layout(num_elements, indices),
        output_layout(num_elements, output_indices),
        backend(data_layout, layout, output_layout) {}

  std::shared_ptr<ComponentGroupBase<T>> clone(
      int num_elements, std::shared_ptr<Vector<int>> data_idx,
      std::shared_ptr<Vector<int>> layout_idx,
      std::shared_ptr<Vector<int>> output_idx) const {
    return std::make_shared<ComponentGroup<T, Components...>>(
        num_elements, data_idx, layout_idx, output_idx);
  }

  // Group compute functions
  T lagrangian(const Vector<T> &data_vec, const Vector<T> &vec) const {
    return backend.lagrangian_kernel(data_layout, layout, data_vec, vec);
  }
  void add_gradient(const Vector<T> &data_vec, const Vector<T> &vec,
                    Vector<T> &res) const {
    backend.add_gradient_kernel(data_layout, layout, data_vec, vec, res);
  }
  void add_hessian_product(const Vector<T> &data_vec, const Vector<T> &vec,
                           const Vector<T> &dir, Vector<T> &res) const {
    backend.add_hessian_product_kernel(data_layout, layout, data_vec, vec, dir,
                                       res);
  }
  void add_hessian(const Vector<T> &data_vec, const Vector<T> &vec,
                   const NodeOwners &owners, CSRMat<T> &jac) const {
    backend.add_hessian_kernel(data_layout, layout, data_vec, vec, owners, jac);
  }

  // Group output functions
  void add_output(const Vector<T> &data_vec, const Vector<T> &vec,
                  Vector<T> &output) const {
    output_backend.add_output_kernel(data_layout, layout, output_layout,
                                     data_vec, vec, output);
  }
  void add_input_jacobian(const Vector<T> &data_vec, const Vector<T> &vec,
                          CSRMat<T> &jac) const {
    output_backend.add_input_jacobian_kernel(data_layout, layout, output_layout,
                                             data_vec, vec, jac);
  }
  void add_data_jacobian(const Vector<T> &data_vec, const Vector<T> &vec,
                         CSRMat<T> &jac) const {
    output_backend.add_data_jacobian_kernel(data_layout, layout, output_layout,
                                            data_vec, vec, jac);
  }

  // Get the ordering information
  void get_data_layout_data(int *num_elements, int *nodes_per_elem,
                            const int **array) const {
    data_layout.get_data(num_elements, nodes_per_elem, array);
  }
  void get_layout_data(int *num_elements, int *nodes_per_elem,
                       const int **array) const {
    layout.get_data(num_elements, nodes_per_elem, array);
  }
  void get_output_layout_data(int *num_elements, int *outputs_per_elem,
                              const int **array) const {
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
