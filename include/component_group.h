#ifndef AMIGO_COMPONENT_GROUP_H
#define AMIGO_COMPONENT_GROUP_H

#include "a2dcore.h"
#include "component_group_base.h"
#include "csr_matrix.h"
#include "layout.h"
#include "ordering_utils.h"
#include "vector.h"

namespace amigo {

template <typename T, class Component>
class SerialGroupBackend {
 public:
  static constexpr int ncomp = Component::ncomp;
  static constexpr int ndata = Component::ndata;
  using Input = typename Component::Input;
  using Data = typename Component::Data;

  SerialGroupBackend(const IndexLayout<ndata> &data_layout,
                     const IndexLayout<ncomp> &layout) {}

  T lagrangian_kernel(const IndexLayout<ndata> &data_layout,
                      const IndexLayout<ncomp> &layout,
                      const Vector<T> &data_vec, const Vector<T> &vec) const {
    Data data;
    Input input;
    T value = 0.0;
    int length = layout.get_length();

    for (int i = 0; i < length; i++) {
      data_layout.get_values(i, data_vec, data);
      layout.get_values(i, vec, input);
      value += Component::lagrange(data, input);
    }
    return value;
  }

  void add_gradient_kernel(const IndexLayout<ndata> &data_layout,
                           const IndexLayout<ncomp> &layout,
                           const Vector<T> &data_vec, const Vector<T> &vec,
                           Vector<T> &res) const {
    Data data;
    Input input, gradient;
    for (int i = 0; i < layout.get_length(); i++) {
      data_layout.get_values(i, data_vec, data);
      gradient.zero();
      layout.get_values(i, vec, input);
      Component::gradient(data, input, gradient);
      layout.add_values(i, gradient, res);
    }
  }

  void add_hessian_product_kernel(const IndexLayout<ndata> &data_layout,
                                  const IndexLayout<ncomp> &layout,
                                  const Vector<T> &data_vec,
                                  const Vector<T> &vec, const Vector<T> &dir,
                                  Vector<T> &res) const {
    Data data;
    Input input, gradient, direction, result;
    for (int i = 0; i < layout.get_length(); i++) {
      data_layout.get_values(i, data_vec, data);
      gradient.zero();
      result.zero();
      layout.get_values(i, vec, input);
      layout.get_values(i, dir, direction);
      Component::hessian(data, input, direction, gradient, result);
      layout.add_values(i, result, res);
    }
  }

  void add_hessian_kernel(const IndexLayout<ndata> &data_layout,
                          const IndexLayout<ncomp> &layout,
                          const Vector<T> &data_vec, const Vector<T> &vec,
                          CSRMat<T> &jac) const {
    Data data;
    Input input, gradient, direction, result;
    for (int i = 0; i < layout.get_length(); i++) {
      data_layout.get_values(i, data_vec, data);
      int index[ncomp];
      layout.get_indices(i, index);
      layout.get_values(i, vec, input);

      for (int j = 0; j < Component::ncomp; j++) {
        direction.zero();
        gradient.zero();
        result.zero();

        direction[j] = 1.0;

        Component::hessian(data, input, direction, gradient, result);

        jac.add_row(index[j], Component::ncomp, index, result);
      }
    }
  }
};

#ifdef AMIGO_USE_OPENMP

template <typename T, class Component>
class OmpGroupBackend {
 public:
  static constexpr int ncomp = Component::ncomp;
  static constexpr int ndata = Component::ndata;
  using Input = typename Component::Input;
  using Data = typename Component::Data;

  OmpGroupBackend(IndexLayout<ndata> &data_layout, IndexLayout<ncomp> &layout)
      : num_colors(0), elem_per_color(nullptr) {
    int length, ncomps;
    const int *array;
    layout.get_data(&length, &ncomps, &array);

    // Create a coloring for the layout
    int *elem_by_color_ptr;
    int *elem_by_color;
    OrderingUtils::color_elements(length, ncomp, array, &num_colors,
                                  &elem_by_color_ptr, &elem_by_color);

    // Re-order the data and the variable layouts
    data_layout.reorder(elem_by_color);
    layout.reorder(elem_by_color);

    elem_per_color = new int[num_colors];
    for (int i = 0; i < num_colors; i++) {
      elem_per_color[i] = elem_by_color_ptr[i + 1] - elem_by_color_ptr[i];
    }

    delete[] elem_by_color_ptr;
    delete[] elem_by_color;

    // Check the coloring
    int max_node = 0;
    for (int i = 0; i < length * ncomps; i++) {
      if (array[i] > max_node) {
        max_node = array[i];
      }
    }
    max_node++;
    // Set the values
    int *temp = new int[max_node];

    int end = 0;
    for (int j = 0; j < num_colors; j++) {
      int start = end;
      end = start + elem_per_color[j];

      std::fill(temp, temp + max_node, 0);
      for (int elem = start; elem < end; elem++) {
        int indices[ncomp];
        layout.get_indices(elem, indices);

        for (int k = 0; k < ncomp; k++) {
          temp[indices[k]] += 1;
        }
      }

      for (int k = 0; k < max_node; k++) {
        if (temp[k] > 1) {
          std::cout << "Error with coloring for node " << k << std::endl;
        }
      }
    }

    delete[] temp;
  }
  ~OmpGroupBackend() { delete[] elem_per_color; }

  T lagrangian_kernel(const IndexLayout<ndata> &data_layout,
                      const IndexLayout<ncomp> &layout,
                      const Vector<T> &data_vec, const Vector<T> &vec) const {
    Data data;
    Input input;
    T value = 0.0;
    int length = layout.get_length();

#pragma omp parallel for reduction(+ : value)
    for (int i = 0; i < length; i++) {
      data_layout.get_values(i, data_vec, data);
      layout.get_values(i, vec, input);
      value += Component::lagrange(data, input);
    }
    return value;
  }

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
  }

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
  }

  void add_hessian_kernel(const IndexLayout<ndata> &data_layout,
                          const IndexLayout<ncomp> &layout,
                          const Vector<T> &data_vec, const Vector<T> &vec,
                          CSRMat<T> &jac) const {
    int end = 0;
    for (int j = 0; j < num_colors; j++) {
      int start = end;
      end = start + elem_per_color[j];
#pragma omp parallel for
      for (int elem = start; elem < end; elem++) {
        Data data;
        Input input, gradient, direction, result;

        data_layout.get_values(elem, data_vec, data);
        int index[ncomp];
        layout.get_indices(elem, index);
        layout.get_values(elem, vec, input);

        for (int k = 0; k < Component::ncomp; k++) {
          direction.zero();
          gradient.zero();
          result.zero();

          direction[k] = 1.0;

          Component::hessian(data, input, direction, gradient, result);

          jac.add_row(index[k], Component::ncomp, index, result);
        }
      }
    }
  }

 private:
  int num_colors;
  int *elem_per_color;
};

template <typename T, class Component>
using DefaultGroupBackend = OmpGroupBackend<T, Component>;
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
  static constexpr int ndata = Component::ndata;
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
template <typename T, class Component>
using DefaultGroupBackend = SerialGroupBackend<T, Component>;
#endif

template <typename T, class Component,
          class Backend = DefaultGroupBackend<T, Component>>
class ComponentGroup : public ComponentGroupBase<T> {
 public:
  static constexpr int ncomp = Component::ncomp;
  static constexpr int ndata = Component::ndata;
  using Input = typename Component::Input;
  using Data = typename Component::Data;

  ComponentGroup(std::shared_ptr<Vector<int>> data_indices,
                 std::shared_ptr<Vector<int>> indices)
      : data_layout(data_indices),
        layout(indices),
        backend(data_layout, layout) {}

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
                   CSRMat<T> &jac) const {
    backend.add_hessian_kernel(data_layout, layout, data_vec, vec, jac);
  }

  void get_layout_data(int *length_, int *ncomp_, const int **array_) const {
    layout.get_data(length_, ncomp_, array_);
  }

 private:
  IndexLayout<ndata> data_layout;
  IndexLayout<ncomp> layout;
  Backend backend;
};

}  // namespace amigo

#endif  // AMIGO_COMPONENT_GROUP_H
