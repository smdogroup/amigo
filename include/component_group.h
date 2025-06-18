#ifndef AMIGO_COMPONENT_GROUP_H
#define AMIGO_COMPONENT_GROUP_H

#include "a2dcore.h"
#include "component_group_base.h"
#include "csr_matrix.h"
#include "layout.h"
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

  OmpGroupBackend(const IndexLayout<ndata> &data_layout,
                  const IndexLayout<ncomp> &layout)
      : num_colors(0), elem_by_color_ptr(nullptr), elem_by_color(nullptr) {
    int length, ncomps;
    const int *array;
    layout.get_data(&length, &ncomps, &array);
    color_elements(length, array);
  }
  ~OmpGroupBackend() {
    delete[] elem_by_color_ptr;
    delete[] elem_by_color;
  }

  void build_element_to_element(int nelems, const int *elem_nodes,
                                int **elem_to_elem_ptr_, int **elem_to_elem_) {
    int max_node = 0;
    for (int i = 0; i < nelems * ncomp; i++) {
      if (elem_nodes[i] > max_node) {
        max_node = elem_nodes[i];
      }
    }
    max_node++;

    // Create a pointer from the nodes back to the elements
    int *node_to_elem_ptr = new int[max_node + 1];
    std::fill(node_to_elem_ptr, node_to_elem_ptr + max_node + 1, 0);

    for (int i = 0; i < nelems * ncomp; i++) {
      node_to_elem_ptr[elem_nodes[i] + 1]++;
    }
    for (int i = 0; i < max_node; i++) {
      node_to_elem_ptr[i + 1] += node_to_elem_ptr[i];
    }

    int *node_to_elem = new int[nelems * ncomp];

    // Fill in the element numbers
    for (int i = 0; i < nelems; i++) {
      for (int j = 0; j < ncomp; j++) {
        int node = elem_nodes[ncomp * i + j];
        node_to_elem[node_to_elem_ptr[node]] = i;
        node_to_elem_ptr[node]++;
      }
    }

    // Fix the now broken node_to_elem_ptr array
    for (int i = max_node - 1; i >= 0; i--) {
      node_to_elem_ptr[i + 1] = node_to_elem_ptr[i];
    }
    node_to_elem_ptr[0] = 0;

    // Compute the element -> element data structure
    int *elem_flags = new int[nelems];
    std::fill(elem_flags, elem_flags + nelems, -1);

    int *elem_to_elem_ptr = new int[nelems + 1];
    for (int i = 0; i < nelems; i++) {
      int count = 0;

      for (int j = 0; j < ncomp; j++) {
        int node = elem_nodes[ncomp * i + j];

        // Find the adjacent elements
        int start = node_to_elem_ptr[node];
        int end = node_to_elem_ptr[node + 1];
        for (int k = start; k < end; k++) {
          int e = node_to_elem[k];

          if (e != i && elem_flags[e] != i) {
            count++;
            elem_flags[e] = i;
          }
        }
      }
      elem_to_elem_ptr[i + 1] = count;
    }

    for (int i = 0; i < nelems; i++) {
      elem_to_elem_ptr[i + 1] += elem_to_elem_ptr[i];
    }

    std::fill(elem_flags, elem_flags + nelems, -1);
    int *elem_to_elem = new int[elem_to_elem_ptr[nelems]];
    for (int i = 0; i < nelems; i++) {
      int count = 0;

      for (int j = 0; j < ncomp; j++) {
        int node = elem_nodes[ncomp * i + j];

        // Find the adjacent elements
        int start = node_to_elem_ptr[node];
        int end = node_to_elem_ptr[node + 1];
        for (int k = start; k < end; k++) {
          int e = node_to_elem[k];

          if (e != i && elem_flags[e] != i) {
            elem_to_elem[elem_to_elem_ptr[i] + count] = e;
            elem_flags[e] = i;
          }
        }
      }
    }

    delete[] node_to_elem;
    delete[] node_to_elem_ptr;
    delete[] elem_flags;

    *elem_to_elem_ptr_ = elem_to_elem_ptr;
    *elem_to_elem_ = elem_to_elem;
  }

  // Input:
  // elem_nodes[e] = list of global node ids for element e
  // Output:
  // elem_color[e] = color assigned to element e
  void color_elements(int nelems, const int *elem_nodes) {
    int *elem_to_elem_ptr, *elem_to_elem;
    build_element_to_element(nelems, elem_nodes, &elem_to_elem_ptr,
                             &elem_to_elem);

    // Greedy coloring
    int *elem_colors = new int[nelems];
    std::fill(elem_colors, elem_colors + nelems, -1);

    // Keep track of the number of colors
    num_colors = 0;

    int *flags = new int[nelems];
    std::fill(flags, flags + nelems, -1);

    for (int e = 0; e < nelems; e++) {
      int start = elem_to_elem_ptr[e];
      int end = elem_to_elem_ptr[e + 1];
      for (int k = start; k < end; k++) {
        int neigh = elem_to_elem[k];
        int c = elem_colors[neigh];

        if (c >= 0) {
          flags[c] = e;
        }
      }

      // Find smallest non-conflicting color if any
      int found = false;
      for (int k = 0; k < num_colors; k++) {
        if (flags[k] != e) {
          elem_colors[e] = k;
          found = true;
          break;
        }
      }

      // No color was found, create a new color
      if (!found) {
        elem_colors[e] = num_colors;
        num_colors++;
      }
    }

    // Step 4: Set up the elements by color
    elem_by_color_ptr = new int[num_colors + 1];
    for (int e = 0; e < nelems; e++) {
      elem_by_color_ptr[elem_colors[e] + 1]++;
    }

    for (int i = 0; i < num_colors; i++) {
      elem_by_color_ptr[i + 1] += elem_by_color_ptr[i];
    }

    for (int e = 0; e < nelems; e++) {
      elem_by_color[elem_by_color_ptr[elem_colors[e]]] = e;
      elem_by_color_ptr[elem_colors[e]]++;
    }

    // Reset the pointer
    for (int i = num_colors - 1; i >= 0; i--) {
      elem_by_color_ptr[i + 1] = elem_by_color_ptr[i];
    }
    elem_by_color_ptr[0] = 0;

    delete[] elem_colors;
    delete[] flags;
  }

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
    Data data;
    Input input, gradient;

    for (int j = 0; j < num_colors; j++) {
      int start = elem_by_color_ptr[j];
      int end = elem_by_color_ptr[j + 1];
#pragma omp parallel for
      for (int i = start; i < end; i++) {
        int elem = elem_by_color[i];
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
    Data data;
    Input input, gradient, direction, result;

    for (int j = 0; j < num_colors; j++) {
      int start = elem_by_color_ptr[j];
      int end = elem_by_color_ptr[j + 1];
#pragma omp parallel for
      for (int i = start; i < end; i++) {
        int elem = elem_by_color[i];
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
    Data data;
    Input input, gradient, direction, result;

    for (int j = 0; j < num_colors; j++) {
      int start = elem_by_color_ptr[j];
      int end = elem_by_color_ptr[j + 1];
#pragma omp parallel for
      for (int i = start; i < end; i++) {
        int elem = elem_by_color[i];
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
  int *elem_by_color_ptr;
  int *elem_by_color;
};

template <typename T, class Component>
using DefaultGroupBackend = OmpGroupBackend<T, Component>;
#elif defined(AMIGO_USE_CUDA)

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
