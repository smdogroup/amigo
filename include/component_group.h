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
      : elem_by_color(layout.get_length()) {
    int length, ncomps;
    const int *array;
    layout.get_data(&length, &ncomps, &array);
    color_elements(length, array);
  }
  int max_color;
  std::vector<int> elem_ptr;
  std::vector<int> elem_by_color;

  // Input:
  // elem_nodes[e] = list of global node ids for element e
  // Output:
  // elem_color[e] = color assigned to element e
  void color_elements(int nelems, const int *elem_nodes) {
    std::unordered_map<int, std::vector<int>> node_to_elems;

    // Step 1: Build node to element mapping
    for (int e = 0; e < nelems; ++e) {
      for (int j = 0; j < ncomp; j++) {
        int node = elem_nodes[ncomp * e + j];
        node_to_elems[node].push_back(e);
      }
    }

    // Step 2: Build element adjacency list
    std::vector<std::unordered_set<int>> elem_neighbors(nelems);
    for (const auto &[node, elems] : node_to_elems) {
      for (size_t i = 0; i < elems.size(); ++i) {
        for (size_t j = 0; j < elems.size(); ++j) {
          if (i != j) {
            elem_neighbors[elems[i]].insert(elems[j]);
          }
        }
      }
    }

    // Step 3: Greedy coloring
    std::vector<int> elem_color(nelems, -1);
    max_color = 0;

    for (int e = 0; e < nelems; ++e) {
      std::unordered_set<int> neighbor_colors;
      for (int neigh : elem_neighbors[e]) {
        int c = elem_color[neigh];
        if (c != -1) {
          neighbor_colors.insert(c);
        }
      }

      // Find smallest non-conflicting color
      int color = 0;
      while (neighbor_colors.count(color)) {
        ++color;
      }

      elem_color[e] = color;
      max_color = std::max(max_color, color);
    }

    // Step 4: Set up the elements by color
    elem_ptr.resize(max_color + 1, 0);
    for (int e = 0; e < nelems; e++) {
      elem_ptr[elem_color[e] + 1]++;
    }

    for (int i = 0; i < max_color; i++) {
      elem_ptr[i + 1] += elem_ptr[i];
    }

    for (int e = 0; e < nelems; e++) {
      elem_by_color[ptr[elem_color[e]]] = e;
      ptr[elem_color[e]]++;
    }

    // Reset the pointer
    for (int i = max_color - 1; i >= 0; i--) {
      elem_ptr[i + 1] = elem_ptr[i];
    }
    elem_ptr[0] = 0;
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

    for (int j = 0; j < max_color; j++) {
      int start = elem_ptr[j];
      int end = elem_ptr[j + 1];
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

    for (int j = 0; j < max_color; j++) {
      int start = elem_ptr[j];
      int end = elem_ptr[j + 1];
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
    Input input, gradient, direction, result[ncomp];

    for (int j = 0; j < max_color; j++) {
      int start = elem_ptr[j];
      int end = elem_ptr[j + 1];
#pragma omp parallel for
      for (int i = start; i < end; i++) {
        int elem = elem_by_color[i];
        data_layout.get_values(elem, data_vec, data);
        int index[ncomp];
        layout.get_indices(elem, index);
        layout.get_values(elem, vec, input);

        for (int j = 0; j < Component::ncomp; j++) {
          direction.zero();
          gradient.zero();
          result[j].zero();

          direction[j] = 1.0;

          Component::hessian(data, input, direction, gradient, result[j]);
        }

        for (int j = 0; j < Component::ncomp; j++) {
          jac.add_row(index[j], Component::ncomp, index, result[j]);
        }
      }
    }
  }
};

template <typename T, class Component>
using DefaultGroupBackend = OmpGroupBackend<T, Component>;
#elif defined(AMIGO_USE_CUDA)

#else

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
