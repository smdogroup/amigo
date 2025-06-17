#ifndef AMIGO_COMPONENT_GROUP_H
#define AMIGO_COMPONENT_GROUP_H

#include "a2dcore.h"
#include "component_group_base.h"
#include "csr_matrix.h"
#include "layout.h"
#include "vector.h"

namespace amigo {

template <typename T, class Component>
class ComponentGroup : public ComponentGroupBase<T> {
 public:
  static constexpr int ncomp = Component::ncomp;
  static constexpr int ndata = Component::ndata;
  using Input = typename Component::Input;
  using Data = typename Component::Data;

  ComponentGroup(std::shared_ptr<Vector<int>> data_indices,
                 std::shared_ptr<Vector<int>> indices)
      : data_layout(data_indices), layout(indices) {}

  int get_max_dof() const {
    int index[ncomp];
    int max_dof = 0;
    for (int i = 0; i < layout.get_length(); i++) {
      layout.get_indices(i, index);
      for (int j = 0; j < ncomp; j++) {
        if (index[j] > max_dof) {
          max_dof = index[j];
        }
      }
    }
    return max_dof;
  }

  T lagrangian(const Vector<T> &data_vec, const Vector<T> &vec) const {
    Data data;
    Input input;
    T value = 0.0;
    for (int i = 0; i < layout.get_length(); i++) {
      data_layout.get_values(i, data_vec, data);
      layout.get_values(i, vec, input);
      value += Component::lagrange(data, input);
    }
    return value;
  }

  void add_gradient(const Vector<T> &data_vec, const Vector<T> &vec,
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

  void add_hessian_product(const Vector<T> &data_vec, const Vector<T> &vec,
                           const Vector<T> &dir, Vector<T> &res) const {
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

  void add_hessian(const Vector<T> &data_vec, const Vector<T> &vec,
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

  void get_layout_data(int *length_, int *ncomp_, const int **array_) const {
    layout.get_data(length_, ncomp_, array_);
  }

 private:
  IndexLayout<ndata> data_layout;
  IndexLayout<ncomp> layout;
};

}  // namespace amigo

#endif  // AMIGO_COMPONENT_GROUP_H