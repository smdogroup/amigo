#ifndef AMIGO_SERIAL_COMPONENT_SET_H
#define AMIGO_SERIAL_COMPONENT_SET_H

#include "a2dcore.h"
#include "component_set.h"
#include "csr_matrix.h"
#include "layout.h"
#include "vector.h"

namespace amigo {

template <typename T, class Component>
class SerialComponentSet : public ComponentSet<T> {
 public:
  static constexpr int ncomp = Component::ncomp;
  using Input = typename Component::Input;

  SerialComponentSet(Vector<int> &indices) : layout(indices) {}

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

  T lagrangian(const Vector<T> &vec) const {
    T value = 0.0;
    for (int i = 0; i < layout.get_length(); i++) {
      Input input;
      layout.get_values(i, vec, input);
      value += Component::lagrange(input);
    }
    return value;
  }

  void add_gradient(const Vector<T> &vec, Vector<T> &res) const {
    Input input, gradient;
    for (int i = 0; i < layout.get_length(); i++) {
      gradient.zero();
      layout.get_values(i, vec, input);
      Component::gradient(input, gradient);
      layout.add_values(i, gradient, res);
    }
  }

  void add_hessian_product(const Vector<T> &vec, const Vector<T> &dir,
                           Vector<T> &res) const {
    Input input, gradient, direction, result;
    for (int i = 0; i < layout.get_length(); i++) {
      gradient.zero();
      result.zero();
      layout.get_values(i, vec, input);
      layout.get_values(i, dir, direction);
      Component::hessian(input, direction, gradient, result);
      layout.add_values(i, result, res);
    }
  }

  void add_nonzero_pattern(std::set<std::pair<int, int>> &map) const {
    for (int i = 0; i < layout.get_length(); i++) {
      int index[ncomp];
      layout.get_indices(i, index);

      for (int j = 0; j < ncomp; j++) {
        for (int k = 0; k < ncomp; k++) {
          map.insert(std::pair<int, int>(index[j], index[k]));
        }
      }
    }
  }

  void add_hessian(const Vector<T> &vec, CSRMat<T> &jac) const {
    Input input, gradient, direction, result;
    for (int i = 0; i < layout.get_length(); i++) {
      int index[ncomp];
      layout.get_indices(i, index);
      layout.get_values(i, vec, input);

      for (int j = 0; j < Component::ncomp; j++) {
        direction.zero();
        gradient.zero();
        result.zero();

        direction[j] = 1.0;

        Component::hessian(input, direction, gradient, result);

        jac.add_row(index[j], Component::ncomp, index, result);
      }
    }
  }

 private:
  Component comp;
  IndexLayout<ncomp> layout;
};

}  // namespace amigo

#endif  // AMIGO_SERIAL_COMPONENT_SET_H