#ifndef MDGO_COMPONENT_COLLECTION_H
#define MDGO_COMPONENT_COLLECTION_H

#include <set>

#include "a2dcore.h"
#include "csr_matrix.h"
#include "vector.h"

namespace mdgo {

template <typename T, class Component, class Layout>
class SerialCollection {
 public:
  static constexpr int ncomp = Component::ncomp;
  using Input = typename Component::template Input<T>;

  SerialCollection(Component &comp, Layout &layout)
      : comp(comp), layout(layout) {}

  T lagrangian(const Vector<T> &vec) const {
    T value = 0.0;
    for (int i = 0; i < layout.get_length(); i++) {
      Input input;
      layout.get_values(i, vec, input);
      value += comp.lagrange(input);
    }
    return value;
  }

  void add_gradient(const Vector<T> &vec, Vector<T> &res) const {
    Input input, gradient;
    for (int i = 0; i < layout.get_length(); i++) {
      gradient.zero();
      layout.get_values(i, vec, input);
      comp.gradient(input, gradient);
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
      comp.hessian_product(input, direction, gradient, result);
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

        comp.hessian_product(input, direction, gradient, result);

        jac.add_row(index[j], Component::ncomp, index, result);
      }
    }
  }

 private:
  Component &comp;
  Layout &layout;
};

}  // namespace mdgo

#endif  // MDGO_COMPONENT_COLLECTION_H