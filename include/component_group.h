#ifndef MDGO_COMPONENT_COLLECTION_H
#define MDGO_COMPONENT_COLLECTION_H

#include "a2dcore.h"
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

 private:
  Component &comp;
  Layout &layout;
};

}  // namespace mdgo

#endif  // MDGO_COMPONENT_COLLECTION_H