#ifndef AMIGO_COMPONENT_EXTERNAL_H
#define AMIGO_COMPONENT_EXTERNAL_H

#include "component_group_base.h"
#include "csr_matrix.h"

namespace amigo {

template <typename T>
class ComponentExtneral : public ComponentGroupBase<T> {
 public:
  ComponentExtneral(std::shared_ptr<Vector<int>> con_indices,
                    std::shared_ptr<Vector<int>> var_indices,
                    std::shared_ptr<Vector<T>> constraints,
                    std::shared_ptr<CSRMat<T>> jacobian)
      : con_indices(con_indices),
        var_indices(var_indices),
        constraints(constraints),
        jacobian(jacobian) {}

  virtual T lagrangian(const Vector<T>& data, const Vector<T>& x) const {
    T value(0.0);
    for (int i = 0; i < size; i++) {
      T lambda = x[con_indices[i]];
      value += lambda * constraints[i];
    }
    return value;
  }
  virtual void add_gradient(const Vector<T>& data, const Vector<T>& x,
                            Vector<T>& g) const {
    // Compute the Hessian-vector product
    for (int i = 0; i < ncon; i++) {
    }
  }
  virtual void add_hessian_product(const Vector<T>& data, const Vector<T>& x,
                                   const Vector<T>& p, Vector<T>& h) const {}
  virtual void add_hessian(const Vector<T>& data, const Vector<T>& x,
                           const NodeOwners& owners, CSRMat<T>& mat) const {}

  std::shared_ptr<Vector<T>> get_constraints() { return constraints; }
  std::shared_ptr<CSRMat<T>> get_jacobian() { return jacobian; }

 private:
  std::shared_ptr<Vector<int>> con_indices;
  std::shared_ptr<Vector<int>> var_indices;
  std::shared_ptr<Vector<T>> constraints;
  std::shared_ptr<CSRMat<T>> jacobian;
};

}  // namespace amigo

#endif  // AMIGO_COMPONENT_EXTERNAL_H