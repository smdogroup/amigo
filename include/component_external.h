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

  // Update the design point - applying the callbacks
  // void update(const Vector<T>& x) { update_callback(x, constraints,
  // jacobian); }

  // Compute the Lagrangian
  T lagrangian(const Vector<T>& data, const Vector<T>& x) const {
    T value(0.0);
    for (int i = 0; i < size; i++) {
      T lambda = x[con_indices[i]];
      value += lambda * constraints[i];
    }
    return value;
  }
  void add_gradient(const Vector<T>& data, const Vector<T>& x,
                    Vector<T>& g) const {
    for (int i = 0; i < nrows; i++) {
      int row = con_indices[i];
      g[row] += constraints[i];

      for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
        int j = cols[jp];
        int col = var_indices[j];

        T lambda = x[row];
        g[col] += A[jp] * lambda;
      }
    }
  }
  void add_hessian_product(const Vector<T>& data, const Vector<T>& x,
                           const Vector<T>& p, Vector<T>& h) const {
    for (int i = 0; i < nrows; i++) {
      int row = con_indices[i];

      for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
        int j = cols[jp];
        int col = var_indices[j];

        T px = p[col];
        T plambda = p[row];
        h[col] += A[jp] * plambda;
        h[row] += A[jp] * px;
      }
    }
  }
  void add_hessian(const Vector<T>& data, const Vector<T>& x,
                   const NodeOwners& owners, CSRMat<T>& mat) const {
    // Add the contributions to the Hessian matrix...
  }

  void get_constraint_csr_data(int* nrows, int* ncols, const int* rows[],
                               const int* columns[], const int* rowp[],
                               const int* cols[]) const {
    if (rows) {
      *rows = con_indices->get_array();
    }
    if (columns) {
      *columns = var_indices->get_array();
    }
    jacobian->get_data(nrows, ncols, rowp, cols);
  }

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