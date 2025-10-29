#ifndef AMIGO_COMPONENT_EXTERNAL_H
#define AMIGO_COMPONENT_EXTERNAL_H

#include "component_group_base.h"
#include "csr_matrix.h"

namespace amigo {

/**
 * @brief Interface for the external constraint evaluation
 *
 * @tparam T
 */
template <typename T>
class ExternalComponentEvaluation {
 public:
  virtual void evaluate_constraints(const std::shared_ptr<Vector<T>>& x,
                                    std::shared_ptr<Vector<T>>& constraints,
                                    std::shared_ptr<CSRMat<T>>& jacobian) = 0;
};

/**
 * @brief An external component that contributes constraint
 *
 * @tparam T
 */
template <typename T>
class ExternalComponent : public ComponentGroupBase<T> {
 public:
  ExternalComponent(std::shared_ptr<Vector<int>> con_indices,
                    std::shared_ptr<Vector<int>> var_indices,
                    std::shared_ptr<Vector<T>> constraints,
                    std::shared_ptr<CSRMat<T>> jacobian,
                    std::shared_ptr<ExternalComponentEvaluation<T>> extrn)
      : con_indices(con_indices),
        var_indices(var_indices),
        constraints(constraints),
        jacobian(jacobian),
        extrn(extrn) {
    jacobian_transpose = jacobian->transpose();
  }

  /**
   * @brief Update the gradient evaluation
   *
   * @param x The design variable values
   */
  void update(const Vector<T>& x) {
    ext->evaluate_constraints(x, constraints, jacobian);
    jacobian_transpose->copy_transpose(jacobian);
  }

  // Compute the Lagrangian
  T lagrangian(const Vector<T>& data, const Vector<T>& x) const {
    T value(0.0);
    for (int i = 0; i < size; i++) {
      T lambda = x[con_indices[i]];
      value += lambda * constraints[i];
    }
    return value;
  }

  /**
   * @brief Add the gradient contributions from the component
   *
   * @param data (not used, but required for matching the signature)
   * @param x (not used)
   * @param g The gradient of the lagrangian
   */
  void add_gradient(const Vector<T>& data, const Vector<T>& x,
                    Vector<T>& g) const {
    int nrows;
    const int *rowp, *cols;
    T* A;
    jacobian->get_data(&nrows, nullptr, nullptr, &rowp, &cols, &A);

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

  /**
   * @brief Compute a Hessian-vector product
   *
   * @param data (not used, but required for matching the signature)
   * @param x (not used)
   * @param p The input direction
   * @param h The output Hessian-vector product
   */
  void add_hessian_product(const Vector<T>& data, const Vector<T>& x,
                           const Vector<T>& p, Vector<T>& h) const {
    int nrows;
    const int *rowp, *cols;
    T* A;
    jacobian->get_data(&nrows, nullptr, nullptr, &rowp, &cols, &A);

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

  /**
   * @brief Add the Hessian contributions to the matrix
   *
   * @param data (not used, but required for matching the signature)
   * @param x (not used)
   * @param owners
   * @param mat
   */
  void add_hessian(const Vector<T>& data, const Vector<T>& x,
                   const NodeOwners& owners, CSRMat<T>& mat) const {
    // Add the contributions to the Hessian matrix...
    mat->add_submatrix(con_indices, var_indices, jacobian);
    mat->add_submatrix(var_indices, con_indices, jacobian_transpose);
  }

  /**
   * @brief Get the data that describes the contributions to the sparse
   * constraints
   *
   * @param nrows Number of constraints
   * @param ncols Number of columns
   * @param rows Row indices
   * @param columns Column indices
   * @param rowp Pointer into the rows of each constraint
   * @param cols Local column indices columns[cols[i]] = column index
   */
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

 private:
  std::shared_ptr<Vector<int>> con_indices;
  std::shared_ptr<Vector<int>> var_indices;
  std::shared_ptr<Vector<T>> constraints;
  std::shared_ptr<CSRMat<T>> jacobian;
  std::shared_ptr<CSRMat<T>> jacobian_transpose;
  std::shared_ptr<ExternalComponentEvaluation<T>> extrn;
};

}  // namespace amigo

#endif  // AMIGO_COMPONENT_EXTERNAL_H