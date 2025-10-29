#ifndef AMIGO_COMPONENT_EXTERNAL_H
#define AMIGO_COMPONENT_EXTERNAL_H

#include "component_group_base.h"
#include "csr_matrix.h"

namespace amigo {

/**
 * @brief Interface for the external constraint evaluation
 *
 * @tparam T Template type for the computation
 */
template <typename T>
class ExternalComponentEvaluation {
 public:
  ExternalComponentEvaluation(int ncon, const int cons[], int nvars,
                              const int vars[], const int rowp[],
                              const int cols[])
      : ncon(ncon), nvars(nvars) {
    con_idx = std::make_shared<Vector<int>>(ncon);
    con_idx->copy(cons);
    var_idx = std::make_shared<Vector<int>>(nvars);
    var_idx->copy(vars);

    constraints = std::make_shared<Vector<T>>(ncon);

    int nnz = cols[ncon];
    jacobian = CSRMat<T>::create_from_csr_data(ncon, nvars, nnz, rowp, cols);
  }
  virtual ~ExternalComponentEvaluation() {}

  int get_num_constraints() const { return ncon; }
  int get_num_variables() const { return nvars; }
  const std::shared_ptr<Vector<int>> get_constraint_indices() const {
    return con_idx;
  }
  const std::shared_ptr<Vector<int>> get_variable_indices() const {
    return var_idx;
  }
  const std::shared_ptr<Vector<T>> get_constraints() const {
    return constraints;
  }
  const std::shared_ptr<CSRMat<T>> get_jacobian() const { return jacobian; }

  virtual void evaluate(const Vector<T>& x) = 0;

 protected:
  int ncon, nvars;
  std::shared_ptr<Vector<int>> con_idx;
  std::shared_ptr<Vector<int>> var_idx;
  std::shared_ptr<Vector<T>> constraints;
  std::shared_ptr<CSRMat<T>> jacobian;
};

/**
 * @brief An external component that contributes constraint
 *
 * @tparam T Template type for the computation
 */
template <typename T>
class ExternalComponent : public ComponentGroupBase<T> {
 public:
  ExternalComponent(std::shared_ptr<ExternalComponentEvaluation<T>> extrn)
      : extrn(extrn) {
    const std::shared_ptr<CSRMat<T>> jacobian = extrn->get_jacobian();
    jacobian_transpose = jacobian->transpose();
  }

  /**
   * @brief This is not a clone-able derived class
   */
  std::shared_ptr<ComponentGroupBase<T>> clone(
      int num_elements, std::shared_ptr<Vector<int>> data_idx,
      std::shared_ptr<Vector<int>> layout_idx,
      std::shared_ptr<Vector<int>> output_idx) const {
    return nullptr;
  }

  /**
   * @brief Update the gradient evaluation
   *
   * @param x The design variable values
   */
  void update(const Vector<T>& x) {
    extrn->evaluate(x);

    const std::shared_ptr<CSRMat<T>> jacobian = extrn->get_jacobian();
    jacobian_transpose->copy_transpose(jacobian);
  }

  // Compute the Lagrangian
  T lagrangian(const Vector<T>& data, const Vector<T>& x) const {
    T value(0.0);

    const Vector<int>& con_indices = *extrn->get_constraint_indices();
    const Vector<T>& constraints = *extrn->get_constraints();
    int ncon = extrn->get_num_constraints();

    for (int i = 0; i < ncon; i++) {
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
    const std::shared_ptr<CSRMat<T>> jacobian = extrn->get_jacobian();
    const Vector<int>& con_indices = *extrn->get_constraint_indices();
    const Vector<int>& var_indices = *extrn->get_variable_indices();
    const Vector<T>& constraints = *extrn->get_constraints();

    int nrows;
    const int *rowp, *cols;
    const T* A;
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
    const std::shared_ptr<CSRMat<T>> jacobian = extrn->get_jacobian();
    const Vector<int>& con_indices = *extrn->get_constraint_indices();
    const Vector<int>& var_indices = *extrn->get_variable_indices();

    int nrows;
    const int *rowp, *cols;
    const T* A;
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
    const std::shared_ptr<CSRMat<T>> jacobian = extrn->get_jacobian();
    const Vector<int>& con_indices = *extrn->get_constraint_indices();
    const Vector<int>& var_indices = *extrn->get_variable_indices();

    mat.add_submatrix(con_indices.get_array(), var_indices.get_array(),
                      jacobian);
    mat.add_submatrix(var_indices.get_array(), con_indices.get_array(),
                      jacobian_transpose);
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
    const std::shared_ptr<CSRMat<T>> jacobian = extrn->get_jacobian();
    const std::shared_ptr<Vector<int>> con_indices =
        extrn->get_constraint_indices();
    const std::shared_ptr<Vector<int>> var_indices =
        extrn->get_variable_indices();

    if (rows) {
      *rows = con_indices->get_array();
    }
    if (columns) {
      *columns = var_indices->get_array();
    }
    jacobian->get_data(nrows, ncols, nullptr, rowp, cols, nullptr);
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