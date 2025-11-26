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
  ExternalComponentEvaluation(int nvars, int ncon, const int jac_rowp[],
                              const int jac_cols[],
                              const int hess_rowp[] = nullptr,
                              const int hess_cols[] = nullptr)
      : ncon(ncon), nvars(nvars) {
    x = std::make_shared<Vector<T>>(nvars + ncon);

    // Objective and constraints
    fobj = T(0.0);
    constraints = std::make_shared<Vector<T>>(ncon);

    // Gradients and Jacobian
    fobj_gradient = std::make_shared<Vector<T>>(nvars);

    int jac_nnz = jac_rowp[ncon];
    jacobian = CSRMat<T>::create_from_csr_data(ncon, nvars, jac_nnz, jac_rowp,
                                               jac_cols);

    // Hessian contribution
    hessian = nullptr;
    if (hess_rowp) {
      int hess_nnz = hess_rowp[nvars];
      hessian = CSRMat<T>::create_from_csr_data(nvars, nvars, hess_nnz,
                                                hess_rowp, hess_cols);
    }
  }
  virtual ~ExternalComponentEvaluation() {}

  int get_num_constraints() const { return ncon; }
  int get_num_variables() const { return nvars; }

  // Get the design variable vector
  std::shared_ptr<Vector<T>> get_variables() { return x; }
  const std::shared_ptr<Vector<T>> get_variables() const { return x; }

  // Grab non-constant pointers
  T& get_objective() { return fobj; }
  std::shared_ptr<Vector<T>> get_constraints() { return constraints; }
  std::shared_ptr<Vector<T>> get_objective_gradient() { return fobj_gradient; }
  std::shared_ptr<CSRMat<T>> get_jacobian() { return jacobian; }
  std::shared_ptr<CSRMat<T>> get_hessian() { return hessian; }

  // Grab constant pointers
  const T& get_objective() const { return fobj; }
  std::shared_ptr<const Vector<T>> get_constraints() const {
    return constraints;
  }
  std::shared_ptr<const Vector<T>> get_objective_gradient() const {
    return fobj_gradient;
  }
  std::shared_ptr<const CSRMat<T>> get_jacobian() const { return jacobian; }
  std::shared_ptr<const CSRMat<T>> get_hessian() const { return hessian; }

  // Evaluate the objective, constraints and
  virtual void evaluate() = 0;

 protected:
  int ncon, nvars;

  // The design variables
  std::shared_ptr<Vector<T>> x;

  // The objective and constraint values
  T fobj;
  std::shared_ptr<Vector<T>> constraints;

  // Gradient and constraint Jacobian
  std::shared_ptr<Vector<T>> fobj_gradient;
  std::shared_ptr<CSRMat<T>> jacobian;

  // Hessian matrix
  std::shared_ptr<CSRMat<T>> hessian;
};

/**
 * @brief An external component that contributes constraint
 *
 * @tparam T Template type for the computation
 */
template <typename T, ExecPolicy policy>
class ExternalComponentGroup : public ComponentGroupBase<T, policy> {
 public:
  ExternalComponentGroup(int nvars, const int vars[], int ncon,
                         const int cons[],
                         std::shared_ptr<ExternalComponentEvaluation<T>> extrn)
      : extrn(extrn) {
    var_indices = std::make_shared<Vector<int>>(nvars);
    var_indices->copy(vars);
    con_indices = std::make_shared<Vector<int>>(ncon);
    con_indices->copy(cons);

    std::shared_ptr<const CSRMat<T>> jacobian = extrn->get_jacobian();
    jacobian_transpose = jacobian->transpose();
  }

  /**
   * @brief This is not a clone-able derived class
   */
  std::shared_ptr<ComponentGroupBase<T, policy>> clone(
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
    int ncon = extrn->get_num_constraints();
    int nvars = extrn->get_num_variables();

    const int* var_idx = var_indices->get_array();
    const int* con_idx = con_indices->get_array();

    // Copy over the design variable values and multipliers
    Vector<T>& xlocal = *extrn->get_variables();
    for (int i = 0; i < nvars; i++) {
      xlocal[i] = x[var_idx[i]];
    }
    for (int i = 0; i < ncon; i++) {
      xlocal[nvars + i] = x[con_idx[i]];
    }

    // Evaluate the objective, constraints, gradient, Jacobian and possibly the
    // Hessian matrix too, if supplied
    extrn->evaluate();

    // Retrieve the constraint Jacobian and compute its transpose
    std::shared_ptr<const CSRMat<T>> jacobian = extrn->get_jacobian();
    jacobian->copy_transpose(jacobian_transpose);

    xlocal = *extrn->get_variables();
  }

  // Compute the Lagrangian
  T lagrangian(T alpha, const Vector<T>& data, const Vector<T>& x) const {
    T value = extrn->get_objective();
    const Vector<T>& constraints = *extrn->get_constraints();

    int ncon = extrn->get_num_constraints();
    const int* con_idx = con_indices->get_array();

    for (int i = 0; i < ncon; i++) {
      T lambda = x[con_idx[i]];
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
  void add_gradient(T alpha, const Vector<T>& data, const Vector<T>& x,
                    Vector<T>& g) const {
    const std::shared_ptr<CSRMat<T>> jacobian = extrn->get_jacobian();
    const Vector<T>& constraints = *extrn->get_constraints();
    const int* var_idx = var_indices->get_array();
    const int* con_idx = con_indices->get_array();

    int nrows;
    const int *rowp, *cols;
    const T* A;
    jacobian->get_data(&nrows, nullptr, nullptr, &rowp, &cols, &A);

    for (int i = 0; i < nrows; i++) {
      int row = con_idx[i];
      g[row] += constraints[i];

      for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
        int j = cols[jp];
        int col = var_idx[j];

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
  void add_hessian_product(T alpha, const Vector<T>& data, const Vector<T>& x,
                           const Vector<T>& p, Vector<T>& h) const {
    const int* var_idx = var_indices->get_array();
    const int* con_idx = con_indices->get_array();

    const std::shared_ptr<CSRMat<T>> jacobian = extrn->get_jacobian();
    int nrows;
    const int *rowp, *cols;
    const T* A;
    jacobian->get_data(&nrows, nullptr, nullptr, &rowp, &cols, &A);

    for (int i = 0; i < nrows; i++) {
      int row = con_idx[i];

      for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
        int j = cols[jp];
        int col = var_idx[j];

        T px = p[col];
        T plambda = p[row];
        h[col] += A[jp] * plambda;
        h[row] += A[jp] * px;
      }
    }

    const std::shared_ptr<CSRMat<T>> hessian = extrn->get_hessian();
    if (hessian) {
      hessian->get_data(&nrows, nullptr, nullptr, &rowp, &cols, &A);

      for (int i = 0; i < nrows; i++) {
        int row = var_idx[i];

        for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
          int j = cols[jp];
          int col = var_idx[j];

          T px = p[col];
          h[row] += A[jp] * px;
        }
      }
    }
  }

  /**
   * @brief Add the Hessian contributions to the matrix
   *
   * @param data (not used, but required for matching the signature)
   * @param x (not used)
   * @param owners (not used)
   * @param mat The terms added to the Hessian matrix
   */
  void add_hessian(T alpha, const Vector<T>& data, const Vector<T>& x,
                   const NodeOwners& owners, CSRMat<T>& mat) const {
    // Add the contributions to the Hessian matrix...
    const std::shared_ptr<CSRMat<T>> jacobian = extrn->get_jacobian();
    const std::shared_ptr<CSRMat<T>> hessian = extrn->get_hessian();

    mat.add_submatrix(con_indices->get_array(), var_indices->get_array(),
                      jacobian);
    mat.add_submatrix(var_indices->get_array(), con_indices->get_array(),
                      jacobian_transpose);

    if (hessian) {
      mat.add_submatrix(var_indices->get_array(), var_indices->get_array(),
                        hessian);
    }
  }

  /**
   * @brief Get the data that describes the contributions to the sparse
   * Hessian matrix from an external component
   *
   * @param nvars Number of design variables
   * @param vars Design variable indices
   * @param ncon Number of constraints
   * @param cons Constraint (multiplier) indices
   * @param jac_rowp Pointer into the rows of each constraint
   * @param jac_cols Local column indices columns[cols[i]] = column index
   * @param hess_rowp Pointer into the rows of each constraint
   * @param hess_cols Local column indices columns[cols[i]] = column index
   */
  void get_csr_data(int* nvars, const int* vars[], int* ncon, const int* cons[],
                    const int* jac_rowp[], const int* jac_cols[],
                    const int* hess_rowp[], const int* hess_cols[]) const {
    const std::shared_ptr<CSRMat<T>> jacobian = extrn->get_jacobian();
    const std::shared_ptr<CSRMat<T>> hessian = extrn->get_hessian();

    if (nvars) {
      *nvars = extrn->get_num_variables();
    }
    if (ncon) {
      *ncon = extrn->get_num_constraints();
    }

    if (vars) {
      *vars = var_indices->get_array();
    }
    if (cons) {
      *cons = con_indices->get_array();
    }
    jacobian->get_data(nullptr, nullptr, nullptr, jac_rowp, jac_cols, nullptr);

    if (hessian) {
      hessian->get_data(nullptr, nullptr, nullptr, hess_rowp, hess_cols,
                        nullptr);
    } else {
      // This object does not define a Hessian matrix
      if (hess_rowp) {
        *hess_rowp = nullptr;
      }
      if (hess_cols) {
        *hess_cols = nullptr;
      }
    }
  }

 private:
  std::shared_ptr<Vector<int>> con_indices;
  std::shared_ptr<Vector<int>> var_indices;

  std::shared_ptr<CSRMat<T>> jacobian_transpose;
  std::shared_ptr<ExternalComponentEvaluation<T>> extrn;
};

}  // namespace amigo

#endif  // AMIGO_COMPONENT_EXTERNAL_H