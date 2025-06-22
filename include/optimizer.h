#ifndef AMIGO_OPTIMIZER_H
#define AMIGO_OPTIMIZER_H

#include "component_group_base.h"

namespace amigo {

/**
 * @brief Class for implementing the primary numerical contributions from an
 * interior point method compatible with Amigo.
 *
 * The problem is
 *
 * min f(x)
 * st. c(x) - s = 0
 * st. lbx <= x <= ubx
 * st. lbc <= s <= ubc
 *
 * The primal-dual residual is computed from the KKT conditions. The residual
 * contains both primal and dual components intermixed.
 *
 * [ grad(x) + A^{T} * lam ] - zlx + zux = 0
 * [                  c(x) ] - s         = 0
 *                      -lam - zlc + zuc = 0
 *        (xs - lb) * zl - barrier_param = 0
 *        (ub - xs) * zu - barrier_param = 0
 *
 * Here xlam represents the full primal-dual vector, zl and zu are the lower and
 * upper bounds for the design variables and slacks.
 *
 * The terms inside the square brackets are computed directly by the
 * optimization problem. The remaining terms are handled by the code in the
 * optimizer class.
 *
 * Linearizing the residual about the primal-dual point gives
 *
 * [ H  |  A^T ][ px   ] - pzlx + pzux = [ bx ]
 * [ A  |    0 ][ plam ] - ps          = [ bc ]
 *                 -plam - pzlc + pzuc = blam
 *          (xs - lb) * pzl + pxs * zl = bzl
 *          (ub - xs) * pzu - pxs * zu = bzu
 *
 * This system of equations can be simplified by solving for the updates pzl and
 * pzu in terms of pxs
 *
 * pzl = - (xs - lb)^{-1} * zl * pxs + (xs - lb)^{-1} * bzl
 * pzu =   (ub - xs)^{-1} * zu * pxs + (ub - xs)^{-1} * bzu
 *
 * Now, substituting these expressions into the equation for plam gives:
 *
 * plam = - pzlc + pzuc + blam
 *      = C * ps + blam - (s - lbc)^{-1} * bzlc + (ubc - s)^{-1} * bzuc
 *
 * Where C is defined as
 *
 * C = ((s - lbc)^{-1} * zlc + (ubc - s)^{-1} * zuc)
 *
 * This can be re-arranged to find
 *
 * ps = C^{-1} * plam -
 *   C^{-1} * (blam - (s - lbc)^{-1} * bzlc + (ubc - s)^{-1} * bzuc)
 *
 * Subsituting this into the second equation gives
 *
 * A * px - C^{-1} * plam +
 * C^{-1} * (blam - (s - lb)^{-1} * bzl + (ub - xs)^{-1} * bzu) = bc
 *
 * This can be expressed as
 *
 * A * px - C^{-1} * plam = rc
 *
 * where rc is:
 *
 * rc = bc - C^{-1} * blam + C^{-1} * (s - lbc)^{-1} * bzlc -
 *      C^{-1} * (ubc - s)^{-1} * bzuc
 *
 * Finding pzlx and pzux from pzl and pzu gives
 *
 * pzlx = - (x - lbx)^{-1} * zlx * px + (x - lbx)^{-1} * bzlx
 * pzux =   (ubx - x)^{-1} * zux * px + (ubx - x)^{-1} * bzux
 *
 * Substituting these into the first expression gives
 *
 * (H + D) * px + A^{T} * plam - (x - lbx)^{-1} * bzlx +
 *       (ubx - x)^{-1} * bzux = bx
 *
 * Simplifying gives
 *
 * (H + D) * px + A^{T} * plam = rx
 *
 * Where rx is
 *
 * rx = bx + (x - lbx)^{-1} * bzlx - (ubx - x)^{-1} * bzux
 *
 * Requiring the solution of the linear system:
 *
 * [(H + D) |   A^{T} ][ px   ] = [ rx ]
 * [A       | -C^{-1} ][ plam ] = [ rc ]
 *
 * @tparam T
 */

template <typename T>
class OptVector {
 public:
  OptVector(std::shared_ptr<OptimizationProblem<T>> problem) {
    x = problem->create_vector();
    xs = problem->create_vector();
    zl = problem->create_vector();
    zu = problem->create_vector();
  }
  std::shared_ptr<Vector<T>> x;   // The primal-dual variables
  std::shared_ptr<Vector<T>> xs;  // Primal-slack variables
  std::shared_ptr<Vector<T>> zl;  // Multipliers for the lower bounds
  std::shared_ptr<Vector<T>> zu;  // Multipliers for the upper bounds
};

template <typename T>
class Optimizer {
 public:
  Optimizer(std::shared_ptr<OptimizationProblem<T>> problem,
            std::shared_ptr<Vector<T>> lb, std::shared_ptr<Vector<T>> ub)
      : problem(problem), lb(lb), ub(ub) {}

  /**
   * @brief Create an instance of the optimization state vector
   *
   * @return std::shared_ptr<OptVector<T>>
   */
  std::shared_ptr<OptVector<T>> create_vector() {
    return std::make_shared<OptVector<T>>(problem);
  }

  /**
   * @brief Compute the primal-dual residual based on the value of the gradient
   *
   * res.x = [ nabla f + A^{T} * lam ] - zlx + zux
   *         [                  c(x) ] - s
   *
   * res.xs = - lam - zlc + zuc
   * res.zl = (xs - lb) * zl - barrier_param
   * res.zu = (ub - sx) * zu - barrier_param
   *
   * @param barrier_param The barrier parameter for the residual
   */
  void compute_residual(T barrier_param,
                        const std::shared_ptr<OptVector<T>> vars,
                        const std::shared_ptr<Vector<T>> grad,
                        std::shared_ptr<OptVector<T>> res) {
    // Compute the residual of the full KKT system
    res->x->copy(grad);
    res->x->axpy();
  }

  /**
   * @brief Compute the reduced residual from the full primal-dual residual
   *
   * @param vars The values of the optimization variables
   * @param res The full residual vector
   * @param reduced The reduced residual vector
   */
  void compute_reduced_residual(const std::shared_ptr<OptVector<T>> vars,
                                const std::shared_vector<OptVector<T>> res,
                                std::shared_ptr<Vector<T>> reduced) {
    // reduced = [ rx ]
    //           [ rc ]
    // rx = bx + (x - lbx)^{-1} * bzlx - (ubx - x)^{-1} * bzux
    // rc = bc - C^{-1} * blam + C^{-1} * (s - lbc)^{-1} * bzlc -
    //         C^{-1} * (ubc - s)^{-1} * bzuc
  }

  /**
   * @brief Compute the update for the full set of primal-dual variables
   *
   * @param vars The values of the optimization variables
   * @param px The update to the reduced design variables
   * @param update The full update of the optimization variables
   */
  void compute_update_from_reduced(const std::shared_ptr<OptVector<T>> vars,
                                   const std::shared_ptr<Vector<T>> px,
                                   std::shared_ptr<OptVector<T>> update) {}

  /**
   * @brief Compute the full KKT matrix, including design variables
   */
  void hessian(const std::shared_ptr<OptVector<T>> vars) {}

 private:
  /**
   * Compute the diagonal entries of the matrix
   *
   * [ D |         ]
   * [ 0 | -C^{-1} ]
   */
  void set_diagonal(const Vector<T> xs, Vector<T> diag) {
    for (int i = 0; i < num_variables; i++) {
      diag[i] = 0.0;
      if (lb[i] < ub[i]) {
        // If the lower bound isn't infinite, add its value
        if (!std::isinf(lb[i])) {
          diag[i] += zl[i] / (xs[i] - lb[i]);
        }

        // If the upper bound isn't infinite, add its value
        if (!std::isinf(ub[i])) {
          diag[i] += zu[i] / (ub[i] - xs[i]);
        }
      }
    }

    // Now, set the values of -C^{-1}
    for (int i = 0; i < num_constraints; i++) {
      int index = constraints[i];
      if (diag[index] != 0.0) {
        diag[index] = -1.0 / diag[index];
      }
    }
  }

  // The optimization problem
  std::shared_ptr<OptimizationProblem<T>> problem;

  // Lower and upper bounds for the design variables
  std::shared_ptr<Vector<T>> lb, ub;
};

}  // namespace amigo

#endif  // AMIGO_OPTIMIZER_H