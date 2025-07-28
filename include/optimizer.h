#ifndef AMIGO_OPTIMIZER_H
#define AMIGO_OPTIMIZER_H

#include <mpi.h>

#include <cstdio>

#include "component_group_base.h"
#include "optimization_problem.h"

namespace amigo {

/**
 * @brief Optimization vector that stores all the info
 *
 * @tparam T Numerical type for the computations
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

  OptVector(std::shared_ptr<Vector<T>> x,
            std::shared_ptr<OptimizationProblem<T>> problem)
      : x(x) {
    xs = problem->create_vector();
    zl = problem->create_vector();
    zu = problem->create_vector();

    // Set the design variable values in xs as well
    Vector<T>& x_ = *x;
    Vector<T>& xs_ = *xs;
    int size = problem->get_num_variables();
    const Vector<int>& is_multiplier = *problem->get_multiplier_indicator();
    for (int i = 0; i < size; i++) {
      if (!is_multiplier[i]) {
        xs_[i] = x_[i];
      }
    }
  }

  void copy(const std::shared_ptr<OptVector<T>> src) {
    x->copy(*src->x);
    xs->copy(*src->xs);
    zl->copy(*src->zl);
    zu->copy(*src->zu);
  }

  std::shared_ptr<Vector<T>> x;   // The primal-dual variables
  std::shared_ptr<Vector<T>> xs;  // Primal-slack variables
  std::shared_ptr<Vector<T>> zl;  // Multipliers for the lower bounds
  std::shared_ptr<Vector<T>> zu;  // Multipliers for the upper bounds
};

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
 * - plam + C * ps = blam + (s - lbc)^{-1} * bzlc - (ubc - s)^{-1} * bzuc
 *
 * Where C is defined as
 *
 * C = ((s - lbc)^{-1} * zlc + (ubc - s)^{-1} * zuc)
 *
 * This can be re-arranged to find
 *
 * ps = C^{-1} * (plam + b2)
 *
 * where b2 = (blam + (s - lbc)^{-1} * bzlc - (ubc - s)^{-1} * bzuc)
 *
 * Subsituting this into the second equation gives
 *
 * A * px - C^{-1} * plam - C^{-1} * b2 = bc
 *
 * This can be expressed as
 *
 * A * px - C^{-1} * plam = rc
 *
 * where rc is:
 *
 * rc = bc + C^{-1} * b2
 *
 * or, expanding terms
 *
 * rc = bc + C^{-1} * (blam + (s - lbc)^{-1} * bzlc - (ubc - s)^{-1} * bzuc)
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
 * @tparam T Numerical type for the computations
 */
template <typename T>
class InteriorPointOptimizer {
 public:
  InteriorPointOptimizer(std::shared_ptr<OptimizationProblem<T>> problem,
                         std::shared_ptr<Vector<T>> lower,
                         std::shared_ptr<Vector<T>> upper)
      : problem(problem), lower(lower), upper(upper) {
    comm = problem->get_mpi_comm();
    num_variables = problem->get_num_variables();
  }

  /**
   * @brief Create an instance of the optimization state vector
   *
   * @return std::shared_ptr<OptVector<T>>
   */
  std::shared_ptr<OptVector<T>> create_opt_vector() const {
    return std::make_shared<OptVector<T>>(problem);
  }

  /**
   * @brief Create an instance of an optimization state vector with the provided
   * initial point
   *
   * @return std::shared_ptr<OptVector<T>>
   */
  std::shared_ptr<OptVector<T>> create_opt_vector(
      std::shared_ptr<Vector<T>> x) const {
    return std::make_shared<OptVector<T>>(x, problem);
  }

  /**
   * @brief Initialize the multipliers and slack variables in the problem
   *
   */
  void initialize_multipliers_and_slacks(
      std::shared_ptr<OptVector<T>> vars) const {
    // Get the multiplier indicator array
    const Vector<int>& is_multiplier = *problem->get_multiplier_indicator();

    // Extract the optimization state vectors to make things easier
    Vector<T>& x = *vars->x;    // Primal and dual variables
    Vector<T>& xs = *vars->xs;  // Primal and slack variables
    Vector<T>& zl = *vars->zl;
    Vector<T>& zu = *vars->zu;

    const Vector<T>& lb = *lower;
    const Vector<T>& ub = *upper;

    for (int i = 0; i < num_variables; i++) {
      if (is_multiplier[i]) {
        // Set the multipliers to 1.0
        x[i] = 1.0;

        // Set the slack variables to either the mid-point, or inside the
        // feasible region
        if (!std::isinf(lb[i]) && !std::isinf(ub[i])) {
          xs[i] = 0.5 * (lb[i] + ub[i]);
        } else if (!std::isinf(lb[i])) {
          if (xs[i] <= lb[i]) {
            xs[i] = lb[i] + 1.0;
          }
        } else if (!std::isinf(ub[i])) {
          if (xs[i] >= ub[i]) {
            xs[i] = ub[i] - 1.0;
          }
        }
      } else {
        if (lb[i] <= x[i] && x[i] <= ub[i]) {
          xs[i] = x[i];
        } else {
          if (!std::isinf(lb[i]) && !std::isinf(ub[i])) {
            xs[i] = x[i] = 0.5 * (lb[i] + ub[i]);
          }
        }
      }

      // Set the lower and upper multipliers to 1.0
      zl[i] = 1.0;
      zu[i] = 1.0;
    }
  }

  /**
   * @brief Make sure that the design variables are consistent between x and xs
   *
   * @param vars The variable vector
   */
  void make_vars_consistent(std::shared_ptr<OptVector<T>> vars) const {
    // Set the design variable values in xs as well
    Vector<T>& x = *vars->x;
    Vector<T>& xs = *vars->xs;
    const Vector<int>& is_multiplier = *problem->get_multiplier_indicator();

    for (int i = 0; i < num_variables; i++) {
      if (!is_multiplier[i]) {
        xs[i] = x[i];
      }
    }
  }

  /**
   * @brief Compute the negative of the primal-dual residuals based on the value
   * of the gradient and the optimizer state variables
   *
   * res.x = -( [ nabla f + A^{T} * lam ] - zlx + zux )
   *          ( [                  c(x) ] - s         )
   *
   * res.xs = -(-lam - zlc + zuc)
   * res.zl = -((xs - lb) * zl - barrier_param)
   * res.zu = -((ub - sx) * zu - barrier_param)
   *
   * @param barrier_param The barrier parameter for the residual
   * @param vars The optimization variables
   * @param grad The gradient computed from the problem
   * @param res The full KKT residual
   */
  T compute_residual(T barrier_param, const std::shared_ptr<OptVector<T>> vars,
                     const std::shared_ptr<Vector<T>> grad,
                     std::shared_ptr<OptVector<T>> res) const {
    const Vector<int>& is_multiplier = *problem->get_multiplier_indicator();

    // Extract the vectors to make things easier
    const Vector<T>& x = *vars->x;    // Primal and dual variables
    const Vector<T>& xs = *vars->xs;  // Primal and slack variables
    const Vector<T>& zl = *vars->zl;
    const Vector<T>& zu = *vars->zu;
    const Vector<T>& lb = *lower;
    const Vector<T>& ub = *upper;
    const Vector<T>& g = *grad;

    // Set references to the residuals
    Vector<T>& rx = *res->x;
    Vector<T>& rxs = *res->xs;
    Vector<T>& rzl = *res->zl;
    Vector<T>& rzu = *res->zu;

    // Compute the residual of the full KKT system
    for (int i = 0; i < num_variables; i++) {
      if (is_multiplier[i]) {
        // This term is -(c(x) - s)
        rx[i] = -(g[i] - xs[i]);
      } else {
        // This term is nabla -(f - A^{T} * lam - zl + zu)
        rx[i] = -(g[i] - zl[i] + zu[i]);
      }

      // This term is -(-lam - zl + zu)
      rxs[i] = 0.0;
      if (is_multiplier[i]) {
        if (lb[i] < ub[i]) {
          rxs[i] = x[i];
          if (!std::isinf(lb[i])) {
            rxs[i] += zl[i];
          }
          if (!std::isinf(ub[i])) {
            rxs[i] -= zu[i];
          }
        }
      }

      // These terms are:
      // rzl = -((xs - lb) * zl - barrier_param)
      // rzu = -((ub - xs) * zu - barrier_param)
      rzl[i] = 0.0;
      rzu[i] = 0.0;
      if (lb[i] < ub[i]) {
        if (!std::isinf(lb[i])) {
          rzl[i] = barrier_param - (xs[i] - lb[i]) * zl[i];
        }
        if (!std::isinf(ub[i])) {
          rzu[i] = barrier_param - (ub[i] - xs[i]) * zu[i];
        }
      }
    }

    // Compute the residual norm
    T local_norm = rx.dot(rx) + rxs.dot(rxs) + rzl.dot(rzl) + rzu.dot(rzu);
    T norm;
    MPI_Allreduce(&local_norm, &norm, 1, get_mpi_type<T>(), MPI_SUM, comm);

    return std::sqrt(norm);
  }

  /**
   * @brief Compute the reduced residual from the full primal-dual residual
   *
   * reduced = [ bx + (x - lbx)^{-1} * bzlx - (ubx - x)^{-1} * bzux ]
   *           [ bc + C^{-1} * (blam + (s - lbc)^{-1} * bzlc
   *                - (ubc - s)^{-1} * bzuc) ]
   *
   * @param vars The values of the optimization variables
   * @param res The full residual vector
   * @param reduced The reduced residual vector
   */
  void compute_reduced_residual(const std::shared_ptr<OptVector<T>> vars,
                                const std::shared_ptr<OptVector<T>> res,
                                std::shared_ptr<Vector<T>> reduced) const {
    const Vector<int>& is_multiplier = *problem->get_multiplier_indicator();

    // Extract the optimization state vectors to make things easier
    const Vector<T>& xs = *vars->xs;  // Primal and slack variables
    const Vector<T>& zl = *vars->zl;
    const Vector<T>& zu = *vars->zu;
    const Vector<T>& lb = *lower;
    const Vector<T>& ub = *upper;

    // Set references to the input residuals
    const Vector<T>& bx = *res->x;
    const Vector<T>& bxs = *res->xs;
    const Vector<T>& bzl = *res->zl;
    const Vector<T>& bzu = *res->zu;

    // Set references for the reduced residuals
    Vector<T>& rx = *reduced;

    for (int i = 0; i < num_variables; i++) {
      rx[i] = bx[i];

      T contrib = 0.0;
      if (lb[i] < ub[i]) {
        if (!std::isinf(lb[i])) {
          contrib += bzl[i] / (xs[i] - lb[i]);
        }
        if (!std::isinf(ub[i])) {
          contrib -= bzu[i] / (ub[i] - xs[i]);
        }
      }

      if (is_multiplier[i]) {
        T Cinv = compute_Cinv(lb[i], ub[i], zl[i], zu[i], xs[i]);
        rx[i] += Cinv * (bxs[i] + contrib);
      } else {
        rx[i] += contrib;
      }
    }
  }

  /**
   * @brief Compute the update for the full set of primal-dual variables
   *
   * @param vars The values of the optimization variables
   * @param res The full space residual
   * @param reduced_update The update to the reduced design variables
   * @param update The full update of the optimization variables
   */
  void compute_update_from_reduced(
      const std::shared_ptr<OptVector<T>> vars,
      const std::shared_ptr<OptVector<T>> res,
      const std::shared_ptr<Vector<T>> reduced_update,
      std::shared_ptr<OptVector<T>> update) const {
    // Get the multiplier indicator array
    const Vector<int>& is_multiplier = *problem->get_multiplier_indicator();

    // Extract the optimization state vectors to make things easier
    const Vector<T>& xs = *vars->xs;  // Primal and slack variables
    const Vector<T>& zl = *vars->zl;
    const Vector<T>& zu = *vars->zu;
    const Vector<T>& lb = *lower;
    const Vector<T>& ub = *upper;

    // Set references to the input residuals
    const Vector<T>& bxs = *res->xs;
    const Vector<T>& bzl = *res->zl;
    const Vector<T>& bzu = *res->zu;

    // Set references for the full update
    Vector<T>& px = *update->x;
    Vector<T>& pxs = *update->xs;
    Vector<T>& pzl = *update->zl;
    Vector<T>& pzu = *update->zu;

    // Copy the update for the design variables and dual variables
    px.copy(*reduced_update);

    // Set the values into the full update
    for (int i = 0; i < num_variables; i++) {
      pxs[i] = 0.0;
      if (is_multiplier[i]) {
        // This is a multipler, compute the update for the corresponding slack
        // variable
        T Cinv = compute_Cinv(lb[i], ub[i], zl[i], zu[i], xs[i]);

        // contrib = (s - lc)^{-1} * bzl - (uc - s)^{-1} * bzu
        T contrib = 0.0;
        if (lb[i] < ub[i]) {
          if (!std::isinf(lb[i])) {
            contrib += bzl[i] / (xs[i] - lb[i]);
          }
          if (!std::isinf(ub[i])) {
            contrib -= bzu[i] / (ub[i] - xs[i]);
          }
        }

        // Compute the update for the slacks, multipliers and bounds
        // ps = C^{-1} * (plam + blam + contrib))
        pxs[i] = Cinv * (px[i] + bxs[i] + contrib);
      } else {
        pxs[i] = px[i];
      }

      pzl[i] = 0.0;
      pzu[i] = 0.0;

      // Compute the update for the design variable bounds
      if (lb[i] < ub[i]) {
        if (!std::isinf(lb[i])) {
          pzl[i] = (bzl[i] - zl[i] * pxs[i]) / (xs[i] - lb[i]);
        }
        if (!std::isinf(ub[i])) {
          pzu[i] = (bzu[i] + zu[i] * pxs[i]) / (ub[i] - xs[i]);
        }
      }
    }
  }

  /**
   * @brief Add the diagonal contributions from the matrix
   *
   * This code computes the components of the matrix:
   *
   * [ D |         ]
   * [ 0 | -C^{-1} ]
   *
   * @param vars The values of the optimization variables
   * @param diag The CSR Matrix for the Hessian
   */
  void compute_diagonal(const std::shared_ptr<OptVector<T>> vars,
                        std::shared_ptr<Vector<T>> diagonal) const {
    const Vector<int>& is_multiplier = *problem->get_multiplier_indicator();

    const Vector<T>& xs = *vars->xs;  // Primal and slack variables
    const Vector<T>& zl = *vars->zl;
    const Vector<T>& zu = *vars->zu;
    const Vector<T>& lb = *lower;
    const Vector<T>& ub = *upper;

    Vector<T>& diag = *diagonal;

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

      if (is_multiplier[i]) {
        // Set the values of -C^{-1}
        if (diag[i] != 0.0) {
          diag[i] = -1.0 / diag[i];
        }
      }
    }
  }

  /**
   * @brief Compute the max step length given the fraction to the boundary
   *
   * @param tau Fractional step to the boundary
   * @param vars The optimization state variables
   * @param update The update to the optimization state variables
   * @param alpha_x_max The step in the primal variables
   * @param alpha_z_max The step in the dual variables
   */
  void compute_max_step(const T tau, const std::shared_ptr<OptVector<T>> vars,
                        const std::shared_ptr<OptVector<T>> update,
                        T& alpha_x_max, T& alpha_z_max) const {
    // Extract the optimization state vectors to make things easier
    const Vector<T>& xs = *vars->xs;  // Primal and slack variables
    const Vector<T>& zl = *vars->zl;
    const Vector<T>& zu = *vars->zu;
    const Vector<T>& lb = *lower;
    const Vector<T>& ub = *upper;

    // Set references for the full update
    Vector<T>& pxs = *update->xs;
    Vector<T>& pzl = *update->zl;
    Vector<T>& pzu = *update->zu;

    // Set the max step for the design variables and multipliers
    alpha_x_max = 1.0;
    alpha_z_max = 1.0;

    // Check for steps lengths for the design variables and slacks
    for (int i = 0; i < num_variables; i++) {
      if (lb[i] < ub[i]) {
        if (!std::isinf(lb[i]) && pxs[i] < 0.0) {
          T numer = xs[i] - lb[i];
          T alpha = -tau * numer / pxs[i];
          if (alpha < alpha_x_max) {
            alpha_x_max = alpha;
          }
        }
        if (!std::isinf(ub[i]) && pxs[i] > 0.0) {
          T numer = ub[i] - xs[i];
          T alpha = tau * numer / pxs[i];
          if (alpha < alpha_x_max) {
            alpha_x_max = alpha;
          }
        }
      }
    }

    // Check step lengths for the multipliers
    for (int i = 0; i < num_variables; i++) {
      if (lb[i] < ub[i]) {
        if (!std::isinf(lb[i]) && pzl[i] < 0.0) {
          T alpha = -tau * zl[i] / pzl[i];
          if (alpha < alpha_z_max) {
            alpha_z_max = alpha;
          }
        }
        if (!std::isinf(ub[i]) && pzu[i] < 0.0) {
          T alpha = -tau * zu[i] / pzu[i];
          if (alpha < alpha_z_max) {
            alpha_z_max = alpha;
          }
        }
      }
    }

    T alphas[2], max_alphas[2];
    alphas[0] = alpha_x_max;
    alphas[1] = alpha_z_max;
    MPI_Allreduce(alphas, max_alphas, 2, get_mpi_type<T>(), MPI_MIN, comm);
    alpha_x_max = max_alphas[0];
    alpha_z_max = max_alphas[1];
  }

  /**
   * @brief Apply the update with the provided step lengths
   *
   * @param alpha_x Step length for the primal variables
   * @param alpha_z Step length for the dual variables
   * @param update Update to the state variables
   * @param vars Input/output variables that are updated
   */
  void apply_step_update(const T alpha_x, const T alpha_z,
                         const std::shared_ptr<OptVector<T>> update,
                         std::shared_ptr<OptVector<T>> vars) const {
    // Get the multiplier indicator array
    const Vector<int>& is_multiplier = *problem->get_multiplier_indicator();

    // Extract the optimization state vectors to make things easier
    Vector<T>& x = *vars->x;    // Primal and dual variables
    Vector<T>& xs = *vars->xs;  // Primal and slack variables
    Vector<T>& zl = *vars->zl;
    Vector<T>& zu = *vars->zu;

    // Set references for the full update
    const Vector<T>& px = *update->x;
    const Vector<T>& pxs = *update->xs;
    const Vector<T>& pzl = *update->zl;
    const Vector<T>& pzu = *update->zu;

    for (int i = 0; i < num_variables; i++) {
      if (is_multiplier[i]) {
        // Apply the update to the multipliers
        x[i] += alpha_z * px[i];

        // Apply the update to the slack variables
        xs[i] += alpha_x * pxs[i];
      } else {
        // Apply the update to the design variables
        x[i] += alpha_x * px[i];

        // Copy the design variable to the (x, s) vector
        xs[i] = x[i];
      }

      // Update the bound multipliers (x and s)
      zl[i] += alpha_z * pzl[i];
      zu[i] += alpha_z * pzu[i];
    }
  }

  /**
   * @brief Check whether the computed updates satisfy the linearization of the
   * optimality conditions
   *
   * @param vars Optimization state variables
   * @param res The full system of residuals
   * @param update The update to the full system
   * @param hessian The Hessian matrix - must be recomputed
   */
  void check_update(const std::shared_ptr<OptVector<T>> vars,
                    const std::shared_ptr<OptVector<T>> res,
                    const std::shared_ptr<OptVector<T>> update,
                    const std::shared_ptr<CSRMat<T>> hessian) {
    // Recompute the Hessian so it doesn't have the diagonal terms
    problem->hessian(vars->x, hessian);

    // Get the multiplier indicator array
    const Vector<int>& is_multiplier = *problem->get_multiplier_indicator();

    // Extract the optimization state vectors to make things easier
    const Vector<T>& xs = *vars->xs;  // Primal and slack variables
    const Vector<T>& zl = *vars->zl;
    const Vector<T>& zu = *vars->zu;
    const Vector<T>& lb = *lower;
    const Vector<T>& ub = *upper;

    // Set references for the full update
    const Vector<T>& px = *update->x;
    const Vector<T>& pxs = *update->xs;
    const Vector<T>& pzl = *update->zl;
    const Vector<T>& pzu = *update->zu;

    // Set references for the right-hand-side
    const Vector<T>& bx = *res->x;
    const Vector<T>& bxs = *res->xs;
    const Vector<T>& bzl = *res->zl;
    const Vector<T>& bzu = *res->zu;

    std::shared_ptr<Vector<T>> tmp = problem->create_vector();
    Vector<T>& t = *tmp;

    // Form the residuals from the different expressions
    // [ H  |  A^T ][ px   ] - pzlx + pzux = [ bx ]
    // [ A  |    0 ][ plam ] - ps          = [ bc ]

    hessian->mult(update->x, tmp);
    for (int i = 0; i < num_variables; i++) {
      if (is_multiplier[i]) {
        t[i] = t[i] - pxs[i] - bx[i];
      } else {
        t[i] = t[i] - pzl[i] + pzu[i] - bx[i];
      }
    }
    std::printf("%-40s %15.6e\n", "||H * px - zl + zu - bx|| ",
                std::sqrt(tmp->dot(*tmp)));

    // -plam - pzlc + pzuc = blam
    for (int i = 0; i < num_variables; i++) {
      t[i] = -bxs[i];
      if (is_multiplier[i]) {
        if (lb[i] < ub[i]) {
          t[i] += px[i];
          if (!std::isinf(lb[i])) {
            t[i] += pzl[i];
          }
          if (!std::isinf(ub[i])) {
            t[i] -= pzu[i];
          }
        }
      }
    }
    std::printf("%-40s %15.6e\n", "||-plam - pzlc + pzuc - blam|| ",
                std::sqrt(tmp->dot(*tmp)));

    // (xs - lb) * pzl + pxs * zl = bzl
    for (int i = 0; i < num_variables; i++) {
      if (lb[i] < ub[i]) {
        if (!std::isinf(lb[i])) {
          t[i] = (xs[i] - lb[i]) * pzl[i] + pxs[i] * zl[i] - bzl[i];
        } else {
          t[i] = 0.0;
        }
      } else {
        t[i] = 0.0;
      }
    }
    std::printf("%-40s %15.6e\n", "||(xs - lb) * pzl + pxs * zl - bzl|| ",
                std::sqrt(tmp->dot(*tmp)));

    // (ub - xs) * pzu - pxs * zu = bzu
    tmp->zero();
    for (int i = 0; i < num_variables; i++) {
      if (lb[i] < ub[i]) {
        if (!std::isinf(ub[i])) {
          t[i] = (ub[i] - xs[i]) * pzu[i] - pxs[i] * zu[i] - bzu[i];
        } else {
          t[i] = 0.0;
        }
      } else {
        t[i] = 0.0;
      }
    }
    std::printf("%-40s %15.6e\n", "||(ub - xs) * pzu + pxs * zu - bzu|| ",
                std::sqrt(tmp->dot(*tmp)));
  }

 private:
  T compute_Cinv(const T lb, const T ub, const T zl, const T zu,
                 const T xs) const {
    T C = 0.0;
    if (lb < ub) {
      // If the lower bound isn't infinite, add its value
      if (!std::isinf(lb)) {
        C += zl / (xs - lb);
      }

      // If the upper bound isn't infinite, add its value
      if (!std::isinf(ub)) {
        C += zu / (ub - xs);
      }
    }

    if (C != 0.0) {
      return 1.0 / C;
    }
    return 0.0;
  }

  // The optimization problem
  std::shared_ptr<OptimizationProblem<T>> problem;

  // Lower and upper bounds for the design variables
  std::shared_ptr<Vector<T>> lower, upper;

  // The MPI communicator
  MPI_Comm comm;

  // The number of primal and dual variables
  int num_variables;
};

}  // namespace amigo

#endif  // AMIGO_OPTIMIZER_H