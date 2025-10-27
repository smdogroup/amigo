#ifndef AMIGO_OPTIMIZER_H
#define AMIGO_OPTIMIZER_H

#include <mpi.h>

#include <cstdio>

#include "a2dcore.h"
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
  OptVector(int num_variables, int num_equalities, int num_inequalities,
            std::shared_ptr<Vector<T>> x)
      : num_variables(num_variables),
        num_equalities(num_equalities),
        num_inequalities(num_inequalities),
        x(x) {
    size = 2 * num_variables + 9 * num_inequalities;
    array = new T[size];
    std::fill(array, array + size, 0.0);
  }
  ~OptVector() { delete[] array; }

  void zero() {
    x->zero();
    std::fill(array, array + size, 0.0);
  }
  void copy(std::shared_ptr<OptVector<T>> src) {
    x->copy(*src->x);
    std::copy(src->array, src->array + size, array);
  }

  void get_bound_duals(T** zl, T** zu) {
    if (zl) {
      *zl = &array[0];
    }
    if (zu) {
      *zu = &array[num_variables];
    }
  }
  void get_bound_duals(const T** zl, const T** zu) const {
    if (zl) {
      *zl = &array[0];
    }
    if (zu) {
      *zu = &array[num_variables];
    }
  }
  void get_slacks(T** s, T** sl, T** tl, T** su, T** tu) {
    const int offset = 2 * num_variables;
    if (s) {
      *s = &array[offset];
    }
    if (sl) {
      *sl = &array[offset + num_inequalities];
    }
    if (tl) {
      *tl = &array[offset + 2 * num_inequalities];
    }
    if (su) {
      *su = &array[offset + 3 * num_inequalities];
    }
    if (tu) {
      *tu = &array[offset + 4 * num_inequalities];
    }
  }
  void get_slacks(const T** s, const T** sl, const T** tl, const T** su,
                  const T** tu) const {
    const int offset = 2 * num_variables;
    if (s) {
      *s = &array[offset];
    }
    if (sl) {
      *sl = &array[offset + num_inequalities];
    }
    if (tl) {
      *tl = &array[offset + 2 * num_inequalities];
    }
    if (su) {
      *su = &array[offset + 3 * num_inequalities];
    }
    if (tu) {
      *tu = &array[offset + 4 * num_inequalities];
    }
  }
  void get_slack_duals(T** zsl, T** ztl, T** zsu, T** ztu) {
    const int offset = 2 * num_variables + 5 * num_inequalities;
    if (zsl) {
      *zsl = &array[offset];
    }
    if (ztl) {
      *ztl = &array[offset + num_inequalities];
    }
    if (zsu) {
      *zsu = &array[offset + 2 * num_inequalities];
    }
    if (ztu) {
      *ztu = &array[offset + 3 * num_inequalities];
    }
  }
  void get_slack_duals(const T** zsl, const T** ztl, const T** zsu,
                       const T** ztu) const {
    const int offset = 2 * num_variables + 5 * num_inequalities;
    if (zsl) {
      *zsl = &array[offset];
    }
    if (ztl) {
      *ztl = &array[offset + num_inequalities];
    }
    if (zsu) {
      *zsu = &array[offset + 2 * num_inequalities];
    }
    if (ztu) {
      *ztu = &array[offset + 3 * num_inequalities];
    }
  }

  std::shared_ptr<Vector<T>> get_solution() { return x; }
  const std::shared_ptr<Vector<T>> get_solution() const { return x; }

 private:
  int num_variables;
  int num_equalities;
  int num_inequalities;
  std::shared_ptr<Vector<T>> x;

  int size;
  T* array;
};

/**
 * @brief Class for implementing the primary numerical contributions from an
 * interior point method compatible with Amigo.
 *
 * The problem is
 *
 * min f(x)
 * st. lbx <= x < ubx
 * st. h(x) - lbh = 0
 * st. lbc <= c(x) <= ubc
 *
 * In Amigo, the design variables and multipliers are mixed in the solution
 * vector. The gradient provided is
 *
 * grad = [ g(x) + Ah^{T} * lamh + A^{T} * lam ]
 *        [                               h(x) ]
 *        [                               c(x) ]
 *
 * The update to the solution of the linear system:
 *
 * [ (H + D) |  Ah^{T} |  A^{T}  ][ px    ] = [ rx ]
 * [ Ah      |  0      |  0      ][ plamh ] = [ rh ]
 * [ A       |  0      | -C^{-1} ][ plam  ] = [ rc ]
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

    // Keep track of the different variable types
    num_variables = 0;
    num_equalities = 0;
    num_inequalities = 0;

    // Get the total number of variables (primal + dual)
    int size = problem->get_num_variables();
    const Vector<int>& is_multiplier = *problem->get_multiplier_indicator();

    // Find how many types of variables
    const Vector<T>& lb = *lower;
    const Vector<T>& ub = *upper;
    for (int i = 0; i < size; i++) {
      if (is_multiplier[i]) {
        if (!std::isinf(lb[i]) && !std::isinf(ub[i]) && lb[i] == ub[i]) {
          num_equalities++;
        } else {
          num_inequalities++;
        }
      } else {
        num_variables++;
      }
    }

    // Allocate vectors
    lbx.resize(num_variables);
    ubx.resize(num_variables);
    design_variable_indices.resize(num_variables);

    lbh.resize(num_equalities);
    equality_indices.resize(num_equalities);

    lbc.resize(num_inequalities);
    ubc.resize(num_inequalities);
    inequality_indices.resize(num_inequalities);

    // Set the variable indices and bounds
    int num_vars = 0, num_eq = 0, num_ineq = 0;
    for (int i = 0; i < size; i++) {
      if (is_multiplier[i]) {
        if (!std::isinf(lb[i]) && !std::isinf(ub[i]) && lb[i] == ub[i]) {
          lbh[num_eq] = lb[i];
          equality_indices[num_eq] = i;
          num_eq++;
        } else {
          lbc[num_ineq] = lb[i];
          ubc[num_ineq] = ub[i];
          inequality_indices[num_ineq] = i;
          num_ineq++;
        }
      } else {
        lbx[num_vars] = lb[i];
        ubx[num_vars] = ub[i];
        design_variable_indices[num_vars] = i;
        num_vars++;
      }
    }
  }

  /**
   * @brief Create an instance of the optimization state vector
   *
   * @return std::shared_ptr<OptVector<T>>
   */
  std::shared_ptr<OptVector<T>> create_opt_vector() const {
    return std::make_shared<OptVector<T>>(num_variables, num_equalities,
                                          num_inequalities,
                                          problem->create_vector());
  }

  /**
   * @brief Create an instance of an optimization state vector with the provided
   * initial point
   *
   * @return std::shared_ptr<OptVector<T>>
   */
  std::shared_ptr<OptVector<T>> create_opt_vector(
      std::shared_ptr<Vector<T>> x) const {
    return std::make_shared<OptVector<T>>(num_variables, num_equalities,
                                          num_inequalities, x);
  }

  /**
   * @brief Initialize the multipliers and slack variables in the problem
   *
   * @param vars All of the optimization variables
   */
  void initialize_multipliers_and_slacks(
      T barrier_param, const std::shared_ptr<Vector<T>> grad,
      std::shared_ptr<OptVector<T>> vars) const {
    // Get the dual values for the bound constraints
    T *zl, *zu;
    vars->get_bound_duals(&zl, &zu);

    // Get the slack variable values
    T *s, *sl, *tl, *su, *tu;
    vars->get_slacks(&s, &sl, &tl, &su, &tu);

    // Get the dual values for the slacks
    T *zsl, *ztl, *zsu, *ztu;
    vars->get_slack_duals(&zsl, &ztl, &zsu, &ztu);

    // Set a pointer to the vector
    const Vector<T>& g = *grad;

    // Initialize the lower and upper bound dual variables
    for (int i = 0; i < num_variables; i++) {
      zl[i] = zu[i] = 0.0;
      if (!std::isinf(lbx[i])) {
        zl[i] = barrier_param;
      }
      if (!std::isinf(ubx[i])) {
        zu[i] = barrier_param;
      }
    }

    // Initialize the slack variables
    for (int i = 0; i < num_inequalities; i++) {
      int index = inequality_indices[i];
      s[i] = -g[index];

      sl[i] = tl[i] = zsl[i] = ztl[i] = 0.0;
      su[i] = tu[i] = zsu[i] = ztu[i] = 0.0;

      if (!std::isinf(lbc[i])) {
        sl[i] = barrier_param;
        tl[i] = barrier_param;
        zsl[i] = barrier_param;
        ztl[i] = barrier_param;
      }
      if (!std::isinf(ubc[i])) {
        su[i] = barrier_param;
        tu[i] = barrier_param;
        zsu[i] = barrier_param;
        ztu[i] = barrier_param;
      }
    }
  }

  /**
   * @brief Compute the negative of the primal-dual residuals based on the value
   * of the gradient and the optimizer state variables
   *
   * @param barrier_param The barrier parameter for the residual
   * @param gamma The penalty parameter
   * @param vars The optimization variables
   * @param grad The gradient computed from the problem
   * @param res The full KKT residual
   */
  T compute_residual(T barrier_param, T gamma,
                     const std::shared_ptr<OptVector<T>> vars,
                     const std::shared_ptr<Vector<T>> grad,
                     std::shared_ptr<Vector<T>> res) const {
    // Get the dual values for the bound constraints
    const T *zl, *zu;
    vars->get_bound_duals(&zl, &zu);

    // Get the slack variable values
    const T *s, *sl, *tl, *su, *tu;
    vars->get_slacks(&s, &sl, &tl, &su, &tu);

    // Get the dual values for the slacks
    const T *zsl, *ztl, *zsu, *ztu;
    vars->get_slack_duals(&zsl, &ztl, &zsu, &ztu);

    // Get the solution vector - design variables and multipliers
    const Vector<T>& xlam = *vars->get_solution();

    // Set the gradient vector
    const Vector<T>& g = *grad;
    Vector<T>& r = *res;

    // Zero the residual
    res->zero();

    // Compute the residual for the variables
    for (int i = 0; i < num_variables; i++) {
      // Get the gradient component corresponding to this variable
      int index = design_variable_indices[i];

      // Extract the design variable value
      T x = xlam[index];

      // Compute the right-hand-side
      T bx = -(g[index] - zl[i] + zu[i]);
      if (!std::isinf(lbx[i])) {
        T bzl = -((x - lbx[i]) * zl[i] - barrier_param);
        bx += bzl / (x - lbx[i]);
      }
      if (!std::isinf(ubx[i])) {
        T bzu = -((ubx[i] - x) * zu[i] - barrier_param);
        bx -= bzu / (ubx[i] - x);
      }

      // Set the right-hand-side
      r[index] = bx;
    }

    // Compute the contributions from the equality constraints
    for (int i = 0; i < num_equalities; i++) {
      int index = equality_indices[i];

      // Set the right-hand-side for the equalities
      r[index] = -g[index];
    }

    // Compute the contributions from the inequality constraints
    for (int i = 0; i < num_inequalities; i++) {
      int index = inequality_indices[i];

      // Extract the multiplier from the solution vector
      T lam = xlam[index];

      // Set the right-hand-side values
      T bc = -(g[index] - s[i]);
      T blam = -(-lam - zsl[i] + zsu[i]);

      // Build the components of C and compute its inverse
      T C = 0.0;
      T d = blam;
      if (!std::isinf(lbc[i])) {
        // Compute the right-hand-sides for the lower bound
        T blaml = -(gamma - zsl[i] - ztl[i]);
        T bsl = -(s[i] - lbc[i] - sl[i] + tl[i]);
        T bzsl = -(sl[i] * zsl[i] - barrier_param);
        T bztl = -(tl[i] * ztl[i] - barrier_param);

        T inv_zsl = 1.0 / zsl[i];
        T inv_ztl = 1.0 / ztl[i];
        T Fl = inv_zsl * sl[i] + inv_ztl * tl[i];
        T dl = bsl + inv_zsl * bzsl - inv_ztl * (bztl + tl[i] * blaml);

        T inv_Fl = 1.0 / Fl;
        d += inv_Fl * dl;
        C += inv_Fl;
      }

      if (!std::isinf(ubc[i])) {
        T blamu = -(gamma - zsu[i] - ztu[i]);
        T bsu = -(ubc[i] - s[i] - su[i] + tu[i]);
        T bzsu = -(su[i] * zsu[i] - barrier_param);
        T bztu = -(tu[i] * ztu[i] - barrier_param);

        T inv_zsu = 1.0 / zsu[i];
        T inv_ztu = 1.0 / ztu[i];
        T Fu = inv_zsu * su[i] + inv_ztu * tu[i];
        T du = bsu + inv_zsu * bzsu - inv_ztu * (bztu + tu[i] * blamu);

        T inv_Fu = 1.0 / Fu;
        d -= inv_Fu * du;
        C += inv_Fu;
      }

      bc += d / C;

      r[index] = bc;
    }

    // Compute the residual norm
    T local_norm = res->dot(*res);
    T norm;
    MPI_Allreduce(&local_norm, &norm, 1, get_mpi_type<T>(), MPI_SUM, comm);

    return std::sqrt(norm);
  }

  /**
   * @brief Compute the update for the full set of primal-dual variables
   *
   * @param barrier_param The barrier parameter for the residual
   * @param gamma The penalty parameter
   * @param vars The values of the optimization variables
   * @param reduced_update The update to the reduced design variables
   * @param update The full update of the optimization variables
   */
  void compute_update(T barrier_param, T gamma,
                      const std::shared_ptr<OptVector<T>> vars,
                      const std::shared_ptr<Vector<T>> reduced_update,
                      std::shared_ptr<OptVector<T>> update) const {
    // Get the dual values for the bound constraints
    const T *zl, *zu;
    T *pzl, *pzu;
    vars->get_bound_duals(&zl, &zu);
    update->get_bound_duals(&pzl, &pzu);

    // Get the slack variable values
    const T *s, *sl, *tl, *su, *tu;
    T *ps, *psl, *ptl, *psu, *ptu;
    vars->get_slacks(&s, &sl, &tl, &su, &tu);
    update->get_slacks(&ps, &psl, &ptl, &psu, &ptu);

    // Get the dual values for the slacks
    const T *zsl, *ztl, *zsu, *ztu;
    T *pzsl, *pztl, *pzsu, *pztu;
    vars->get_slack_duals(&zsl, &ztl, &zsu, &ztu);
    update->get_slack_duals(&pzsl, &pztl, &pzsu, &pztu);

    // Get the solution update
    const Vector<T>& xlam = *vars->get_solution();
    Vector<T>& pxlam = *update->get_solution();

    // Copy the update for the design variables and dual variables
    pxlam.copy(*reduced_update);

    for (int i = 0; i < num_variables; i++) {
      // Get the gradient component corresponding to this variable
      int index = design_variable_indices[i];

      // Extract the design variable
      T x = xlam[index];
      T px = pxlam[index];

      // Compute the update step
      if (!std::isinf(lbx[i])) {
        T bzl = -((x - lbx[i]) * zl[i] - barrier_param);
        pzl[i] = (bzl - zl[i] * px) / (x - lbx[i]);
      }
      if (!std::isinf(ubx[i])) {
        T bzu = -((ubx[i] - x) * zu[i] - barrier_param);
        pzu[i] = (bzu + zu[i] * px) / (ubx[i] - x);
      }
    }

    for (int i = 0; i < num_inequalities; i++) {
      int index = inequality_indices[i];

      // Extract the multiplier from the solution vector
      T lam = xlam[index];
      T plam = pxlam[index];

      // Compute all the contributions to the update
      T blam = -(-lam - zsl[i] + zsu[i]);

      // Build the components of C and compute its inverse
      T C = 0.0;
      T d = blam;
      T Fl = 0.0, dl = 0.0, blaml = 0.0, bsl = 0.0, bzsl = 0.0, bztl = 0.0;
      T Fu = 0.0, du = 0.0, blamu = 0.0, bsu = 0.0, bzsu = 0.0, bztu = 0.0;

      if (!std::isinf(lbc[i])) {
        // Compute the right-hand-sides for the lower bound
        blaml = -(gamma - zsl[i] - ztl[i]);
        bsl = -(s[i] - lbc[i] - sl[i] + tl[i]);
        bzsl = -(sl[i] * zsl[i] - barrier_param);
        bztl = -(tl[i] * ztl[i] - barrier_param);

        T inv_zsl = 1.0 / zsl[i];
        T inv_ztl = 1.0 / ztl[i];
        Fl = inv_zsl * sl[i] + inv_ztl * tl[i];
        dl = bsl + inv_zsl * bzsl - inv_ztl * (bztl + tl[i] * blaml);

        T inv_Fl = 1.0 / Fl;
        d += inv_Fl * dl;
        C += inv_Fl;
      }

      if (!std::isinf(ubc[i])) {
        blamu = -(gamma - zsu[i] - ztu[i]);
        bsu = -(ubc[i] - s[i] - su[i] + tu[i]);
        bzsu = -(su[i] * zsu[i] - barrier_param);
        bztu = -(tu[i] * ztu[i] - barrier_param);

        T inv_zsu = 1.0 / zsu[i];
        T inv_ztu = 1.0 / ztu[i];
        Fu = inv_zsu * su[i] + inv_ztu * tu[i];
        du = bsu + inv_zsu * bzsu - inv_ztu * (bztu + tu[i] * blamu);

        T inv_Fu = 1.0 / Fu;
        d -= inv_Fu * du;
        C += inv_Fu;
      }

      ps[i] = (plam + d) / C;

      if (!std::isinf(lbc[i])) {
        pzsl[i] = (-ps[i] + dl) / Fl;
        pztl[i] = -blaml - pzsl[i];
        psl[i] = (bzsl - sl[i] * pzsl[i]) / zsl[i];
        ptl[i] = (bztl - tl[i] * pztl[i]) / ztl[i];
      }
      if (!std::isinf(ubc[i])) {
        pzsu[i] = (ps[i] + du) / Fu;
        pztu[i] = -blamu - pzsu[i];
        psu[i] = (bzsu - su[i] * pzsu[i]) / zsu[i];
        ptu[i] = (bztu - tu[i] * pztu[i]) / ztu[i];
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
   * @param diag The vector containing the diagonal components of the matrix
   */
  void compute_diagonal(const std::shared_ptr<OptVector<T>> vars,
                        std::shared_ptr<Vector<T>> diagonal) const {
    // Get the dual values for the bound constraints
    const T *zl, *zu;
    vars->get_bound_duals(&zl, &zu);

    // Get the slack variable values
    const T *sl, *tl, *su, *tu;
    vars->get_slacks(nullptr, &sl, &tl, &su, &tu);

    // Get the dual values for the slacks
    const T *zsl, *ztl, *zsu, *ztu;
    vars->get_slack_duals(&zsl, &ztl, &zsu, &ztu);

    // Get the solution vector - design variables and multipliers
    const Vector<T>& xlam = *vars->get_solution();

    // Zero the residual
    diagonal->zero();
    Vector<T>& diag = *diagonal;

    for (int i = 0; i < num_variables; i++) {
      // Get the gradient component corresponding to this variable
      int index = design_variable_indices[i];

      T x = xlam[index];

      // If the lower bound isn't infinite, add its value
      if (!std::isinf(lbx[i])) {
        diag[index] += zl[i] / (x - lbx[i]);
      }

      // If the upper bound isn't infinite, add its value
      if (!std::isinf(ubx[i])) {
        diag[index] += zu[i] / (ubx[i] - x);
      }
    }

    for (int i = 0; i < num_inequalities; i++) {
      int index = inequality_indices[i];

      // Build the components of C and compute its inverse
      T C = 0.0;
      if (!std::isinf(lbc[i])) {
        T Fl = sl[i] / zsl[i] + tl[i] / ztl[i];
        C += 1.0 / Fl;
      }
      if (!std::isinf(ubc[i])) {
        T Fu = su[i] / zsu[i] + tu[i] / ztu[i];
        C += 1.0 / Fu;
      }

      if (C != 0.0) {
        diag[index] = -1.0 / C;
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
   * @param x_index The index of the variable that is limiting the step
   * @param alpha_z_max The step in the dual variables
   * @param z_index The index of the multiplier limiting the step
   */
  void compute_max_step(const T tau, const std::shared_ptr<OptVector<T>> vars,
                        const std::shared_ptr<OptVector<T>> update,
                        T& alpha_x_max, int& x_index, T& alpha_z_max,
                        int& z_index) const {
    // Get the dual values for the bound constraints
    T *zl, *zu;
    const T *pzl, *pzu;
    vars->get_bound_duals(&zl, &zu);
    update->get_bound_duals(&pzl, &pzu);

    // Get the slack variable values
    T *sl, *tl, *su, *tu;
    const T *psl, *ptl, *psu, *ptu;
    vars->get_slacks(nullptr, &sl, &tl, &su, &tu);
    update->get_slacks(nullptr, &psl, &ptl, &psu, &ptu);

    // Get the dual values for the slacks
    T *zsl, *ztl, *zsu, *ztu;
    const T *pzsl, *pztl, *pzsu, *pztu;
    vars->get_slack_duals(&zsl, &ztl, &zsu, &ztu);
    update->get_slack_duals(&pzsl, &pztl, &pzsu, &pztu);

    // Get the solution update
    const Vector<T>& xlam = *vars->get_solution();
    const Vector<T>& pxlam = *update->get_solution();

    // Set the max step for the design variables and multipliers
    alpha_x_max = 1.0;
    x_index = -1;
    alpha_z_max = 1.0;
    z_index = -1;

    // Check for steps lengths for the design variables and slacks
    for (int i = 0; i < num_variables; i++) {
      // Get the gradient component corresponding to this variable
      int index = design_variable_indices[i];
      T x = xlam[index];
      T px = pxlam[index];

      if (!std::isinf(lbx[i])) {
        if (px < 0.0) {
          T numer = x - lbx[i];
          T alpha = -tau * numer / px;
          if (alpha < alpha_x_max) {
            alpha_x_max = alpha;
            x_index = index;
          }
        }
        if (pzl[i] < 0.0) {
          T alpha = -tau * zl[i] / pzl[i];
          if (alpha < alpha_z_max) {
            alpha_z_max = alpha;
            z_index = index;
          }
        }
      }
      if (!std::isinf(ubx[i])) {
        if (px > 0.0) {
          T numer = ubx[i] - x;
          T alpha = tau * numer / px;
          if (alpha < alpha_x_max) {
            alpha_x_max = alpha;
            x_index = index;
          }
        }
        if (pzu[i] < 0.0) {
          T alpha = -tau * zu[i] / pzu[i];
          if (alpha < alpha_z_max) {
            alpha_z_max = alpha;
            z_index = index;
          }
        }
      }
    }

    // Check step lengths for the multipliers
    for (int i = 0; i < num_inequalities; i++) {
      int index = inequality_indices[i];

      if (!std::isinf(lbc[i])) {
        // Slack variables
        if (psl[i] < 0.0) {
          T alpha = -tau * sl[i] / psl[i];
          if (alpha < alpha_x_max) {
            alpha_x_max = alpha;
            x_index = index;
          }
        }
        if (ptl[i] < 0.0) {
          T alpha = -tau * tl[i] / ptl[i];
          if (alpha < alpha_x_max) {
            alpha_x_max = alpha;
            x_index = index;
          }
        }

        // Dual variables
        if (pzsl[i] < 0.0) {
          T alpha = -tau * zsl[i] / pzsl[i];
          if (alpha < alpha_z_max) {
            alpha_z_max = alpha;
            z_index = index;
          }
        }
        if (pztl[i] < 0.0) {
          T alpha = -tau * ztl[i] / pztl[i];
          if (alpha < alpha_z_max) {
            alpha_z_max = alpha;
            z_index = index;
          }
        }
      }

      if (!std::isinf(ubc[i])) {
        // Slack variables
        if (psu[i] < 0.0) {
          T alpha = -tau * su[i] / psu[i];
          if (alpha < alpha_x_max) {
            alpha_x_max = alpha;
            x_index = index;
          }
        }
        if (ptu[i] < 0.0) {
          T alpha = -tau * tu[i] / ptu[i];
          if (alpha < alpha_x_max) {
            alpha_x_max = alpha;
            x_index = index;
          }
        }

        // Dual variables
        if (pzsu[i] < 0.0) {
          T alpha = -tau * zsu[i] / pzsu[i];
          if (alpha < alpha_z_max) {
            alpha_z_max = alpha;
            z_index = index;
          }
        }
        if (pztu[i] < 0.0) {
          T alpha = -tau * ztu[i] / pztu[i];
          if (alpha < alpha_z_max) {
            alpha_z_max = alpha;
            z_index = index;
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
   * @param vars Input/output variables that are updated
   * @param update Update to the state variables
   * @param temp The resulting variable values after the update
   */
  void apply_step_update(const T alpha_x, const T alpha_z,
                         const std::shared_ptr<OptVector<T>> vars,
                         const std::shared_ptr<OptVector<T>> update,
                         std::shared_ptr<OptVector<T>> temp) const {
    // Get the dual values for the bound constraints
    const T *zl, *zu;
    const T *pzl, *pzu;
    T *nzl, *nzu;
    vars->get_bound_duals(&zl, &zu);
    update->get_bound_duals(&pzl, &pzu);
    temp->get_bound_duals(&nzl, &nzu);

    // Get the slack variable values
    const T *s, *sl, *tl, *su, *tu;
    const T *ps, *psl, *ptl, *psu, *ptu;
    T *ns, *nsl, *ntl, *nsu, *ntu;
    vars->get_slacks(&s, &sl, &tl, &su, &tu);
    update->get_slacks(&ps, &psl, &ptl, &psu, &ptu);
    temp->get_slacks(&ns, &nsl, &ntl, &nsu, &ntu);

    // Get the dual values for the slacks
    const T *zsl, *ztl, *zsu, *ztu;
    const T *pzsl, *pztl, *pzsu, *pztu;
    T *nzsl, *nztl, *nzsu, *nztu;
    vars->get_slack_duals(&zsl, &ztl, &zsu, &ztu);
    update->get_slack_duals(&pzsl, &pztl, &pzsu, &pztu);
    temp->get_slack_duals(&nzsl, &nztl, &nzsu, &nztu);

    // Get the solution update
    const Vector<T>& xlam = *vars->get_solution();
    const Vector<T>& pxlam = *update->get_solution();
    Vector<T>& nxlam = *temp->get_solution();

    nxlam.copy(xlam);
    nxlam.axpy(alpha_x, pxlam);

    for (int i = 0; i < num_variables; i++) {
      // Update the dual variables
      if (!std::isinf(lbx[i])) {
        nzl[i] = zl[i] + alpha_z * pzl[i];
      }
      if (!std::isinf(ubx[i])) {
        nzu[i] = zu[i] + alpha_z * pzu[i];
      }
    }

    // Update the slack variables and remaining dual variables
    for (int i = 0; i < num_inequalities; i++) {
      ns[i] = s[i] + alpha_x * ps[i];
      if (!std::isinf(lbc[i])) {
        nsl[i] = sl[i] + alpha_x * psl[i];
        ntl[i] = tl[i] + alpha_x * ptl[i];
        nzsl[i] = zsl[i] + alpha_z * pzsl[i];
        nztl[i] = ztl[i] + alpha_z * pztl[i];
      }
      if (!std::isinf(ubc[i])) {
        nsu[i] = su[i] + alpha_x * psu[i];
        ntu[i] = tu[i] + alpha_x * ptu[i];
        nzsu[i] = zsu[i] + alpha_z * pzsu[i];
        nztu[i] = ztu[i] + alpha_z * pztu[i];
      }
    }
  }

  /**
   * @brief Compute the complementarity value for all inequalities
   *
   * This method computes the average complementarity and the
   * uniformity measure ξ = min_i [w_i y_i / (y^T w / m)]
   *
   * @param vars The optimization variables
   * @param uniformity_measure Pointer to store the uniformity measure
   * @return T The average complementarity value
   */
  T compute_complementarity(const std::shared_ptr<OptVector<T>> vars,
                            T* uniformity_measure) const {
    // Get the dual values for the bound constraints
    const T *zl, *zu;
    vars->get_bound_duals(&zl, &zu);

    // Get the slack variable values
    const T *sl, *tl, *su, *tu;
    vars->get_slacks(nullptr, &sl, &tl, &su, &tu);

    // Get the dual values for the slacks
    const T *zsl, *ztl, *zsu, *ztu;
    vars->get_slack_duals(&zsl, &ztl, &zsu, &ztu);

    Vector<T>& xlam = *vars->get_solution();

    T partial_sum[2] = {0.0, 0.0};
    T local_min = std::numeric_limits<T>::max();

    for (int i = 0; i < num_variables; i++) {
      // Extract the design variable value
      int index = design_variable_indices[i];
      T x = xlam[index];

      if (!std::isinf(lbx[i])) {
        T comp = (x - lbx[i]) * zl[i];
        partial_sum[0] += comp;
        partial_sum[1] += 1.0;
        local_min = A2D::min2(local_min, comp);
      }
      if (!std::isinf(ubx[i])) {
        T comp = (ubx[i] - x) * zu[i];
        partial_sum[0] += comp;
        partial_sum[1] += 1.0;
        local_min = A2D::min2(local_min, comp);
      }
    }

    for (int i = 0; i < num_inequalities; i++) {
      if (!std::isinf(lbc[i])) {
        T comp_sl = sl[i] * zsl[i];
        T comp_tl = tl[i] * ztl[i];
        partial_sum[0] += comp_sl + comp_tl;
        partial_sum[1] += 2.0;
        local_min = A2D::min2(local_min, A2D::min2(comp_sl, comp_tl));
      }

      if (!std::isinf(ubc[i])) {
        T comp_su = su[i] * zsu[i];
        T comp_tu = tu[i] * ztu[i];
        partial_sum[0] += comp_su + comp_tu;
        partial_sum[1] += 2.0;
        local_min = A2D::min2(local_min, A2D::min2(comp_su, comp_tu));
      }
    }

    // Compute the complementarity value across all processors
    T sum[2];
    MPI_Allreduce(partial_sum, sum, 2, get_mpi_type<T>(), MPI_SUM, comm);

    // Compute average complementarity
    T avg_complementarity = (sum[1] == 0.0) ? 0.0 : sum[0] / sum[1];

    // Compute the uniformity measure
    T global_min;
    MPI_Allreduce(&local_min, &global_min, 1, get_mpi_type<T>(), MPI_MIN, comm);

    if (avg_complementarity <= 0.0) {
      *uniformity_measure = 1.0;
    } else {
      T uniformity = global_min / avg_complementarity;
      *uniformity_measure = A2D::max2(0.0, A2D::min2(1.0, uniformity));
    }

    return avg_complementarity;
  }

  /**
   * @brief Compute the variable values for the new starting point
   *
   * @param beta_min The minimum value of the multiplier or slack variable
   * @param vars The values of the variables at the initialization point
   * @param update The affine update computed with barrier_param = 0.0
   * @param temp The resulting variable values after the update
   */
  void compute_affine_start_point(T beta_min,
                                  const std::shared_ptr<OptVector<T>> vars,
                                  const std::shared_ptr<OptVector<T>> update,
                                  std::shared_ptr<OptVector<T>> temp) {
    // Get the dual values for the bound constraints
    const T *zl, *zu;
    const T *pzl, *pzu;
    T *nzl, *nzu;
    vars->get_bound_duals(&zl, &zu);
    update->get_bound_duals(&pzl, &pzu);
    temp->get_bound_duals(&nzl, &nzu);

    // Get the slack variable values
    const T *sl, *tl, *su, *tu;
    const T *psl, *ptl, *psu, *ptu;
    T *nsl, *ntl, *nsu, *ntu;
    vars->get_slacks(nullptr, &sl, &tl, &su, &tu);
    update->get_slacks(nullptr, &psl, &ptl, &psu, &ptu);
    temp->get_slacks(nullptr, &nsl, &ntl, &nsu, &ntu);

    // Get the dual values for the slacks
    const T *zsl, *ztl, *zsu, *ztu;
    const T *pzsl, *pztl, *pzsu, *pztu;
    T *nzsl, *nztl, *nzsu, *nztu;
    vars->get_slack_duals(&zsl, &ztl, &zsu, &ztu);
    update->get_slack_duals(&pzsl, &pztl, &pzsu, &pztu);
    temp->get_slack_duals(&nzsl, &nztl, &nzsu, &nztu);

    for (int i = 0; i < num_variables; i++) {
      nzl[i] = std::max(beta_min, std::fabs(zl[i] + pzl[i]));
      nzu[i] = std::max(beta_min, std::fabs(zu[i] + pzu[i]));
    }

    for (int i = 0; i < num_inequalities; i++) {
      if (!std::isinf(lbc[i])) {
        nsl[i] = std::max(beta_min, std::fabs(sl[i] + psl[i]));
        ntl[i] = std::max(beta_min, std::fabs(tl[i] + ptl[i]));
        nzsl[i] = std::max(beta_min, std::fabs(zsl[i] + pzsl[i]));
        nztl[i] = std::max(beta_min, std::fabs(ztl[i] + pztl[i]));
      }

      if (!std::isinf(ubc[i])) {
        nsu[i] = std::max(beta_min, std::fabs(su[i] + psu[i]));
        ntu[i] = std::max(beta_min, std::fabs(tu[i] + ptu[i]));
        nzsu[i] = std::max(beta_min, std::fabs(zsu[i] + pzsu[i]));
        nztu[i] = std::max(beta_min, std::fabs(ztu[i] + pztu[i]));
      }
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
  void check_update(T barrier_param, T gamma,
                    const std::shared_ptr<Vector<T>> grad,
                    const std::shared_ptr<OptVector<T>> vars,
                    const std::shared_ptr<OptVector<T>> update,
                    const std::shared_ptr<CSRMat<T>> hessian) {
    // Recompute the Hessian so it doesn't have the diagonal terms
    problem->hessian(vars->get_solution(), hessian);

    // Get the dual values for the bound constraints
    const T *zl, *zu;
    const T *pzl, *pzu;
    vars->get_bound_duals(&zl, &zu);
    update->get_bound_duals(&pzl, &pzu);

    // Get the slack variable values
    const T *s, *sl, *tl, *su, *tu;
    const T *ps, *psl, *ptl, *psu, *ptu;
    vars->get_slacks(&s, &sl, &tl, &su, &tu);
    update->get_slacks(&ps, &psl, &ptl, &psu, &ptu);

    // Get the dual values for the slacks
    const T *zsl, *ztl, *zsu, *ztu;
    const T *pzsl, *pztl, *pzsu, *pztu;
    vars->get_slack_duals(&zsl, &ztl, &zsu, &ztu);
    update->get_slack_duals(&pzsl, &pztl, &pzsu, &pztu);

    // Get the solution update
    const Vector<T>& xlam = *vars->get_solution();
    const Vector<T>& pxlam = *update->get_solution();

    // Set the gradient vector
    const Vector<T>& g = *grad;

    std::shared_ptr<Vector<T>> tmp = problem->create_vector();
    Vector<T>& t = *tmp;

    std::printf("KKT step check\n");

    // Form the residuals from the different expressions
    // [ H  |  A^T ][ px   ] - pzlx + pzux = [ bx ]
    // [ A  |    0 ][ plam ] - ps          = [ bc ]

    hessian->mult(update->get_solution(), tmp);
    for (int i = 0; i < num_variables; i++) {
      int index = design_variable_indices[i];

      // Compute the residual
      T rx = (g[index] - zl[i] + zu[i]);
      t[index] = t[index] - pzl[i] + pzu[i] + rx;
    }
    for (int i = 0; i < num_equalities; i++) {
      int index = equality_indices[i];
      T rh = g[index];
      t[index] = t[index] + rh;
    }
    for (int i = 0; i < num_inequalities; i++) {
      int index = inequality_indices[i];
      T rc = g[index] - s[i];
      t[index] = t[index] - ps[i] + rc;
    }
    std::printf("%-40s %15.6e\n", "||H * px - pzl + pzu + rx|| ",
                std::sqrt(tmp->dot(*tmp)));

    // (xs - lb) * pzl + pxs * zl + rzl
    t.zero();
    for (int i = 0; i < num_variables; i++) {
      int index = design_variable_indices[i];
      if (!std::isinf(lbx[i])) {
        T rzl = (xlam[index] - lbx[i]) * zl[i] - barrier_param;
        t[i] = (xlam[index] - lbx[i]) * pzl[i] + pxlam[index] * zl[i] + rzl;
      }
    }
    std::printf("%-40s %15.6e\n", "||(x - lbx) * pzl + px * zl + rzl|| ",
                std::sqrt(tmp->dot(*tmp)));

    // (ub - xs) * pzu - pxs * zu = bzu
    t.zero();
    for (int i = 0; i < num_variables; i++) {
      int index = design_variable_indices[i];
      if (!std::isinf(ubx[i])) {
        T rzu = (ubx[i] - xlam[index]) * zu[i] - barrier_param;
        t[i] = (ubx[i] - xlam[index]) * pzu[i] - zu[i] * pxlam[index] + rzu;
      }
    }
    std::printf("%-40s %15.6e\n", "||(ubx - x) * pzu - px * zu + rzu|| ",
                std::sqrt(tmp->dot(*tmp)));

    // -plam - pzlc + pzuc = blam
    t.zero();
    for (int i = 0; i < num_inequalities; i++) {
      int index = inequality_indices[i];

      T lam = xlam[index];
      T plam = pxlam[index];
      T rlam = (-lam - zsl[i] + zsu[i]);
      t[index] = (-plam - pzsl[i] + pzsu[i]) + rlam;
    }
    std::printf("%-40s %15.6e\n", "||-plam - pzlc + pzuc - blam|| ",
                std::sqrt(tmp->dot(*tmp)));

    // Test equations for the lower bounds
    // rsl = s - lc - sl + tl
    t.zero();
    for (int i = 0; i < num_inequalities; i++) {
      int index = inequality_indices[i];
      if (!std::isinf(lbc[i])) {
        T rsl = s[i] - lbc[i] - sl[i] + tl[i];
        t[index] = ps[i] - psl[i] + ptl[i] + rsl;
      }
    }
    std::printf("%-40s %15.6e\n", "||ps - psl + ptl + rsl|| ",
                std::sqrt(tmp->dot(*tmp)));

    // rlaml = gamma - zsl - ztl
    t.zero();
    for (int i = 0; i < num_inequalities; i++) {
      int index = inequality_indices[i];
      if (!std::isinf(lbc[i])) {
        T rlaml = gamma - zsl[i] - ztl[i];
        t[index] = -pzsl[i] - pztl[i] + rlaml;
      }
    }
    std::printf("%-40s %15.6e\n", "||-pzsl - pztl + rlaml|| ",
                std::sqrt(tmp->dot(*tmp)));

    // rzsl = Sl * zsl - barrier_param
    t.zero();
    for (int i = 0; i < num_inequalities; i++) {
      int index = inequality_indices[i];
      if (!std::isinf(lbc[i])) {
        T rzsl = sl[i] * zsl[i] - barrier_param;
        t[index] = sl[i] * pzsl[i] + zsl[i] * psl[i] + rzsl;
      }
    }
    std::printf("%-40s %15.6e\n", "||sl * pzsl + zsl * psl + rzsl|| ",
                std::sqrt(tmp->dot(*tmp)));

    // rzsl = Tl * ztl - barrier_param
    t.zero();
    for (int i = 0; i < num_inequalities; i++) {
      int index = inequality_indices[i];
      if (!std::isinf(lbc[i])) {
        T rztl = tl[i] * ztl[i] - barrier_param;
        t[index] = tl[i] * pztl[i] + ztl[i] * ptl[i] + rztl;
      }
    }
    std::printf("%-40s %15.6e\n", "||tl * pztl + ztl * ptl + rztl|| ",
                std::sqrt(tmp->dot(*tmp)));

    // Test equations for the upper bounds
    // rsu = ubc - s - su + tu
    t.zero();
    for (int i = 0; i < num_inequalities; i++) {
      int index = inequality_indices[i];
      if (!std::isinf(ubc[i])) {
        T rsu = ubc[i] - s[i] - su[i] + tu[i];
        t[index] = -ps[i] - psu[i] + ptu[i] + rsu;
      }
    }
    std::printf("%-40s %15.6e\n", "||-ps - psu + ptu + rsu|| ",
                std::sqrt(tmp->dot(*tmp)));

    // rlamu = gamma - zsu - ztu
    t.zero();
    for (int i = 0; i < num_inequalities; i++) {
      int index = inequality_indices[i];
      if (!std::isinf(ubc[i])) {
        T rlamu = gamma - zsu[i] - ztu[i];
        t[index] = -pzsu[i] - pztu[i] + rlamu;
      }
    }
    std::printf("%-40s %15.6e\n", "||-pzsu - pztu + rlamu|| ",
                std::sqrt(tmp->dot(*tmp)));

    // rzsu = Su * zsu - barrier_param
    t.zero();
    for (int i = 0; i < num_inequalities; i++) {
      int index = inequality_indices[i];
      if (!std::isinf(ubc[i])) {
        T rzsu = su[i] * zsu[i] - barrier_param;
        t[index] = su[i] * pzsu[i] + zsu[i] * psu[i] + rzsu;
      }
    }
    std::printf("%-40s %15.6e\n", "||su * pzsu + zsu * psu + rzsu|| ",
                std::sqrt(tmp->dot(*tmp)));

    // rzsu = Tu * ztu - barrier_param
    t.zero();
    for (int i = 0; i < num_inequalities; i++) {
      int index = inequality_indices[i];
      if (!std::isinf(ubc[i])) {
        T rztu = tu[i] * ztu[i] - barrier_param;
        t[index] = tu[i] * pztu[i] + ztu[i] * ptu[i] + rztu;
      }
    }
    std::printf("%-40s %15.6e\n", "||tu * pztu + ztu * ptu + rztu|| ",
                std::sqrt(tmp->dot(*tmp)));
  }

 private:
  // The optimization problem
  std::shared_ptr<OptimizationProblem<T>> problem;

  // Lower and upper bounds for the design variables
  std::shared_ptr<Vector<T>> lower, upper;

  // The MPI communicator
  MPI_Comm comm;

  int num_variables;     // Number of design variables
  int num_equalities;    // The number of equalities
  int num_inequalities;  // The number of inequalities

  // Information about the location of the design variables and
  // multipliers/constraints within the solution vector
  std::vector<int> design_variable_indices;
  std::vector<int> equality_indices;
  std::vector<int> inequality_indices;

  // Store local copies of the lower/upper bounds
  std::vector<T> lbx, ubx;  // Design variables
  std::vector<T> lbc, ubc;  // Inequality bounds
  std::vector<T> lbh;       // Equality constraints
};

}  // namespace amigo

#endif  // AMIGO_OPTIMIZER_H