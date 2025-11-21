#ifndef AMIGO_OPTIMIZER_H
#define AMIGO_OPTIMIZER_H

#include <mpi.h>

#include <cstdio>

#include "a2dcore.h"
#include "amigo.h"
#include "optimization_problem.h"
#include "optimizer_backend.h"

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
    int size = 2 * num_variables + 9 * num_inequalities;
    duals = std::make_shared<Vector<T>>(size, 0, x->get_memory_location());
  }
  ~OptVector() {}

  void zero() {
    x->zero();
    duals->zero();
  }
  void copy(std::shared_ptr<OptVector<T>> src) {
    x->copy(*src->x);
    duals->copy(*src->duals);
  }

  template <ExecPolicy policy>
  void get_bound_duals(T** zl, T** zu) {
    T* array = duals->template get_array<policy>();
    if (zl) {
      *zl = &array[0];
    }
    if (zu) {
      *zu = &array[num_variables];
    }
  }
  template <ExecPolicy policy>
  void get_bound_duals(const T** zl, const T** zu) const {
    const T* array = duals->template get_array<policy>();
    if (zl) {
      *zl = &array[0];
    }
    if (zu) {
      *zu = &array[num_variables];
    }
  }
  template <ExecPolicy policy>
  void get_slacks(T** s, T** sl, T** tl, T** su, T** tu) {
    T* array = duals->template get_array<policy>();
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
  template <ExecPolicy policy>
  void get_slacks(const T** s, const T** sl, const T** tl, const T** su,
                  const T** tu) const {
    const T* array = duals->template get_array<policy>();
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
  template <ExecPolicy policy>
  void get_slack_duals(T** zsl, T** ztl, T** zsu, T** ztu) {
    T* array = duals->template get_array<policy>();
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
  template <ExecPolicy policy>
  void get_slack_duals(const T** zsl, const T** ztl, const T** zsu,
                       const T** ztu) const {
    const T* array = duals->template get_array<policy>();
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

  template <ExecPolicy policy>
  T* get_solution_array() {
    return x->template get_array<policy>();
  }
  template <ExecPolicy policy>
  const T* get_solution_array() const {
    return x->template get_array<policy>();
  }

  // Get the underlying solution vector
  std::shared_ptr<Vector<T>> get_solution() { return x; }
  const std::shared_ptr<Vector<T>> get_solution() const { return x; }

 private:
  int num_variables;
  int num_equalities;
  int num_inequalities;
  std::shared_ptr<Vector<T>> x;
  std::shared_ptr<Vector<T>> duals;
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
template <typename T, ExecPolicy policy>
class InteriorPointOptimizer {
 public:
  InteriorPointOptimizer(
      std::shared_ptr<OptimizationProblem<T, policy>> problem,
      std::shared_ptr<Vector<T>> lower, std::shared_ptr<Vector<T>> upper)
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

    // Set the memory location depending on the policy
    MemoryLocation mem_loc = MemoryLocation::HOST_ONLY;
    if (policy == ExecPolicy::CUDA) {
      mem_loc = MemoryLocation::HOST_AND_DEVICE;
    }

    // Allocate vectors
    lbx = std::make_shared<Vector<T>>(num_variables, 0, mem_loc);
    ubx = std::make_shared<Vector<T>>(num_variables, 0, mem_loc);
    design_variable_indices =
        std::make_shared<Vector<int>>(num_variables, 0, mem_loc);

    lbh = std::make_shared<Vector<T>>(num_equalities, 0, mem_loc);
    equality_indices =
        std::make_shared<Vector<int>>(num_equalities, 0, mem_loc);

    lbc = std::make_shared<Vector<T>>(num_inequalities, 0, mem_loc);
    ubc = std::make_shared<Vector<T>>(num_inequalities, 0, mem_loc);
    inequality_indices =
        std::make_shared<Vector<int>>(num_inequalities, 0, mem_loc);

    // Set the variable indices and bounds
    int num_vars = 0, num_eq = 0, num_ineq = 0;
    for (int i = 0; i < size; i++) {
      if (is_multiplier[i]) {
        if (!std::isinf(lb[i]) && !std::isinf(ub[i]) && lb[i] == ub[i]) {
          (*lbh)[num_eq] = lb[i];
          (*equality_indices)[num_eq] = i;
          num_eq++;
        } else {
          (*lbc)[num_ineq] = lb[i];
          (*ubc)[num_ineq] = ub[i];
          (*inequality_indices)[num_ineq] = i;
          num_ineq++;
        }
      } else {
        (*lbx)[num_vars] = lb[i];
        (*ubx)[num_vars] = ub[i];
        (*design_variable_indices)[num_vars] = i;
        num_vars++;
      }
    }

    // Sync everything between the host and device
    lbx->copy_host_to_device();
    ubx->copy_host_to_device();
    design_variable_indices->copy_host_to_device();

    lbh->copy_host_to_device();
    equality_indices->copy_host_to_device();

    lbc->copy_host_to_device();
    ubc->copy_host_to_device();
    inequality_indices->copy_host_to_device();

    // Set the info class
    info.num_variables = num_variables;
    info.num_equalities = num_equalities;
    info.num_inequalities = num_inequalities;

    info.design_variable_indices =
        design_variable_indices->template get_array<policy>();
    info.equality_indices = equality_indices->template get_array<policy>();
    info.inequality_indices = inequality_indices->template get_array<policy>();

    info.lbx = lbx->template get_array<policy>();
    info.ubx = ubx->template get_array<policy>();
    info.lbc = lbc->template get_array<policy>();
    info.ubc = ubc->template get_array<policy>();
    info.lbh = lbh->template get_array<policy>();
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
   * @brief Set the multipliers to the specified value
   *
   * @param value Value to place into the multiplier components
   * @param x Vector
   */
  void set_multipliers_value(T value, std::shared_ptr<Vector<T>> x) const {
    T* x_array = x->template get_array<policy>();
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::set_multipliers_value(info, value, x_array);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::set_multipliers_value_cuda(info, value, x_array);
    }
#endif
  }

  /**
   * @brief Set the design variables to the specified value
   *
   * @param value Value to place into the design variable components
   * @param x Vector
   */
  void set_design_vars_value(T value, std::shared_ptr<Vector<T>> x) const {
    T* x_array = x->template get_array<policy>();
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::set_design_vars_value(info, value, x_array);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::set_design_vars_value_cuda(info, value, x_array);
    }
#endif
  }

  /**
   * @brief Copy only the multipliers from the src to the dest vector
   *
   * @param dest Destination vector
   * @param src Source vector
   */
  void copy_multipliers(std::shared_ptr<Vector<T>> dest,
                        std::shared_ptr<Vector<T>> src) const {
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::copy_multipliers(info, src->template get_array<policy>(),
                               dest->template get_array<policy>());
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::copy_multipliers_cuda(info, src->template get_array<policy>(),
                                    dest->template get_array<policy>());
    }
#endif
  }

  /**
   * @brief Copy only the design variables from the src to the dest vector
   *
   * @param dest Destination vector
   * @param src Source vector
   */
  void copy_design_vars(std::shared_ptr<Vector<T>> dest,
                        std::shared_ptr<Vector<T>> src) const {
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::copy_design_vars(info, src->template get_array<policy>(),
                               dest->template get_array<policy>());
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::copy_design_vars_cuda(info, src->template get_array<policy>(),
                                    dest->template get_array<policy>());
    }
#endif
  }

  /**
   * @brief Initialize the multipliers and slack variables in the problem
   *
   * @param vars All of the optimization variables
   */
  void initialize_multipliers_and_slacks(
      T barrier_param, const std::shared_ptr<Vector<T>> grad,
      std::shared_ptr<OptVector<T>> vars) const {
    // Set state values
    detail::OptStateData<T> pt =
        detail::OptStateData<T>::template make<policy>(vars);

    // Set a pointer to the vector
    const T* g = grad->template get_array<policy>();

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::initialize_multipliers_and_slacks(barrier_param, info, g, pt);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::initialize_multipliers_and_slacks_cuda(barrier_param, info, g,
                                                     pt);
    }
#endif
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
    // Set state values
    detail::OptStateData<const T> pt =
        detail::OptStateData<const T>::template make<policy>(vars);

    // Set the gradient vector
    const T* g = grad->template get_array<policy>();
    T* r = res->template get_array<policy>();

    // Zero the residual
    res->zero();

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::add_residual(barrier_param, gamma, info, pt, g, r);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::add_residual_cuda(barrier_param, gamma, info, pt, g, r);
    }
#endif

    // Compute the residual norm
    T local_norm = res->template dot<policy>(*res);
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
    // Set state values
    detail::OptStateData<const T> pt =
        detail::OptStateData<const T>::template make<policy>(vars);
    detail::OptStateData<T> up =
        detail::OptStateData<T>::template make<policy>(update);

    // Copy the update for the design variables and dual variables
    update->get_solution()->copy(*reduced_update);

    // Perform the update
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::compute_update(barrier_param, gamma, info, pt, up);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::compute_update_cuda(barrier_param, gamma, info, pt, up);
    }
#endif
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
    // Zero the residual
    diagonal->zero();

    detail::OptStateData<const T> pt =
        detail::OptStateData<const T>::template make<policy>(vars);
    T* diag = diagonal->template get_array<policy>();

    // Perform the update
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::compute_diagonal(info, pt, diag);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::compute_diagonal_cuda(info, pt, diag);
    }
#endif
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
    // Set the values
    detail::OptStateData<const T> pt =
        detail::OptStateData<const T>::template make<policy>(vars);
    detail::OptStateData<const T> up =
        detail::OptStateData<const T>::template make<policy>(update);

    // Set the max step for the design variables and multipliers
    alpha_x_max = 1.0;
    x_index = -1;
    alpha_z_max = 1.0;
    z_index = -1;

    // Compute the max step lengths
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::compute_max_step(tau, info, pt, up, alpha_x_max, x_index,
                               alpha_z_max, z_index);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::compute_max_step_cuda(tau, info, pt, up, alpha_x_max, x_index,
                                    alpha_z_max, z_index);
    }
#endif  // AMIGO_USE_CUDA

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
    // Set the values
    detail::OptStateData<const T> pt =
        detail::OptStateData<const T>::template make<policy>(vars);
    detail::OptStateData<const T> up =
        detail::OptStateData<const T>::template make<policy>(update);
    detail::OptStateData<T> tmp =
        detail::OptStateData<T>::template make<policy>(temp);

    // nxlam = xlam + alpha * pxlam
    temp->get_solution()->copy(*vars->get_solution());
    temp->get_solution()->template axpy<policy>(alpha_x,
                                                *update->get_solution());

    // Compute the max step lengths
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::apply_step_update(alpha_x, alpha_z, info, pt, up, tmp);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::apply_step_update_cuda(alpha_x, alpha_z, info, pt, up, tmp);
    }
#endif  // AMIGO_USE_CUDA
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
    // // Get the dual values for the bound constraints
    // const T *zl, *zu;
    // vars->get_bound_duals(&zl, &zu);

    // // Get the slack variable values
    // const T *sl, *tl, *su, *tu;
    // vars->get_slacks(nullptr, &sl, &tl, &su, &tu);

    // // Get the dual values for the slacks
    // const T *zsl, *ztl, *zsu, *ztu;
    // vars->get_slack_duals(&zsl, &ztl, &zsu, &ztu);

    // T* xlam = vars->get_solution_array();

    // T partial_sum[2] = {0.0, 0.0};
    // T local_min = std::numeric_limits<T>::max();

    // for (int i = 0; i < num_variables; i++) {
    //   // Extract the design variable value
    //   int index = design_variable_indices[i];
    //   T x = xlam[index];

    //   if (!std::isinf(lbx[i])) {
    //     T comp = (x - lbx[i]) * zl[i];
    //     partial_sum[0] += comp;
    //     partial_sum[1] += 1.0;
    //     local_min = A2D::min2(local_min, comp);
    //   }
    //   if (!std::isinf(ubx[i])) {
    //     T comp = (ubx[i] - x) * zu[i];
    //     partial_sum[0] += comp;
    //     partial_sum[1] += 1.0;
    //     local_min = A2D::min2(local_min, comp);
    //   }
    // }

    // for (int i = 0; i < num_inequalities; i++) {
    //   if (!std::isinf(lbc[i])) {
    //     T comp_sl = sl[i] * zsl[i];
    //     T comp_tl = tl[i] * ztl[i];
    //     partial_sum[0] += comp_sl + comp_tl;
    //     partial_sum[1] += 2.0;
    //     local_min = A2D::min2(local_min, A2D::min2(comp_sl, comp_tl));
    //   }

    //   if (!std::isinf(ubc[i])) {
    //     T comp_su = su[i] * zsu[i];
    //     T comp_tu = tu[i] * ztu[i];
    //     partial_sum[0] += comp_su + comp_tu;
    //     partial_sum[1] += 2.0;
    //     local_min = A2D::min2(local_min, A2D::min2(comp_su, comp_tu));
    //   }
    // }

    // // Compute the complementarity value across all processors
    // T sum[2];
    // MPI_Allreduce(partial_sum, sum, 2, get_mpi_type<T>(), MPI_SUM, comm);

    // // Compute average complementarity
    // T avg_complementarity = (sum[1] == 0.0) ? 0.0 : sum[0] / sum[1];

    // // Compute the uniformity measure
    // T global_min;
    // MPI_Allreduce(&local_min, &global_min, 1, get_mpi_type<T>(), MPI_MIN,
    // comm);

    // if (avg_complementarity <= 0.0) {
    //   *uniformity_measure = 1.0;
    // } else {
    //   T uniformity = global_min / avg_complementarity;
    //   *uniformity_measure = A2D::max2(0.0, A2D::min2(1.0, uniformity));
    // }

    T avg_complementarity = 0.0;

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
    // Set the values
    detail::OptStateData<const T> pt =
        detail::OptStateData<const T>::template make<policy>(vars);
    detail::OptStateData<const T> up =
        detail::OptStateData<const T>::template make<policy>(update);
    detail::OptStateData<T> tmp =
        detail::OptStateData<T>::template make<policy>(temp);

    // Compute the max step lengths
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::compute_affine_start_point(beta_min, info, pt, up, tmp);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::compute_affine_start_point_cuda(beta_min, info, pt, up, tmp);
    }
#endif  // AMIGO_USE_CUDA
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
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      // Recompute the Hessian so it doesn't have the diagonal terms
      problem->hessian(T(1.0), vars->get_solution(), hessian);

      // Get the dual values for the bound constraints
      const T *zl, *zu;
      const T *pzl, *pzu;
      vars->template get_bound_duals<policy>(&zl, &zu);
      update->template get_bound_duals<policy>(&pzl, &pzu);

      // Get the slack variable values
      const T *s, *sl, *tl, *su, *tu;
      const T *ps, *psl, *ptl, *psu, *ptu;
      vars->template get_slacks<policy>(&s, &sl, &tl, &su, &tu);
      update->template get_slacks<policy>(&ps, &psl, &ptl, &psu, &ptu);

      // Get the dual values for the slacks
      const T *zsl, *ztl, *zsu, *ztu;
      const T *pzsl, *pztl, *pzsu, *pztu;
      vars->template get_slack_duals<policy>(&zsl, &ztl, &zsu, &ztu);
      update->template get_slack_duals<policy>(&pzsl, &pztl, &pzsu, &pztu);

      // Get the solution update
      const T* xlam = vars->template get_solution_array<policy>();
      const T* pxlam = update->template get_solution_array<policy>();

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
        int index = info.design_variable_indices[i];

        // Compute the residual
        T rx = (g[index] - zl[i] + zu[i]);
        t[index] = t[index] - pzl[i] + pzu[i] + rx;
      }
      for (int i = 0; i < num_equalities; i++) {
        int index = info.equality_indices[i];
        T rh = g[index];
        t[index] = t[index] + rh;
      }
      for (int i = 0; i < num_inequalities; i++) {
        int index = info.inequality_indices[i];
        T rc = g[index] - s[i];
        t[index] = t[index] - ps[i] + rc;
      }
      std::printf("%-40s %15.6e\n", "||H * px - pzl + pzu + rx|| ",
                  std::sqrt(tmp->template dot<policy>(*tmp)));

      // (xs - lb) * pzl + pxs * zl + rzl
      t.zero();
      for (int i = 0; i < num_variables; i++) {
        int index = info.design_variable_indices[i];
        if (!std::isinf(info.lbx[i])) {
          T rzl = (xlam[index] - info.lbx[i]) * zl[i] - barrier_param;
          t[i] =
              (xlam[index] - info.lbx[i]) * pzl[i] + pxlam[index] * zl[i] + rzl;
        }
      }
      std::printf("%-40s %15.6e\n", "||(x - lbx) * pzl + px * zl + rzl|| ",
                  std::sqrt(tmp->template dot<policy>(*tmp)));

      // (ub - xs) * pzu - pxs * zu = bzu
      t.zero();
      for (int i = 0; i < num_variables; i++) {
        int index = info.design_variable_indices[i];
        if (!std::isinf(info.ubx[i])) {
          T rzu = (info.ubx[i] - xlam[index]) * zu[i] - barrier_param;
          t[i] =
              (info.ubx[i] - xlam[index]) * pzu[i] - zu[i] * pxlam[index] + rzu;
        }
      }
      std::printf("%-40s %15.6e\n", "||(ubx - x) * pzu - px * zu + rzu|| ",
                  std::sqrt(tmp->template dot<policy>(*tmp)));

      // -plam - pzlc + pzuc = blam
      t.zero();
      for (int i = 0; i < num_inequalities; i++) {
        int index = info.inequality_indices[i];

        T lam = xlam[index];
        T plam = pxlam[index];
        T rlam = (-lam - zsl[i] + zsu[i]);
        t[index] = (-plam - pzsl[i] + pzsu[i]) + rlam;
      }
      std::printf("%-40s %15.6e\n", "||-plam - pzlc + pzuc - blam|| ",
                  std::sqrt(tmp->template dot<policy>(*tmp)));

      // Test equations for the lower bounds
      // rsl = s - lc - sl + tl
      t.zero();
      for (int i = 0; i < num_inequalities; i++) {
        int index = info.inequality_indices[i];
        if (!std::isinf(info.lbc[i])) {
          T rsl = s[i] - info.lbc[i] - sl[i] + tl[i];
          t[index] = ps[i] - psl[i] + ptl[i] + rsl;
        }
      }
      std::printf("%-40s %15.6e\n", "||ps - psl + ptl + rsl|| ",
                  std::sqrt(tmp->template dot<policy>(*tmp)));

      // rlaml = gamma - zsl - ztl
      t.zero();
      for (int i = 0; i < num_inequalities; i++) {
        int index = info.inequality_indices[i];
        if (!std::isinf(info.lbc[i])) {
          T rlaml = gamma - zsl[i] - ztl[i];
          t[index] = -pzsl[i] - pztl[i] + rlaml;
        }
      }
      std::printf("%-40s %15.6e\n", "||-pzsl - pztl + rlaml|| ",
                  std::sqrt(tmp->template dot<policy>(*tmp)));

      // rzsl = Sl * zsl - barrier_param
      t.zero();
      for (int i = 0; i < num_inequalities; i++) {
        int index = info.inequality_indices[i];
        if (!std::isinf(info.lbc[i])) {
          T rzsl = sl[i] * zsl[i] - barrier_param;
          t[index] = sl[i] * pzsl[i] + zsl[i] * psl[i] + rzsl;
        }
      }
      std::printf("%-40s %15.6e\n", "||sl * pzsl + zsl * psl + rzsl|| ",
                  std::sqrt(tmp->template dot<policy>(*tmp)));

      // rzsl = Tl * ztl - barrier_param
      t.zero();
      for (int i = 0; i < num_inequalities; i++) {
        int index = info.inequality_indices[i];
        if (!std::isinf(info.lbc[i])) {
          T rztl = tl[i] * ztl[i] - barrier_param;
          t[index] = tl[i] * pztl[i] + ztl[i] * ptl[i] + rztl;
        }
      }
      std::printf("%-40s %15.6e\n", "||tl * pztl + ztl * ptl + rztl|| ",
                  std::sqrt(tmp->template dot<policy>(*tmp)));

      // Test equations for the upper bounds
      // rsu = ubc - s - su + tu
      t.zero();
      for (int i = 0; i < num_inequalities; i++) {
        int index = info.inequality_indices[i];
        if (!std::isinf(info.ubc[i])) {
          T rsu = info.ubc[i] - s[i] - su[i] + tu[i];
          t[index] = -ps[i] - psu[i] + ptu[i] + rsu;
        }
      }
      std::printf("%-40s %15.6e\n", "||-ps - psu + ptu + rsu|| ",
                  std::sqrt(tmp->template dot<policy>(*tmp)));

      // rlamu = gamma - zsu - ztu
      t.zero();
      for (int i = 0; i < num_inequalities; i++) {
        int index = info.inequality_indices[i];
        if (!std::isinf(info.ubc[i])) {
          T rlamu = gamma - zsu[i] - ztu[i];
          t[index] = -pzsu[i] - pztu[i] + rlamu;
        }
      }
      std::printf("%-40s %15.6e\n", "||-pzsu - pztu + rlamu|| ",
                  std::sqrt(tmp->template dot<policy>(*tmp)));

      // rzsu = Su * zsu - barrier_param
      t.zero();
      for (int i = 0; i < num_inequalities; i++) {
        int index = info.inequality_indices[i];
        if (!std::isinf(info.ubc[i])) {
          T rzsu = su[i] * zsu[i] - barrier_param;
          t[index] = su[i] * pzsu[i] + zsu[i] * psu[i] + rzsu;
        }
      }
      std::printf("%-40s %15.6e\n", "||su * pzsu + zsu * psu + rzsu|| ",
                  std::sqrt(tmp->template dot<policy>(*tmp)));

      // rzsu = Tu * ztu - barrier_param
      t.zero();
      for (int i = 0; i < num_inequalities; i++) {
        int index = info.inequality_indices[i];
        if (!std::isinf(info.ubc[i])) {
          T rztu = tu[i] * ztu[i] - barrier_param;
          t[index] = tu[i] * pztu[i] + ztu[i] * ptu[i] + rztu;
        }
      }
      std::printf("%-40s %15.6e\n", "||tu * pztu + ztu * ptu + rztu|| ",
                  std::sqrt(tmp->template dot<policy>(*tmp)));
    }
  }

 private:
  // The optimization problem
  std::shared_ptr<OptimizationProblem<T, policy>> problem;

  // Lower and upper bounds for the design variables
  std::shared_ptr<Vector<T>> lower, upper;

  // The MPI communicator
  MPI_Comm comm;

  int num_variables;     // Number of design variables
  int num_equalities;    // The number of equalities
  int num_inequalities;  // The number of inequalities

  // Information about the location of the design variables and
  // multipliers/constraints within the solution vector
  std::shared_ptr<Vector<int>> design_variable_indices;
  std::shared_ptr<Vector<int>> equality_indices;
  std::shared_ptr<Vector<int>> inequality_indices;

  // Store local copies of the lower/upper bounds
  std::shared_ptr<Vector<T>> lbx, ubx;  // Design variables
  std::shared_ptr<Vector<T>> lbc, ubc;  // Inequality bounds
  std::shared_ptr<Vector<T>> lbh;       // Equality constraints

  // Information about the problem (host or device side depending
  // on the execution policy that is set)
  detail::OptInfo<T> info;
};

}  // namespace amigo

#endif  // AMIGO_OPTIMIZER_H