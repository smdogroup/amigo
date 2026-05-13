#ifndef AMIGO_INTERIOR_POINT_OPTIMIZER_H
#define AMIGO_INTERIOR_POINT_OPTIMIZER_H

#include <mpi.h>

#include <memory>

#include "amigo.h"
#include "interior_point_backend.h"
#include "optimization_problem.h"

namespace amigo {

/**
 * Optimization state: solution vector + bound duals.
 *
 * Storage:
 *   x_     = xlam vector (primals + multipliers)
 *   duals_ = [zl(n_primal) | zu(n_primal)]
 */
template <typename T>
class OptVector {
 public:
  OptVector(int n_primal, int n_constraints, std::shared_ptr<Vector<T>> x)
      : np_(n_primal), nc_(n_constraints), x_(x) {
    duals_ = std::make_shared<Vector<T>>(2 * np_, 0, x->get_memory_location());
    slacks_ = std::make_shared<Vector<T>>(2 * np_, 0, x->get_memory_location());
  }

  void zero() {
    x_->zero();
    duals_->zero();
    slacks_->zero();
  }

  void copy(std::shared_ptr<OptVector<T>> src) {
    x_->copy(*src->x_);
    duals_->copy(*src->duals_);
    slacks_->copy(*src->slacks_);
  }

  template <ExecPolicy policy>
  void get_bound_duals(T** zl, T** zu) {
    T* a = duals_->template get_array<policy>();
    if (zl) *zl = a;
    if (zu) *zu = a + np_;
  }
  template <ExecPolicy policy>
  void get_bound_duals(const T** zl, const T** zu) const {
    const T* a = duals_->template get_array<policy>();
    if (zl) *zl = a;
    if (zu) *zu = a + np_;
  }

  template <ExecPolicy policy>
  void get_bound_slacks(T** sl, T** su) {
    T* a = slacks_->template get_array<policy>();
    if (sl) *sl = a;
    if (su) *su = a + np_;
  }
  template <ExecPolicy policy>
  void get_bound_slacks(const T** sl, const T** su) const {
    const T* a = slacks_->template get_array<policy>();
    if (sl) *sl = a;
    if (su) *su = a + np_;
  }

  template <ExecPolicy policy>
  T* get_solution_array() {
    return x_->template get_array<policy>();
  }
  template <ExecPolicy policy>
  const T* get_solution_array() const {
    return x_->template get_array<policy>();
  }

  std::shared_ptr<Vector<T>> get_solution() { return x_; }
  const std::shared_ptr<Vector<T>> get_solution() const { return x_; }

  std::shared_ptr<Vector<T>> get_slacks() { return slacks_; }

  int get_n_primal() const { return np_; }
  int get_n_constraints() const { return nc_; }

 private:
  int np_, nc_;
  std::shared_ptr<Vector<T>> x_;
  std::shared_ptr<Vector<T>> duals_;
  std::shared_ptr<Vector<T>> slacks_;
};

// State factory
namespace ipm {
template <typename T>
template <ExecPolicy policy, typename R>
State<T> State<T>::make(std::shared_ptr<OptVector<R>> v) {
  State<T> s{};
  s.xlam = v->template get_solution_array<policy>();
  v->template get_bound_duals<policy>(&s.zl, &s.zu);
  v->template get_bound_slacks<policy>(&s.sl, &s.su);
  return s;
}
}  // namespace ipm

/**
 * Interior-point optimizer for the 2x2 augmented system.
 *
 * Every variable is either a bounded primal or an equality constraint.
 * Wraps ipm:: backend functions for the Python/pybind interface.
 */
template <typename T, ExecPolicy policy>
class InteriorPointOptimizer {
 public:
  InteriorPointOptimizer(
      std::shared_ptr<OptimizationProblem<T, policy>> problem,
      std::shared_ptr<Vector<T>> lower, std::shared_ptr<Vector<T>> upper)
      : problem_(problem), lower_(lower), upper_(upper) {
    comm_ = problem->get_mpi_comm();

    int size = problem->get_num_variables();
    const Vector<int>& mult = *problem->get_multiplier_indicator();
    const Vector<T>& lb = *lower;
    const Vector<T>& ub = *upper;

    int np = 0, nc = 0;
    for (int i = 0; i < size; i++) mult[i] ? nc++ : np++;
    np_ = np;
    nc_ = nc;

    auto ml = (policy == ExecPolicy::CUDA) ? MemoryLocation::HOST_AND_DEVICE
                                           : MemoryLocation::HOST_ONLY;

    pidx_ = std::make_shared<Vector<int>>(np_, 0, ml);
    cidx_ = std::make_shared<Vector<int>>(nc_, 0, ml);
    lbx_ = std::make_shared<Vector<T>>(np_, 0, ml);
    ubx_ = std::make_shared<Vector<T>>(np_, 0, ml);
    lbh_ = std::make_shared<Vector<T>>(nc_, 0, ml);

    int pi = 0, ci = 0;
    for (int i = 0; i < size; i++) {
      if (mult[i]) {
        (*cidx_)[ci] = i;
        (*lbh_)[ci] = lb[i];
        ci++;
      } else {
        (*pidx_)[pi] = i;
        (*lbx_)[pi] = lb[i];
        (*ubx_)[pi] = ub[i];
        pi++;
      }
    }

    pidx_->copy_host_to_device();
    cidx_->copy_host_to_device();
    lbx_->copy_host_to_device();
    ubx_->copy_host_to_device();
    lbh_->copy_host_to_device();

    info_.n_primal = np_;
    info_.n_constraints = nc_;
    info_.primal_indices = pidx_->template get_array<policy>();
    info_.constraint_indices = cidx_->template get_array<policy>();
    info_.lbx = lbx_->template get_array<policy>();
    info_.ubx = ubx_->template get_array<policy>();
    info_.lbh = lbh_->template get_array<policy>();
  }

  // Factory

  std::shared_ptr<OptVector<T>> create_opt_vector() const {
    return std::make_shared<OptVector<T>>(np_, nc_, problem_->create_vector());
  }
  std::shared_ptr<OptVector<T>> create_opt_vector(
      std::shared_ptr<Vector<T>> x) const {
    return std::make_shared<OptVector<T>>(np_, nc_, x);
  }

  // Delegation

  void set_multipliers_value(T v, std::shared_ptr<Vector<T>> x) const {
    ipm::set_constraint_value(info_, v, x->template get_array<policy>());
  }
  void set_design_vars_value(T v, std::shared_ptr<Vector<T>> x) const {
    ipm::set_primal_value(info_, v, x->template get_array<policy>());
  }
  void copy_multipliers(std::shared_ptr<Vector<T>> d,
                        std::shared_ptr<Vector<T>> s) const {
    ipm::copy_constraints(info_, s->template get_array<policy>(),
                          d->template get_array<policy>());
  }
  void copy_design_vars(std::shared_ptr<Vector<T>> d,
                        std::shared_ptr<Vector<T>> s) const {
    ipm::copy_primals(info_, s->template get_array<policy>(),
                      d->template get_array<policy>());
  }

  void initialize_multipliers_and_slacks(
      T mu, const std::shared_ptr<Vector<T>>,
      std::shared_ptr<OptVector<T>> vars) const {
    // Project all primals into strict interior of bounds (Section 3.6),
    // then initialize bound duals and slacks from the projected values.
    T* xlam = vars->template get_solution_array<policy>();
    ipm::project_primals_into_interior(info_, xlam);
    T *zl, *zu;
    vars->template get_bound_duals<policy>(&zl, &zu);
    ipm::initialize_bound_duals(mu, info_, xlam, zl, zu);
    T *sl, *su;
    vars->template get_bound_slacks<policy>(&sl, &su);
    ipm::initialize_slacks(info_, xlam, sl, su);
  }

  // Condensed augmented system RHS (8-block to 4-block).
  // Returns L2 norm of the condensed residual.
  T compute_residual(T mu, const std::shared_ptr<OptVector<T>> vars,
                     const std::shared_ptr<Vector<T>> grad,
                     std::shared_ptr<Vector<T>> res) const {
    ipm::State<const T> s = ipm::State<const T>::template make<policy>(vars);
    res->zero();
    ipm::compute_residual(mu, info_, s, grad->template get_array<policy>(),
                          res->template get_array<policy>());
    T local = res->template dot<policy>(*res);
    T total;
    MPI_Allreduce(&local, &total, 1, get_mpi_type<T>(), MPI_SUM, comm_);
    return std::sqrt(total);
  }

  void compute_residual_and_infeasibility(
      T mu, const std::shared_ptr<OptVector<T>> vars,
      const std::shared_ptr<Vector<T>> grad, std::shared_ptr<Vector<T>> res,
      T& dsq, T& psq) const {
    ipm::State<const T> s = ipm::State<const T>::template make<policy>(vars);
    res->zero();
    ipm::compute_residual_and_infeasibility(
        mu, info_, s, grad->template get_array<policy>(),
        res->template get_array<policy>(), dsq, psq);
  }

  void compute_diagonal(const std::shared_ptr<OptVector<T>> vars,
                        std::shared_ptr<Vector<T>> diag) const {
    ipm::State<const T> s = ipm::State<const T>::template make<policy>(vars);
    ipm::compute_diagonal(info_, s, diag->template get_array<policy>());
  }

  // Copy augmented solution into update, then back-substitute for bound duals.
  void compute_update(T mu, const std::shared_ptr<OptVector<T>> vars,
                      const std::shared_ptr<Vector<T>> px,
                      std::shared_ptr<OptVector<T>> upd) const {
    ipm::State<const T> s = ipm::State<const T>::template make<policy>(vars);
    upd->get_solution()->copy(*px);
    T *dzl, *dzu;
    upd->template get_bound_duals<policy>(&dzl, &dzu);
    ipm::compute_bound_dual_step(mu, info_, s, px->template get_array<policy>(),
                                 dzl, dzu);
  }

  void compute_max_step(T tau, const std::shared_ptr<OptVector<T>> vars,
                        const std::shared_ptr<OptVector<T>> upd, T& ax, int& xi,
                        T& az, int& zi) const {
    ipm::State<const T> s = ipm::State<const T>::template make<policy>(vars);
    const T *dzl, *dzu;
    upd->template get_bound_duals<policy>(&dzl, &dzu);
    ipm::compute_max_step(tau, info_, s,
                          upd->template get_solution_array<policy>(), dzl, dzu,
                          ax, xi, az, zi);
  }

  void apply_step_update(T ax, T az, const std::shared_ptr<OptVector<T>> vars,
                         const std::shared_ptr<OptVector<T>> upd,
                         std::shared_ptr<OptVector<T>> tmp) const {
    ipm::State<const T> s = ipm::State<const T>::template make<policy>(vars);
    const T* dxlam = upd->template get_solution_array<policy>();
    const T *dzl, *dzu;
    upd->template get_bound_duals<policy>(&dzl, &dzu);
    T* xlam_n = tmp->template get_solution_array<policy>();
    T *zl_n, *zu_n;
    tmp->template get_bound_duals<policy>(&zl_n, &zu_n);
    T *sl_n, *su_n;
    tmp->template get_bound_slacks<policy>(&sl_n, &su_n);
    int n = info_.n_primal + info_.n_constraints;
    ipm::apply_step(ax, az, info_, s, dxlam, dzl, dzu, xlam_n, n, zl_n, zu_n,
                    sl_n, su_n);
  }

  // Python: avg_comp, xi = optimizer.compute_complementarity(vars)
  void compute_complementarity(const std::shared_ptr<OptVector<T>> vars, T& avg,
                               T& xi) const {
    ipm::State<const T> s = ipm::State<const T>::template make<policy>(vars);
    T ps[2] = {0, 0};
    T lm = std::numeric_limits<T>::max();
    ipm::compute_complementarity(info_, s, ps, lm);
    T gps[2];
    T gm;
    MPI_Allreduce(ps, gps, 2, get_mpi_type<T>(), MPI_SUM, comm_);
    MPI_Allreduce(&lm, &gm, 1, get_mpi_type<T>(), MPI_MIN, comm_);
    avg = (gps[1] > 0) ? gps[0] / gps[1] : 0.0;
    xi = (avg > 0) ? A2D::max2(T(0), A2D::min2(T(1), gm / avg)) : T(1);
  }

  // Python: scalar = optimizer.compute_complementarity_sq(vars)
  void compute_complementarity_sq(const std::shared_ptr<OptVector<T>> vars,
                                  T& sq) const {
    ipm::State<const T> s = ipm::State<const T>::template make<policy>(vars);
    T local = 0;
    ipm::compute_complementarity_sq(info_, s, T(0), local);
    MPI_Allreduce(&local, &sq, 1, get_mpi_type<T>(), MPI_SUM, comm_);
  }

  // Python: dev = optimizer.compute_max_comp_deviation(vars, mu)
  void compute_max_comp_deviation(const std::shared_ptr<OptVector<T>> vars,
                                  T mu, T& md) const {
    ipm::State<const T> s = ipm::State<const T>::template make<policy>(vars);
    T local;
    ipm::compute_max_comp_deviation(info_, s, mu, local);
    MPI_Allreduce(&local, &md, 1, get_mpi_type<T>(), MPI_MAX, comm_);
  }

  // Python: d_sq, p_sq, c_sq = optimizer.compute_kkt_error(vars, grad)
  void compute_kkt_error(const std::shared_ptr<OptVector<T>> vars,
                         const std::shared_ptr<Vector<T>> grad, T& d_sq,
                         T& p_sq, T& c_sq) const {
    ipm::State<const T> s = ipm::State<const T>::template make<policy>(vars);
    T ld = 0, lp = 0, lc = 0;
    ipm::compute_kkt_error_sq(info_, s, grad->template get_array<policy>(), ld,
                              lp, lc);
    T lv[3] = {ld, lp, lc}, gv[3];
    MPI_Allreduce(lv, gv, 3, get_mpi_type<T>(), MPI_SUM, comm_);
    d_sq = gv[0];
    p_sq = gv[1];
    c_sq = gv[2];
  }

  // Python: theta = optimizer.compute_constraint_violation_1norm(vars, grad)
  // Constraint violation 1-norm for filter line search.
  T compute_constraint_violation_1norm(
      const std::shared_ptr<OptVector<T>> vars,
      const std::shared_ptr<Vector<T>> grad) const {
    ipm::State<const T> s = ipm::State<const T>::template make<policy>(vars);
    T local = ipm::compute_constraint_violation_1norm(
        info_, s, grad->template get_array<policy>());
    T result;
    MPI_Allreduce(&local, &result, 1, get_mpi_type<T>(), MPI_SUM, comm_);
    return result;
  }

  // Python: d_inf, p_inf, c_inf = optimizer.compute_kkt_error_mu(mu, vars,
  // grad) Eq. 5: infinity-norm KKT error with barrier complementarity.
  void compute_kkt_error_mu(T mu, const std::shared_ptr<OptVector<T>> vars,
                            const std::shared_ptr<Vector<T>> grad, T& d_inf,
                            T& p_inf, T& c_inf) const {
    ipm::State<const T> s = ipm::State<const T>::template make<policy>(vars);
    T ld = 0, lp = 0, lc = 0;
    ipm::compute_kkt_error(mu, info_, s, grad->template get_array<policy>(), ld,
                           lp, lc);
    T lv[3] = {ld, lp, lc}, gv[3];
    MPI_Allreduce(lv, gv, 3, get_mpi_type<T>(), MPI_MAX, comm_);
    d_inf = gv[0];
    p_inf = gv[1];
    c_inf = gv[2];
  }

  // Python: dphi = optimizer.compute_barrier_dphi(mu, vars, update, res, px,
  // diag) KKT residual form (used by QF oracle for quality function
  // evaluation).
  T compute_barrier_dphi(T mu, const std::shared_ptr<OptVector<T>> vars,
                         const std::shared_ptr<OptVector<T>> update,
                         const std::shared_ptr<Vector<T>> res,
                         const std::shared_ptr<Vector<T>> px,
                         const std::shared_ptr<Vector<T>> diag) const {
    ipm::State<const T> s = ipm::State<const T>::template make<policy>(vars);
    T local = ipm::compute_barrier_dphi_from_kkt(
        info_, s, res->template get_array<policy>(),
        px->template get_array<policy>(), diag->template get_array<policy>());
    T result;
    MPI_Allreduce(&local, &result, 1, get_mpi_type<T>(), MPI_SUM, comm_);
    return result;
  }

  // Python: dphi = optimizer.compute_barrier_dphi_direct(mu, vars, grad, px)
  // Direct form: grad_barrier^T * dx = sum(grad_f*dx - mu*dx/sl + mu*dx/su)
  // Direct barrier directional derivative for the filter line search.
  T compute_barrier_dphi_direct(T mu, const std::shared_ptr<OptVector<T>> vars,
                                const std::shared_ptr<Vector<T>> grad,
                                const std::shared_ptr<Vector<T>> px) const {
    ipm::State<const T> s = ipm::State<const T>::template make<policy>(vars);
    T local = ipm::compute_barrier_dphi(mu, info_, s,
                                        grad->template get_array<policy>(),
                                        px->template get_array<policy>());
    T result;
    MPI_Allreduce(&local, &result, 1, get_mpi_type<T>(), MPI_SUM, comm_);
    return result;
  }

  T compute_barrier_log_sum(T mu,
                            const std::shared_ptr<OptVector<T>> vars) const {
    ipm::State<const T> s = ipm::State<const T>::template make<policy>(vars);
    T local = ipm::compute_barrier_log_sum(mu, info_, s);
    T result;
    MPI_Allreduce(&local, &result, 1, get_mpi_type<T>(), MPI_SUM, comm_);
    return result;
  }

  // Python: optimizer.reset_bound_multipliers(mu, kappa, vars) -- in-place
  void reset_bound_multipliers(T mu, T kappa,
                               std::shared_ptr<OptVector<T>> vars) const {
    ipm::State<const T> s = ipm::State<const T>::template make<policy>(vars);
    T *zl, *zu;
    vars->template get_bound_duals<policy>(&zl, &zu);
    ipm::reset_bound_multipliers(mu, kappa, info_, s, zl, zu);
  }

  void compute_affine_start_point(T bm,
                                  const std::shared_ptr<OptVector<T>> vars,
                                  const std::shared_ptr<OptVector<T>> upd,
                                  std::shared_ptr<OptVector<T>> dst) const {
    ipm::State<const T> s = ipm::State<const T>::template make<policy>(vars);
    const T *dzl, *dzu;
    upd->template get_bound_duals<policy>(&dzl, &dzu);
    T *zl_o, *zu_o;
    dst->template get_bound_duals<policy>(&zl_o, &zu_o);
    ipm::compute_affine_start_point(bm, info_, s, dzl, dzu, zl_o, zu_o);
  }

  void compute_dual_residual_vector(const std::shared_ptr<OptVector<T>> vars,
                                    const std::shared_ptr<Vector<T>> grad,
                                    std::shared_ptr<Vector<T>> out) const {
    ipm::State<const T> s = ipm::State<const T>::template make<policy>(vars);
    ipm::compute_dual_residual_vector(
        info_, s, grad->template get_array<policy>(),
        out->template get_array<policy>(), out->get_size());
  }

  void get_kkt_element_counts(int& n_d, int& n_p, int& n_c) const {
    n_d = np_;  // dual stationarity has n_primal components
    n_p = nc_;  // primal feasibility has n_constraints components
    n_c = 0;    // complementarity count: sum of finite bounds
    for (int i = 0; i < np_; i++) {
      if (!std::isinf((*lbx_)[i])) n_c++;
      if (!std::isinf((*ubx_)[i])) n_c++;
    }
  }

  // Debug: verify KKT system solve by forming full unreduced residual
  void check_update(T mu, const std::shared_ptr<Vector<T>> grad,
                    const std::shared_ptr<OptVector<T>> vars,
                    const std::shared_ptr<OptVector<T>> update,
                    const std::shared_ptr<CSRMat<T>> hess) const {
    // TODO: implement debug verification for new 2x2 system
  }

  // Slack mapping
  // Register which primals are slacks and which constraints they correspond to.
  // After this call, initialize_slacks() becomes available.
  //   slack_global:  global variable indices of slack variables
  //   constr_global: global variable indices of the inequality constraints
  // Both arrays must have the same length (n_slacks).
  void set_slack_mapping(int n_slacks, const int* slack_global,
                         const int* constr_global) {
    auto ml = (policy == ExecPolicy::CUDA) ? MemoryLocation::HOST_AND_DEVICE
                                           : MemoryLocation::HOST_ONLY;
    n_slacks_ = n_slacks;
    slack_global_ = std::make_shared<Vector<int>>(n_slacks, 0, ml);
    constr_global_ = std::make_shared<Vector<int>>(n_slacks, 0, ml);
    for (int k = 0; k < n_slacks; k++) {
      (*slack_global_)[k] = slack_global[k];
      (*constr_global_)[k] = constr_global[k];
    }
    slack_global_->copy_host_to_device();
    constr_global_->copy_host_to_device();
  }

  // Initialize slacks to s = d(x).
  //
  // Each inequality is reformulated as c_k(x) - s_k = 0.  The gradient
  // at constraint index ci holds the primal residual c_k(x) - s_k.
  // Recovering the constraint body: d_k(x) = (c_k(x) - s_k) + s_k.
  //
  // Requires gradient to have been evaluated at the current (x, s).
  // After this call, use initialize_multipliers_and_slacks() to push
  // the new slacks into bounds and reset bound duals.
  void initialize_slacks(const std::shared_ptr<Vector<T>> grad,
                         std::shared_ptr<OptVector<T>> vars) const {
    if (n_slacks_ == 0) return;
    T* xlam = vars->template get_solution_array<policy>();
    const T* g = grad->template get_array<policy>();
    for (int k = 0; k < n_slacks_; k++) {
      int si = (*slack_global_)[k];
      int ci = (*constr_global_)[k];
      T residual = g[ci];           // c_k(x) - s_k
      T s_old = xlam[si];           // current slack
      xlam[si] = residual + s_old;  // d_k(x) = c_k(x)
    }
  }

  bool has_slacks() const { return n_slacks_ > 0; }
  int get_num_slacks() const { return n_slacks_; }

  // NLP scaling: gradient-based.  Computed once at the initial point.
  // obj_scale_ scales the objective via alpha; constr_scale_ scales
  // Jacobian rows via D*H*D and gradient post-processing.

  /// Compute scaling factors from initial gradient and Jacobian row norms.
  void compute_nlp_scaling(const std::shared_ptr<Vector<T>> x,
                           const std::shared_ptr<Vector<T>> grad,
                           T max_gradient = T(100), T min_value = T(1e-8)) {
    const T* g = grad->template get_array<policy>();

    // Objective: df = min(1, max_gradient / ||grad_f||_inf)
    T obj_max = T(0);
    for (int i = 0; i < np_; i++) {
      T v = std::abs(g[(*pidx_)[i]]);
      if (v > obj_max) obj_max = v;
    }
    T global_obj_max;
    MPI_Allreduce(&obj_max, &global_obj_max, 1, get_mpi_type<T>(), MPI_MAX,
                  comm_);

    obj_scale_ =
        (global_obj_max > max_gradient) ? max_gradient / global_obj_max : T(1);
    if (obj_scale_ < min_value) obj_scale_ = min_value;

    // Constraint: dc[j] = min(1, max_gradient / ||J_row_j||_inf)
    // Assemble KKT at (x, lam=0) to read Jacobian rows.
    auto hess = problem_->create_matrix();
    problem_->hessian(T(1), x, hess);
    hess->copy_data_device_to_host();

    int nr, nc_mat, nnz_mat;
    const int *rp, *cl;
    T* dt;
    hess->get_data(&nr, &nc_mat, &nnz_mat, &rp, &cl, &dt);

    auto ml = (policy == ExecPolicy::CUDA) ? MemoryLocation::HOST_AND_DEVICE
                                           : MemoryLocation::HOST_ONLY;
    constr_scale_ = std::make_shared<Vector<T>>(nc_, 0, ml);

    // Scan each constraint row for its inf-norm
    for (int j = 0; j < nc_; j++) {
      int row = (*cidx_)[j];
      T row_max = T(0);
      for (int k = rp[row]; k < rp[row + 1]; k++) {
        T v = std::abs(dt[k]);
        if (v > row_max) row_max = v;
      }
      T dc = (row_max > max_gradient) ? max_gradient / row_max : T(1);
      if (dc < min_value) dc = min_value;
      (*constr_scale_)[j] = dc;
    }

    // Scale constraint targets for consistency
    for (int j = 0; j < nc_; j++) {
      T dc = (*constr_scale_)[j];
      if (dc != T(1)) (*lbh_)[j] *= dc;
    }
    lbh_->copy_host_to_device();
    info_.lbh = lbh_->template get_array<policy>();

    // Per-variable scaling: 1.0 for primals, dc[j] for constraints.
    scale_vec_.resize(nr, T(1));
    for (int j = 0; j < nc_; j++) scale_vec_[(*cidx_)[j]] = (*constr_scale_)[j];

    constr_scale_->copy_host_to_device();
    scaling_active_ = true;
  }

  /// Scale constraint rows of the gradient in-place.
  void apply_gradient_scaling(std::shared_ptr<Vector<T>> grad) const {
    if (!scaling_active_) return;
    T* g = grad->template get_array<policy>();
    for (int j = 0; j < nc_; j++) g[(*cidx_)[j]] *= (*constr_scale_)[j];
  }

  /// D*H*D similarity transform on the KKT matrix (scales Jacobian blocks).
  void apply_hessian_scaling(std::shared_ptr<CSRMat<T>> hess) const {
    if (!scaling_active_) return;
    int nr, nc_mat, nnz_mat;
    const int *rp, *cl;
    T* dt;
    hess->get_data(&nr, &nc_mat, &nnz_mat, &rp, &cl, &dt);
    const T* sv = scale_vec_.data();
    for (int i = 0; i < nr; i++) {
      T di = sv[i];
      for (int k = rp[i]; k < rp[i + 1]; k++) dt[k] *= di * sv[cl[k]];
    }
  }

  /// Scale constraint multipliers into scaled space: y[j] *= dc[j].
  void scale_multipliers(std::shared_ptr<Vector<T>> x) const {
    if (!scaling_active_) return;
    T* xlam = x->template get_array<policy>();
    for (int j = 0; j < nc_; j++) xlam[(*cidx_)[j]] *= (*constr_scale_)[j];
  }

  /// Unscale constraint multipliers: y[j] /= dc[j].
  void unscale_multipliers(std::shared_ptr<Vector<T>> x) const {
    if (!scaling_active_) return;
    T* xlam = x->template get_array<policy>();
    for (int j = 0; j < nc_; j++) xlam[(*cidx_)[j]] /= (*constr_scale_)[j];
  }

  T get_obj_scale() const { return obj_scale_; }
  bool has_scaling() const { return scaling_active_; }

  // Accessors
  int get_num_design_variables() const { return np_; }
  int get_num_constraints() const { return nc_; }
  int get_num_equalities() const { return nc_; }
  int get_num_inequalities() const { return 0; }

  std::shared_ptr<Vector<T>> get_lbx() const { return lbx_; }
  std::shared_ptr<Vector<T>> get_ubx() const { return ubx_; }

  // Return relaxed bounds if available, otherwise original bounds.
  // These are the bounds actually used by the IPM backend (info_.lbx/ubx).
  std::shared_ptr<Vector<T>> get_lbx_relaxed() const {
    return lbx_relaxed_ ? lbx_relaxed_ : lbx_;
  }
  std::shared_ptr<Vector<T>> get_ubx_relaxed() const {
    return ubx_relaxed_ ? ubx_relaxed_ : ubx_;
  }

  // Relax bounds by bound_relax_factor (default 1e-8).
  // Must be called before initialize_multipliers_and_slacks.
  void relax_bounds(T factor = 1e-8, T constr_viol_tol = 1e-4) {
    if (factor <= 0) return;
    lbx_relaxed_ =
        std::make_shared<Vector<T>>(np_, 0, lbx_->get_memory_location());
    ubx_relaxed_ =
        std::make_shared<Vector<T>>(np_, 0, ubx_->get_memory_location());
    T* lb_buf = lbx_relaxed_->template get_array<policy>();
    T* ub_buf = ubx_relaxed_->template get_array<policy>();
    ipm::relax_bounds(info_, lb_buf, ub_buf, factor, constr_viol_tol);
    lbx_relaxed_->copy_host_to_device();
    ubx_relaxed_->copy_host_to_device();
  }

 private:
  std::shared_ptr<OptimizationProblem<T, policy>> problem_;
  std::shared_ptr<Vector<T>> lower_, upper_;
  MPI_Comm comm_;
  int np_, nc_;
  std::shared_ptr<Vector<int>> pidx_, cidx_;
  std::shared_ptr<Vector<T>> lbx_, ubx_, lbh_;
  std::shared_ptr<Vector<T>> lbx_relaxed_, ubx_relaxed_;
  ipm::ProblemInfo<T> info_;

  // Slack-to-constraint mapping (set via set_slack_mapping)
  int n_slacks_ = 0;
  std::shared_ptr<Vector<int>> slack_global_;
  std::shared_ptr<Vector<int>> constr_global_;

  // NLP scaling state
  T obj_scale_ = T(1);
  std::shared_ptr<Vector<T>> constr_scale_;
  std::vector<T> scale_vec_;  // per-variable: 1.0 primals, dc[j] constraints
  bool scaling_active_ = false;
};

}  // namespace amigo

#endif  // AMIGO_INTERIOR_POINT_OPTIMIZER_H
