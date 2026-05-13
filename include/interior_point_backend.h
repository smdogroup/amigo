#ifndef AMIGO_INTERIOR_POINT_BACKEND_H
#define AMIGO_INTERIOR_POINT_BACKEND_H

/*
  Primal-dual interior-point backend for the full-space solver.

  Problem formulation (after slack introduction for inequalities):

    min  f(x)
    s.t. c(x) = 0          (equalities, including c_k(x) - s_k = 0 for ineq)
         xL <= x <= xU      (bounds on all primals: design vars + slacks)

  Full 8-block Newton system:

    Block 1 (x stationarity):     rhs_x  = grad_f + J^T y - z_L + z_U
    Block 2 (s stationarity):     rhs_s  = -y_d - v_L + v_U
    Block 3 (eq feasibility):     rhs_yc = c(x)
    Block 4 (ineq feasibility):   rhs_yd = d(x) - s
    Block 5 (lower compl, x):     rhs_zL = gap_xL * z_L - mu
    Block 6 (upper compl, x):     rhs_zU = gap_xU * z_U - mu
    Block 7 (lower compl, s):     rhs_vL = gap_sL * v_L - mu
    Block 8 (upper compl, s):     rhs_vU = gap_sU * v_U - mu

  Condensation to 4-block augmented system:

    augRhs_x = rhs_x + rhs_zL / gap_xL - rhs_zU / gap_xU
    augRhs_s = rhs_s + rhs_vL / gap_sL - rhs_vU / gap_sU
    (= simplifies to: -grad + mu/gap_L - mu/gap_U, as z terms cancel)

  In amigo, x and s share the primal vector, and y_c and y_d share
  the constraint vector, so the 4-block reduces to a 2-block solve:

    [ W + Sigma + delta_w*I    J^T      ] [ dx   ]   [ -augRhs ]
    [       J              -delta_c*I   ] [ dlam ] = [ -rhs_c  ]

  Back-substitution for bound dual steps:
    dz_L = -(rhs_zL + z_L * dx) / gap_xL
    dz_U = -(rhs_zU - z_U * dx) / gap_xU
*/

#include <cmath>
#include <memory>

#include "a2dcore.h"
#include "amigo.h"

namespace amigo {

template <typename T>
class OptVector;

namespace ipm {

// Two categories: bounded primals and equality constraints.
template <typename T>
struct ProblemInfo {
  int n_primal = 0;
  int n_constraints = 0;
  const int* primal_indices = nullptr;
  const int* constraint_indices = nullptr;
  const T *lbx = nullptr, *ubx = nullptr;
  const T* lbh = nullptr;
};

// Pointers into OptVector storage. T may be const-qualified for read-only
// access.
template <typename T>
struct State {
  T* xlam = nullptr;
  T* zl = nullptr;
  T* zu = nullptr;
  T* sl = nullptr;  // lower bound slack: sl[i] ≈ x[i] - lbx[i], always > 0
  T* su = nullptr;  // upper bound slack: su[i] ≈ ubx[i] - x[i], always > 0

  template <ExecPolicy policy, typename R>
  static State make(std::shared_ptr<OptVector<R>> vars);
};

// Initialize stored bound slacks from current primal values.
// Must be called after project_primals_into_interior.
template <typename T>
void initialize_slacks(const ProblemInfo<T>& p, const T* xlam, T* sl, T* su) {
  for (int i = 0; i < p.n_primal; i++) {
    T x = xlam[p.primal_indices[i]];
    sl[i] = std::isinf(p.lbx[i]) ? T(1) : (x - p.lbx[i]);
    su[i] = std::isinf(p.ubx[i]) ? T(1) : (p.ubx[i] - x);
  }
}

// Utilities
template <typename T>
void set_primal_value(const ProblemInfo<T>& p, T val, T* xlam) {
  for (int i = 0; i < p.n_primal; i++) {
    xlam[p.primal_indices[i]] = val;
  }
}

template <typename T>
void set_constraint_value(const ProblemInfo<T>& p, T val, T* xlam) {
  for (int i = 0; i < p.n_constraints; i++) {
    xlam[p.constraint_indices[i]] = val;
  }
}

template <typename T>
void copy_primals(const ProblemInfo<T>& p, const T* src, T* dst) {
  for (int i = 0; i < p.n_primal; i++) {
    int k = p.primal_indices[i];
    dst[k] = src[k];
  }
}

template <typename T>
void copy_constraints(const ProblemInfo<T>& p, const T* src, T* dst) {
  for (int i = 0; i < p.n_constraints; i++) {
    int k = p.constraint_indices[i];
    dst[k] = src[k];
  }
}

// Relax bounds by a small factor to avoid numerical issues at exact bounds.
//   x_L -= min(constr_viol_tol, factor * max(1, |x_L|))
//   x_U += min(constr_viol_tol, factor * max(1, |x_U|))
// Default: bound_relax_factor = 1e-8, constr_viol_tol = 1e-4.
template <typename T>
void relax_bounds(ProblemInfo<T>& p, T* lbx_buf, T* ubx_buf, T factor = 1e-8,
                  T constr_viol_tol = 1e-4) {
  for (int i = 0; i < p.n_primal; i++) {
    if (!std::isinf(p.lbx[i])) {
      T delta = A2D::min2(constr_viol_tol,
                          factor * A2D::max2(T(1), std::abs(p.lbx[i])));
      lbx_buf[i] = p.lbx[i] - delta;
    } else {
      lbx_buf[i] = p.lbx[i];
    }
    if (!std::isinf(p.ubx[i])) {
      T delta = A2D::min2(constr_viol_tol,
                          factor * A2D::max2(T(1), std::abs(p.ubx[i])));
      ubx_buf[i] = p.ubx[i] + delta;
    } else {
      ubx_buf[i] = p.ubx[i];
    }
  }
  p.lbx = lbx_buf;
  p.ubx = ubx_buf;
}

// Project all primals into the strict interior of their bounds (Section 3.6).
// For one-sided bounds: x <- max(x, lb + kappa1 * max(1, |lb|))  or similar.
// For two-sided bounds: x is projected into [lb + p_l, ub - p_u] where
//   p_l = min(kappa1 * max(1, |lb|), kappa2 * (ub - lb))
//   p_u = min(kappa1 * max(1, |ub|), kappa2 * (ub - lb))
// Defaults: kappa1 = 0.01, kappa2 = 0.01.
// Must be called before initialize_bound_duals.
template <typename T>
void project_primals_into_interior(const ProblemInfo<T>& p, T* xlam,
                                   T kappa1 = 1e-2, T kappa2 = 1e-2) {
  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    T x = xlam[idx];
    T lb = p.lbx[i];
    T ub = p.ubx[i];
    bool has_lb = !std::isinf(lb);
    bool has_ub = !std::isinf(ub);

    if (has_lb && has_ub) {
      T range = ub - lb;
      T pl = A2D::min2(kappa1 * A2D::max2(T(1), std::abs(lb)), kappa2 * range);
      T pu = A2D::min2(kappa1 * A2D::max2(T(1), std::abs(ub)), kappa2 * range);
      xlam[idx] = A2D::max2(A2D::min2(x, ub - pu), lb + pl);
    } else if (has_lb) {
      xlam[idx] = A2D::max2(x, lb + kappa1 * A2D::max2(T(1), std::abs(lb)));
    } else if (has_ub) {
      xlam[idx] = A2D::min2(x, ub - kappa1 * A2D::max2(T(1), std::abs(ub)));
    }
  }
}

// Initialize bound duals to 1.0 for all finite bounds (Section 3.6).
// Must be called after project_primals_into_interior.
template <typename T>
void initialize_bound_duals(T mu, const ProblemInfo<T>& p, const T* xlam, T* zl,
                            T* zu) {
  for (int i = 0; i < p.n_primal; i++) {
    zl[i] = std::isinf(p.lbx[i]) ? T(0) : T(1);
    zu[i] = std::isinf(p.ubx[i]) ? T(0) : T(1);
  }
}

// Augmented system RHS via 8-block to 4-block condensation.
//
// For each primal i, the 8-block RHS has three relevant blocks:
//   rhs_stat   = grad[i] - zl[i] + zu[i]       (stationarity, Blocks 1-2)
//   rhs_complL = gap_L * zl - mu                (complementarity, Block 5/7)
//   rhs_complU = gap_U * zu - mu                (complementarity, Block 6/8)
//
// Condensation folds complementarity into stationarity:
//   augRhs = rhs_stat + rhs_complL/gap_L - rhs_complU/gap_U
//
// Output is negated: res = -augRhs  (convention: K * px = res gives Newton
// step)
template <typename T>
void compute_residual(T mu, const ProblemInfo<T>& p, State<const T>& s,
                      const T* grad, T* res) {
  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];

    // Stationarity residual (Blocks 1-2: rhs_x or rhs_s)
    T rhs_stat = grad[idx] - s.zl[i] + s.zu[i];

    // Condense complementarity into stationarity
    T aug = rhs_stat;
    if (!std::isinf(p.lbx[i])) {
      T gap = s.sl[i];
      aug += (gap * s.zl[i] - mu) / gap;  // +rhs_complL / gap_L
    }
    if (!std::isinf(p.ubx[i])) {
      T gap = s.su[i];
      aug -= (gap * s.zu[i] - mu) / gap;  // -rhs_complU / gap_U
    }
    res[idx] = -aug;
  }

  // Constraint feasibility (Blocks 3-4: rhs_yc, rhs_yd)
  for (int j = 0; j < p.n_constraints; j++) {
    int idx = p.constraint_indices[j];
    res[idx] = -(grad[idx] - p.lbh[j]);
  }
}

// Same as compute_residual, but also accumulates squared norms of the
// un-condensed dual and primal infeasibility for convergence monitoring.
template <typename T>
void compute_residual_and_infeasibility(T mu, const ProblemInfo<T>& p,
                                        State<const T>& s, const T* grad,
                                        T* res, T& dual_sq, T& primal_sq) {
  dual_sq = 0.0;
  primal_sq = 0.0;

  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];

    // Stationarity residual (un-condensed, for monitoring)
    T rhs_stat = grad[idx] - s.zl[i] + s.zu[i];
    dual_sq += rhs_stat * rhs_stat;

    // Condense complementarity into stationarity
    T aug = rhs_stat;
    if (!std::isinf(p.lbx[i])) {
      T gap = s.sl[i];
      aug += (gap * s.zl[i] - mu) / gap;
    }
    if (!std::isinf(p.ubx[i])) {
      T gap = s.su[i];
      aug -= (gap * s.zu[i] - mu) / gap;
    }
    res[idx] = -aug;
  }

  for (int j = 0; j < p.n_constraints; j++) {
    int idx = p.constraint_indices[j];
    T rp = grad[idx] - p.lbh[j];
    primal_sq += rp * rp;
    res[idx] = -rp;
  }
}

// Barrier diagonal Sigma for the augmented system.
//
//   sigma_x[i] = z_L[i]/gap_xL[i] + z_U[i]/gap_xU[i]  (Block 1,1)
//   sigma_s[k] = v_L[k]/gap_sL[k] + v_U[k]/gap_sU[k]  (Block 2,2)
//
// In amigo, x and s share the primal vector, so both use the same loop.
// Constraint diagonal entries are zero (regularization delta_c added
// separately).
template <typename T>
void compute_diagonal(const ProblemInfo<T>& p, State<const T>& s, T* diag) {
  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    T sigma = T(0);
    if (!std::isinf(p.lbx[i])) sigma += s.zl[i] / s.sl[i];
    if (!std::isinf(p.ubx[i])) sigma += s.zu[i] / s.su[i];
    diag[idx] = sigma;
  }
  for (int j = 0; j < p.n_constraints; j++) {
    diag[p.constraint_indices[j]] = T(0);
  }
}

// Bound dual back-substitution.
//
// After solving the augmented system for (dx, dlam), recover the bound
// dual steps that were eliminated during RHS condensation.
//
// The complementarity blocks give:
//   rhs_complL = gap_L * zl - mu     (Block 5/7)
//   rhs_complU = gap_U * zu - mu     (Block 6/8)
//
// Back-substitution with sign correction:
//   dzl = -(rhs_complL + zl * dx) / gap_L
//   dzu = -(rhs_complU - zu * dx) / gap_U
template <typename T>
void compute_bound_dual_step(T mu, const ProblemInfo<T>& p, State<const T>& s,
                             const T* px, T* dzl, T* dzu) {
  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    T dx = px[idx];
    dzl[i] = dzu[i] = 0.0;

    if (!std::isinf(p.lbx[i])) {
      T gap = s.sl[i];
      T rhs_complL = gap * s.zl[i] - mu;
      dzl[i] = -(rhs_complL + s.zl[i] * dx) / gap;
    }
    if (!std::isinf(p.ubx[i])) {
      T gap = s.su[i];
      T rhs_complU = gap * s.zu[i] - mu;
      dzu[i] = -(rhs_complU - s.zu[i] * dx) / gap;
    }
  }
}

// Fraction-to-the-boundary rule. Finds the largest step alpha in (0,1]
// such that all primals stay within bounds and all duals stay positive:
//   x + alpha*dx >= (1-tau)*(x - lb)  for each finite lower bound
//   ub - (x + alpha*dx) >= (1-tau)*(ub - x)  for each finite upper bound
//   zl + alpha*dzl >= (1-tau)*zl, zu + alpha*dzu >= (1-tau)*zu
template <typename T>
void compute_max_step(T tau, const ProblemInfo<T>& p, State<const T>& s,
                      const T* px, const T* dzl, const T* dzu, T& ax, int& xi,
                      T& az, int& zi) {
  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    T dx = px[idx];

    if (!std::isinf(p.lbx[i])) {
      if (dx < 0.0) {
        T a = -tau * s.sl[i] / dx;
        if (a < ax) {
          ax = a;
          xi = idx;
        }
      }
      if (dzl[i] < 0.0) {
        T a = -tau * s.zl[i] / dzl[i];
        if (a < az) {
          az = a;
          zi = idx;
        }
      }
    }
    if (!std::isinf(p.ubx[i])) {
      if (dx > 0.0) {
        T a = tau * s.su[i] / dx;
        if (a < ax) {
          ax = a;
          xi = idx;
        }
      }
      if (dzu[i] < 0.0) {
        T a = -tau * s.zu[i] / dzu[i];
        if (a < az) {
          az = a;
          zi = idx;
        }
      }
    }
  }
}

// Apply the full primal-dual-bound trial step (eq. 14-15).
//   xlam_new = xlam + alpha_x * dxlam   (primals + multipliers)
//   zl_new   = zl   + alpha_z * dzl     (lower bound duals)
//   zu_new   = zu   + alpha_z * dzu     (upper bound duals)
template <typename T>
void apply_step(T ax, T az, const ProblemInfo<T>& p, State<const T>& s,
                const T* dxlam, const T* dzl, const T* dzu, T* xlam_new,
                int n_xlam, T* zl_new, T* zu_new, T* sl_new, T* su_new) {
  for (int i = 0; i < n_xlam; i++) {
    xlam_new[i] = s.xlam[i] + ax * dxlam[i];
  }
  for (int i = 0; i < p.n_primal; i++) {
    if (!std::isinf(p.lbx[i])) zl_new[i] = s.zl[i] + az * dzl[i];
    if (!std::isinf(p.ubx[i])) zu_new[i] = s.zu[i] + az * dzu[i];
  }
  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    T dx = ax * dxlam[idx];
    if (!std::isinf(p.lbx[i])) sl_new[i] = s.sl[i] + dx;
    if (!std::isinf(p.ubx[i])) su_new[i] = s.su[i] - dx;
  }
}

// Average complementarity mu_avg = sum(gap*z) / n_bounds, and
// minimum complementarity product (for uniformity measure xi).
template <typename T>
void compute_complementarity(const ProblemInfo<T>& p, State<const T>& s,
                             T partial_sum[], T& local_min) {
  for (int i = 0; i < p.n_primal; i++) {
    if (!std::isinf(p.lbx[i])) {
      T c = s.sl[i] * s.zl[i];
      partial_sum[0] += c;
      partial_sum[1] += 1.0;
      local_min = A2D::min2(local_min, c);
    }
    if (!std::isinf(p.ubx[i])) {
      T c = s.su[i] * s.zu[i];
      partial_sum[0] += c;
      partial_sum[1] += 1.0;
      local_min = A2D::min2(local_min, c);
    }
  }
}

// Maximum deviation of individual complementarity products from mu.
template <typename T>
void compute_max_comp_deviation(const ProblemInfo<T>& p, State<const T>& s,
                                T mu, T& max_dev) {
  max_dev = 0.0;
  for (int i = 0; i < p.n_primal; i++) {
    if (!std::isinf(p.lbx[i]))
      max_dev = A2D::max2(max_dev, std::abs(s.sl[i] * s.zl[i] - mu));
    if (!std::isinf(p.ubx[i]))
      max_dev = A2D::max2(max_dev, std::abs(s.su[i] * s.zu[i] - mu));
  }
}

// Sum of squared complementarity products (for quality function evaluation).
template <typename T>
void compute_complementarity_sq(const ProblemInfo<T>& p, State<const T>& s,
                                T mu, T& sq) {
  sq = 0.0;
  for (int i = 0; i < p.n_primal; i++) {
    if (!std::isinf(p.lbx[i])) {
      T r = s.sl[i] * s.zl[i] - mu;
      sq += r * r;
    }
    if (!std::isinf(p.ubx[i])) {
      T r = s.su[i] * s.zu[i] - mu;
      sq += r * r;
    }
  }
}

// Optimality error E_mu with three components (infinity norms):
//   dual    = max |grad_i - zl_i + zu_i|           (stationarity)
//   primal  = max |c_j(x) - target_j|              (feasibility)
//   comp    = max |gap_i * z_i - mu|                (complementarity)
template <typename T>
void compute_kkt_error(T mu, const ProblemInfo<T>& p, State<const T>& s,
                       const T* grad, T& dual, T& primal, T& comp) {
  dual = primal = comp = 0.0;

  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    dual = A2D::max2(dual, std::abs(grad[idx] - s.zl[i] + s.zu[i]));
    if (!std::isinf(p.lbx[i]))
      comp = A2D::max2(comp, std::abs(s.sl[i] * s.zl[i] - mu));
    if (!std::isinf(p.ubx[i]))
      comp = A2D::max2(comp, std::abs(s.su[i] * s.zu[i] - mu));
  }
  for (int j = 0; j < p.n_constraints; j++) {
    int idx = p.constraint_indices[j];
    primal = A2D::max2(primal, std::abs(grad[idx] - p.lbh[j]));
  }
}

// Barrier log-sum: -mu * sum_i ln(x_i - lb_i) - mu * sum_i ln(ub_i - x_i).
// Added to the objective f(x) to form the barrier objective phi_mu(x).
template <typename T>
T compute_barrier_log_sum(T mu, const ProblemInfo<T>& p, State<const T>& s) {
  T b = 0.0;
  for (int i = 0; i < p.n_primal; i++) {
    if (!std::isinf(p.lbx[i])) {
      T g = s.sl[i];
      if (g > 0) b -= mu * std::log(g);
    }
    if (!std::isinf(p.ubx[i])) {
      T g = s.su[i];
      if (g > 0) b -= mu * std::log(g);
    }
  }
  return b;
}

// Directional derivative of the barrier objective along the search direction:
//   dphi = sum_i (grad_i * dx_i - mu * dx_i / gap_l_i + mu * dx_i / gap_u_i)
// Used in the Armijo condition and switching condition of the filter line
// search.
template <typename T>
T compute_barrier_dphi(T mu, const ProblemInfo<T>& p, State<const T>& s,
                       const T* grad, const T* px) {
  T d = 0.0;
  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    T dx = px[idx];
    d += grad[idx] * dx;
    if (!std::isinf(p.lbx[i])) d -= mu * dx / s.sl[i];
    if (!std::isinf(p.ubx[i])) d += mu * dx / s.su[i];
  }
  return d;
}

// Bound multiplier reset: clamp each z_i to [mu/(kappa*gap), kappa*mu/gap].
// Prevents the ratio Sigma_i = z_i/gap from deviating too far from mu/gap^2,
// which is needed for the convergence proof of primal-dual methods.
template <typename T>
void reset_bound_multipliers(T mu, T kappa, const ProblemInfo<T>& p,
                             State<const T>& s, T* zl_out, T* zu_out) {
  for (int i = 0; i < p.n_primal; i++) {
    if (!std::isinf(p.lbx[i])) {
      T g = s.sl[i];
      zl_out[i] =
          A2D::max2(A2D::min2(s.zl[i], kappa * mu / g), mu / (kappa * g));
    }
    if (!std::isinf(p.ubx[i])) {
      T g = s.su[i];
      zu_out[i] =
          A2D::max2(A2D::min2(s.zu[i], kappa * mu / g), mu / (kappa * g));
    }
  }
}

// Project bound duals to at least beta_min after the affine scaling step.
template <typename T>
void compute_affine_start_point(T beta_min, const ProblemInfo<T>& p,
                                State<const T>& s, const T* dzl, const T* dzu,
                                T* zl_out, T* zu_out) {
  for (int i = 0; i < p.n_primal; i++) {
    if (!std::isinf(p.lbx[i]))
      zl_out[i] = A2D::max2(s.zl[i] + dzl[i], beta_min);
    if (!std::isinf(p.ubx[i]))
      zu_out[i] = A2D::max2(s.zu[i] + dzu[i], beta_min);
  }
}

// Squared KKT error norms (sum-of-squares, for quality function / convergence).
template <typename T>
void compute_kkt_error_sq(const ProblemInfo<T>& p, State<const T>& s,
                          const T* grad, T& dual_sq, T& primal_sq, T& comp_sq) {
  dual_sq = primal_sq = comp_sq = 0.0;

  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    T rd = grad[idx] - s.zl[i] + s.zu[i];
    dual_sq += rd * rd;
    if (!std::isinf(p.lbx[i])) {
      T c = s.sl[i] * s.zl[i];
      comp_sq += c * c;
    }
    if (!std::isinf(p.ubx[i])) {
      T c = s.su[i] * s.zu[i];
      comp_sq += c * c;
    }
  }
  for (int j = 0; j < p.n_constraints; j++) {
    int idx = p.constraint_indices[j];
    T rp = grad[idx] - p.lbh[j];
    primal_sq += rp * rp;
  }
}

// Constraint violation 1-norm: theta = sum_j |c_j(x)|.
template <typename T>
T compute_constraint_violation_1norm(const ProblemInfo<T>& p, State<const T>& s,
                                     const T* grad) {
  T result = 0.0;
  for (int j = 0; j < p.n_constraints; j++) {
    int idx = p.constraint_indices[j];
    result += std::abs(grad[idx] - p.lbh[j]);
  }
  return result;
}

// Dual residual vector: r_d[i] = grad[i] - zl[i] + zu[i] for primals, 0
// elsewhere. Used by the quality function to compute the cross term r_d^T *
// (Hessian_mod * dx).
template <typename T>
void compute_dual_residual_vector(const ProblemInfo<T>& p, State<const T>& s,
                                  const T* grad, T* out, int size) {
  for (int i = 0; i < size; i++) out[i] = 0.0;
  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    out[idx] = grad[idx] - s.zl[i] + s.zu[i];
  }
}

// Barrier directional derivative from the condensed KKT residual and solution.
//
// dphi = -augRhs_primal^T * px - lam^T * (J * dx)
//
// J*dx is reconstructed from constraint rows: J*dx = res_j - diag_j * px_j,
// which removes the regularization contribution (delta_c * dlam).
//
// Used in the switching condition of the filter line search (Eq. 19).
template <typename T>
T compute_barrier_dphi_from_kkt(const ProblemInfo<T>& p, State<const T>& s,
                                const T* res, const T* px, const T* diag) {
  T dphi = 0.0;
  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    dphi -= res[idx] * px[idx];
  }
  for (int j = 0; j < p.n_constraints; j++) {
    int idx = p.constraint_indices[j];
    T lam = s.xlam[idx];
    T jdx = res[idx] - diag[idx] * px[idx];
    dphi -= lam * jdx;
  }
  return dphi;
}

}  // namespace ipm
}  // namespace amigo

#endif  // AMIGO_INTERIOR_POINT_BACKEND_H
