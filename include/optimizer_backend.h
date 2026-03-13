#ifndef AMIGO_OPTIMIZER_BACKEND_H
#define AMIGO_OPTIMIZER_BACKEND_H

#include <memory>

#include "amigo.h"

namespace amigo {

template <typename T>
class OptVector;

namespace detail {

template <typename T>
class OptInfo {
 public:
  int num_variables = 0;
  int num_equalities = 0;
  int num_inequalities = 0;

  const int* design_variable_indices = nullptr;
  const int* equality_indices = nullptr;
  const int* inequality_indices = nullptr;

  // bounds (can be +/-inf)
  const T *lbx = nullptr, *ubx = nullptr;  // x bounds for variables
  const T *lbc = nullptr, *ubc = nullptr;  // constraint bounds
  const T* lbh = nullptr;                  // constraint bounds
};

template <typename T>
class OptStateData {
 public:
  template <ExecPolicy policy, typename R>
  static OptStateData make(std::shared_ptr<OptVector<R>> vars) {
    OptStateData self{};

    vars->template get_bound_duals<policy>(&self.zl, &self.zu);
    vars->template get_slacks<policy>(&self.s);
    vars->template get_slack_duals<policy>(&self.zsl, &self.zsu);
    self.xlam = vars->template get_solution_array<policy>();

    return self;
  }

  // primal/dual/slacks
  T* xlam = nullptr;  // full solution vector (design vars + multipliers)
  T *zl = nullptr, *zu = nullptr;  // variable duals (size num_variables)

  T* s = nullptr;      // slack for inequalities (size num_inequalities)
  T *zsl = nullptr;    // dual for slack lower bound (s >= lbc)
  T *zsu = nullptr;    // dual for slack upper bound (s <= ubc)
};

template <typename T>
void set_multipliers_value(const OptInfo<T>& info, T value, T* x) {
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_equalities; i++) {
    int idx = info.equality_indices[i];
    x[idx] = value;
  }

#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_inequalities; i++) {
    int idx = info.inequality_indices[i];
    x[idx] = value;
  }
}

template <typename T>
void set_design_vars_value(const OptInfo<T>& info, T value, T* x) {
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_variables; i++) {
    int idx = info.design_variable_indices[i];
    x[idx] = value;
  }
}

template <typename T>
void copy_multipliers(const OptInfo<T>& info, const T* d_src, T* d_dest) {
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_equalities; i++) {
    int idx = info.equality_indices[i];
    d_dest[idx] = d_src[idx];
  }

#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_inequalities; i++) {
    int idx = info.inequality_indices[i];
    d_dest[idx] = d_src[idx];
  }
}

template <typename T>
void copy_design_vars(const OptInfo<T>& info, const T* d_src, T* d_dest) {
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_variables; i++) {
    int idx = info.design_variable_indices[i];
    d_dest[idx] = d_src[idx];
  }
}

template <typename T>
void initialize_multipliers_and_slacks(T barrier_param, const OptInfo<T>& info,
                                       const T* g, OptStateData<T>& pt) {
  // Initialize the lower and upper bound dual variables
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_variables; i++) {
    pt.zl[i] = pt.zu[i] = 0.0;
    if (!std::isinf(info.lbx[i])) {
      pt.zl[i] = barrier_param;
    }
    if (!std::isinf(info.ubx[i])) {
      pt.zu[i] = barrier_param;
    }
  }

  // Initialize the slack variables
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_inequalities; i++) {
    int index = info.inequality_indices[i];
    pt.s[i] = g[index];

    // Project slack into strict interior of [lbc, ubc]
    if (!std::isinf(info.lbc[i])) {
      pt.s[i] = A2D::max2(pt.s[i], info.lbc[i] + barrier_param);
    }
    if (!std::isinf(info.ubc[i])) {
      pt.s[i] = A2D::min2(pt.s[i], info.ubc[i] - barrier_param);
    }

    pt.zsl[i] = pt.zsu[i] = 0.0;

    if (!std::isinf(info.lbc[i])) {
      T gap = A2D::max2(pt.s[i] - info.lbc[i], barrier_param);
      pt.zsl[i] = barrier_param / gap;
    }
    if (!std::isinf(info.ubc[i])) {
      T gap = A2D::max2(info.ubc[i] - pt.s[i], barrier_param);
      pt.zsu[i] = barrier_param / gap;
    }
  }
}

template <typename T>
void add_residual(T barrier_param, const OptInfo<T>& info,
                  OptStateData<const T>& pt, const T* g, T* r) {
  // Compute the residual for the variables
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_variables; i++) {
    // Get the gradient component corresponding to this variable
    int index = info.design_variable_indices[i];

    // Extract the design variable value
    T x = pt.xlam[index];

    // Compute the right-hand-side
    T bx = -(g[index] - pt.zl[i] + pt.zu[i]);
    if (!std::isinf(info.lbx[i])) {
      T bzl = -((x - info.lbx[i]) * pt.zl[i] - barrier_param);
      bx += bzl / (x - info.lbx[i]);
    }
    if (!std::isinf(info.ubx[i])) {
      T bzu = -((info.ubx[i] - x) * pt.zu[i] - barrier_param);
      bx -= bzu / (info.ubx[i] - x);
    }

    // Set the right-hand-side
    r[index] = bx;
  }

  // Compute the contributions from the equality constraints
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_equalities; i++) {
    int index = info.equality_indices[i];

    // Set the right-hand-side for the equalities
    r[index] = -(g[index] - info.lbh[i]);
  }

  // Compute the contributions from the inequality constraints
  // Slack s has direct bounds: lbc <= s <= ubc
  // Complementarity: (s-lbc)*zsl = mu, (ubc-s)*zsu = mu
  // Mirrors variable bound handling exactly
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_inequalities; i++) {
    int index = info.inequality_indices[i];

    // Extract the multiplier from the solution vector
    T lam = pt.xlam[index];

    // Primal feasibility: -(c(x) - s)
    T bc = -(g[index] - pt.s[i]);
    // Slack stationarity: -(-lam - zsl + zsu)
    T blam = -(-lam - pt.zsl[i] + pt.zsu[i]);

    T Sigma_s = 0.0;
    T d = blam;
    if (!std::isinf(info.lbc[i])) {
      T gap = pt.s[i] - info.lbc[i];
      T bzsl = -(gap * pt.zsl[i] - barrier_param);
      d += bzsl / gap;
      Sigma_s += pt.zsl[i] / gap;
    }
    if (!std::isinf(info.ubc[i])) {
      T gap = info.ubc[i] - pt.s[i];
      T bzsu = -(gap * pt.zsu[i] - barrier_param);
      d -= bzsu / gap;
      Sigma_s += pt.zsu[i] / gap;
    }

    bc += d / Sigma_s;

    r[index] = bc;
  }
}

// Fills the condensed Newton RHS (identical to add_residual) AND accumulates
// the true dual/primal infeasibility squared norms needed by the quality
// function (Nocedal-Wachter-Waltz 2009, eq 4.2).
//
//   dual_infeas_sq   = ||g - zl + zu||^2   (full stationarity)
//   primal_infeas_sq = ||h - lbh||^2 + ||c - s||^2
//
// Using the condensed residual at mu=0 for these quantities is wrong:
// variable rows give -g[i] (bound multipliers cancelled out), not -(g-zl+zu).
template <typename T>
void compute_residual_and_infeasibility(T barrier_param,
                                        const OptInfo<T>& info,
                                        OptStateData<const T>& pt,
                                        const T* g, T* r,
                                        T& dual_infeas_sq,
                                        T& primal_infeas_sq) {
  dual_infeas_sq = 0.0;
  primal_infeas_sq = 0.0;

  // Variable rows
  for (int i = 0; i < info.num_variables; i++) {
    int index = info.design_variable_indices[i];
    T x = pt.xlam[index];

    // True stationarity (g - zl + zu) before bound-complementarity condensation
    T rd = g[index] - pt.zl[i] + pt.zu[i];
    dual_infeas_sq += rd * rd;

    // Condensed Newton RHS (identical to add_residual)
    T bx = -rd;
    if (!std::isinf(info.lbx[i])) {
      T bzl = -((x - info.lbx[i]) * pt.zl[i] - barrier_param);
      bx += bzl / (x - info.lbx[i]);
    }
    if (!std::isinf(info.ubx[i])) {
      T bzu = -((info.ubx[i] - x) * pt.zu[i] - barrier_param);
      bx -= bzu / (info.ubx[i] - x);
    }
    r[index] = bx;
  }

  // Equality rows
  for (int i = 0; i < info.num_equalities; i++) {
    int index = info.equality_indices[i];
    T rp = g[index] - info.lbh[i];
    primal_infeas_sq += rp * rp;
    r[index] = -rp;
  }

  // Inequality rows
  for (int i = 0; i < info.num_inequalities; i++) {
    int index = info.inequality_indices[i];
    T lam = pt.xlam[index];

    // True primal infeasibility c(x) - s
    T rp = g[index] - pt.s[i];
    primal_infeas_sq += rp * rp;

    // Condensed Newton RHS (identical to add_residual)
    T bc = -rp;
    T blam = -(-lam - pt.zsl[i] + pt.zsu[i]);
    T Sigma_s = 0.0;
    T d = blam;
    if (!std::isinf(info.lbc[i])) {
      T gap = pt.s[i] - info.lbc[i];
      T bzsl = -(gap * pt.zsl[i] - barrier_param);
      d += bzsl / gap;
      Sigma_s += pt.zsl[i] / gap;
    }
    if (!std::isinf(info.ubc[i])) {
      T gap = info.ubc[i] - pt.s[i];
      T bzsu = -(gap * pt.zsu[i] - barrier_param);
      d -= bzsu / gap;
      Sigma_s += pt.zsu[i] / gap;
    }
    bc += d / Sigma_s;
    r[index] = bc;
  }
}

template <typename T>
void compute_update(T barrier_param, const OptInfo<T>& info,
                    OptStateData<const T>& pt, OptStateData<T>& up) {
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_variables; i++) {
    // Get the gradient component corresponding to this variable
    int index = info.design_variable_indices[i];

    // Extract the design variable
    T x = pt.xlam[index];
    T px = up.xlam[index];

    // Compute the update step
    if (!std::isinf(info.lbx[i])) {
      T bzl = -((x - info.lbx[i]) * pt.zl[i] - barrier_param);
      up.zl[i] = (bzl - pt.zl[i] * px) / (x - info.lbx[i]);
    }
    if (!std::isinf(info.ubx[i])) {
      T bzu = -((info.ubx[i] - x) * pt.zu[i] - barrier_param);
      up.zu[i] = (bzu + pt.zu[i] * px) / (info.ubx[i] - x);
    }
  }

#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_inequalities; i++) {
    int index = info.inequality_indices[i];

    // Extract the multiplier from the solution vector
    T lam = pt.xlam[index];
    T plam = up.xlam[index];

    // Slack stationarity residual
    T blam = -(-lam - pt.zsl[i] + pt.zsu[i]);

    // Build Sigma_s and condensed RHS (mirrors add_residual)
    T Sigma_s = 0.0;
    T d = blam;

    if (!std::isinf(info.lbc[i])) {
      T gap = pt.s[i] - info.lbc[i];
      T bzsl = -(gap * pt.zsl[i] - barrier_param);
      d += bzsl / gap;
      Sigma_s += pt.zsl[i] / gap;
    }
    if (!std::isinf(info.ubc[i])) {
      T gap = info.ubc[i] - pt.s[i];
      T bzsu = -(gap * pt.zsu[i] - barrier_param);
      d -= bzsu / gap;
      Sigma_s += pt.zsu[i] / gap;
    }

    // Recover slack step from condensed system
    up.s[i] = (plam + d) / Sigma_s;

    // Back-substitute for slack bound duals (mirrors variable bound duals)
    if (!std::isinf(info.lbc[i])) {
      T gap = pt.s[i] - info.lbc[i];
      T bzsl = -(gap * pt.zsl[i] - barrier_param);
      up.zsl[i] = (bzsl - pt.zsl[i] * up.s[i]) / gap;
    }
    if (!std::isinf(info.ubc[i])) {
      T gap = info.ubc[i] - pt.s[i];
      T bzsu = -(gap * pt.zsu[i] - barrier_param);
      up.zsu[i] = (bzsu + pt.zsu[i] * up.s[i]) / gap;
    }
  }
}

template <typename T>
void compute_diagonal(const OptInfo<T>& info, OptStateData<const T>& pt,
                      T* diag) {
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_variables; i++) {
    // Get the gradient component corresponding to this variable
    int index = info.design_variable_indices[i];

    T x = pt.xlam[index];

    // If the lower bound isn't infinite, add its value
    if (!std::isinf(info.lbx[i])) {
      diag[index] += pt.zl[i] / (x - info.lbx[i]);
    }

    // If the upper bound isn't infinite, add its value
    if (!std::isinf(info.ubx[i])) {
      diag[index] += pt.zu[i] / (info.ubx[i] - x);
    }
  }

#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_inequalities; i++) {
    int index = info.inequality_indices[i];

    // Sigma_s = zsl/(s-lbc) + zsu/(ubc-s), mirrors variable bound diagonal
    T Sigma_s = 0.0;
    if (!std::isinf(info.lbc[i])) {
      Sigma_s += pt.zsl[i] / (pt.s[i] - info.lbc[i]);
    }
    if (!std::isinf(info.ubc[i])) {
      Sigma_s += pt.zsu[i] / (info.ubc[i] - pt.s[i]);
    }

    if (Sigma_s != 0.0) {
      diag[index] = -1.0 / Sigma_s;
    }
  }
}

template <typename T>
void compute_max_step(const T tau, const OptInfo<T>& info,
                      OptStateData<const T>& pt, OptStateData<const T>& up,
                      T& alpha_x_max, int& x_index, T& alpha_z_max,
                      int& z_index) {
  // Check for steps lengths for the design variables and slacks
  for (int i = 0; i < info.num_variables; i++) {
    // Get the gradient component corresponding to this variable
    int index = info.design_variable_indices[i];
    T x = pt.xlam[index];
    T px = up.xlam[index];

    if (!std::isinf(info.lbx[i])) {
      if (px < 0.0) {
        T numer = x - info.lbx[i];
        T alpha = -tau * numer / px;
        if (alpha < alpha_x_max) {
          alpha_x_max = alpha;
          x_index = index;
        }
      }
      if (up.zl[i] < 0.0) {
        T alpha = -tau * pt.zl[i] / up.zl[i];
        if (alpha < alpha_z_max) {
          alpha_z_max = alpha;
          z_index = index;
        }
      }
    }
    if (!std::isinf(info.ubx[i])) {
      if (px > 0.0) {
        T numer = info.ubx[i] - x;
        T alpha = tau * numer / px;
        if (alpha < alpha_x_max) {
          alpha_x_max = alpha;
          x_index = index;
        }
      }
      if (up.zu[i] < 0.0) {
        T alpha = -tau * pt.zu[i] / up.zu[i];
        if (alpha < alpha_z_max) {
          alpha_z_max = alpha;
          z_index = index;
        }
      }
    }
  }

  // Check step lengths for the slack variables and their duals
  for (int i = 0; i < info.num_inequalities; i++) {
    int index = info.inequality_indices[i];

    // Slack primal: s + alpha*ds must stay within [lbc, ubc]
    if (!std::isinf(info.lbc[i])) {
      if (up.s[i] < 0.0) {
        T alpha = -tau * (pt.s[i] - info.lbc[i]) / up.s[i];
        if (alpha < alpha_x_max) {
          alpha_x_max = alpha;
          x_index = index;
        }
      }
      // Dual: zsl must stay positive
      if (up.zsl[i] < 0.0) {
        T alpha = -tau * pt.zsl[i] / up.zsl[i];
        if (alpha < alpha_z_max) {
          alpha_z_max = alpha;
          z_index = index;
        }
      }
    }

    if (!std::isinf(info.ubc[i])) {
      if (up.s[i] > 0.0) {
        T alpha = tau * (info.ubc[i] - pt.s[i]) / up.s[i];
        if (alpha < alpha_x_max) {
          alpha_x_max = alpha;
          x_index = index;
        }
      }
      // Dual: zsu must stay positive
      if (up.zsu[i] < 0.0) {
        T alpha = -tau * pt.zsu[i] / up.zsu[i];
        if (alpha < alpha_z_max) {
          alpha_z_max = alpha;
          z_index = index;
        }
      }
    }
  }
}

template <typename T>
void apply_step_update(const T alpha_x, const T alpha_z, const OptInfo<T>& info,
                       OptStateData<const T>& pt, OptStateData<const T>& up,
                       OptStateData<T>& tmp) {
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_variables; i++) {
    // Update the dual variables
    if (!std::isinf(info.lbx[i])) {
      tmp.zl[i] = pt.zl[i] + alpha_z * up.zl[i];
    }
    if (!std::isinf(info.ubx[i])) {
      tmp.zu[i] = pt.zu[i] + alpha_z * up.zu[i];
    }
  }

  // Update the slack variables and their duals
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_inequalities; i++) {
    tmp.s[i] = pt.s[i] + alpha_x * up.s[i];
    if (!std::isinf(info.lbc[i])) {
      tmp.zsl[i] = pt.zsl[i] + alpha_z * up.zsl[i];
    }
    if (!std::isinf(info.ubc[i])) {
      tmp.zsu[i] = pt.zsu[i] + alpha_z * up.zsu[i];
    }
  }
}

template <typename T>
void compute_complementarity_pairs(const OptInfo<T>& info,
                                   OptStateData<const T>& pt, T partial_sum[],
                                   T& local_min) {
  for (int i = 0; i < info.num_variables; i++) {
    // Extract the design variable value
    int index = info.design_variable_indices[i];
    T x = pt.xlam[index];

    if (!std::isinf(info.lbx[i])) {
      T comp = (x - info.lbx[i]) * pt.zl[i];
      partial_sum[0] += comp;
      partial_sum[1] += 1.0;
      local_min = A2D::min2(local_min, comp);
    }
    if (!std::isinf(info.ubx[i])) {
      T comp = (info.ubx[i] - x) * pt.zu[i];
      partial_sum[0] += comp;
      partial_sum[1] += 1.0;
      local_min = A2D::min2(local_min, comp);
    }
  }

  for (int i = 0; i < info.num_inequalities; i++) {
    if (!std::isinf(info.lbc[i])) {
      T comp = (pt.s[i] - info.lbc[i]) * pt.zsl[i];
      partial_sum[0] += comp;
      partial_sum[1] += 1.0;
      local_min = A2D::min2(local_min, comp);
    }

    if (!std::isinf(info.ubc[i])) {
      T comp = (info.ubc[i] - pt.s[i]) * pt.zsu[i];
      partial_sum[0] += comp;
      partial_sum[1] += 1.0;
      local_min = A2D::min2(local_min, comp);
    }
  }
}

template <typename T>
void compute_kkt_error_components(const OptInfo<T>& info,
                                   OptStateData<const T>& pt, const T* g,
                                   T& dual_infeas_sq, T& primal_infeas_sq,
                                   T& comp_sq) {
  dual_infeas_sq = 0.0;
  primal_infeas_sq = 0.0;
  comp_sq = 0.0;

  // Design variables: dual feasibility and complementarity
  for (int i = 0; i < info.num_variables; i++) {
    int index = info.design_variable_indices[i];
    T x = pt.xlam[index];

    // Dual feasibility: stationarity of Lagrangian wrt x
    // g[index] = grad_f + J_h^T lam_h + J_c^T lam_c (computed by problem)
    // KKT: g[index] - zl + zu = 0
    T rd = g[index] - pt.zl[i] + pt.zu[i];
    dual_infeas_sq += rd * rd;

    // Complementarity: (x - lb) * zl, (ub - x) * zu
    if (!std::isinf(info.lbx[i])) {
      T c = (x - info.lbx[i]) * pt.zl[i];
      comp_sq += c * c;
    }
    if (!std::isinf(info.ubx[i])) {
      T c = (info.ubx[i] - x) * pt.zu[i];
      comp_sq += c * c;
    }
  }

  // Equality constraints: primal feasibility h(x) - lbh = 0
  for (int i = 0; i < info.num_equalities; i++) {
    int index = info.equality_indices[i];
    T rp = g[index] - info.lbh[i];
    primal_infeas_sq += rp * rp;
  }

  // Inequality constraints: primal feasibility c(x) = s
  // Plus complementarity from slack decomposition
  for (int i = 0; i < info.num_inequalities; i++) {
    int index = info.inequality_indices[i];

    // Primal feasibility: c(x) - s = 0
    T rp = g[index] - pt.s[i];
    primal_infeas_sq += rp * rp;

    // Complementarity from slack bounds
    if (!std::isinf(info.lbc[i])) {
      T c = (pt.s[i] - info.lbc[i]) * pt.zsl[i];
      comp_sq += c * c;
    }
    if (!std::isinf(info.ubc[i])) {
      T c = (info.ubc[i] - pt.s[i]) * pt.zsu[i];
      comp_sq += c * c;
    }
  }
}

// Compute ||ZXe||^2 = sum of squared complementarity products at point pt.
// Equivalent to the comp_sq term in compute_kkt_error_components but requires
// no gradient vector. Used by the linear quality function (NWW 2009 eq 4.2).
template <typename T>
T compute_complementarity_sq(const OptInfo<T>& info,
                             OptStateData<const T>& pt) {
  T comp_sq = 0.0;

  for (int i = 0; i < info.num_variables; i++) {
    int index = info.design_variable_indices[i];
    T x = pt.xlam[index];
    if (!std::isinf(info.lbx[i])) {
      T c = (x - info.lbx[i]) * pt.zl[i];
      comp_sq += c * c;
    }
    if (!std::isinf(info.ubx[i])) {
      T c = (info.ubx[i] - x) * pt.zu[i];
      comp_sq += c * c;
    }
  }

  for (int i = 0; i < info.num_inequalities; i++) {
    if (!std::isinf(info.lbc[i])) {
      T c = (pt.s[i] - info.lbc[i]) * pt.zsl[i];
      comp_sq += c * c;
    }
    if (!std::isinf(info.ubc[i])) {
      T c = (info.ubc[i] - pt.s[i]) * pt.zsu[i];
      comp_sq += c * c;
    }
  }

  return comp_sq;
}

// Compute the barrier log-sum: -mu * sum(ln(barrier_variables)).
// This is the barrier contribution to the IPOPT barrier objective:
//   phi_mu(x) = f(x) + barrier_log_sum
// Barrier variables are: (x-lb), (ub-x) for bounded design variables,
// and (s-lbc), (ubc-s) for bounded inequality slacks.
template <typename T>
T compute_barrier_log_sum(T barrier_param, const OptInfo<T>& info,
                          OptStateData<const T>& pt) {
  T log_sum = 0.0;

  for (int i = 0; i < info.num_variables; i++) {
    int index = info.design_variable_indices[i];
    T x = pt.xlam[index];
    if (!std::isinf(info.lbx[i])) {
      T slack = x - info.lbx[i];
      if (slack > 0) {
        log_sum += std::log(slack);
      } else {
        return 1e20;  // infeasible point
      }
    }
    if (!std::isinf(info.ubx[i])) {
      T slack = info.ubx[i] - x;
      if (slack > 0) {
        log_sum += std::log(slack);
      } else {
        return 1e20;
      }
    }
  }

  for (int i = 0; i < info.num_inequalities; i++) {
    if (!std::isinf(info.lbc[i])) {
      T gap = pt.s[i] - info.lbc[i];
      if (gap > 0) {
        log_sum += std::log(gap);
      } else {
        return 1e20;
      }
    }
    if (!std::isinf(info.ubc[i])) {
      T gap = info.ubc[i] - pt.s[i];
      if (gap > 0) {
        log_sum += std::log(gap);
      } else {
        return 1e20;
      }
    }
  }

  return -barrier_param * log_sum;
}

// Compute the analytical directional derivative of the barrier objective
// along the Newton search direction:
//   dphi = nabla_x phi_mu^T dx + nabla_s phi_mu^T ds
//
// where phi_mu = f(x) - mu * sum(ln(barrier_vars)).
//
// Uses the identity: for design variable i,
//   g[i] + (-mu/(x-l) + mu/(u-x)) = -r[i]   (condensed residual)
// so the design contribution is -r_design^T dx - lambda^T J dx.
// The lambda^T J dx term is extracted from the KKT constraint rows:
//   J dx = r_constr - D_eff * d_lambda
// where D_eff = diag[constr] (the constraint diagonal used in factorization).
//
// The slack contribution uses the barrier gradient w.r.t. inequality slacks:
//   nabla_s phi_mu = -mu/(s-lbc) + mu/(ubc-s)
template <typename T>
T compute_barrier_dphi(T barrier_param, const OptInfo<T>& info,
                       OptStateData<const T>& pt,
                       OptStateData<const T>& up,
                       const T* r, const T* p, const T* diag) {
  T dphi = 0.0;

  // Term 1: -res_design^T dx
  // This equals (g + barrier_x_grad)^T dx = (nabla_x L + barrier_x)^T dx
  for (int i = 0; i < info.num_variables; i++) {
    int index = info.design_variable_indices[i];
    dphi -= r[index] * p[index];
  }

  // Term 2: -lambda^T J dx
  // From KKT constraint rows: J dx = r_constr - diag_constr * d_lambda
  // so lambda^T J dx = sum_j lambda_j * (r_j - diag_j * p_j)
  for (int i = 0; i < info.num_equalities; i++) {
    int index = info.equality_indices[i];
    T lam = pt.xlam[index];
    T jdx_j = r[index] - diag[index] * p[index];
    dphi -= lam * jdx_j;
  }
  for (int i = 0; i < info.num_inequalities; i++) {
    int index = info.inequality_indices[i];
    T lam = pt.xlam[index];
    T jdx_j = r[index] - diag[index] * p[index];
    dphi -= lam * jdx_j;
  }

  // Term 3: barrier_s_grad^T ds (slack barrier contribution)
  // nabla_s phi_mu = -mu/(s-lbc) + mu/(ubc-s)
  for (int i = 0; i < info.num_inequalities; i++) {
    T ds = up.s[i];
    T barrier_s = 0.0;
    if (!std::isinf(info.lbc[i])) {
      T gap = pt.s[i] - info.lbc[i];
      if (gap > 0) barrier_s -= barrier_param / gap;
    }
    if (!std::isinf(info.ubc[i])) {
      T gap = info.ubc[i] - pt.s[i];
      if (gap > 0) barrier_s += barrier_param / gap;
    }
    dphi += barrier_s * ds;
  }

  return dphi;
}

// IPOPT Eq. 16: Reset bound multipliers to keep them within
// [mu / (kappa_sigma * gap), kappa_sigma * mu / gap].
// This prevents bound multipliers from diverging and keeps
// sigma_i = z_i * gap_i bounded in [mu/kappa_sigma, kappa_sigma * mu].
template <typename T>
void reset_bound_multipliers(T barrier_param, T kappa_sigma,
                             const OptInfo<T>& info,
                             OptStateData<T>& pt) {
  for (int i = 0; i < info.num_variables; i++) {
    int index = info.design_variable_indices[i];
    T x = pt.xlam[index];

    if (!std::isinf(info.lbx[i])) {
      T gap = x - info.lbx[i];
      if (gap > 0) {
        T z_max = kappa_sigma * barrier_param / gap;
        T z_min = barrier_param / (kappa_sigma * gap);
        pt.zl[i] = std::max(std::min(pt.zl[i], z_max), z_min);
      }
    }
    if (!std::isinf(info.ubx[i])) {
      T gap = info.ubx[i] - x;
      if (gap > 0) {
        T z_max = kappa_sigma * barrier_param / gap;
        T z_min = barrier_param / (kappa_sigma * gap);
        pt.zu[i] = std::max(std::min(pt.zu[i], z_max), z_min);
      }
    }
  }

  for (int i = 0; i < info.num_inequalities; i++) {
    if (!std::isinf(info.lbc[i])) {
      T gap = pt.s[i] - info.lbc[i];
      if (gap > 0) {
        T z_max = kappa_sigma * barrier_param / gap;
        T z_min = barrier_param / (kappa_sigma * gap);
        pt.zsl[i] = std::max(std::min(pt.zsl[i], z_max), z_min);
      }
    }
    if (!std::isinf(info.ubc[i])) {
      T gap = info.ubc[i] - pt.s[i];
      if (gap > 0) {
        T z_max = kappa_sigma * barrier_param / gap;
        T z_min = barrier_param / (kappa_sigma * gap);
        pt.zsu[i] = std::max(std::min(pt.zsu[i], z_max), z_min);
      }
    }
  }
}

template <typename T>
void compute_affine_start_point(T beta_min, const OptInfo<T>& info,
                                OptStateData<const T>& pt,
                                OptStateData<const T>& up,
                                OptStateData<T>& tmp) {
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_variables; i++) {
    tmp.zl[i] = A2D::max2(beta_min, A2D::fabs(pt.zl[i] + up.zl[i]));
    tmp.zu[i] = A2D::max2(beta_min, A2D::fabs(pt.zu[i] + up.zu[i]));
  }

#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_inequalities; i++) {
    if (!std::isinf(info.lbc[i])) {
      tmp.zsl[i] = A2D::max2(beta_min, A2D::fabs(pt.zsl[i] + up.zsl[i]));
    }
    if (!std::isinf(info.ubc[i])) {
      tmp.zsu[i] = A2D::max2(beta_min, A2D::fabs(pt.zsu[i] + up.zsu[i]));
    }
  }
}

// Template delarations
#ifdef AMIGO_USE_CUDA
template <typename T>
void set_multipliers_value_cuda(const OptInfo<T>& info, const T value, T* d_x,
                                cudaStream_t stream = 0);

template <typename T>
void set_design_vars_value_cuda(const OptInfo<T>& info, const T value, T* d_x,
                                cudaStream_t stream = 0);

template <typename T>
void copy_multipliers_cuda(const OptInfo<T>& info, const T* d_src, T* d_dest,
                           cudaStream_t stream = 0);

template <typename T>
void copy_design_vars_cuda(const OptInfo<T>& info, const T* d_src, T* d_dest,
                           cudaStream_t stream = 0);

template <typename T>
void initialize_multipliers_and_slacks_cuda(T barrier_param,
                                            const OptInfo<T>& info,
                                            const T* d_g, OptStateData<T>& pt,
                                            cudaStream_t stream = 0);

template <typename T>
void add_residual_cuda(T barrier_param, const OptInfo<T>& info,
                       OptStateData<const T>& pt, const T* g, T* r,
                       cudaStream_t stream = 0);

template <typename T>
void compute_update_cuda(T barrier_param, const OptInfo<T>& info,
                         OptStateData<const T>& pt, OptStateData<T>& up,
                         cudaStream_t stream = 0);

template <typename T>
void compute_diagonal_cuda(const OptInfo<T>& info, OptStateData<const T>& pt,
                           T* diag, cudaStream_t stream = 0);

template <typename T>
void compute_max_step_cuda(const T tau, const OptInfo<T>& info,
                           OptStateData<const T>& pt, OptStateData<const T>& up,
                           T& alpha_x_max, int& x_index, T& alpha_z_max,
                           int& z_index, cudaStream_t stream = 0);

template <typename T>
void apply_step_update_cuda(const T alpha_x, const T alpha_z,
                            const OptInfo<T>& info, OptStateData<const T>& pt,
                            OptStateData<const T>& up, OptStateData<T>& tmp,
                            cudaStream_t stream = 0);

template <typename T>
void compute_affine_start_point_cuda(T beta_min, const OptInfo<T>& info,
                                     OptStateData<const T>& pt,
                                     OptStateData<const T>& up,
                                     OptStateData<T>& tmp,
                                     cudaStream_t stream = 0);

template <typename T>
void compute_complementarity_pairs_cuda(const OptInfo<T>& info,
                                        const OptStateData<const T>& pt,
                                        T partial_sum[], T& local_min);
#endif

}  // namespace detail

}  // namespace amigo

#endif  // AMIGO_OPTIMIZER_BACKEND_H