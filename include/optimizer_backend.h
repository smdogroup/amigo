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
    vars->template get_slacks<policy>(&self.s, &self.sl, &self.tl, &self.su,
                                      &self.tu);
    vars->template get_slack_duals<policy>(&self.zsl, &self.ztl, &self.zsu,
                                           &self.ztu);
    self.xlam = vars->template get_solution_array<policy>();

    return self;
  }

  // primal/dual/slacks
  T* xlam = nullptr;  // full solution vector (design vars + multipliers)
  T *zl = nullptr, *zu = nullptr;  // variable duals (size num_variables)

  T* s = nullptr;  // slack for inequalities (size num_inequalities)
  T *sl = nullptr, *tl = nullptr;
  T *su = nullptr, *tu = nullptr;  // lower/upper slack splits
  T *zsl = nullptr, *ztl = nullptr;
  T *zsu = nullptr, *ztu = nullptr;  // multipliers for slacks
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
    pt.s[i] = -g[index];

    pt.sl[i] = pt.tl[i] = pt.zsl[i] = pt.ztl[i] = 0.0;
    pt.su[i] = pt.tu[i] = pt.zsu[i] = pt.ztu[i] = 0.0;

    if (!std::isinf(info.lbc[i])) {
      pt.sl[i] = barrier_param;
      pt.tl[i] = barrier_param;
      pt.zsl[i] = barrier_param;
      pt.ztl[i] = barrier_param;
    }
    if (!std::isinf(info.ubc[i])) {
      pt.su[i] = barrier_param;
      pt.tu[i] = barrier_param;
      pt.zsu[i] = barrier_param;
      pt.ztu[i] = barrier_param;
    }
  }
}

template <typename T>
void add_residual(T barrier_param, T gamma, const OptInfo<T>& info,
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
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_inequalities; i++) {
    int index = info.inequality_indices[i];

    // Extract the multiplier from the solution vector
    T lam = pt.xlam[index];

    // Set the right-hand-side values
    T bc = -(g[index] - pt.s[i]);
    T blam = -(-lam - pt.zsl[i] + pt.zsu[i]);

    // Build the components of C and compute its inverse
    T C = 0.0;
    T d = blam;
    if (!std::isinf(info.lbc[i])) {
      // Compute the right-hand-sides for the lower bound
      T blaml = -(gamma - pt.zsl[i] - pt.ztl[i]);
      T bsl = -(pt.s[i] - info.lbc[i] - pt.sl[i] + pt.tl[i]);
      T bzsl = -(pt.sl[i] * pt.zsl[i] - barrier_param);
      T bztl = -(pt.tl[i] * pt.ztl[i] - barrier_param);

      T inv_zsl = 1.0 / pt.zsl[i];
      T inv_ztl = 1.0 / pt.ztl[i];
      T Fl = inv_zsl * pt.sl[i] + inv_ztl * pt.tl[i];
      T dl = bsl + inv_zsl * bzsl - inv_ztl * (bztl + pt.tl[i] * blaml);

      T inv_Fl = 1.0 / Fl;
      d += inv_Fl * dl;
      C += inv_Fl;
    }

    if (!std::isinf(info.ubc[i])) {
      T blamu = -(gamma - pt.zsu[i] - pt.ztu[i]);
      T bsu = -(info.ubc[i] - pt.s[i] - pt.su[i] + pt.tu[i]);
      T bzsu = -(pt.su[i] * pt.zsu[i] - barrier_param);
      T bztu = -(pt.tu[i] * pt.ztu[i] - barrier_param);

      T inv_zsu = 1.0 / pt.zsu[i];
      T inv_ztu = 1.0 / pt.ztu[i];
      T Fu = inv_zsu * pt.su[i] + inv_ztu * pt.tu[i];
      T du = bsu + inv_zsu * bzsu - inv_ztu * (bztu + pt.tu[i] * blamu);

      T inv_Fu = 1.0 / Fu;
      d -= inv_Fu * du;
      C += inv_Fu;
    }

    bc += d / C;

    r[index] = bc;
  }
}

template <typename T>
void compute_update(T barrier_param, T gamma, const OptInfo<T>& info,
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

    // Compute all the contributions to the update
    T blam = -(-lam - pt.zsl[i] + pt.zsu[i]);

    // Build the components of C and compute its inverse
    T C = 0.0;
    T d = blam;
    T Fl = 0.0, dl = 0.0, blaml = 0.0, bsl = 0.0, bzsl = 0.0, bztl = 0.0;
    T Fu = 0.0, du = 0.0, blamu = 0.0, bsu = 0.0, bzsu = 0.0, bztu = 0.0;

    if (!std::isinf(info.lbc[i])) {
      // Compute the right-hand-sides for the lower bound
      blaml = -(gamma - pt.zsl[i] - pt.ztl[i]);
      bsl = -(pt.s[i] - info.lbc[i] - pt.sl[i] + pt.tl[i]);
      bzsl = -(pt.sl[i] * pt.zsl[i] - barrier_param);
      bztl = -(pt.tl[i] * pt.ztl[i] - barrier_param);

      T inv_zsl = 1.0 / pt.zsl[i];
      T inv_ztl = 1.0 / pt.ztl[i];
      Fl = inv_zsl * pt.sl[i] + inv_ztl * pt.tl[i];
      dl = bsl + inv_zsl * bzsl - inv_ztl * (bztl + pt.tl[i] * blaml);

      T inv_Fl = 1.0 / Fl;
      d += inv_Fl * dl;
      C += inv_Fl;
    }

    if (!std::isinf(info.ubc[i])) {
      blamu = -(gamma - pt.zsu[i] - pt.ztu[i]);
      bsu = -(info.ubc[i] - pt.s[i] - pt.su[i] + pt.tu[i]);
      bzsu = -(pt.su[i] * pt.zsu[i] - barrier_param);
      bztu = -(pt.tu[i] * pt.ztu[i] - barrier_param);

      T inv_zsu = 1.0 / pt.zsu[i];
      T inv_ztu = 1.0 / pt.ztu[i];
      Fu = inv_zsu * pt.su[i] + inv_ztu * pt.tu[i];
      du = bsu + inv_zsu * bzsu - inv_ztu * (bztu + pt.tu[i] * blamu);

      T inv_Fu = 1.0 / Fu;
      d -= inv_Fu * du;
      C += inv_Fu;
    }

    up.s[i] = (plam + d) / C;

    if (!std::isinf(info.lbc[i])) {
      up.zsl[i] = (-up.s[i] + dl) / Fl;
      up.ztl[i] = -blaml - up.zsl[i];
      up.sl[i] = (bzsl - pt.sl[i] * up.zsl[i]) / pt.zsl[i];
      up.tl[i] = (bztl - pt.tl[i] * up.ztl[i]) / pt.ztl[i];
    }
    if (!std::isinf(info.ubc[i])) {
      up.zsu[i] = (up.s[i] + du) / Fu;
      up.ztu[i] = -blamu - up.zsu[i];
      up.su[i] = (bzsu - pt.su[i] * up.zsu[i]) / pt.zsu[i];
      up.tu[i] = (bztu - pt.tu[i] * up.ztu[i]) / pt.ztu[i];
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

    // Build the components of C and compute its inverse
    T C = 0.0;
    if (!std::isinf(info.lbc[i])) {
      T Fl = pt.sl[i] / pt.zsl[i] + pt.tl[i] / pt.ztl[i];
      C += 1.0 / Fl;
    }
    if (!std::isinf(info.ubc[i])) {
      T Fu = pt.su[i] / pt.zsu[i] + pt.tu[i] / pt.ztu[i];
      C += 1.0 / Fu;
    }

    if (C != 0.0) {
      diag[index] = -1.0 / C;
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

  // Check step lengths for the multipliers
  for (int i = 0; i < info.num_inequalities; i++) {
    int index = info.inequality_indices[i];

    if (!std::isinf(info.lbc[i])) {
      // Slack variables
      if (up.sl[i] < 0.0) {
        T alpha = -tau * pt.sl[i] / up.sl[i];
        if (alpha < alpha_x_max) {
          alpha_x_max = alpha;
          x_index = index;
        }
      }
      if (up.tl[i] < 0.0) {
        T alpha = -tau * pt.tl[i] / up.tl[i];
        if (alpha < alpha_x_max) {
          alpha_x_max = alpha;
          x_index = index;
        }
      }

      // Dual variables
      if (up.zsl[i] < 0.0) {
        T alpha = -tau * pt.zsl[i] / up.zsl[i];
        if (alpha < alpha_z_max) {
          alpha_z_max = alpha;
          z_index = index;
        }
      }
      if (up.ztl[i] < 0.0) {
        T alpha = -tau * pt.ztl[i] / up.ztl[i];
        if (alpha < alpha_z_max) {
          alpha_z_max = alpha;
          z_index = index;
        }
      }
    }

    if (!std::isinf(info.ubc[i])) {
      // Slack variables
      if (up.su[i] < 0.0) {
        T alpha = -tau * pt.su[i] / up.su[i];
        if (alpha < alpha_x_max) {
          alpha_x_max = alpha;
          x_index = index;
        }
      }
      if (up.tu[i] < 0.0) {
        T alpha = -tau * pt.tu[i] / up.tu[i];
        if (alpha < alpha_x_max) {
          alpha_x_max = alpha;
          x_index = index;
        }
      }

      // Dual variables
      if (up.zsu[i] < 0.0) {
        T alpha = -tau * pt.zsu[i] / up.zsu[i];
        if (alpha < alpha_z_max) {
          alpha_z_max = alpha;
          z_index = index;
        }
      }
      if (up.ztu[i] < 0.0) {
        T alpha = -tau * pt.ztu[i] / up.ztu[i];
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

  // Update the slack variables and remaining dual variables
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for
#endif  // AMIGO_USE_OPENMP
  for (int i = 0; i < info.num_inequalities; i++) {
    tmp.s[i] = pt.s[i] + alpha_x * up.s[i];
    if (!std::isinf(info.lbc[i])) {
      tmp.sl[i] = pt.sl[i] + alpha_x * up.sl[i];
      tmp.tl[i] = pt.tl[i] + alpha_x * up.tl[i];
      tmp.zsl[i] = pt.zsl[i] + alpha_z * up.zsl[i];
      tmp.ztl[i] = pt.ztl[i] + alpha_z * up.ztl[i];
    }
    if (!std::isinf(info.ubc[i])) {
      tmp.su[i] = pt.su[i] + alpha_x * up.su[i];
      tmp.tu[i] = pt.tu[i] + alpha_x * up.tu[i];
      tmp.zsu[i] = pt.zsu[i] + alpha_z * up.zsu[i];
      tmp.ztu[i] = pt.ztu[i] + alpha_z * up.ztu[i];
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
      tmp.sl[i] = A2D::max2(beta_min, A2D::fabs(pt.sl[i] + up.sl[i]));
      tmp.tl[i] = A2D::max2(beta_min, A2D::fabs(pt.tl[i] + up.tl[i]));
      tmp.zsl[i] = A2D::max2(beta_min, A2D::fabs(pt.zsl[i] + up.zsl[i]));
      tmp.ztl[i] = A2D::max2(beta_min, A2D::fabs(pt.ztl[i] + up.ztl[i]));
    }

    if (!std::isinf(info.ubc[i])) {
      tmp.su[i] = A2D::max2(beta_min, A2D::fabs(pt.su[i] + up.su[i]));
      tmp.tu[i] = A2D::max2(beta_min, A2D::fabs(pt.tu[i] + up.tu[i]));
      tmp.zsu[i] = A2D::max2(beta_min, A2D::fabs(pt.zsu[i] + up.zsu[i]));
      tmp.ztu[i] = A2D::max2(beta_min, A2D::fabs(pt.ztu[i] + up.ztu[i]));
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
void add_residual_cuda(T barrier_param, T gamma, const OptInfo<T>& info,
                       OptStateData<const T>& pt, const T* g, T* r,
                       cudaStream_t stream = 0);

template <typename T>
void compute_update_cuda(T barrier_param, T gamma, const OptInfo<T>& info,
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
#endif

}  // namespace detail

}  // namespace amigo

#endif  // AMIGO_OPTIMIZER_BACKEND_H