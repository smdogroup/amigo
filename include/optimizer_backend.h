#ifndef AMIGO_OPTIMIZER_BACKEND_H
#define AMIGO_OPTIMIZER_BACKEND_H

#include "amigo.h"

namespace amigo {

namespace detail {

template <typename T>
class OptInfo {
 public:
  int num_variables, num_equalities, num_inequalities;

  const int* design_variable_indices;
  const int* equality_indices;
  const int* inequality_indices;

  // bounds (can be +/-inf)
  const T *lbx, *ubx;  // x bounds for variables (size num_variables)
  const T *lbc, *ubc;  // constraint bounds (size num_inequalities)
  const T* lbh;        // constraint bounds (size num_equalities)
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
void initialize_multipliers_and_slacks(T barrier_param, const T* g,
                                       OptStateData<T>& pt) {
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
      pzl[i] = (bzl - pt.zl[i] * px) / (x - info.lbx[i]);
    }
    if (!std::isinf(ubx[i])) {
      T bzu = -((info.ubx[i] - x) * zu[i] - barrier_param);
      pzu[i] = (bzu + pt.zu[i] * px) / (info.ubx[i] - x);
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
    int index = design_variable_indices[i];

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

}  // namespace detail

}  // namespace amigo

#endif  // AMIGO_OPTIMIZER_BACKEND_H