#ifndef AMIGO_OPTIMIZER_CUDA_BACKEND_H
#define AMIGO_OPTIMIZER_CUDA_BACKEND_H

#include <limits>

#include "amigo.h"
#include "optimizer_backend.h"

// #ifdef AMIGO_USE_CUDA

namespace amgio {

namespace detail {

template <typename T>
AMIGO_KERNEL void initialize_multipliers_kernel(T barrier_param,
                                                OptInfo<T> info,
                                                OptStateData<T> pt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= info.num_variables) {
    return;
  }

  const T inf = std::numeric_limits<T>::infinity();

  pt.zl[i] = T(0);
  pt.zu[i] = T(0);

  if (info.lbx[i] != -inf) {
    pt.zl[i] = barrier_param;
  }
  if (info.ubx[i] != inf) {
    pt.zu[i] = barrier_param;
  }
}

template <typename T>
AMIGO_KERNEL void initialize_slacks_kernel(T barrier_param, OptInfo<T> info,
                                           const T* __restrict__ g,
                                           OptStateData<T> pt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= info.num_inequalities) {
    return;
  }

  const T inf = std::numeric_limits<T>::infinity();

  int index = info.inequality_indices[i];

  pt.s[i] = -g[index];

  pt.sl[i] = T(0);
  pt.tl[i] = T(0);
  pt.zsl[i] = T(0);
  pt.ztl[i] = T(0);

  pt.su[i] = T(0);
  pt.tu[i] = T(0);
  pt.zsu[i] = T(0);
  pt.ztu[i] = T(0);

  if (info.lbc[i] != -inf) {
    pt.sl[i] = barrier_param;
    pt.tl[i] = barrier_param;
    pt.zsl[i] = barrier_param;
    pt.ztl[i] = barrier_param;
  }

  if (info.ubc[i] != inf) {
    pt.su[i] = barrier_param;
    pt.tu[i] = barrier_param;
    pt.zsu[i] = barrier_param;
    pt.ztu[i] = barrier_param;
  }
}

template <typename T>
void initialize_multipliers_and_slacks_cuda(T barrier_param,
                                            const OptInfo<T>& info,
                                            const T* d_g, OptStateData<T>& pt,
                                            cudaStream_t stream = 0) {
  constexpr int TPB = 256;

  // Variables
  int grid_vars = (dinfo.num_variables + TPB - 1) / TPB;
  initialize_multipliers_kernel<<<grid_vars, TPB, 0, stream>>>(barrier_param,
                                                               info, pt);

  // Inequalities
  int grid_ineq = (dinfo.num_inequalities + TPB - 1) / TPB;
  initialize_slacks_kernel<<<grid_ineq, TPB, 0, stream>>>(barrier_param, info,
                                                          d_g, pt);
}

template <class T>
AMIGO_KERNEL void add_residual_vars_kernel(T barrier_param,
                                           const OptInfo<T>& info,
                                           OptStateData<const T>& pt,
                                           const T* g, T* r) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= info.num_variables) {
    return;
  }

  const T inf = std::numeric_limits<T>::infinity();

  // Get the gradient component corresponding to this variable
  int index = info.design_variable_indices[i];

  // Extract the design variable value
  T x = pt.xlam[index];

  // Compute the right-hand-side
  T bx = -(g[index] - pt.zl[i] + pt.zu[i]);
  if (info.lbx[i] != -inf) {
    T bzl = -((x - info.lbx[i]) * pt.zl[i] - barrier_param);
    bx += bzl / (x - info.lbx[i]);
  }
  if (info.ubx[i] != inf) {
    T bzu = -((info.ubx[i] - x) * pt.zu[i] - barrier_param);
    bx -= bzu / (info.ubx[i] - x);
  }

  // Set the right-hand-side
  r[index] = bx;
}

template <class T>
AMIGO_KERNEL void add_residual_eq_kernel(const OptInfo<T>& info,
                                         OptStateData<const T>& pt, const T* g,
                                         T* r) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= info.num_equalities) {
    return;
  }

  int index = info.equality_indices[i];

  // Set the right-hand-side for the equalities
  r[index] = -(g[index] - info.lbh[i]);
}

template <class T>
AMIGO_KERNEL void add_residual_ineq_kernel(T barrier_param,
                                           const OptInfo<T>& info,
                                           OptStateData<const T>& pt,
                                           const T* g, T* r) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= info.num_inequalities) {
    return;
  }

  const T inf = std::numeric_limits<T>::infinity();

  int index = info.inequality_indices[i];

  // Extract the multiplier from the solution vector
  T lam = pt.xlam[index];

  // Set the right-hand-side values
  T bc = -(g[index] - pt.s[i]);
  T blam = -(-lam - pt.zsl[i] + pt.zsu[i]);

  // Build the components of C and compute its inverse
  T C = 0.0;
  T d = blam;
  if (info.lbc[i] != -inf) {
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

  if (info.ubc[i] != inf) {
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

template <class T>
void add_residual_cuda(T barrier_param, const OptInfo<T>& info,
                       OptStateData<const T>& pt, const T* g, T* r,
                       cudaStream_t stream = 0) {
  constexpr int TPB = 256;
  int gv = (A.num_variables + TPB - 1) / TPB;
  int ge = (A.num_equalities + TPB - 1) / TPB;
  int gi = (A.num_inequalities + TPB - 1) / TPB;
  add_residual_vars_kernel<T>
      <<<gv, TPB, 0, stream>>>(barrier_param, info, pt, g, r);
  add_residual_eq_kernel<T><<<ge, TPB, 0, stream>>>(info, pt, g, r);
  add_residual_ineq_kernel<T>
      <<<gi, TPB, 0, stream>>>(barrier_param, info, pt, g, r);
}

}  // namespace detail

}  // namespace amgio

#endif  // AMIGO_USE_CUDA

#endif  // AMIGO_OPTIMIZER_CUDA_BACKEND_H