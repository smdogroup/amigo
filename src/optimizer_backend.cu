#ifndef AMIGO_OPTIMIZER_CUDA_BACKEND_H
#define AMIGO_OPTIMIZER_CUDA_BACKEND_H

#include <cmath>

#include "a2dcore.h"
#include "amigo.h"
#include "optimizer_backend.h"

namespace amigo {

namespace detail {

template <typename T>
AMIGO_KERNEL void set_array_value(int num_variables, const int* indices,
                                  T value, T* array) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_variables) {
    return;
  }

  int idx = indices[i];
  array[idx] = value;
}

template <typename T>
AMIGO_KERNEL void copy_array_values(int num_variables, const int* indices,
                                    const T* d_src, T* d_dest) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_variables) {
    return;
  }

  int idx = indices[i];
  d_dest[idx] = d_src[idx];
}

template <typename T>
void set_multipliers_value_cuda(const OptInfo<T>& info, const T value, T* d_x,
                                cudaStream_t stream) {
  constexpr int TPB = 256;

  int grid_eq = (info.num_equalities + TPB - 1) / TPB;
  set_array_value<T><<<grid_eq, TPB, 0, stream>>>(
      info.num_equalities, info.equality_indices, value, d_x);

  int grid_ineq = (info.num_inequalities + TPB - 1) / TPB;
  set_array_value<T><<<grid_ineq, TPB, 0, stream>>>(
      info.num_inequalities, info.inequality_indices, value, d_x);
}

template <typename T>
void set_design_vars_value_cuda(const OptInfo<T>& info, const T value, T* d_x,
                                cudaStream_t stream) {
  constexpr int TPB = 256;

  int grid_vars = (info.num_variables + TPB - 1) / TPB;
  set_array_value<T><<<grid_vars, TPB, 0, stream>>>(
      info.num_variables, info.design_variable_indices, value, d_x);
}

template <typename T>
void copy_multipliers_cuda(const OptInfo<T>& info, const T* d_src, T* d_dest,
                           cudaStream_t stream) {
  constexpr int TPB = 256;

  int grid_eq = (info.num_equalities + TPB - 1) / TPB;
  copy_array_values<T><<<grid_eq, TPB, 0, stream>>>(
      info.num_equalities, info.equality_indices, d_src, d_dest);

  int grid_ineq = (info.num_inequalities + TPB - 1) / TPB;
  copy_array_values<T><<<grid_ineq, TPB, 0, stream>>>(
      info.num_inequalities, info.inequality_indices, d_src, d_dest);
}

template <typename T>
void copy_design_vars_cuda(const OptInfo<T>& info, const T* d_src, T* d_dest,
                           cudaStream_t stream) {
  constexpr int TPB = 256;

  int grid_vars = (info.num_variables + TPB - 1) / TPB;
  copy_array_values<T><<<grid_vars, TPB, 0, stream>>>(
      info.num_variables, info.design_variable_indices, d_src, d_dest);
}

template <typename T>
AMIGO_KERNEL void initialize_multipliers_kernel(int num_variables,
                                                T barrier_param,
                                                OptInfo<T> info,
                                                OptStateData<T> pt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_variables) {
    return;
  }

  pt.zl[i] = T(0);
  pt.zu[i] = T(0);

  if (!::isinf(info.lbx[i])) {
    pt.zl[i] = barrier_param;
  }
  if (!::isinf(info.ubx[i])) {
    pt.zu[i] = barrier_param;
  }
}

template <typename T>
AMIGO_KERNEL void initialize_slacks_kernel(int num_inequalities,
                                           T barrier_param, OptInfo<T> info,
                                           const T* __restrict__ g,
                                           OptStateData<T> pt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_inequalities) {
    return;
  }

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

  if (!::isinf(info.lbc[i])) {
    pt.sl[i] = barrier_param;
    pt.tl[i] = barrier_param;
    pt.zsl[i] = barrier_param;
    pt.ztl[i] = barrier_param;
  }

  if (!::isinf(info.ubc[i])) {
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
                                            cudaStream_t stream) {
  constexpr int TPB = 256;

  // Variables
  int grid_vars = (info.num_variables + TPB - 1) / TPB;
  initialize_multipliers_kernel<<<grid_vars, TPB, 0, stream>>>(
      info.num_variables, barrier_param, info, pt);

  // Inequalities
  int grid_ineq = (info.num_inequalities + TPB - 1) / TPB;
  initialize_slacks_kernel<<<grid_ineq, TPB, 0, stream>>>(
      info.num_inequalities, barrier_param, info, d_g, pt);
}

template <typename T>
AMIGO_KERNEL void add_residual_vars_kernel(int num_variables, T barrier_param,
                                           const OptInfo<T> info,
                                           OptStateData<const T> pt, const T* g,
                                           T* r) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= num_variables) {
    return;
  }

  // Get the gradient component corresponding to this variable
  int index = info.design_variable_indices[i];

  // Extract the design variable value
  T x = pt.xlam[index];

  // Compute the right-hand-side
  T bx = -(g[index] - pt.zl[i] + pt.zu[i]);
  if (!::isinf(info.lbx[i])) {
    T bzl = -((x - info.lbx[i]) * pt.zl[i] - barrier_param);
    bx += bzl / (x - info.lbx[i]);
  }
  if (!::isinf(info.ubx[i])) {
    T bzu = -((info.ubx[i] - x) * pt.zu[i] - barrier_param);
    bx -= bzu / (info.ubx[i] - x);
  }

  // Set the right-hand-side
  r[index] = bx;
}

template <typename T>
AMIGO_KERNEL void add_residual_eq_kernel(int num_equalities,
                                         const OptInfo<T> info,
                                         OptStateData<const T> pt, const T* g,
                                         T* r) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= num_equalities) {
    return;
  }

  int index = info.equality_indices[i];

  // Set the right-hand-side for the equalities
  r[index] = -(g[index] - info.lbh[i]);
}

template <typename T>
AMIGO_KERNEL void add_residual_ineq_kernel(int num_inequalities,
                                           T barrier_param, T gamma,
                                           const OptInfo<T> info,
                                           OptStateData<const T> pt, const T* g,
                                           T* r) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= num_inequalities) {
    return;
  }

  int index = info.inequality_indices[i];

  // Extract the multiplier from the solution vector
  T lam = pt.xlam[index];

  // Set the right-hand-side values
  T bc = -(g[index] - pt.s[i]);
  T blam = -(-lam - pt.zsl[i] + pt.zsu[i]);

  // Build the components of C and compute its inverse
  T C = 0.0;
  T d = blam;
  if (!::isinf(info.lbc[i])) {
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

  if (!::isinf(info.ubc[i])) {
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

template <typename T>
void add_residual_cuda(T barrier_param, T gamma, const OptInfo<T>& info,
                       OptStateData<const T>& pt, const T* g, T* r,
                       cudaStream_t stream) {
  constexpr int TPB = 256;
  int gv = (info.num_variables + TPB - 1) / TPB;
  int ge = (info.num_equalities + TPB - 1) / TPB;
  int gi = (info.num_inequalities + TPB - 1) / TPB;
  add_residual_vars_kernel<T><<<gv, TPB, 0, stream>>>(
      info.num_variables, barrier_param, info, pt, g, r);
  add_residual_eq_kernel<T>
      <<<ge, TPB, 0, stream>>>(info.num_equalities, info, pt, g, r);
  add_residual_ineq_kernel<T><<<gi, TPB, 0, stream>>>(
      info.num_inequalities, barrier_param, gamma, info, pt, g, r);
}

template <typename T>
AMIGO_KERNEL void compute_update_vars_kernel(int num_variables, T barrier_param,
                                             const OptInfo<T> info,
                                             OptStateData<const T> pt,
                                             OptStateData<T> up) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= num_variables) {
    return;
  }

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

template <typename T>
AMIGO_KERNEL void compute_update_ineq_kernel(int num_inequalities,
                                             T barrier_param, T gamma,
                                             const OptInfo<T> info,
                                             OptStateData<const T> pt,
                                             OptStateData<T> up) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= num_inequalities) {
    return;
  }

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

  if (!::isinf(info.lbc[i])) {
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

  if (!::isinf(info.ubc[i])) {
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

  if (!::isinf(info.lbc[i])) {
    up.zsl[i] = (-up.s[i] + dl) / Fl;
    up.ztl[i] = -blaml - up.zsl[i];
    up.sl[i] = (bzsl - pt.sl[i] * up.zsl[i]) / pt.zsl[i];
    up.tl[i] = (bztl - pt.tl[i] * up.ztl[i]) / pt.ztl[i];
  }
  if (!::isinf(info.ubc[i])) {
    up.zsu[i] = (up.s[i] + du) / Fu;
    up.ztu[i] = -blamu - up.zsu[i];
    up.su[i] = (bzsu - pt.su[i] * up.zsu[i]) / pt.zsu[i];
    up.tu[i] = (bztu - pt.tu[i] * up.ztu[i]) / pt.ztu[i];
  }
}

template <typename T>
void compute_update_cuda(T barrier_param, T gamma, const OptInfo<T>& info,
                         OptStateData<const T>& pt, OptStateData<T>& up,
                         cudaStream_t stream) {
  constexpr int TPB = 256;
  int gv = (info.num_variables + TPB - 1) / TPB;
  int gi = (info.num_inequalities + TPB - 1) / TPB;
  compute_update_vars_kernel<T>
      <<<gv, TPB, 0, stream>>>(info.num_variables, barrier_param, info, pt, up);
  compute_update_ineq_kernel<T><<<gi, TPB, 0, stream>>>(
      info.num_inequalities, barrier_param, gamma, info, pt, up);
}

template <typename T>
AMIGO_KERNEL void compute_diagonal_vars_kernel(int num_variables,
                                               const OptInfo<T> info,
                                               OptStateData<const T> pt,
                                               T* diag) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= num_variables) {
    return;
  }

  // Get the gradient component corresponding to this variable
  int index = info.design_variable_indices[i];

  T x = pt.xlam[index];

  // If the lower bound isn't infinite, add its value
  if (!::isinf(info.lbx[i])) {
    diag[index] += pt.zl[i] / (x - info.lbx[i]);
  }

  // If the upper bound isn't infinite, add its value
  if (!::isinf(info.ubx[i])) {
    diag[index] += pt.zu[i] / (info.ubx[i] - x);
  }
}

template <typename T>
AMIGO_KERNEL void compute_diagonal_slack_kernel(int num_inequalities,
                                                const OptInfo<T> info,
                                                OptStateData<const T> pt,
                                                T* diag) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= num_inequalities) {
    return;
  }

  int index = info.inequality_indices[i];

  // Build the components of C and compute its inverse
  T C = 0.0;
  if (!::isinf(info.lbc[i])) {
    T Fl = pt.sl[i] / pt.zsl[i] + pt.tl[i] / pt.ztl[i];
    C += 1.0 / Fl;
  }
  if (!::isinf(info.ubc[i])) {
    T Fu = pt.su[i] / pt.zsu[i] + pt.tu[i] / pt.ztu[i];
    C += 1.0 / Fu;
  }

  if (C != 0.0) {
    diag[index] = -1.0 / C;
  }
}

template <typename T>
void compute_diagonal_cuda(const OptInfo<T>& info, OptStateData<const T>& pt,
                           T* diag, cudaStream_t stream) {
  constexpr int TPB = 256;
  int gv = (info.num_variables + TPB - 1) / TPB;
  int gi = (info.num_inequalities + TPB - 1) / TPB;
  compute_diagonal_vars_kernel<T>
      <<<gv, TPB, 0, stream>>>(info.num_variables, info, pt, diag);
  compute_diagonal_slack_kernel<T>
      <<<gi, TPB, 0, stream>>>(info.num_inequalities, info, pt, diag);
}

template <typename T>
AMIGO_KERNEL void compute_max_step_vars_kernel(
    int num_variables, const T tau, const OptInfo<T> info,
    OptStateData<const T> pt, OptStateData<const T> up, T init_alpha_x,
    T init_alpha_z, T* alpha_x_max_out, int* x_index_out, T* alpha_z_max_out,
    int* z_index_out) {
  extern __shared__ unsigned char smem[];

  // Layout shared memory as: alpha_x[blockDim], x_idx[blockDim],
  // alpha_z[blockDim], z_idx[blockDim]
  T* s_alpha_x = reinterpret_cast<T*>(smem);
  int* s_x_idx = reinterpret_cast<int*>(s_alpha_x + blockDim.x);
  T* s_alpha_z = reinterpret_cast<T*>(s_x_idx + blockDim.x);
  int* s_z_idx = reinterpret_cast<int*>(s_alpha_z + blockDim.x);

  int tid = threadIdx.x;
  int stride = blockDim.x;  // gridDim.x = 1

  // Local bests
  T local_alpha_x = init_alpha_x;
  int local_x_idx = -1;
  T local_alpha_z = init_alpha_z;
  int local_z_idx = -1;

  for (int i = tid; i < num_variables; i += stride) {
    int index = info.design_variable_indices[i];
    T x = pt.xlam[index];
    T px = up.xlam[index];

    // Lower bound
    if (!::isinf(info.lbx[i])) {
      if (px < 0.0) {
        T numer = x - info.lbx[i];
        T alpha = -tau * numer / px;
        if (alpha < local_alpha_x) {
          local_alpha_x = alpha;
          local_x_idx = index;
        }
      }
      if (up.zl[i] < 0.0) {
        T alpha = -tau * pt.zl[i] / up.zl[i];
        if (alpha < local_alpha_z) {
          local_alpha_z = alpha;
          local_z_idx = index;
        }
      }
    }

    // Upper bound
    if (!::isinf(info.ubx[i])) {
      if (px > 0.0) {
        T numer = info.ubx[i] - x;
        T alpha = tau * numer / px;
        if (alpha < local_alpha_x) {
          local_alpha_x = alpha;
          local_x_idx = index;
        }
      }
      if (up.zu[i] < 0.0) {
        T alpha = -tau * pt.zu[i] / up.zu[i];
        if (alpha < local_alpha_z) {
          local_alpha_z = alpha;
          local_z_idx = index;
        }
      }
    }
  }

  // Write to shared memory
  s_alpha_x[tid] = local_alpha_x;
  s_x_idx[tid] = local_x_idx;
  s_alpha_z[tid] = local_alpha_z;
  s_z_idx[tid] = local_z_idx;

  __syncthreads();

  // Block reduction (min with argmin)
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      // For x
      if (s_alpha_x[tid + offset] < s_alpha_x[tid]) {
        s_alpha_x[tid] = s_alpha_x[tid + offset];
        s_x_idx[tid] = s_x_idx[tid + offset];
      }
      // For z
      if (s_alpha_z[tid + offset] < s_alpha_z[tid]) {
        s_alpha_z[tid] = s_alpha_z[tid + offset];
        s_z_idx[tid] = s_z_idx[tid + offset];
      }
    }
    __syncthreads();
  }

  // Thread 0 writes out
  if (tid == 0) {
    *alpha_x_max_out = s_alpha_x[0];
    *x_index_out = s_x_idx[0];
    *alpha_z_max_out = s_alpha_z[0];
    *z_index_out = s_z_idx[0];
  }
}

template <typename T>
AMIGO_KERNEL void compute_max_step_slack_kernel(
    int num_inequalities, const T tau, const OptInfo<T> info,
    OptStateData<const T> pt, OptStateData<const T> up, T* alpha_x_max_out,
    int* x_index_out, T* alpha_z_max_out, int* z_index_out) {
  extern __shared__ unsigned char smem[];

  // Layout shared memory as: alpha_x[blockDim], x_idx[blockDim],
  // alpha_z[blockDim], z_idx[blockDim]
  T* s_alpha_x = reinterpret_cast<T*>(smem);
  int* s_x_idx = reinterpret_cast<int*>(s_alpha_x + blockDim.x);
  T* s_alpha_z = reinterpret_cast<T*>(s_x_idx + blockDim.x);
  int* s_z_idx = reinterpret_cast<int*>(s_alpha_z + blockDim.x);

  int tid = threadIdx.x;
  int stride = blockDim.x;  // gridDim.x = 1

  // Local bests
  T local_alpha_x = *alpha_x_max_out;
  int local_x_idx = *x_index_out;
  T local_alpha_z = *alpha_z_max_out;
  int local_z_idx = *z_index_out;

  for (int i = tid; i < num_inequalities; i += stride) {
    int idx = info.inequality_indices[i];

    if (!::isinf(info.lbc[i])) {
      // Slack variables
      if (up.sl[i] < 0.0) {
        T alpha = -tau * pt.sl[i] / up.sl[i];
        if (alpha < local_alpha_x) {
          local_alpha_x = alpha;
          local_x_idx = idx;
        }
      }
      if (up.tl[i] < 0.0) {
        T alpha = -tau * pt.tl[i] / up.tl[i];
        if (alpha < local_alpha_x) {
          local_alpha_x = alpha;
          local_x_idx = idx;
        }
      }

      // Dual variables
      if (up.zsl[i] < 0.0) {
        T alpha = -tau * pt.zsl[i] / up.zsl[i];
        if (alpha < local_alpha_z) {
          local_alpha_z = alpha;
          local_z_idx = idx;
        }
      }
      if (up.ztl[i] < 0.0) {
        T alpha = -tau * pt.ztl[i] / up.ztl[i];
        if (alpha < local_alpha_z) {
          local_alpha_z = alpha;
          local_z_idx = idx;
        }
      }
    }

    if (!::isinf(info.ubc[i])) {
      // Slack variables
      if (up.su[i] < 0.0) {
        T alpha = -tau * pt.su[i] / up.su[i];
        if (alpha < local_alpha_x) {
          local_alpha_x = alpha;
          local_x_idx = idx;
        }
      }
      if (up.tu[i] < 0.0) {
        T alpha = -tau * pt.tu[i] / up.tu[i];
        if (alpha < local_alpha_x) {
          local_alpha_x = alpha;
          local_x_idx = idx;
        }
      }

      // Dual variables
      if (up.zsu[i] < 0.0) {
        T alpha = -tau * pt.zsu[i] / up.zsu[i];
        if (alpha < local_alpha_z) {
          local_alpha_z = alpha;
          local_z_idx = idx;
        }
      }
      if (up.ztu[i] < 0.0) {
        T alpha = -tau * pt.ztu[i] / up.ztu[i];
        if (alpha < local_alpha_z) {
          local_alpha_z = alpha;
          local_z_idx = idx;
        }
      }
    }
  }

  // Write to shared memory
  s_alpha_x[tid] = local_alpha_x;
  s_x_idx[tid] = local_x_idx;
  s_alpha_z[tid] = local_alpha_z;
  s_z_idx[tid] = local_z_idx;

  __syncthreads();

  // Block reduction (min with argmin)
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      // For x
      if (s_alpha_x[tid + offset] < s_alpha_x[tid]) {
        s_alpha_x[tid] = s_alpha_x[tid + offset];
        s_x_idx[tid] = s_x_idx[tid + offset];
      }
      // For z
      if (s_alpha_z[tid + offset] < s_alpha_z[tid]) {
        s_alpha_z[tid] = s_alpha_z[tid + offset];
        s_z_idx[tid] = s_z_idx[tid + offset];
      }
    }
    __syncthreads();
  }

  // Thread 0 writes out
  if (tid == 0) {
    *alpha_x_max_out = s_alpha_x[0];
    *x_index_out = s_x_idx[0];
    *alpha_z_max_out = s_alpha_z[0];
    *z_index_out = s_z_idx[0];
  }
}

template <typename T>
void compute_max_step_cuda(const T tau, const OptInfo<T>& info,
                           OptStateData<const T>& pt, OptStateData<const T>& up,
                           T& alpha_x_max, int& x_index, T& alpha_z_max,
                           int& z_index, cudaStream_t stream) {
  // Allocate device scalars for results
  T *d_alpha_x, *d_alpha_z;
  int *d_x_idx, *d_z_idx;
  AMIGO_CHECK_CUDA(cudaMalloc(&d_alpha_x, sizeof(T)));
  AMIGO_CHECK_CUDA(cudaMalloc(&d_alpha_z, sizeof(T)));
  AMIGO_CHECK_CUDA(cudaMalloc(&d_x_idx, sizeof(int)));
  AMIGO_CHECK_CUDA(cudaMalloc(&d_z_idx, sizeof(int)));

  // Initial values
  T init_alpha_x = alpha_x_max, init_alpha_z = alpha_z_max;

  // Launch kernel
  int block_size = 256;
  int grid_size = 1;  // single block
  size_t shmem_size = 2 * block_size * (sizeof(T) + sizeof(int));

  compute_max_step_vars_kernel<T>
      <<<grid_size, block_size, shmem_size, stream>>>(
          info.num_variables, tau, info, pt, up, init_alpha_x, init_alpha_z,
          d_alpha_x, d_x_idx, d_alpha_z, d_z_idx);

  AMIGO_CHECK_CUDA(cudaDeviceSynchronize());

  compute_max_step_slack_kernel<T>
      <<<grid_size, block_size, shmem_size, stream>>>(
          info.num_inequalities, tau, info, pt, up, d_alpha_x, d_x_idx,
          d_alpha_z, d_z_idx);

  AMIGO_CHECK_CUDA(cudaDeviceSynchronize());

  // Copy back results
  AMIGO_CHECK_CUDA(
      cudaMemcpy(&alpha_x_max, d_alpha_x, sizeof(T), cudaMemcpyDeviceToHost));
  AMIGO_CHECK_CUDA(
      cudaMemcpy(&x_index, d_x_idx, sizeof(int), cudaMemcpyDeviceToHost));
  AMIGO_CHECK_CUDA(
      cudaMemcpy(&alpha_z_max, d_alpha_z, sizeof(T), cudaMemcpyDeviceToHost));
  AMIGO_CHECK_CUDA(
      cudaMemcpy(&z_index, d_z_idx, sizeof(int), cudaMemcpyDeviceToHost));

  cudaFree(d_alpha_x);
  cudaFree(d_alpha_z);
  cudaFree(d_x_idx);
  cudaFree(d_z_idx);
}

template <typename T>
AMIGO_KERNEL void apply_step_update_vars_kernel(
    int num_variables, const T alpha_z, const OptInfo<T> info,
    OptStateData<const T> pt, OptStateData<const T> up, OptStateData<T> tmp) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= num_variables) {
    return;
  }

  // Update the dual variables
  if (!::isinf(info.lbx[i])) {
    tmp.zl[i] = pt.zl[i] + alpha_z * up.zl[i];
  }
  if (!::isinf(info.ubx[i])) {
    tmp.zu[i] = pt.zu[i] + alpha_z * up.zu[i];
  }
}

template <typename T>
AMIGO_KERNEL void apply_step_update_slack_kernel(
    int num_inequalities, const T alpha_x, const T alpha_z,
    const OptInfo<T> info, OptStateData<const T> pt, OptStateData<const T> up,
    OptStateData<T> tmp) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= num_inequalities) {
    return;
  }

  tmp.s[i] = pt.s[i] + alpha_x * up.s[i];
  if (!::isinf(info.lbc[i])) {
    tmp.sl[i] = pt.sl[i] + alpha_x * up.sl[i];
    tmp.tl[i] = pt.tl[i] + alpha_x * up.tl[i];
    tmp.zsl[i] = pt.zsl[i] + alpha_z * up.zsl[i];
    tmp.ztl[i] = pt.ztl[i] + alpha_z * up.ztl[i];
  }
  if (!::isinf(info.ubc[i])) {
    tmp.su[i] = pt.su[i] + alpha_x * up.su[i];
    tmp.tu[i] = pt.tu[i] + alpha_x * up.tu[i];
    tmp.zsu[i] = pt.zsu[i] + alpha_z * up.zsu[i];
    tmp.ztu[i] = pt.ztu[i] + alpha_z * up.ztu[i];
  }
}

template <typename T>
void apply_step_update_cuda(const T alpha_x, const T alpha_z,
                            const OptInfo<T>& info, OptStateData<const T>& pt,
                            OptStateData<const T>& up, OptStateData<T>& tmp,
                            cudaStream_t stream) {
  constexpr int TPB = 256;
  int gv = (info.num_variables + TPB - 1) / TPB;
  int gi = (info.num_inequalities + TPB - 1) / TPB;
  apply_step_update_vars_kernel<T>
      <<<gv, TPB, 0, stream>>>(info.num_variables, alpha_z, info, pt, up, tmp);
  apply_step_update_slack_kernel<T><<<gi, TPB, 0, stream>>>(
      info.num_inequalities, alpha_x, alpha_z, info, pt, up, tmp);
}

template <typename T>
AMIGO_KERNEL void compute_affine_start_point_vars_kernel(
    int num_variables, T beta_min, OptStateData<const T> pt,
    OptStateData<const T> up, OptStateData<T> tmp) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= num_variables) {
    return;
  }

  tmp.zl[i] = A2D::max2(beta_min, A2D::fabs(pt.zl[i] + up.zl[i]));
  tmp.zu[i] = A2D::max2(beta_min, A2D::fabs(pt.zu[i] + up.zu[i]));
}

template <typename T>
AMIGO_KERNEL void compute_affine_start_point_ineq_kernel(
    int num_inequalities, T beta_min, const OptInfo<T> info,
    OptStateData<const T> pt, OptStateData<const T> up, OptStateData<T> tmp) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= num_inequalities) {
    return;
  }
  if (!::isinf(info.lbc[i])) {
    tmp.sl[i] = A2D::max2(beta_min, A2D::fabs(pt.sl[i] + up.sl[i]));
    tmp.tl[i] = A2D::max2(beta_min, A2D::fabs(pt.tl[i] + up.tl[i]));
    tmp.zsl[i] = A2D::max2(beta_min, A2D::fabs(pt.zsl[i] + up.zsl[i]));
    tmp.ztl[i] = A2D::max2(beta_min, A2D::fabs(pt.ztl[i] + up.ztl[i]));
  }

  if (!::isinf(info.ubc[i])) {
    tmp.su[i] = A2D::max2(beta_min, A2D::fabs(pt.su[i] + up.su[i]));
    tmp.tu[i] = A2D::max2(beta_min, A2D::fabs(pt.tu[i] + up.tu[i]));
    tmp.zsu[i] = A2D::max2(beta_min, A2D::fabs(pt.zsu[i] + up.zsu[i]));
    tmp.ztu[i] = A2D::max2(beta_min, A2D::fabs(pt.ztu[i] + up.ztu[i]));
  }
}

template <typename T>
void compute_affine_start_point_cuda(T beta_min, const OptInfo<T>& info,
                                     OptStateData<const T>& pt,
                                     OptStateData<const T>& up,
                                     OptStateData<T>& tmp,
                                     cudaStream_t stream) {
  constexpr int TPB = 256;
  int gv = (info.num_variables + TPB - 1) / TPB;
  int gi = (info.num_inequalities + TPB - 1) / TPB;
  compute_affine_start_point_vars_kernel<T>
      <<<gv, TPB, 0, stream>>>(info.num_variables, beta_min, pt, up, tmp);
  compute_affine_start_point_ineq_kernel<T><<<gi, TPB, 0, stream>>>(
      info.num_inequalities, beta_min, info, pt, up, tmp);
}

/**
 *  Explicit instantiations for T = double
 */

template void set_multipliers_value_cuda<double>(const OptInfo<double>& info,
                                                 double value, double* d_x,
                                                 cudaStream_t stream);

template void set_design_vars_value_cuda<double>(const OptInfo<double>& info,
                                                 double value, double* d_x,
                                                 cudaStream_t stream);

template void copy_multipliers_cuda<double>(const OptInfo<double>& info,
                                            const double* d_src, double* d_dest,
                                            cudaStream_t stream);

template void copy_design_vars_cuda<double>(const OptInfo<double>& info,
                                            const double* d_src, double* d_dest,
                                            cudaStream_t stream);

template void initialize_multipliers_and_slacks_cuda<double>(
    double barrier_param, const OptInfo<double>& info, const double* d_g,
    OptStateData<double>& pt, cudaStream_t stream);

template void add_residual_cuda<double>(double barrier_param, double gamma,
                                        const OptInfo<double>& info,
                                        OptStateData<const double>& pt,
                                        const double* g, double* r,
                                        cudaStream_t stream);

template void compute_update_cuda<double>(double barrier_param, double gamma,
                                          const OptInfo<double>& info,
                                          OptStateData<const double>& pt,
                                          OptStateData<double>& up,
                                          cudaStream_t stream);

template void compute_diagonal_cuda<double>(const OptInfo<double>& info,
                                            OptStateData<const double>& pt,
                                            double* diag, cudaStream_t stream);

template void compute_max_step_cuda<double>(const double tau,
                                            const OptInfo<double>& info,
                                            OptStateData<const double>& pt,
                                            OptStateData<const double>& up,
                                            double& alpha_x_max, int& x_index,
                                            double& alpha_z_max, int& z_index,
                                            cudaStream_t stream);

template void apply_step_update_cuda<double>(
    const double alpha_x, const double alpha_z, const OptInfo<double>& info,
    OptStateData<const double>& pt, OptStateData<const double>& up,
    OptStateData<double>& tmp, cudaStream_t stream);

template void compute_affine_start_point_cuda<double>(
    double beta_min, const OptInfo<double>& info,
    OptStateData<const double>& pt, OptStateData<const double>& up,
    OptStateData<double>& tmp, cudaStream_t stream);

/**
 *  Explicit instantiations for T = float
 */
template void set_multipliers_value_cuda<float>(const OptInfo<float>& info,
                                                float value, float* d_x,
                                                cudaStream_t stream);

template void set_design_vars_value_cuda<float>(const OptInfo<float>& info,
                                                float value, float* d_x,
                                                cudaStream_t stream);

template void copy_multipliers_cuda<float>(const OptInfo<float>& info,
                                           const float* d_src, float* d_dest,
                                           cudaStream_t stream);

template void copy_design_vars_cuda<float>(const OptInfo<float>& info,
                                           const float* d_src, float* d_dest,
                                           cudaStream_t stream);

template void initialize_multipliers_and_slacks_cuda<float>(
    float barrier_param, const OptInfo<float>& info, const float* d_g,
    OptStateData<float>& pt, cudaStream_t stream);

template void add_residual_cuda<float>(float barrier_param, float gamma,
                                       const OptInfo<float>& info,
                                       OptStateData<const float>& pt,
                                       const float* g, float* r,
                                       cudaStream_t stream);

template void compute_update_cuda<float>(float barrier_param, float gamma,
                                         const OptInfo<float>& info,
                                         OptStateData<const float>& pt,
                                         OptStateData<float>& up,
                                         cudaStream_t stream);

template void compute_diagonal_cuda<float>(const OptInfo<float>& info,
                                           OptStateData<const float>& pt,
                                           float* diag, cudaStream_t stream);

template void compute_max_step_cuda<float>(
    const float tau, const OptInfo<float>& info, OptStateData<const float>& pt,
    OptStateData<const float>& up, float& alpha_x_max, int& x_index,
    float& alpha_z_max, int& z_index, cudaStream_t stream);

template void apply_step_update_cuda<float>(
    const float alpha_x, const float alpha_z, const OptInfo<float>& info,
    OptStateData<const float>& pt, OptStateData<const float>& up,
    OptStateData<float>& tmp, cudaStream_t stream);

template void compute_affine_start_point_cuda<float>(
    float beta_min, const OptInfo<float>& info, OptStateData<const float>& pt,
    OptStateData<const float>& up, OptStateData<float>& tmp,
    cudaStream_t stream);

}  // namespace detail

}  // namespace amigo

#endif  // AMIGO_OPTIMIZER_CUDA_BACKEND_H