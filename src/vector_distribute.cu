#ifndef AMGIO_VECTOR_DISTRIBUTE_CUDA_H
#define AMGIO_VECTOR_DISTRIBUTE_CUDA_H

#include "amigo.h"

namespace amigo {

namespace detail {

template <typename T>
AMIGO_KERNEL void set_buffer_values(int nnodes, const int* nodes,
                                    const T* array, T* buffer) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nnodes) {
    return;
  }

  int idx = nodes[i];
  buffer[i] = array[idx];
}

template <typename T>
AMIGO_KERNEL void add_buffer_values(int nnodes, const int* nodes,
                                    const T* buffer, T* array) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nnodes) {
    return;
  }

  int idx = nodes[i];
  array[idx] += buffer[i];
}

template <typename T>
void set_buffer_values_kernel_cuda(int nnodes, const int* nodes, const T* array,
                                   T* buffer) {
  constexpr int TPB = 256;
  int grid = (nnodes + TPB - 1) / TPB;
  set_buffer_values<T><<<grid, TPB>>>(nnodes, nodes, array, buffer);
}

template <typename T>
void add_buffer_values_kernel_cuda(int nnodes, const int* nodes,
                                   const T* buffer, T* array) {
  constexpr int TPB = 256;
  int grid = (nnodes + TPB - 1) / TPB;
  add_buffer_values<T><<<grid, TPB>>>(nnodes, nodes, buffer, array);
}

// Instatiations for T = double
template void set_buffer_values_kernel_cuda<double>(int, const int*,
                                                    const double*, double*);

template void add_buffer_values_kernel_cuda<double>(int, const int*,
                                                    const double*, double*);

// Instantiations for T = float
template void set_buffer_values_kernel_cuda<float>(int, const int*,
                                                   const float*, float*);

template void add_buffer_values_kernel_cuda<float>(int, const int*,
                                                   const float*, float*);

// Instantiations for T = int
template void set_buffer_values_kernel_cuda<int>(int, const int*, const int*,
                                                 int*);

template void add_buffer_values_kernel_cuda<int>(int, const int*, const int*,
                                                 int*);

}  // namespace detail

}  // namespace amigo

#endif  // AMGIO_VECTOR_DISTRIBUTE_CUDA_H