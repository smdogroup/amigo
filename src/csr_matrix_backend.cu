#ifndef AMIGO_CSR_MATRIX_CUDA_BACKEND_H
#define AMIGO_CSR_MATRIX_CUDA_BACKEND_H

#include "amigo.h"

namespace amigo {

namespace detail {
template <typename T>
AMIGO_KERNEL void add_array_values(int num_variables, const int* indices,
                                   const T* d_src, T* d_dest) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_variables) {
    return;
  }

  int idx = indices[i];
  if (idx >= 0) {
    d_dest[idx] += d_src[i];
  }
}

template <typename T>
void add_diagonal_cuda(int nrows, const int* d_indices, const T* d_values,
                       T* d_data, cudaStream_t stream) {
  constexpr int TPB = 256;

  int grid = (nrows + TPB - 1) / TPB;
  add_array_values<T>
      <<<grid, TPB, 0, stream>>>(nrows, d_indices, d_values, d_data);
}

template void add_diagonal_cuda<double>(int nrows, const int* d_indices,
                                        const double* d_values, double* d_data,
                                        cudaStream_t stream);

template void add_diagonal_cuda<float>(int nrows, const int* d_indices,
                                       const float* d_values, float* d_data,
                                       cudaStream_t stream);

}  // namespace detail

}  // namespace amigo

#endif