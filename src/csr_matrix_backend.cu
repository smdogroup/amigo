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

template <typename T>
AMIGO_KERNEL void zero_at_indices(int nentries, const int* d_indices,
                                  double* d_array) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nentries;
       i += blockDim.x * gridDim.x) {
    d_array[d_indices[i]] = 0.0;
  }
}

template <typename T>
void zero_at_indices_cuda(int nentries, const int* d_indices, T* d_array,
                          cudaStream_t stream) {
  constexpr int TPB = 256;

  int grid = (nrows + TPB - 1) / TPB;
  zero_at_indices<T><<<grid, TPB, 0, stream>>>(nentries, d_indices, d_array);
}

template <typename T>
AMIGO_KERNEL void set_value_at_indices(const T value, int nentries,
                                       const int* d_indices, T* d_array) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nentries;
       i += blockDim.x * gridDim.x) {
    d_array[d_indices[i]] = value;
  }
}

template <typename T>
void set_value_at_indices_cuda(const T value, int nentries,
                               const int* d_indices, T* d_array,
                               cudaStream_t stream) {
  constexpr int TPB = 256;

  int grid = (nrows + TPB - 1) / TPB;
  set_value_at_indices<T>
      <<<grid, TPB, 0, stream>>>(value, nentries, d_indices, d_array);
}

template void add_diagonal_cuda<double>(int nrows, const int* d_indices,
                                        const double* d_values, double* d_data,
                                        cudaStream_t stream);

template void add_diagonal_cuda<float>(int nrows, const int* d_indices,
                                       const float* d_values, float* d_data,
                                       cudaStream_t stream);

template void zero_at_indices_cuda<double>(int nentries, const int* d_indices,
                                           double* d_array,
                                           cudaStream_t stream);

template void zero_at_indices_cuda<float>(int nentries, const int* d_indices,
                                          float* d_array, cudaStream_t stream);

template void set_value_at_indices_cuda<double>(double value, int nentries,
                                                const int* d_indices,
                                                double* d_array,
                                                cudaStream_t stream);

template void set_value_at_indices_cuda<float>(float value, int nentries,
                                               const int* d_indices,
                                               float* d_array,
                                               cudaStream_t stream);

}  // namespace detail

}  // namespace amigo

#endif