#ifndef AMIGO_CUDA_COMPONENT_GROUP_H
#define AMIGO_CUDA_COMPONENT_GROUP_H

#include "csr_matrix.h"
#include "layout.h"
#include "vector.h"

namespace amigo {

template <typename T, int ncomp, class Input, int ndata, class Data,
          class... Components>
class CudaGroupBackend {
 public:
  template <int noutputs>
  CudaGroupBackend(IndexLayout<ndata>& data_layout, IndexLayout<ncomp>& layout,
                   IndexLayout<noutputs>& outputs) {
    data_layout.copy_host_to_device();
    layout.copy_host_to_device();
    outputs.copy_host_to_device();

    d_hess_pos = nullptr;
  }
  ~CudaGroupBackend() {
    if (d_hess_pos) {
      cudaFree(d_hess_pos);
    }
  }

  void initialize(IndexLayout<ndata>& data_layout, IndexLayout<ncomp>& layout,
                  IndexLayout<noutputs>& outputs, 
                  const NodeOwners& owners, CSRMat<T>& mat) {
    int num_elements;
    const int* vec_indices;
    layout->get_data(&num_elements, nullptr, &vec_indices);

    const int *rowp, *cols;
    mat->get_data(nullptr, nullptr, nullptr, &rowp, &cols, nullptr);

    // Allocate space for the positions of the Hessian entries
    int size = num_elements * ncomp * ncomp;
    int* hess_pos = new int[size];

    // Locate the positions within the CSRMatrix
    for (int i = 0; i < num_elements; i++) {
      int rows[ncomp], columns[ncomp];
      for (int j = 0; j < ncomp; j++ ){
        rows[j] = vec_indices[ncomp * index + j];
      }

      owners.local_to_global(ncomp, rows, columns);

      for (int j = 0; j < ncomp; j++) {
        int row = rows[j];
        int row_size = rowp[row + 1] - rowp[row];
        int* start = &cols[rowp[row]];
        int* end = start + row_size;

        for (int k = 0; k < ncomp; k++) {
          auto* it = std::lower_bound(start, end, columns[k]);

          if (it != end && *it == columns[k]) {
            hess_pos[ncomp * (ncomp * index + j) + k] = it - cols;
          }
        }
      }
    }

    // Copy the result to the device
    cudaMalloc(&d_hess_pos, size * sizeof(int));
    cudaMemcpy(d_hess_pos, hess_pos, size * sizeof(int),
               cudaMemcpyHostToDevice);

    // This is not needed on the host
    delete[] hess_pos;
  }

  // This isn't handled properly yet...
  T lagrangian_kernel(const IndexLayout<ndata>& data_layout,
                      const IndexLayout<ncomp>& layout,
                      const Vector<T>& data_vec, const Vector<T>& vec) const {
    return 0.0;
  }

  void add_gradient_kernel(const IndexLayout<ndata>& data_layout,
                           const IndexLayout<ncomp>& layout,
                           const Vector<T>& data_vec, const Vector<T>& vec,
                           Vector<T>& res) const {
    const int TPB = 256;
    int num_elements;
    const int* data_indices;
    const int* vec_indices;
    data_layout->get_device_data(&num_elements, nullptr, &data_indices);
    layout->get_device_data(nullptr, nullptr, &vec_indices);

    const T* data_values = data_vec->get_device_array();
    const T* vec_values = vec->get_device_array();
    T* res_values = vec->get_device_array();

    dim3 grid((num_elements + TPB - 1) / TPB);
    dim3 block(TPB);

    gradient_kernel_atomic<<<grid, block>>>(num_elements, data_indices,
                                            vec_indices, data_values,
                                            vec_values, res_values);
  }

  void add_hessian_product_kernel(const IndexLayout<ndata>& data_layout,
                                  const IndexLayout<ncomp>& layout,
                                  const Vector<T>& data_vec,
                                  const Vector<T>& vec, const Vector<T>& dir,
                                  Vector<T>& res) const {
    const int TPB = 256;
    int num_elements;
    const int* data_indices;
    const int* vec_indices;
    data_layout->get_device_data(&num_elements, nullptr, &data_indices);
    layout->get_device_data(nullptr, nullptr, &vec_indices);

    dim3 grid((num_elements + TPB - 1) / TPB);
    dim3 block(TPB);

    hessian_product_kernel_atomic<<<grid, block>>>(num_elements, data_indices,
                                                   vec_indices, data_values,
                                                   vec_values, res_values);
  }

  // Need to add the hessian...
  void add_hessian_kernel(const IndexLayout<ndata>& data_layout,
                          const IndexLayout<ncomp>& layout,
                          const Vector<T>& data_vec, const Vector<T>& vec,
                          CSRMat<T>& jac) const {
    const int TPB = 256;
    int num_elements;
    const int* data_indices;
    const int* vec_indices;
    data_layout->get_device_data(&num_elements, nullptr, &data_indices);
    layout->get_device_data(nullptr, nullptr, &vec_indices);

    dim3 grid((num_elements + TPB - 1) / TPB, ncomp);
    dim3 block(TPB);

    T* csr_data;
    mat->get_device_data(nullptr, nullptr, nullptr, nullptr, nullptr,
                         &csr_data);

    hessian_kernel_atomic<<<grid, block>>>(num_elements, data_indices,
                                           vec_indices, d_hess_pos, data_values,
                                           vec_values, csr_data);
  }

 private:
  int* d_hess_pos;
  // int *d_jac_pos;

  static AMIGO_KERNEL void gradient_kernel_atomic(
      int num_elements, const int* data_indices, const int* vec_indices,
      const T* data_values, const T* vec_values, T* grad_values) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > num_elements) {
      return;
    }

    Input input, grad;
    Data data;
    IndexLayout<ndata>::get_values_device(index, data_indices, data_values,
                                          data);
    IndexLayout<ncomp>::get_values_device(index, vec_indices, vec_values,
                                          input);

    // Compute the gradient
    grad.zero();
    add_gradient<Data, Input, Components...>(data, input, grad);

    // Add the values to grad_values
    IndexLayout<ndomp>::add_values_atomic(index, vec_indices, grad,
                                          grad_values);
  }

  template <class Component, class... Remain>
  static AMIGO_DEVICE void add_gradient(const Data& data, const Input& input,
                                        Input& grad) {
    if constexpr (!Component::is_compute_empty) {
      Component::gradient(data, input, grad);
    }

    if constexpr (sizeof...(Remain) > 0) {
      add_gradient<Data, Input, Remain...>(data, input, grad);
    }
  }

  static AMIGO_KERNEL void hessian_product_kernel_atomic(
      int num_elements, const int* data_indices, const int* vec_indices,
      const T* data_values, const T* vec_values, const T* dir_values,
      T* grad_values) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > num_elements) {
      return;
    }

    Input input, dir, h;
    Data data;
    IndexLayout<ndata>::get_values_device(index, data_indices, data_values,
                                          data);
    IndexLayout<ncomp>::get_values_device(index, vec_indices, vec_values,
                                          input);
    IndexLayout<ncomp>::get_values_device(index, vec_indices, dir_values, dir);

    // Compute the gradient
    h.zero();
    add_hessian_product<Data, Input, Components...>(data, input, dir, grad);

    // Add the values to grad_values
    IndexLayout<ndomp>::add_values_atomic(index, vec_indices, grad,
                                          grad_values);
  }

  template <class Component, class... Remain>
  static AMIGO_DEVICE void add_hessian_product(const Data& data,
                                               const Input& input,
                                               const Input& dir, Input& h) {
    if constexpr (!Component::is_compute_empty) {
      Input grad;
      Component::hessian(data, input, dir, grad, h);
    }

    if constexpr (sizeof...(Remain) > 0) {
      add_hessian_product<Data, Input, Remain...>(data, input, grad);
    }
  }

  static AMIGO_KERNEL void hessian_kernel_atomic(
      int num_elements, const int* data_indices, const int* vec_indices,
      const int* csr_pos, const T* data_values, const T* vec_values,
      T* csr_data) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > num_elements) {
      return;
    }
    int row = blockIdx.y;

    Input input, dir, h;
    Data data;
    IndexLayout<ndata>::get_values_device(index, data_indices, data_values,
                                          data);
    IndexLayout<ncomp>::get_values_device(index, vec_indices, vec_values,
                                          input);

    h.zero();
    dir.zero();
    dir[row] = 1.0;
    add_hessian_product<Data, Input, Components...>(data, input, dir, h);

    // Now add the Hessian contribution
    for (int i = 0; i < ncomp; i++) {
      int pos = csr_pos[(index * ncomp + local_row) * ncomp + i];
      if (pos >= 0) {
        atomicAdd(&csr_data[pos], h[i]);
      }
    }
  }
};

}  // namespace amigo

#endif  // AMIGO_CUDA_COMPONENT_GROUP_H