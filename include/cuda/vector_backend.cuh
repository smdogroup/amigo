#ifndef AMIGO_CUDA_VECTOR_BACKEND_H
#define AMIGO_CUDA_VECTOR_BACKEND_H

#include <cuda_runtime.h>

#include "amigo.h"

namespace amigo {

template <typename T>
class CudaVecBackend {
 public:
  CudaVecBackend() : size(0), d_ptr(nullptr) {}
  ~CudaVecBackend() {
    if (d_ptr) {
      cudaFree(d_ptr);
    }
  }

  void allocate(int size_) {
    if (d_ptr) {
      cudaFree(d_ptr);
    }
    size = size_;
    AMIGO_CHECK_CUDA(cudaMalloc(&d_ptr, size * sizeof(T)));
  }

  void copy_host_to_device(const T* h_ptr) {
    AMIGO_CHECK_CUDA(
        cudaMemcpy(d_ptr, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice));
  }

  void copy_device_to_host(T* h_ptr) {
    AMIGO_CHECK_CUDA(
        cudaMemcpy(h_ptr, d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
  }

  void copy(const T* d_src) {
    AMIGO_CHECK_CUDA(
        cudaMemcpy(d_ptr, d_src, size * sizeof(T), cudaMemcpyDeviceToDevice));
  }

  void zero() { AMIGO_CHECK_CUDA(cudaMemset(d_ptr, 0, size * sizeof(T))); }

  T* get_device_ptr() { return d_ptr; }
  const T* get_device_ptr() const { return d_ptr; }

 private:
  int size;
  T* d_ptr;
};

}  // namespace amigo

#endif  // AMIGO_CUDA_VECTOR_BACKEND_H