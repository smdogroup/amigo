#ifndef AMIGO_CUDA_VECTOR_BACKEND_H
#define AMIGO_CUDA_VECTOR_BACKEND_H

#include <cuda_runtime.h>

#include "amigo.h"

namespace amigo {

template <typename T>
class CudaVecBackend {
 public:
  CudaVecBackend() : size(0), device_ptr(nullptr) {}
  ~CudaVecBackend() {
    if (device_ptr) {
      cudaFree(device_ptr);
    }
  }

  void allocate(int size_) {
    if (device_ptr) {
      cudaFree(device_ptr);
    }
    size = size_;
    AMIGO_CHECK_CUDA(cudaMalloc(&device_ptr, size * sizeof(T)));
  }

  void copy_host_to_device(T* host_ptr) {
    AMIGO_CHECK_CUDA(cudaMemcpy(device_ptr, host_ptr, size * sizeof(T),
                                cudaMemcpyHostToDevice));
  }

  void copy_device_to_host(T* host_ptr) {
    AMIGO_CHECK_CUDA(cudaMemcpy(host_ptr, device_ptr, size * sizeof(T),
                                cudaMemcpyDeviceToHost));
  }

  void zero() { AMIGO_CHECK_CUDA(cudaMemset(device_ptr, 0, size * sizeof(T))); }

  T* get_device_ptr() { return device_ptr; }
  const T* get_device_ptr() const { return device_ptr; }

 private:
  int size;
  T* device_ptr;
};

}  // namespace amigo

#endif  // AMIGO_CUDA_VECTOR_BACKEND_H