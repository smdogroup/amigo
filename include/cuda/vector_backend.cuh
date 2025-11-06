#include "amigo.h"

template <typename T>
AMIGO_KERNEL void zero_kernel(int size, T* x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    x[i] = 0.0;
  }
}

template <typename T>
AMIGO_KERNEL void axpy_kernel(int size, T alpha, const T* x, T* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    y[i] = y[i] + alpha * x[i];
  }
}

template <typename T>
AMIGO_KERNEL T dot_kernel(int size, T* x, T* y) {
  T value = 0.0;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    value += x[i] * y[i];
  }
  return value;
}

template <typename T>
AMIGO_KERNEL void scale_kernel(int size, T alpha, T* x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    x[i] *= alpha;
  }
}

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
    cudaMalloc(&device_ptr, size);
  }

  void copy_host_to_device(T* host_ptr) {
    cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice);
  }

  void copy_device_to_host(T* host_ptr) {
    cudaMemcpy(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost);
  }

  T* get_device_ptr() { return device_ptr; }
  const T* get_device_ptr() const { return device_ptr; }

  // Launch vector-specific kernels
  void axpy(T alpha, const T* x_device_ptr) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    axpy_kernel<<<numBlocks, blockSize>>>(size, x_device_ptr, device_ptr);
  }

 private:
  int size;
  T* device_ptr;
};
