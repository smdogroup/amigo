#ifndef AMIGO_VECTOR_H
#define AMIGO_VECTOR_H

#include <algorithm>
#include <random>

#include "amigo.h"

namespace amigo {

template <typename T>
class EmptyBackend {
 public:
  EmptyBackend() {}
  ~EmptyBackend() {}

  void allocate(int size_) {}
  void copy_to_host(T* host_dest) {}
  void copy_to_device(T* host_src) {}
  T* get_device_ptr() { return nullptr; }
  const T* get_device_ptr() const { return nullptr; }

  // Kernel functions
  void axpy_kernel(T alpha, const T* x_device_ptr) {}
};

// class GpuLaunchParams {
// public:
// static constexpr ;
// };

#ifdef AMIGO_USE_CUDA

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
class CudaBackend {
 public:
  GpuVecBackend() : size(0), device_ptr(nullptr) {}
  ~GpuVecBackend() {
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
  void axpy_kernel(T alpha, const T* x_device_ptr) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    axpy_kernel<<<numBlocks, blockSize>>>(size, x_device_ptr, device_ptr);
  }

 private:
  int size;
  T* device_ptr;
};

template <typename T>
using DefaultBackend = CudaBackend<T>;
#else
template <typename T>
using DefaultBackend = EmptyBackend<T>;
#endif  // AMIGO_USE_CUDA

enum class VectorLocation { HOST_ONLY, DEVICE_ONLY, HOST_AND_DEVICE };

template <typename T, class Backend = DefaultBackend<T>>
class Vector {
 public:
  Vector(int size, VectorLocation vtype = VectorLocation::HOST_AND_DEVICE)
      : size(size), vtype(vtype) {
    if (vtype == VectorLocation::HOST_AND_DEVICE ||
        vtype == VectorLocation::HOST_ONLY) {
      array = new T[size];
      std::fill(array, array + size, T(0.0));
    }
    if (vtype == VectorLocation::HOST_AND_DEVICE ||
        vtype == VectorLocation::DEVICE_ONLY) {
      backend.allocate(size);
    }
  }

  ~Vector() {
    if (array) {
      delete[] array;
    }
  }

  void copy_host_to_device() {
    if (vtype == VectorLocation::HOST_ONLY) {
      backend.allocate(size);
      vtype = VectorLocation::HOST_AND_DEVICE;
    }

    if (vtype == VectorLocation::HOST_AND_DEVICE) {
      backend.copy_host_to_device(array);
    }
  }

  void copy_device_to_host() {
    if (vtype == VectorLocation::DEVICE_ONLY) {
      array = new T[size];
      vtype = VectorLocation::HOST_AND_DEVICE;
    }

    if (vtype == VectorLocation::HOST_AND_DEVICE) {
      backend.copy_device_to_host(array);
    }
  }

  void zero() {
    if (array) {
      memset(array, 0, size * sizeof(T));
    }
  }

  void set_random(T low = -1.0, T high = 1.0) {
    if (array) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<T> dis(low, high);

      for (int i = 0; i < size; i++) {
        array[i] = dis(gen);
      }
    }
  }

  void axpy(T alpha, const Vector<T>& x) {
    if ((vtype == VectorLocation::DEVICE_ONLY ||
         vtype == VectorLocation::HOST_AND_DEVICE) &&
        (x.vtype == VectorLocation::DEVICE_ONLY ||
         x.vtype == VectorLocation::HOST_AND_DEVICE)) {
      backend.axpy_kernel(alpha, x.get_device_array());
    } else if (array && x.array) {
      for (int i = 0; i < size; i++) {
        array[i] += alpha * x.array[i];
      }
    }
  }

  T dot(const Vector<T>& x) const {
    T value = 0.0;
    if (array && x.array) {
      for (int i = 0; i < size; i++) {
        value += array[i] * x.array[i];
      }
    }
    return value;
  }

  void scale(T alpha) {
    if (array) {
      for (int i = 0; i < size; i++) {
        array[i] *= alpha;
      }
    }
  }

  T& operator[](int i) { return array[i]; }
  const T& operator[](int i) const { return array[i]; }

  int get_size() const { return size; }

  T* get_array() { return array; }
  const T* get_array() const { return array; }

  T* get_device_array() { return backend.get_device_ptr(); }
  const T* get_device_array() const { return backend.get_device_ptr(); }

 private:
  int size;  // Size of the vector
  VectorLocation vtype;
  T* array;  // Host array
  Backend backend;
};

}  // namespace amigo

#endif  // AMIGO_VECTOR_H