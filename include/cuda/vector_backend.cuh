#ifndef AMIGO_CUDA_VECTOR_BACKEND_H
#define AMIGO_CUDA_VECTOR_BACKEND_H

#include <cuda_runtime.h>

#include "amigo.h"

namespace amigo {

namespace detail {

template <typename T>
struct CublasVecOps;  // primary template left undefined on purpose

template <>
struct CublasVecOps<float> {
  static cublasStatus_t dot(cublasHandle_t h, int n, const float* x, int incx,
                            const float* y, int incy, float* result) {
    return cublasSdot(h, n, x, incx, y, incy, result);
  }

  static cublasStatus_t axpy(cublasHandle_t h, int n, const float* alpha,
                             const float* x, int incx, float* y, int incy) {
    return cublasSaxpy(h, n, alpha, x, incx, y, incy);
  }

  static cublasStatus_t scal(cublasHandle_t h, int n, const float* alpha,
                             float* x, int incx) {
    return cublasSscal(h, n, alpha, x, incx);
  }
};

template <>
struct CublasVecOps<double> {
  static cublasStatus_t dot(cublasHandle_t h, int n, const double* x, int incx,
                            const double* y, int incy, double* result) {
    return cublasDdot(h, n, x, incx, y, incy, result);
  }

  static cublasStatus_t axpy(cublasHandle_t h, int n, const double* alpha,
                             const double* x, int incx, double* y, int incy) {
    return cublasDaxpy(h, n, alpha, x, incx, y, incy);
  }

  static cublasStatus_t scal(cublasHandle_t h, int n, const double* alpha,
                             double* x, int incx) {
    return cublasDscal(h, n, alpha, x, incx);
  }
};

}  // namespace detail

template <typename T>
class CudaVecBackend {
 public:
  CudaVecBackend() : size(0), d_ptr(nullptr), handle(nullptr) {
    AMIGO_CHECK_CUBLAS(cublasCreate(&handle));
  }
  ~CudaVecBackend() {
    if (d_ptr) {
      cudaFree(d_ptr);
    }
    if (handle) {
      cublasDestroy(handle);
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

  T dot(const T* d_src) const {
    T result{};
    AMIGO_CHECK_CUBLAS(detail::CublasVecOps<T>::dot(handle, size, d_ptr, 1,
                                                    d_src, 1, &result));
    return result;  // host scalar
  }

  void axpy(T alpha, const T* d_x) const {
    AMIGO_CHECK_CUBLAS(
        detail::CublasVecOps<T>::axpy(handle, size, &alpha, d_x, 1, d_ptr, 1));
  }

  void scale(T alpha) {
    AMIGO_CHECK_CUBLAS(
        detail::CublasVecOps<T>::scal(handle, size, &alpha, d_ptr, 1));
  }

  T* get_device_ptr() { return d_ptr; }
  const T* get_device_ptr() const { return d_ptr; }

 private:
  int size;
  T* d_ptr;
  cublasHandle_t handle;
};

}  // namespace amigo

#endif  // AMIGO_CUDA_VECTOR_BACKEND_H