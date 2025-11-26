#ifndef AMIGO_H
#define AMIGO_H

namespace amigo {

/**
 * @brief Tag where the memory is located host, device or both
 */
enum class MemoryLocation { HOST_ONLY, DEVICE_ONLY, HOST_AND_DEVICE };

/**
 * @brief Tag where to execute the code: host or device
 */
enum class ExecPolicy { SERIAL, OPENMP, CUDA };

inline bool check_consistent_policy_and_location(ExecPolicy policy,
                                                 MemoryLocation loc) {
  if ((policy == ExecPolicy::SERIAL || policy == ExecPolicy::OPENMP) &&
      loc == MemoryLocation::DEVICE_ONLY) {
    return false;
  } else if (policy == ExecPolicy::CUDA && loc == MemoryLocation::HOST_ONLY) {
    return false;
  }
  return true;
}

}  // namespace amigo

#ifdef AMIGO_USE_CUDA
#define AMIGO_KERNEL __global__
#define AMIGO_DEVICE __device__
#define AMIGO_HOST_DEVICE __host__ __device__
#define AMIGO_RESTRICT __restrict__

#include <cublas_v2.h>
#include <cuda_runtime.h>

#if __has_include(<cudss.h>)
#include <cudss.h>
#define AMIGO_USE_CUDSS 1
#endif

#ifndef AMIGO_CHECK_CUDA
#define AMIGO_CHECK_CUDA(call)                                           \
  do {                                                                   \
    cudaError_t err__ = (call);                                          \
    if (err__ != cudaSuccess) {                                          \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err__));                           \
      std::abort();                                                      \
    }                                                                    \
  } while (0)
#endif

#ifndef AMIGO_CHECK_CUSPARSE
#define AMIGO_CHECK_CUSPARSE(call)                                           \
  do {                                                                       \
    auto err__ = (call);                                                     \
    if (err__ != CUSPARSE_STATUS_SUCCESS) {                                  \
      std::fprintf(stderr, "cuSPARSE error %s:%d: %d\n", __FILE__, __LINE__, \
                   (int)err__);                                              \
      std::abort();                                                          \
    }                                                                        \
  } while (0)
#endif

#ifndef AMIGO_CHECK_CUSOLVER
#define AMIGO_CHECK_CUSOLVER(call)                                           \
  do {                                                                       \
    auto err__ = (call);                                                     \
    if (err__ != CUSOLVER_STATUS_SUCCESS) {                                  \
      std::fprintf(stderr, "cuSOLVER error %s:%d: %d\n", __FILE__, __LINE__, \
                   (int)err__);                                              \
      std::abort();                                                          \
    }                                                                        \
  } while (0)
#endif

#ifdef AMIGO_USE_CUDSS
#ifndef AMIGO_CHECK_CUDSS
#define AMIGO_CHECK_CUDSS(call)                                           \
  do {                                                                    \
    auto err__ = (call);                                                  \
    if (err__ != CUDSS_STATUS_SUCCESS) {                                  \
      std::fprintf(stderr, "cuDSS error %s:%d: %d\n", __FILE__, __LINE__, \
                   int(err__));                                           \
      std::abort();                                                       \
    }                                                                     \
  } while (0)
#endif
#endif

#ifndef AMIGO_CHECK_CUBLAS
#define AMIGO_CHECK_CUBLAS(call)                            \
  do {                                                      \
    cublasStatus_t _st = (call);                            \
    if (_st != CUBLAS_STATUS_SUCCESS) {                     \
      std::fprintf(stderr, "cuBLAS error: %d\n", int(_st)); \
      std::abort();                                         \
    }                                                       \
  } while (0)
#endif

#else
#define AMIGO_KERNEL
#define AMGIO_DEVICE
#define AMIGO_HOST_DEVICE
#define AMIGO_RESTRICT
#endif  // AMIGO_USE_CUDA

#endif  // AMIGO_H