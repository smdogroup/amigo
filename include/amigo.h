#ifndef AMIGO_H
#define AMIGO_H

namespace amigo {

enum class MemoryLocation { HOST_ONLY, DEVICE_ONLY, HOST_AND_DEVICE };

}

#ifdef AMIGO_USE_CUDA
#define AMIGO_KERNEL __global__
#define AMIGO_DEVICE __device__
#define AMIGO_HOST_DEVICE __host__ __device__
#define AMIGO_RESTRICT __restrict__

#include <cuda_runtime.h>
#include <cudss.h>

#ifndef AMIGO_CHECK_CUDA
#define AMIGO_CHECK_CUDA(call)                                      \
  do {                                                              \
    cudaError_t err__ = (call);                                     \
    if (err__ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err__));                           \
      std::abort();                                                 \
    }                                                               \
  } while (0)
#endif

#ifndef AMIGO_CHECK_CUSPARSE
#define AMIGO_CHECK_CUSPARSE(call)                                             \
  do {                                                                         \
    auto err__ = (call);                                                       \
    if (err__ != CUSPARSE_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "cuSPARSE %s:%d: %d\n", __FILE__, __LINE__, (int)err__); \
      std::abort();                                                            \
    }                                                                          \
  } while (0)
#endif

#ifndef AMIGO_CHECK_CUSOLVER
#define AMIGO_CHECK_CUSOLVER(call)                                             \
  do {                                                                         \
    auto err__ = (call);                                                       \
    if (err__ != CUSOLVER_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "cuSOLVER %s:%d: %d\n", __FILE__, __LINE__, (int)err__); \
      std::abort();                                                            \
    }                                                                          \
  } while (0)
#endif

#ifdef AMIGO_CHECK_CUDSS
#define AMIGO_CHECK_CUDSS(call)                    \
  do {                                             \
    auto err__ = (call);                           \
    if (s != CUDSS_STATUS_SUCCESS) {               \
      fprintf(stderr, "cuDSS error %d\n", int(s)); \
      std::abort();                                \
    }                                              \
  } while (0)

#else
#define AMIGO_KERNEL
#define AMGIO_DEVICE
#define AMIGO_HOST_DEVICE
#define AMIGO_RESTRICT
#endif  // AMIGO_USE_CUDA

#endif  // AMIGO_H