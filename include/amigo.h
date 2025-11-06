#ifndef AMIGO_H
#define AMIGO_H

#ifdef AMIGO_USE_CUDA
#define AMIGO_KERNEL __global__
#define AMGIO_DEVICE __device__
#define AMIGO_HOST_DEVICE __host__ __device___

#include <cuda_runtime.h>

#ifndef AMIGO_CUDA_CHECK
#define AMIGO_CUDA_CHECK(call)                                      \
  do {                                                              \
    cudaError_t err__ = (call);                                     \
    if (err__ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err__));                           \
      std::abort();                                                 \
    }                                                               \
  } while (0)
#endif

#else
#define AMIGO_KERNEL
#define AMGIO_DEVICE
#define AMIGO_HOST_DEVICE
#endif  // AMIGO_USE_CUDA

#endif  // AMIGO_H