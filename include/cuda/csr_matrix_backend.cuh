#ifndef AMIGO_CUDA_CSR_MATRIX_BACKEND_H
#define AMIGO_CUDA_CSR_MATRIX_BACKEND_H

#include <cuda_runtime.h>

#include "amigo.h"

namespace amigo {

template <typename T>
class CudaCSRMatBackend {
 public:
  CudaCSRMatBackend()
      : nrows(0),
        ncols(0),
        nnz(0),
        d_rowp(nullptr),
        d_cols(nullptr),
        d_data(nullptr) {}
  ~CudaCSRMatBackend() {
    if (d_rowp) {
      cudaFree(d_rowp);
    }
    if (d_cols) {
      cudaFree(d_cols);
    }
    if (d_data) {
      cudaFree(d_data);
    }
  }

  void allocate(int nrows_, int ncols_, int nnz_) {
    if (d_rowp) {
      cudaFree(d_rowp);
    }
    if (d_cols) {
      cudaFree(d_cols);
    }
    if (d_data) {
      cudaFree(d_data);
    }
    nrows = nrows_;
    ncols = ncols_;
    nnz = nnz_;

    AMIGO_CHECK_CUDA(cudaMalloc(&d_rowp, (nrows + 1) * sizeof(int)));
    AMIGO_CHECK_CUDA(cudaMalloc(&d_cols, nnz * sizeof(int)));
    AMIGO_CHECK_CUDA(cudaMalloc(&d_data, nnz * sizeof(T)));
  }

  void copy_pattern_host_to_device(const int* rowp, const int* cols) {
    AMIGO_CHECK_CUDA(cudaMemcpy(d_rowp, rowp, (nrows + 1) * sizeof(int),
                                cudaMemcpyHostToDevice));
    AMIGO_CHECK_CUDA(
        cudaMemcpy(d_cols, cols, nnz * sizeof(int), cudaMemcpyHostToDevice));
  }

  void copy_data_device_to_host(T* data) {
    AMIGO_CHECK_CUDA(
        cudaMemcpy(data, d_data, nnz * sizeof(T), cudaMemcpyDeviceToHost));
  }

  void copy_data_device_to_host(int ext_offset, int ext_size, T* ext_data) {
    if (ext_size <= nnz - ext_offset) {
      AMIGO_CHECK_CUDA(cudaMemcpy(ext_data, d_data + ext_offset,
                                  ext_size * sizeof(T),
                                  cudaMemcpyDeviceToHost));
    }
  }

  void zero() { cudaMemset(d_data, 0, nnz * sizeof(T)); }

  void get_device_data(const int* rowp[], const int* cols[], T* data[]) {
    if (rowp) {
      *rowp = d_rowp;
    }
    if (cols) {
      *cols = d_cols;
    }
    if (data) {
      *data = d_data;
    }
  }

 private:
  int nrows, num_local_rows;  // Number of rows and number of local rows
  int ncols;                  // Number of columns
  int nnz;                    // Total number of non-zeros
  int ext_size;  // External data size - data to be sent to other MPI-connected
                 // devices
  int *d_rowp, *d_cols;  // Non-zero pattern
  T* d_data;             // Device data
  T* ext_data;           // External data buffer
};

}  // namespace amigo

#endif  // AMIGO_CUDA_CSR_MATRIX_BACKEND_H