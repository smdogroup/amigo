#ifndef AMIGO_CUDA_CSR_MATRIX_BACKEND_H
#define AMIGO_CUDA_CSR_MATRIX_BACKEND_H

#include <cuda_runtime.h>

#include "amigo.h"

namespace amigo {

template <typename T>
class CudaCSRMatBackend {
 public:
  CudaCSRMatBackend() : nrows(0), ncols(0), nnz(0), d_rowp(nullptr), d_cols(nullptr), d_data(nullptr) {}
  ~CudaCSRMatBackend() {
    if (d_rowp) {
      cudaFree(d_rowp);
    }
    if (d_cols){
      cudaFree(d_cols);
    }
    if (d_data){
      cudaFree(d_data);
    }
  }

  void allocate(int nrows_, int ncols_, int nnz_) {
    if (d_rowp) {
      cudaFree(d_rowp);
    }
    if (d_cols){
      cudaFree(d_cols);
    }
    if (d_data){
      cudaFree(d_data);
    }
    nrows = nrows_;
    ncols = ncols_;
    nnz = nnz_;
    
    cudaMalloc(&d_rowp, (nrows + 1) * sizeof(int));
    cudaMalloc(&d_cols, nnz * sizeof(int));
    cudaMalloc(&d_data, nnz * sizeof(T));
  }

  void copy_pattern_host_to_device(const int *rowp, const int *cols) {
    cudaMemcpy(d_rowp, rowp, (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, cols, nnz * sizeof(int), cudaMemcpyHostToDevice);
  }

  void copy_data_device_to_host(T* data) {
    cudaMemcpy(data, d_data, nnz * sizeof(T), cudaMemcpyDeviceToHost);
  }

  void get_device_data(const int *rowp[], const int *cols[], T *data[]) { 
    if (rowp){
      *rowp = d_rowp;
    }
    if (cols){
      *cols = d_cols;
    }
    if (data){
      *data = d_data;
    }
  }

 private:
  int nrows, ncols, nnz;
  int *d_rowp, *d_cols;
  T *d_data;
};

}  // namespace amigo

#endif  // AMIGO_CUDA_CSR_MATRIX_BACKEND_H