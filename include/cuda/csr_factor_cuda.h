#ifndef AMIGO_CUDA_MATRIX_FACTOR_H
#define AMIGO_CUDA_MATRIX_FACTOR_H

#include "amigo.h"
#include "csr_matrix.h"
#include "vector.h"

namespace amigo {

class CudssFactorBackend;
class CuSolverFactorBackend;

class CSRMatFactorCuda {
 public:
  CSRMatFactorCuda(std::shared_ptr<CSRMat<double>> m,
                   double pivot_tol_in = 1e-12);
  ~CSRMatFactorCuda();

  void factor();

  void solve(std::shared_ptr<Vector<double>> b,
             std::shared_ptr<Vector<double>> x);

 private:
  // Pointer to the CSR matrix
  std::shared_ptr<CSRMat<double>> mat;

#ifdef AMIGO_USE_CUDSS
  CudssFactorBackend* obj;
#else
  CuSolverFactorBackend* obj;
#endif
};

}  // namespace amigo

#endif  // AMIGO_CUDA_MATRIX_FACTOR_H