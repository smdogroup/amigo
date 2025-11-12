#ifndef AMIGO_CUDA_MATRIX_FACTOR_H
#define AMIGO_CUDA_MATRIX_FACTOR_H

#include "amigo.h"
#include "csr_matrix.h"
#include "vector.h"

#ifdef AMIGO_USE_CUDSS

namespace amigo {

#include <cudss.h>

class CSRMatFactorCuda {
 public:
  CSRMatFactorCuda(std::shared_ptr<CSRMat<double>> m,
                   double pivot_tol_in = 1e-12)
      : mat(std::move(m)),
        pivot_tol(pivot_tol_in),
        n(0),
        nnz(0),
        handle(nullptr),
        A(nullptr),
        info(nullptr),
        config(nullptr) {
    mat->get_data(&n, nullptr, &nnz, nullptr, nullptr, nullptr);

    // Handle for cudss
    AMIGO_CHECK_CUDSS(cudssCreate(&handle));

    // Create matrix
    const int *d_rowp, *d_cols;
    double* d_data;
    mat->get_device_data(&d_rowp, &d_cols, &d_data);
    AMIGO_CHECK_CUDSS(cudssCreateCsr(&A, n, n, nnz, d_rowp, d_cols, d_data,
                                     CUDSS_INDEX_32I, CUDSS_R_64F,
                                     CUDSS_INDEX_BASE_ZERO));

    // Create the configuration settings
    AMIGO_CHECK_CUDSS(cudssCreateConfig(&config));

    // Set configuration options
    cudssAlgSolverType_t solver_type = CUDSS_ALG_SOLVER_TYPE_LU;
    AMIGO_CHECK_CUDSS(cudssConfigSet(handle, config,
                                     CUDSS_CONFIG_ALG_SOLVER_TYPE, &solver_type,
                                     sizeof(solver_type)));

    cudssMatrixType_t mat_type = CUDSS_MATRIX_TYPE_GENERAL;
    AMIGO_CHECK_CUDSS(cudssConfigSet(handle, config, CUDSS_CONFIG_MATRIX_TYPE,
                                     &mat_type, sizeof(mat_type)));

    cudssReorderMethod_t reorder = CUDSS_REORDER_METIS;
    AMIGO_CHECK_CUDSS(cudssConfigSet(handle, config,
                                     CUDSS_CONFIG_REORDER_METHOD, &reorder,
                                     sizeof(reorder)));

    AMIGO_CHECK_CUDSS(cudssConfigSet(handle, config,
                                     CUDSS_CONFIG_PIVOT_TOLERANCE, &pivot_eps,
                                     sizeof(pivot_eps)));

    // Symbolic analysis
    AMIGO_CHECK_CUDSS(cudssCreateAnalysisInfo(&info));
    AMIGO_CHECK_CUDSS(cudssAnalyze(handle, A, config, info));
  }
  ~CSRMatFactorCuda() {
    if (config) {
      cudssDestroyConfig(config);
    }
    if (info) {
      cudssDestroyAnalysisInfo(info);
    }
    if (A) {
      cudssDestroyMatrix(A);
    }
    if (handle) {
      cudssDestroy(handle);
    }
  }

  void factor() {
    // Numerical factorization
    AMIGO_CHECK_CUDSS(cudssFactorize(handle, A, config, info));
  }

  void solve(std::shared_ptr<Vector<double>> b,
             std::shared_ptr<Vector<double>> x) {
    double* d_b = b->get_device_array();
    double* d_x = x->get_device_array();

    cudssDense_t B, X;
    AMIGO_CHECK_CUDSS(cudssCreateDn(&B, n, 1, d_b, CUDSS_R_64F));
    AMIGO_CHECK_CUDSS(cudssCreateDn(&X, n, 1, d_x, CUDSS_R_64F));

    AMIGO_CHECK_CUDSS(cudssSolve(handle, CUDSS_OP_N, A, B, X, config, info));

    cudssDestroyDense(B);
    cudssDestroyDense(X);
  }

 private:
  // Pointer to the CSR matrix
  std::shared_ptr<CSRMat<double>> mat;

  // The pivot tolerance
  double pivot_tol;
  int n, nnz;

  // Handle for the matrix
  cudssHandle_t handle;

  // The matrix itself
  cudssMatrix_t A;

  // Info about the analysis
  cudssAnalysisInfo_t info;

  // Set the configuration options
  cudssConfig_t config;
};

}  // namespace amigo

#endif  // AMIGO_USE_CUDSS

#endif  // AMIGO_CUDA_MATRIX_FACTOR_H