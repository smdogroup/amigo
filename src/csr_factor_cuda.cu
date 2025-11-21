#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusparse.h>

#include "amigo.h"
#include "cuda/csr_factor_cuda.h"

namespace amigo {

class CudssFactorBackend {
 public:
  CudssFactorBackend(std::shared_ptr<CSRMat<double>> m,
                     double pivot_tol_in = 1e-12)
      : mat(m), pivot_tol(pivot_tol_in), n(0), nnz(0) {
#ifdef AMIGO_USE_CUDSS
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
#endif  // AMIGO_USE_CUDSS
  }

  ~CudssFactorBackend() {
#ifdef AMIGO_USE_CUDSS
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
#endif
  }

  void factor() {
#ifdef AMIGO_USE_CUDSS
    // Numerical factorization
    AMIGO_CHECK_CUDSS(cudssFactorize(handle, A, config, info));
#endif
  }

  void solve(std::shared_ptr<Vector<double>> b,
             std::shared_ptr<Vector<double>> x) {
#ifdef AMIGO_USE_CUDSS
    double* d_b = b->get_device_array();
    double* d_x = x->get_device_array();

    cudssDense_t B, X;
    AMIGO_CHECK_CUDSS(cudssCreateDn(&B, n, 1, d_b, CUDSS_R_64F));
    AMIGO_CHECK_CUDSS(cudssCreateDn(&X, n, 1, d_x, CUDSS_R_64F));

    AMIGO_CHECK_CUDSS(cudssSolve(handle, CUDSS_OP_N, A, B, X, config, info));

    cudssDestroyDense(B);
    cudssDestroyDense(X);
#endif
  }

 private:
  std::shared_ptr<CSRMat<double>> mat;

  // The pivot tolerance
  double pivot_tol;
  int n, nnz;

#ifdef AMIGO_USE_CUDSS
  // Handle for the matrix
  cudssHandle_t handle;

  // The matrix itself
  cudssMatrix_t A;

  // Info about the analysis
  cudssAnalysisInfo_t info;

  // Set the configuration options
  cudssConfig_t config;
#endif  // AMIGO_USE_CUDSS
};

class CuSolverFactorBackend {
 public:
  CuSolverFactorBackend(std::shared_ptr<CSRMat<double>> m, int reorder_in = 3,
                        double pivot_tol_in = 1e-12)
      : mat(m), reorder(reorder_in), pivot_tol(pivot_tol_in), n(0), nnz(0) {
    // Get the matrix sizes
    mat->get_data(&n, nullptr, &nnz, nullptr, nullptr, nullptr);

    AMIGO_CHECK_CUSOLVER(cusolverSpCreate(&handle));

    // Descriptor for the original matrix A
    AMIGO_CHECK_CUSPARSE(cusparseCreateMatDescr(&descA));
    AMIGO_CHECK_CUSPARSE(
        cusparseSetMatType(descA, CUSPARSE_MATRIX_TYPE_GENERAL));
    AMIGO_CHECK_CUSPARSE(
        cusparseSetMatIndexBase(descA, CUSPARSE_INDEX_BASE_ZERO));
  }

  ~CuSolverFactorBackend() {
    cusparseDestroyMatDescr(descA);
    cusolverSpDestroy(handle);
  }

  void factor() {}

  void solve(std::shared_ptr<Vector<double>> b,
             std::shared_ptr<Vector<double>> x) {
    // Get device-side data
    const int* d_rowp = nullptr;
    const int* d_cols = nullptr;
    double* d_data = nullptr;
    mat->get_device_data(&d_rowp, &d_cols, &d_data);

    double* d_b = b->get_device_array();
    double* d_x = x->get_device_array();

    int singularity = -1;
    AMIGO_CHECK_CUSOLVER(cusolverSpDcsrlsvqr(handle, n, nnz, descA, d_data,
                                             d_rowp, d_cols, d_b, pivot_tol,
                                             reorder, d_x, &singularity));
  }

 private:
  std::shared_ptr<CSRMat<double>> mat;

  int reorder;
  double pivot_tol;
  int n, nnz;

  cusolverSpHandle_t handle;
  cusparseMatDescr_t descA;
};

CSRMatFactorCuda::CSRMatFactorCuda(std::shared_ptr<CSRMat<double>> m,
                                   double pivot_tol)
    : mat(m) {
#ifdef AMIGO_USE_CUDSS
  obj = new CudssFactorBackend(m, pivot_tol);
#else
  obj = new CuSolverFactorBackend(m, pivot_tol);
#endif
}

CSRMatFactorCuda::~CSRMatFactorCuda() { delete obj; }

void CSRMatFactorCuda::factor() { obj->factor(); }

void CSRMatFactorCuda::solve(std::shared_ptr<Vector<double>> b,
                             std::shared_ptr<Vector<double>> x) {
  obj->solve(b, x);
}

}  // namespace amigo
