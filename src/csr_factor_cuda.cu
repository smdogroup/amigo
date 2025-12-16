#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusparse.h>

#ifdef AMIGO_USE_CUDSS
#include <cudss.h>

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

// cuDSS in CUDA <= 12.9: uses classic CSR
#if CUDA_VERSION < 13000
#define AMIGO_CUDSS_USE_CLASSIC_CSR
#endif

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

    // Create the data object
    AMIGO_CHECK_CUDSS(cudssDataCreate(handle, &data));

    // Create matrix
    int* d_rowp = nullptr;
    int* d_cols = nullptr;
    double* d_data = nullptr;
    mat->get_device_data(&d_rowp, &d_cols, &d_data);

    d_start_rowp = nullptr, d_end_rowp = nullptr;

#ifdef AMIGO_CUDSS_USE_CLASSIC_CSR
    AMIGO_CHECK_CUDSS(cudssMatrixCreateCsr(
        &A, (int64_t)n, (int64_t)n, (int64_t)nnz, d_rowp, nullptr, d_cols,
        d_data, CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
        CUDSS_BASE_ZERO));
#else
    AMIGO_CHECK_CUDA(cudaMalloc(&d_start_rowp, n * sizeof(int)));
    AMIGO_CHECK_CUDA(cudaMalloc(&d_end_rowp, n * sizeof(int)));
    AMIGO_CHECK_CUDA(cudaMemcpy(d_start_rowp, d_rowp, n * sizeof(int),
                                cudaMemcpyDeviceToDevice));
    AMIGO_CHECK_CUDA(cudaMemcpy(d_end_rowp, d_rowp + 1, n * sizeof(int),
                                cudaMemcpyDeviceToDevice));

    AMIGO_CHECK_CUDSS(cudssMatrixCreateCsr(
        &A, (int64_t)n, (int64_t)n, (int64_t)nnz, d_rowp, nullptr, d_cols,
        d_data, CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
        CUDSS_BASE_ZERO));
#endif  // AMIGO_CUDSS_USE_CLASSIC_CSR

    // Create the configuration settings
    AMIGO_CHECK_CUDSS(cudssConfigCreate(&config));

    // Example configuration: reordering algorithm, etc.
    cudssAlgType_t reorder = CUDSS_ALG_DEFAULT;
    AMIGO_CHECK_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG,
                                     &reorder, sizeof(reorder)));

    AMIGO_CHECK_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_PIVOT_EPSILON,
                                     &pivot_tol, sizeof(pivot_tol)));

    AMIGO_CHECK_CUDA(cudaMalloc(&d_X, n * sizeof(double)));
    AMIGO_CHECK_CUDA(cudaMalloc(&d_B, n * sizeof(double)));
    AMIGO_CHECK_CUDSS(cudssMatrixCreateDn(&X, n, 1, n, d_X, CUDA_R_64F,
                                          CUDSS_LAYOUT_COL_MAJOR));
    AMIGO_CHECK_CUDSS(cudssMatrixCreateDn(&B, n, 1, n, d_B, CUDA_R_64F,
                                          CUDSS_LAYOUT_COL_MAJOR));

    AMIGO_CHECK_CUDSS(
        cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, A, X, B));

#endif  // AMIGO_USE_CUDSS
  }

  ~CudssFactorBackend() {
#ifdef AMIGO_USE_CUDSS
    if (d_start_rowp) {
      cudaFree(d_start_rowp);
    }
    if (d_end_rowp) {
      cudaFree(d_end_rowp);
    }
    cudssMatrixDestroy(A);
    cudssMatrixDestroy(B);
    cudssMatrixDestroy(X);
    cudaFree(d_X);
    cudaFree(d_B);
    cudssConfigDestroy(config);
    cudssDataDestroy(handle, data);
    cudssDestroy(handle);
#endif
  }

  void factor() {
#ifdef AMIGO_USE_CUDSS
    AMIGO_CHECK_CUDSS(
        cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, A, X, B));
#endif
  }

  void solve(std::shared_ptr<Vector<double>> b,
             std::shared_ptr<Vector<double>> x) {
#ifdef AMIGO_USE_CUDSS
    double* d_b = b->get_device_array();
    double* d_x = x->get_device_array();

    AMIGO_CHECK_CUDA(
        cudaMemcpy(d_B, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice));

    AMIGO_CHECK_CUDSS(
        cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, A, X, B));

    AMIGO_CHECK_CUDA(
        cudaMemcpy(d_x, d_X, n * sizeof(double), cudaMemcpyDeviceToDevice));
#endif
  }

 private:
  std::shared_ptr<CSRMat<double>> mat;

  // The pivot tolerance
  double pivot_tol;
  int n, nnz;

  // The end of each row
  int* d_start_rowp;
  int* d_end_rowp;

#ifdef AMIGO_USE_CUDSS
  // Handle for the matrix
  cudssHandle_t handle;

  // Set the configuration options
  cudssConfig_t config;

  // Data created during the different phases
  cudssData_t data;

  // The matrix itself
  cudssMatrix_t A;
  cudssMatrix_t X, B;
  double *d_X, *d_B;
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
    int* d_rowp = nullptr;
    int* d_cols = nullptr;
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
