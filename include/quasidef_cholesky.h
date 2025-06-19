#ifndef AMIGO_QUASIDEF_CHOLESKY_H
#define AMIGO_QUASIDEF_CHOLESKY_H

#include <complex>

#include "csr_matrix.h"

extern "C" {
// Compute C := alpha*A*A**T + beta*C or C := alpha*A**T*A + beta*C
extern void dsyrk_(const char *uplo, const char *trans, int *n, int *k,
                   double *alpha, double *a, int *lda, double *beta, double *c,
                   int *ldc);

// Compute C := alpha*op( A )*op( B ) + beta*C,
extern void dgemm_(const char *ta, const char *tb, int *m, int *n, int *k,
                   double *alpha, double *a, int *lda, double *b, int *ldb,
                   double *beta, double *c, int *ldc);

// Solve A*x = b or A^T*x = b where A is in packed format
extern void dtpsv_(const char *uplo, const char *transa, const char *diag,
                   int *n, double *a, double *x, int *incx);

// Factorization of packed storage matrices
extern void dpptrf_(const char *c, int *n, double *ap, int *info);

// Compute C := alpha*A*A**T + beta*C or C := alpha*A**T*A + beta*C
extern void zsyrk_(const char *uplo, const char *trans, int *n, int *k,
                   std::complex<double> *alpha, std::complex<double> *a,
                   int *lda, std::complex<double> *beta,
                   std::complex<double> *c, int *ldc);

// Compute C := alpha*op( A )*op( B ) + beta*C,
extern void zgemm_(const char *ta, const char *tb, int *m, int *n, int *k,
                   std::complex<double> *alpha, std::complex<double> *a,
                   int *lda, std::complex<double> *b, int *ldb,
                   std::complex<double> *beta, std::complex<double> *c,
                   int *ldc);

// Solve A*x = b or A^T*x = b where A is in packed format
extern void ztpsv_(const char *uplo, const char *transa, const char *diag,
                   int *n, std::complex<double> *a, std::complex<double> *x,
                   int *incx);

// Factorization of packed storage matrices
extern void zpptrf_(const char *c, int *n, std::complex<double> *ap, int *info);
}

namespace amigo {

template <typename T>
void blas_syrk(const char *uplo, const char *trans, int *n, int *k, T *alpha,
               T *a, int *lda, T *beta, T *c, int *ldc) {
  if constexpr (std::is_same<T, double>::value) {
    dsyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    zsyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_syrk only supports double and std::complex<double>");
  }
}

template <typename T>
void blas_gemm(const char *ta, const char *tb, int *m, int *n, int *k, T *alpha,
               T *a, int *lda, T *b, int *ldb, T *beta, T *c, int *ldc) {
  if constexpr (std::is_same<T, double>::value) {
    dgemm_(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    zgemm_(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_gemm only supports double and std::complex<double>");
  }
}

template <typename T>
void blas_tpsv(const char *uplo, const char *transa, const char *diag, int *n,
               T *a, T *x, int *incx) {
  if constexpr (std::is_same<T, double>::value) {
    dtpsv_(uplo, transa, diag, n, a, x, incx);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    ztpsv_(uplo, transa, diag, n, a, x, incx);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_tpsv only supports double and std::complex<double>");
  }
}

template <typename T>
void lapack_pptrf(const char *c, int *n, T *ap, int *info) {
  if constexpr (std::is_same<T, double>::value) {
    dpptrf_(c, n, ap, info);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    zpptrf_(c, n, ap, info);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "lapack_pptrf only supports double and std::complex<double>");
  }
}

/**
 * @brief A Cholesky factorization of a sparse symmetric quasi-definite matrix.
 *
 * A symmetric quasi-definite matrix consists of positive definite sub-matrices
 * A and C and matrix B that takes the following form
 *
 * [ A  B^{T} ]
 * [ B     -C ]
 *
 * This matrix can be factorized using a Cholesky factorization
 *
 * [ A  B^{T} ] = [ L                ][ L^{T}      L^{-1} * B^{T} ]
 * [ B     -C ] = [ B * L^{-T}   -F  ][                     F^{T} ]
 *
 * where A = L * L^{T} is the Cholesky factorization of A and F is the Cholesky
 * factorization of (C + B * A^{-1} * B) given by
 *
 * F * F^{T} = C + B * L^{-1} * L^{-1} * B^{T} = C + B * A^{-1} * B^{T}
 *
 * @tparam T Typename for the computations
 */
template <typename T>
class QuasidefCholesky {
 public:
  QuasidefCholesky(std::shared_ptr<CSRMat<T>> mat) : mat(mat) {
    // Set the values
    size = mat->nrows;

    // Perform a symbolic analysis to determine the size of the factorization
    int *parent = new int[size];  // Space for the etree
    int *Lnz = new int[size];     // Nonzeros below the diagonal
    build_tree(mat->rowp, mat->cols, parent, Lnz);

    // Find the supernodes in the matrix
    var_to_snode = new int[size];
    num_snodes = init_supernodes(parent, Lnz, var_to_snode);

    // Set the remainder of the data based on the var to snode data
    snode_size = new int[num_snodes];
    snode_to_first_var = new int[num_snodes];
    for (int i = 0, j = 0; i < num_snodes; i++) {
      int jstart = j;
      while (j < size && var_to_snode[j] == i) {
        j++;
      }
      snode_size[i] = j - jstart;
      snode_to_first_var[i] = jstart;
    }

    // Set the data for the entries in the matrix
    colp = new int[num_snodes + 1];
    data_ptr = new int[num_snodes + 1];

    // Compute the non-zeros in the list
    colp[0] = 0;
    data_ptr[0] = 0;
    for (int i = 0; i < num_snodes; i++) {
      int ssize = snode_size[i];
      int var = snode_to_first_var[i];
      int n = 1 + Lnz[var] - ssize;

      data_ptr[i + 1] = data_ptr[i] + (ssize * (ssize + 1) / 2) + n * ssize;
      colp[i + 1] = colp[i] + n;
    }

    rows = new int[colp[num_snodes]];

    // Now we can build the non-zero pattern
    build_nonzero_pattern(mat->rowp, mat->cols, parent, Lnz);
    delete[] parent;
    delete[] Lnz;

    data = new T[data_ptr[num_snodes]];

    // Compute the work size
    work_size = 0;
    for (int i = 0; i < num_snodes; i++) {
      int col_size = snode_size[i] * (colp[i + 1] - colp[i]);
      if (col_size > work_size) {
        work_size = col_size;
      }
      int diag_size = snode_size[i] * snode_size[i];
      if (diag_size > work_size) {
        work_size = diag_size;
      }
    }
  }

  ~QuasidefCholesky() {
    delete[] snode_size;
    delete[] var_to_snode;
    delete[] snode_to_first_var;
    delete[] data_ptr;
    delete[] data;
  }

  /*
    Compute the Cholesky decomposition using a left-looking supernode approach

    At the current iteration L11, L21 and L31 are computed. We want to compute
    the update to L22 and L32. This leads to the relationship

    [ L11   0    0  ][ L11^T  L21^T  L31^T ] = [ x         ]
    [ L21  L22   0  ][  0     L22^T  L32^T ] = [ x  A22    ]
    [ L31  L32  L33 ][  0       0    L33^T ] = [ x  A32  x ]

    The diagonal block gives A22 = L22 * L22^{T} + L21 * L21^{T}, which leads
    to the factorization

    L22 * L22^{T} = A22 - L21 * L21^{T}

    The A32 block gives A32 = L31 * L21^{T} + L32 * L22^{T}

    L32 = (A32 - L31 * L21^{T}) * L22^{-T}

    This leads to the following steps in the algorithm:

    (1) Compute the diagonal update: A22 <- A22 - L21 * L21^{T}

    (2) Compute the column update: A32 <- A32 - L32 * L21. This is a two
    step process whereby we first accumulate the numerical results in a
    temporary work vector then apply them to the A32/L32 data.

    After all columns in the row have completed

    (3) Factor the diagonal to obtain L22

    (4) Apply the factor to the column L32 <- (A32 - L32 * L21) * L22^{-T}
  */
  int factor() {
    // Set values from the matrix
    set_values(mat->rowp, mat->cols, mat->data);

    int rflag = 0;
    int *list = new int[num_snodes];  // List pointer
    int *first = new int[num_snodes];

    // Temporary numeric workspace for stuff
    T *work_temp = new T[work_size];

    // Initialize the linked list and copy the diagonal values
    for (int j = 0; j < num_snodes; j++) {
      list[j] = -1;
    }

    for (int j = 0; j < num_snodes; j++) {
      // Keep track of the size of the supernode on the diagonal
      int diag_size = snode_size[j];
      T *diag = get_diag_pointer(j);

      // First variable associated with this supernode
      int jfirst_var = snode_to_first_var[j];

      // Set the pointer to the current column indices
      const int *jrows = &rows[colp[j]];
      T *jptr = get_factor_pointer(j, diag_size);

      // Go through the linked list of supernodes to find the super node columns
      // k with non-zero entries in row j
      int k = list[j];
      while (k != -1) {
        // Store the next supernode that we will visit
        int next_k = list[k];

        // Width of this supernode
        int ksize = snode_size[k];

        // The array "first" is indexed by supernode and points into the row
        // indices such that rows[first[list[j]]] = j
        int ip_start = first[k];
        int ip_end = colp[k + 1];

        // Find the extent of the variables associated with this supernode
        int ip_next = ip_start + 1;
        while (ip_next < ip_end && var_to_snode[rows[ip_next]] == j) {
          ip_next++;
        }

        // Set the value for the next column in the list
        if (ip_next < ip_end) {
          int snode = var_to_snode[rows[ip_next]];

          // Update the first/list data structure
          first[k] = ip_next;
          list[k] = list[snode];
          list[snode] = k;
        }

        // The number of rows in L21
        int nkrows = ip_next - ip_start;
        const int *krows = &rows[ip_start];
        T *kvals = get_factor_pointer(k, ksize, ip_start);

        // Perform the update to the diagonal by computing
        // diag <- diag - L21 * L21^{T}
        update_diag(ksize, nkrows, jfirst_var, krows, kvals, diag_size, diag,
                    work_temp);

        // Perform the update for the column by computing
        // work_temp = L31 * L21^{T}
        int iremain = ip_end - ip_next;
        update_work_column(ksize, nkrows,
                           get_factor_pointer(k, ksize, ip_start), iremain,
                           get_factor_pointer(k, ksize, ip_next), work_temp);

        // Add the temporary column to the remainder
        // updateColumn(nkrows, iremain, &rows[ip_next], work_temp, jrows,
        // jptr);
        update_column(diag_size, nkrows, jfirst_var, krows, iremain,
                      &rows[ip_next], work_temp, jrows, jptr);

        // Move to the next k non-zero
        k = next_k;
      }

      // Facgtor the diagonal and copy the entries back to the diagonal
      rflag = factor_diag(diag_size, diag);
      if (rflag != 0) {
        rflag += jfirst_var;
        break;
      }

      // Compute (A32 - L32 * L21 ) * L21^{-T}
      int nrhs = colp[j + 1] - colp[j];
      solve_diag(diag_size, diag, nrhs, jptr);

      // Update the list for this column
      if (colp[j] < colp[j + 1]) {
        int snode = var_to_snode[rows[colp[j]]];
        first[j] = colp[j];
        list[j] = list[snode];
        list[snode] = j;
      }
    }

    delete[] list;
    delete[] first;
    delete[] work_temp;

    return rflag;
  }

  /*
    Solve the system of equations with the Cholesky factorization
  */
  void solve(Vector<T> *x) {
    T *xt = x->get_array();

    // Solve L * x = x
    for (int j = 0; j < num_snodes; j++) {
      const int jsize = snode_size[j];
      T *D = get_diag_pointer(j);
      T *y = &xt[snode_to_first_var[j]];
      solve_diag(jsize, D, 1, y);

      // Apply the update from the whole column
      const T *L = get_factor_pointer(j, jsize);

      int ip_end = colp[j + 1];
      for (int ip = colp[j]; ip < ip_end; ip++) {
        T val = 0.0;
        for (int ii = 0; ii < jsize; ii++) {
          val += L[ii] * y[ii];
        }
        xt[rows[ip]] -= val;
        L += jsize;
      }
    }

    // Solve L^{T} * x = x
    for (int j = num_snodes - 1; j >= 0; j--) {
      const int jsize = snode_size[j];
      T *y = &xt[snode_to_first_var[j]];
      T *L = get_factor_pointer(j, jsize);

      int ip_end = colp[j + 1];
      for (int ip = colp[j]; ip < ip_end; ip++) {
        for (int ii = 0; ii < jsize; ii++) {
          y[ii] -= L[ii] * xt[rows[ip]];
        }
        L += jsize;
      }

      T *D = get_diag_pointer(j);
      solve_diag_transpose(jsize, D, 1, y);
    }
  }

 private:
  /**
    Set the values into the matrix.

    @param n The number of columns in the input
    @param Acolp Pointer into the columns
    @param Arows Row indices of the nonzero entries
    @param Avals The numerical values
  */
  void set_values(const int Acolp[], const int Arows[], const T Avals[]) {
    std::fill(data, data + data_ptr[num_snodes], T(0.0));

    for (int j = 0; j < size; j++) {
      int sj = var_to_snode[j];
      int jfirst = snode_to_first_var[sj];
      int jsize = snode_size[sj];

      int ip_end = Acolp[j + 1];
      for (int ip = Acolp[j]; ip < ip_end; ip++) {
        int i = Arows[ip];

        if (i >= j) {
          // Check if this is a diagonal element
          if (i < jfirst + jsize) {
            int jj = j - jfirst;
            int ii = i - jfirst;

            T *D = get_diag_pointer(sj);
            D[get_diag_index(ii, jj)] += Avals[ip];
          } else {
            int jj = j - jfirst;

            // Look for the row
            for (int kp = colp[sj]; kp < colp[sj + 1]; kp++) {
              if (rows[kp] == i) {
                T *L = get_factor_pointer(sj, jsize, kp);
                L[jj] += Avals[ip];

                break;
              }
            }
          }
        }
      }
    }
  }

  /**
    Build the elimination tree and compute the number of non-zeros in
    each column.

    @param Acolp The pointer into each column
    @param Arows The row indices for each matrix entry
    @param parent The elimination tree/forest
    @param Lnz The number of non-zeros in each column
  */
  void build_tree(const int Acolp[], const int Arows[], int parent[],
                  int Lnz[]) {
    int *flag = new int[size];

    for (int k = 0; k < size; k++) {
      parent[k] = -1;
      flag[k] = k;
      Lnz[k] = 0;

      // Loop over the k-th column of the original matrix
      int ip_end = Acolp[k + 1];
      for (int ip = Acolp[k]; ip < ip_end; ip++) {
        int i = Arows[ip];

        if (i < k) {
          // Scan up the etree
          for (; flag[i] != k; i = parent[i]) {
            if (parent[i] == -1) {
              parent[i] = k;
            }

            // L[k, i] is non-zero
            Lnz[i]++;
            flag[i] = k;
          }
        }
      }
    }

    delete[] flag;
  }

  /**
    Initialize the supernodes in the matrix

    The supernodes share the same column non-zero pattern

    @param parent The elimination tree data
    @param Lnz The number of non-zeros per variable
    @param vtn The array of supernodes for each variable
  */
  int init_supernodes(const int parent[], const int Lnz[], int vtn[]) {
    int snode = 0;
    for (int i = 0; i < size;) {
      vtn[i] = snode;
      i++;

      while (i < size && (parent[i - 1] == i) && Lnz[i] == Lnz[i - 1] - 1) {
        vtn[i] = snode;
        i++;
      }
      snode++;
    }

    return snode;
  }

  /**
    Build the non-zero pattern for the supernodes in the matrix

    This follows a similar logic to buildForest(), but uses the parent data.
    This must be called after the supernodes are constructed.

    @param Acolp The pointer into each column
    @param Arows The row indices for each matrix entry
    @param parent The elimination tree/forest
    @param Lnz The number of non-zeros in each column
  */
  void build_nonzero_pattern(const int Acolp[], const int Arows[],
                             const int parent[], int Lnz[]) {
    int *flag = new int[size];

    for (int k = 0; k < size; k++) {
      flag[k] = k;
      Lnz[k] = 0;

      // Loop over the k-th column
      int ip_end = Acolp[k + 1];
      for (int ip = Acolp[k]; ip < ip_end; ip++) {
        int i = Arows[ip];

        if (i < k) {
          // Scan up the etree
          for (; flag[i] != k; i = parent[i]) {
            int si = var_to_snode[i];
            int ivar = snode_to_first_var[si];
            if (i == ivar) {
              int isize = snode_size[si];
              if (k >= ivar + isize) {
                rows[colp[si] + Lnz[i]] = k;
                Lnz[i]++;
              }
            }

            flag[i] = k;
          }
        }
      }
    }

    delete[] flag;
  }

  /**
    Add the diagonal update

    Update the entries of the diagonal matrix

    D <- D - L * L^{T}
  */
  void update_diag(const int lsize, const int nlrows, const int lfirst_var,
                   const int *lrows, T *L, const int diag_size, T *diag,
                   T *work) {
    // Compute L * L^{T}
    int n = nlrows;
    int k = lsize;
    T alpha = 1.0, beta = 0.0;
    blas_syrk<T>("L", "T", &n, &k, &alpha, L, &k, &beta, work, &n);

    // Add D <- D - L * L^{T}
    for (int jj = 0; jj < nlrows; jj++) {
      int j = lrows[jj] - lfirst_var;
      for (int ii = 0; ii < jj + 1; ii++) {
        int i = lrows[ii] - lfirst_var;
        diag[i + j * (j + 1) / 2] -= work[jj + nlrows * ii];
      }
    }
  }

  /**
    Perform a column update into the work column.

    The column T has dimensions of the number of non-zero rows in L32 by the
    number of non-zero rows in L21.

    Given the current factorization at step *, where the matrix is of the form

    [ L11  0   ]
    [ L21  *   ]
    [ L31  Tmp ]

    Compute the result

    Tmp = L31 * L21^{T}

    @param lwidth The number of columns in L21 and L32
    @param n21rows The number of non-zero rows in L21
    @param L21 The numerical values of L21 in row-major order
    @param n31rows The number of non-zero rows in L32
    @param L31 The numerical values of L32 in row-major order
    @param Tmp The temporary vector
  */
  void update_work_column(int lwidth, int n21rows, T *L21, int n31rows, T *L31,
                          T *Tmp) {
    // These matrices are stored in row-major order. To compute the result we
    // use LAPACK with the computation: T^{T} = L21 * L31^{T}
    // dimension of T^{T} is n21rows X n32rows
    // dimension of L21 is n21rows X lwidth
    // dimension of L31^{T} is lwidth X n31rows
    T alpha = 1.0, beta = 0.0;
    blas_gemm<T>("T", "N", &n21rows, &n31rows, &lwidth, &alpha, L21, &lwidth,
                 L31, &lwidth, &beta, Tmp, &n21rows);
  }

  /**
    Subtract a sparse column from another sparse column

    L32 = L32 - Tmp

    The row indices in the input brows must be a subset of the rows in arows.
    Both the input arows and brows must be sorted. All indices in arows must
    exist in brows.

    @param lwidth The width of the L32 column
    @param nrows The number of the rows to update
    @param lrows The indices of the L32 column
    @param A The A values of the column
    @param brows The indices of the B column
    @param B The B values of the column
  */
  void update_column(const int lwidth, const int nlcols, const int lfirst_var,
                     const int *lrows, int nrows, const int *arows, const T *A,
                     const int *brows, T *B) {
    for (int i = 0, bi = 0; i < nrows; i++) {
      while (brows[bi] < arows[i]) {
        bi++;
        B += lwidth;
      }

      for (int jj = 0; jj < nlcols; jj++) {
        int j = lrows[jj] - lfirst_var;
        B[j] -= A[jj];
      }
      A += nlcols;
    }
  }

  /*
    Perform the dense Cholesky factorization of the diagonal components
  */
  int factor_diag(const int diag_size, T *D) {
    int n = diag_size, info;
    lapack_pptrf<T>("U", &n, D, &info);
    return info;
  }

  /*
    Solve L * y = x and output x = y
  */
  void solve_diag(int diag_size, T *L, int nrhs, T *x) {
    int incr = 1;
    for (int k = 0; k < nrhs; k++) {
      blas_tpsv<T>("U", "T", "N", &diag_size, L, x, &incr);
      x += diag_size;
    }
  }

  /*
    Solve L^{T} * y = x and output x = y
  */
  void solve_diag_transpose(int diag_size, T *L, int nrhs, T *x) {
    int incr = 1;
    for (int k = 0; k < nrhs; k++) {
      blas_tpsv<T>("U", "N", "N", &diag_size, L, x, &incr);
      x += diag_size;
    }
  }

  // The following are short cut inline functions.
  // Get the diagonal block index
  inline int get_diag_index(const int i, const int j) {
    if (i >= j) {
      return j + i * (i + 1) / 2;
    } else {
      return i + j * (j + 1) / 2;
    }
  }

  // Given the supernode index, return the pointer to the diagonal matrix
  inline T *get_diag_pointer(const int i) { return &data[data_ptr[i]]; }

  // Given the supernode index, the supernode size and the index into the rows
  // data, return the pointer to the lower factor
  inline T *get_factor_pointer(const int i, const int size, const int index) {
    const int dsize = size * (size + 1) / 2;
    const int offset = index - colp[i];
    return &data[data_ptr[i] + dsize + size * offset];
  }

  // Given the supernode index, the supernode size and the index into the rows
  // data, return the pointer to the lower factor
  inline T *get_factor_pointer(const int i, const int size) {
    const int dsize = size * (size + 1) / 2;
    return &data[data_ptr[i] + dsize];
  }

  // The matrix
  std::shared_ptr<CSRMat<T>> mat;

  // The dimension of the square matrix
  int size;

  // The row indices for the strict lower-diagonal entries of each super node.
  // This does not contain the row indices for the supernode itself. Only
  // entries below the supernode.
  int *rows;

  // Pointer into the row indices for the strict lower block of the
  // matrix. This does not include the row indices for the supernode.
  int *colp;

  // Number of supernodes
  int num_snodes;

  // Supernode sizes - How many consecutive variables belong to this
  // supernode? sum_{i=1}^{num_snodes} snode_size = size
  int *snode_size;

  // Given the variable index, what is the corresponding supernode?
  int *var_to_snode;

  // Given the supernode, what is the first variable in the node?
  int *snode_to_first_var;

  // Given the supernode index, a pointer into the supernode data
  // This is computed as the following for k = 0 ... num_snodes
  // data_ptr[k] = sum_{i = 0}^{k} snode_size[i] * ((snode_size[i] + 1)/2 +
  // colp[i + 1] - colp[i])
  int *data_ptr;

  // Work_size = max(max_{i} (snode_size[i] * (colp[i+1] - colp[i]))
  //                 max_{i} snode_size[i]**2)
  int work_size;

  // The numerical data for all entries size = data_ptr[num_snodes]
  T *data;
};

}  // namespace amigo

#endif  // AMIGO_QUASIDEF_CHOLESKY_H