#ifndef AMIGO_SPARSE_CHOLESKY_H
#define AMIGO_SPARSE_CHOLESKY_H

#include "blas_interface.h"
#include "csr_matrix.h"

namespace amigo {

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
 * [ A  B^{T} ] = [ L               ][ L^{T}    L^{-1} * B^{T} ]
 * [ B     -C ] = [ B * L^{-T}   -F ][                   F^{T} ]
 *
 * where A = L * L^{T} is the Cholesky factorization of A and F is the Cholesky
 * factorization of (C + B * A^{-1} * B) given by
 *
 * F * F^{T} = C + B * L^{-1} * L^{-1} * B^{T} = C + B * A^{-1} * B^{T}
 *
 * [ L11             ] [ L11^{T}  L21^{T}  L31^{T} ]   [ *     *   * ]
 * [ L21  -F11       ] [          F11^{T}  F21^{T} ] = [ *  -C11   * ]
 * [ L31  -F21  -F22 ] [                   F22^{T} ]   [ *  -C21   * ]
 *
 * C contributions, the updates chage. The C values recieve positive
 * contributions for the diagonal update from L and negative contributions from
 * the F components of the factored matrix.
 *
 * The diagonal update for C22 takes the form:
 *
 * C22 <- C22 + L31 * L31^{T} - F21 * F21^{T}
 *
 * Followed by the Cholesky factorization F22 * F22^{T} = C22.
 *
 * For the below-diagonal contributions, it's straightforward to observe that
 *
 * L31 * L21^{T} - F21 * F11^{T} = - C21
 *
 * So that the F21 contribution can be computed as:
 *
 * F21 = (C21 + L31 * L21^{T}) * F11^{-T}
 *
 * The lower update takes the form:
 *
 * L41 * L31^{T} - F31 * F21^{T} - F32^{T} * F22^{T} = - C32
 *
 * So that the lower block is
 *
 * F32 = (C32 + L41 * L31^{T} - F31 * F21^{T}) * F22^{-T}
 *
 * Note that the L-L contributions signs are reversed (now positive) compared to
 * regular Cholesky factorization, while the C-C contributions take the regular
 * sign.
 *
 * @tparam T Typename for the computations
 */
template <typename T>
class SparseCholesky {
 public:
  SparseCholesky(std::shared_ptr<CSRMat<T>> mat) : mat(mat) {
    // Get the non-zero pattern
    const int *mat_rowp, *mat_cols;
    mat->get_data(&size, nullptr, nullptr, &mat_rowp, &mat_cols, nullptr);

    // Set the first variable which belongs to the SQD matrix
    sqdef_index = mat->get_sqdef_index();
    if (sqdef_index < 0 || sqdef_index > size) {
      sqdef_index = size;
    }

    // Perform a symbolic analysis to determine the size of the factorization
    int* parent = new int[size];  // Space for the etree
    int* Lnz = new int[size];     // Nonzeros below the diagonal
    build_tree(mat_rowp, mat_cols, parent, Lnz);

    // Find the supernodes in the matrix
    var_to_snode = new int[size];
    num_snodes = init_snodes(parent, Lnz, var_to_snode);

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
    build_nonzero_pattern(mat_rowp, mat_cols, parent, Lnz);
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

  ~SparseCholesky() {
    delete[] snode_size;
    delete[] var_to_snode;
    delete[] snode_to_first_var;
    delete[] data_ptr;
    delete[] data;
  }

  /**
   * @brief Compute the Cholesky decomposition using a left-looking supernode
   * approach
   *
   * At the current iteration L11, L21 and L31 are computed. We want to compute
   * the update to L22 and L32. This leads to the relationship
   *
   * [ L11   0    0  ][ L11^T  L21^T  L31^T ] = [ x         ]
   * [ L21  L22   0  ][  0     L22^T  L32^T ] = [ x  A22    ]
   * [ L31  L32  L33 ][  0       0    L33^T ] = [ x  A32  x ]
   *
   * The diagonal block gives A22 = L22 * L22^{T} + L21 * L21^{T}, which leads
   * to the factorization
   *
   * L22 * L22^{T} = A22 - L21 * L21^{T}
   *
   * The A32 block gives A32 = L31 * L21^{T} + L32 * L22^{T}
   *
   * L32 = (A32 - L31 * L21^{T}) * L22^{-T}
   *
   * This leads to the following steps in the algorithm:
   *
   * (1) Compute the diagonal update: A22 <- A22 - L21 * L21^{T}
   *
   * (2) Compute the column update: A32 <- A32 - L31 * L21^{T}. This is a two
   * step process whereby we first accumulate the numerical results in a
   * temporary work vector then apply them to the A32/L32 data.
   *
   * After all columns in the row have completed
   *
   * (3) Factor the diagonal to obtain L22
   *
   * (4) Apply the factor to the column L32 <- (A32 - L31 * L21^{T}) * L22^{-T}
   */
  int factor() {
    // Set values from the matrix
    const int *mat_rowp, *mat_cols;
    T* mat_data;
    mat->get_data(&size, nullptr, nullptr, &mat_rowp, &mat_cols, &mat_data);

    set_values(mat_rowp, mat_cols, mat_data);

    int rflag = 0;
    int* list = new int[num_snodes];  // List pointer
    int* first = new int[num_snodes];

    // Temporary numeric workspace for stuff
    T* work_temp = new T[work_size];

    // Initialize the linked list and copy the diagonal values
    for (int j = 0; j < num_snodes; j++) {
      list[j] = -1;
    }

    for (int j = 0; j < num_snodes; j++) {
      // Keep track of the size of the supernode on the diagonal
      int diag_size = snode_size[j];
      T* diag = get_diag_pointer(j);

      // First variable associated with this supernode
      int jfirst_var = snode_to_first_var[j];

      // Set the pointer to the current column indices
      const int* jrows = &rows[colp[j]];
      T* jptr = get_factor_pointer(j, diag_size);

      // Go through the linked list of supernodes to find the super node columns
      // k with non-zero entries in row j
      int k = list[j];
      while (k != -1) {
        // Store the next supernode that we will visit
        int next_k = list[k];
        int kfirst_var = snode_to_first_var[k];

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
        const int* krows = &rows[ip_start];
        T* kvals = get_factor_pointer(k, ksize, ip_start);

        if (jfirst_var >= sqdef_index && kfirst_var < sqdef_index) {
          // We are factoring C and adding a contribution from L so we need to
          // add values, not subtract them here...
          add_diag_update(ksize, nkrows, jfirst_var, krows, kvals, diag_size,
                          diag, work_temp);
        } else {
          // Perform the update to the diagonal by computing
          // diag <- diag - L21 * L21^{T}
          sub_diag_update(ksize, nkrows, jfirst_var, krows, kvals, diag_size,
                          diag, work_temp);
        }

        // Perform the update for the column by computing
        // work_temp = L31 * L21^{T}
        int iremain = ip_end - ip_next;
        compute_work_update(ksize, nkrows,
                            get_factor_pointer(k, ksize, ip_start), iremain,
                            get_factor_pointer(k, ksize, ip_next), work_temp);

        // Add the temporary column to the remainder
        if (jfirst_var >= sqdef_index && kfirst_var < sqdef_index) {
          add_column_update(diag_size, nkrows, jfirst_var, krows, iremain,
                            &rows[ip_next], work_temp, jrows, jptr);
        } else {
          sub_column_update(diag_size, nkrows, jfirst_var, krows, iremain,
                            &rows[ip_next], work_temp, jrows, jptr);
        }

        // Move to the next k non-zero
        k = next_k;
      }

      // Facgtor the diagonal and copy the entries back to the diagonal
      rflag = factor_diag(diag_size, diag);
      if (rflag != 0) {
        rflag += jfirst_var;
        break;
      }

      // Compute (A32 - L31 * L21^{T}) * L22^{-T}
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
   * Solve the system of equations with the Cholesky factorization
   */
  void solve(Vector<T>* x) {
    T* xt = x->get_array();

    // Solve L * x = x
    for (int j = 0; j < num_snodes; j++) {
      const int jsize = snode_size[j];
      T* D = get_diag_pointer(j);
      int jvar = snode_to_first_var[j];
      T* y = &xt[jvar];

      if (jvar < sqdef_index) {
        solve_diag(jsize, D, 1, y);

        // Apply the update from the whole column
        const T* L = get_factor_pointer(j, jsize);

        int ip_end = colp[j + 1];
        for (int ip = colp[j]; ip < ip_end; ip++) {
          T val = 0.0;
          for (int ii = 0; ii < jsize; ii++) {
            val += L[ii] * y[ii];
          }
          xt[rows[ip]] -= val;
          L += jsize;
        }
      } else {
        // This is the C/F part of the matrix
        solve_diag(jsize, D, 1, y);
        for (int k = 0; k < jsize; k++) {
          y[k] *= -1.0;
        }

        // Apply the update from the whole column
        const T* L = get_factor_pointer(j, jsize);

        int ip_end = colp[j + 1];
        for (int ip = colp[j]; ip < ip_end; ip++) {
          T val = 0.0;
          for (int ii = 0; ii < jsize; ii++) {
            val += L[ii] * y[ii];
          }
          xt[rows[ip]] += val;
          L += jsize;
        }
      }
    }

    // Solve L^{T} * x = x
    for (int j = num_snodes - 1; j >= 0; j--) {
      const int jsize = snode_size[j];
      T* y = &xt[snode_to_first_var[j]];
      T* L = get_factor_pointer(j, jsize);

      int ip_end = colp[j + 1];
      for (int ip = colp[j]; ip < ip_end; ip++) {
        for (int ii = 0; ii < jsize; ii++) {
          y[ii] -= L[ii] * xt[rows[ip]];
        }
        L += jsize;
      }

      T* D = get_diag_pointer(j);
      solve_diag_transpose(jsize, D, 1, y);
    }
  }

 private:
  /**
   * @brief Set the values into the matrix.
   *
   * @param Acolp Pointer into the columns
   * @param Arows Row indices of the nonzero entries
   * @param Avals The numerical values
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
          if (i >= sqdef_index && j >= sqdef_index) {
            // This is part of the F matrix - from C.
            // Store +C by taking the negative of the values in the matrix.
            // These values will often be zero or not in the sparsity pattern at
            // all.
            if (i < jfirst + jsize) {
              int jj = j - jfirst;
              int ii = i - jfirst;

              T* D = get_diag_pointer(sj);
              D[get_diag_index(ii, jj)] -= Avals[ip];
            } else {
              int jj = j - jfirst;

              // Look for the row
              for (int kp = colp[sj]; kp < colp[sj + 1]; kp++) {
                if (rows[kp] == i) {
                  T* L = get_factor_pointer(sj, jsize, kp);
                  L[jj] -= Avals[ip];

                  break;
                }
              }
            }
          } else {
            // This is part of the L matrix - from A or B. Add the values into
            // the L matrix components.
            if (i < jfirst + jsize) {
              int jj = j - jfirst;
              int ii = i - jfirst;

              T* D = get_diag_pointer(sj);
              D[get_diag_index(ii, jj)] += Avals[ip];
            } else {
              int jj = j - jfirst;

              // Look for the row
              for (int kp = colp[sj]; kp < colp[sj + 1]; kp++) {
                if (rows[kp] == i) {
                  T* L = get_factor_pointer(sj, jsize, kp);
                  L[jj] += Avals[ip];

                  break;
                }
              }
            }
          }
        }
      }
    }
  }

  /**
   * @brief Build the elimination tree and compute the number of non-zeros in
   * each column.
   *
   * @param Acolp The pointer into each column
   * @param Arows The row indices for each matrix entry
   * @param parent The elimination tree/forest
   * @param Lnz The number of non-zeros in each column
   */
  void build_tree(const int Acolp[], const int Arows[], int parent[],
                  int Lnz[]) {
    int* flag = new int[size];

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
          while (flag[i] != k) {
            if (parent[i] == -1) {
              parent[i] = k;
            }

            // L[k, i] is non-zero
            Lnz[i]++;
            flag[i] = k;

            i = parent[i];
          }
        }
      }
    }

    delete[] flag;
  }

  /**
   * @brief Initialize the supernodes in the matrix
   *
   * The supernodes share the same column non-zero pattern
   *
   * @param parent The elimination tree data
   * @param Lnz The number of non-zeros per variable
   * @param vtn The array of supernodes for each variable
   */
  int init_snodes(const int parent[], const int Lnz[], int vtn[]) {
    int snode = 0;

    // First find the supernodes belonging strictly to L
    int i = 0;
    while (i < sqdef_index) {
      vtn[i] = snode;
      i++;

      while (i < sqdef_index && (parent[i - 1] == i) &&
             Lnz[i] == Lnz[i - 1] - 1) {
        vtn[i] = snode;
        i++;
      }
      snode++;
    }

    // Now find superndoes belonging strictly to C
    while (i < size) {
      vtn[i] = snode;
      i++;

      while (i < sqdef_index && (parent[i - 1] == i) &&
             Lnz[i] == Lnz[i - 1] - 1) {
        vtn[i] = snode;
        i++;
      }
      snode++;
    }

    return snode;
  }

  /**
   * @brief Build the non-zero pattern for the supernodes in the matrix
   *
   * This follows a similar logic to buildForest(), but uses the parent data.
   * This must be called after the supernodes are constructed.
   *
   * @param Acolp The pointer into each column
   * @param Arows The row indices for each matrix entry
   * @param parent The elimination tree/forest
   * @param Lnz The number of non-zeros in each column
   */
  void build_nonzero_pattern(const int Acolp[], const int Arows[],
                             const int parent[], int Lnz[]) {
    int* flag = new int[size];

    for (int k = 0; k < size; k++) {
      flag[k] = k;
      Lnz[k] = 0;

      // Loop over the k-th column
      int ip_end = Acolp[k + 1];
      for (int ip = Acolp[k]; ip < ip_end; ip++) {
        int i = Arows[ip];

        if (i < k) {
          // Scan up the etree
          while (flag[i] != k) {
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
            i = parent[i];
          }
        }
      }
    }

    delete[] flag;
  }

  /**
   * Add the diagonal update
   *
   * Update the entries of the diagonal matrix
   *
   * D <- D - L * L^{T}
   */
  void sub_diag_update(const int lsize, const int nlrows, const int lfirst_var,
                       const int* lrows, T* L, const int diag_size, T* diag,
                       T* work) {
    // Compute L * L^{T}
    int n = nlrows;
    int k = lsize;
    blas_syrk<T>("L", "T", n, k, 1.0, L, k, 0.0, work, n);

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
   * Add the diagonal update
   *
   * Update the entries of the diagonal matrix
   *
   * D <- D + L * L^{T}
   */
  void add_diag_update(const int lsize, const int nlrows, const int lfirst_var,
                       const int* lrows, T* L, const int diag_size, T* diag,
                       T* work) {
    // Compute L * L^{T}
    int n = nlrows;
    int k = lsize;
    blas_syrk<T>("L", "T", n, k, 1.0, L, k, 0.0, work, n);

    // Add D <- D - L * L^{T}
    for (int jj = 0; jj < nlrows; jj++) {
      int j = lrows[jj] - lfirst_var;
      for (int ii = 0; ii < jj + 1; ii++) {
        int i = lrows[ii] - lfirst_var;
        diag[i + j * (j + 1) / 2] += work[jj + nlrows * ii];
      }
    }
  }

  /**
   * @brief Perform a column update into the work column.
   *
   * The column T has dimensions of the number of non-zero rows in L32 by the
   * number of non-zero rows in L21.
   *
   * Given the current factorization at step *, where the matrix is of the form
   *
   * [ L11  0   ]
   * [ L21  *   ]
   * [ L31  Tmp ]
   *
   * Compute the result
   *
   * Tmp = L31 * L21^{T}
   *
   * @param lwidth The number of columns in L21 and L32
   * @param n21rows The number of non-zero rows in L21
   * @param L21 The numerical values of L21 in row-major order
   * @param n31rows The number of non-zero rows in L32
   * @param L31 The numerical values of L32 in row-major order
   * @param Tmp The temporary vector
   */
  void compute_work_update(int lwidth, int n21rows, T* L21, int n31rows, T* L31,
                           T* Tmp) {
    // These matrices are stored in row-major order. To compute the result we
    // use LAPACK with the computation: Tmp^{T} = L21 * L31^{T}
    // dimension of Tmp^{T} is n21rows X n32rows
    // dimension of L21 is n21rows X lwidth
    // dimension of L31^{T} is lwidth X n31rows
    blas_gemm<T>("T", "N", n21rows, n31rows, lwidth, 1.0, L21, lwidth, L31,
                 lwidth, 0.0, Tmp, n21rows);
  }

  /**
   * @brief  Subtract a sparse column update from another sparse column
   *
   *  L32 = L32 - Tmp
   *
   *  The row indices in the input brows must be a subset of the rows in arows.
   *  Both the input arows and brows must be sorted. All indices in arows must
   *  exist in brows.
   *
   *  @param lwidth The width of the L32 column
   *  @param nrows The number of the rows to update
   *  @param lrows The indices of the L32 column
   *  @param A The A values of the column
   *  @param brows The indices of the B column
   *  @param B The B values of the column
   */
  void sub_column_update(const int lwidth, const int nlcols,
                         const int lfirst_var, const int* lrows, int nrows,
                         const int* arows, const T* A, const int* brows, T* B) {
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

  /**
   * @brief Add a sparse column update from another sparse column
   *
   * L32 = L32 + Tmp
   *
   * The row indices in the input brows must be a subset of the rows in arows.
   * Both the input arows and brows must be sorted. All indices in arows must
   * exist in brows.
   *
   * @param lwidth The width of the L32 column
   * @param nrows The number of the rows to update
   * @param lrows The indices of the L32 column
   * @param A The A values of the column
   * @param brows The indices of the B column
   * @param B The B values of the column
   */
  void add_column_update(const int lwidth, const int nlcols,
                         const int lfirst_var, const int* lrows, int nrows,
                         const int* arows, const T* A, const int* brows, T* B) {
    for (int i = 0, bi = 0; i < nrows; i++) {
      while (brows[bi] < arows[i]) {
        bi++;
        B += lwidth;
      }

      for (int jj = 0; jj < nlcols; jj++) {
        int j = lrows[jj] - lfirst_var;
        B[j] += A[jj];
      }
      A += nlcols;
    }
  }

  /**
   * Perform the dense Cholesky factorization of the diagonal components
   */
  int factor_diag(const int diag_size, T* D) {
    int n = diag_size, info;
    lapack_pptrf<T>("U", n, D, &info);
    return info;
  }

  /**
   * Solve L * y = x and output x = y
   */
  int solve_diag(int diag_size, T* L, int nrhs, T* x) {
    int info = 0;
    blas_tptrs<T>("U", "T", "N", diag_size, nrhs, L, x, diag_size, &info);
    return info;
  }

  /**
   * Solve L^{T} * y = x and output x = y
   */
  int solve_diag_transpose(int diag_size, T* L, int nrhs, T* x) {
    int info = 0;
    blas_tptrs<T>("U", "N", "N", diag_size, nrhs, L, x, diag_size, &info);
    return info;
  }

  // Get the diagonal block index
  inline int get_diag_index(const int i, const int j) {
    if (i >= j) {
      return j + i * (i + 1) / 2;
    } else {
      return i + j * (j + 1) / 2;
    }
  }

  // Given the supernode index, return the pointer to the diagonal matrix
  inline T* get_diag_pointer(const int i) { return &data[data_ptr[i]]; }

  // Given the supernode index, the supernode size and the index into the rows
  // data, return the pointer to the lower factor
  inline T* get_factor_pointer(const int i, const int node_size,
                               const int index) {
    const int dsize = node_size * (node_size + 1) / 2;
    const int offset = index - colp[i];
    return &data[data_ptr[i] + dsize + node_size * offset];
  }

  // Given the supernode index, the supernode size and the index into the rows
  // data, return the pointer to the lower factor
  inline T* get_factor_pointer(const int i, const int node_size) {
    const int dsize = node_size * (node_size + 1) / 2;
    return &data[data_ptr[i] + dsize];
  }

  // The matrix
  std::shared_ptr<CSRMat<T>> mat;

  // The dimension of the square matrix
  int size;

  // SQD index indicating the first variable which belongs to the -C matrix
  int sqdef_index;

  // The row indices for the strict lower-diagonal entries of each super node.
  // This does not contain the row indices for the supernode itself. Only
  // entries below the supernode.
  int* rows;

  // Pointer into the row indices for the strict lower block of the
  // matrix. This does not include the row indices for the supernode.
  int* colp;

  // Number of supernodes
  int num_snodes;

  // Supernode sizes - How many consecutive variables belong to this
  // supernode? sum_{i=1}^{num_snodes} snode_size = size
  int* snode_size;

  // Given the variable index, what is the corresponding supernode?
  int* var_to_snode;

  // Given the supernode, what is the first variable in the node?
  int* snode_to_first_var;

  // Given the supernode index, a pointer into the supernode data
  // This is computed as the following for k = 0 ... num_snodes
  // data_ptr[k] = sum_{i = 0}^{k} snode_size[i] * ((snode_size[i] + 1)/2 +
  // colp[i + 1] - colp[i])
  int* data_ptr;

  // Work_size = max(max_{i} (snode_size[i] * (colp[i+1] - colp[i]))
  //                 max_{i} snode_size[i]**2)
  int work_size;

  // The numerical data for all entries size = data_ptr[num_snodes]
  T* data;
};

}  // namespace amigo

#endif  // AMIGO_SPARSE_CHOLESKY_H