#ifndef AMIGO_SPARSE_LDL_H
#define AMIGO_SPARSE_LDL_H

#include "blas_interface.h"
#include "csr_matrix.h"
#include "ordering_utils.h"

namespace amigo {

template <typename T>
class SparseLDL {
 public:
  enum class SolverType { LDL, CHOLESKY };

  /**
   * @brief Construct the data needed for the LDL factorization with the given
   * matrix and perform the symbolic factorization phase
   *
   * Perform a right-looking multifrontal factorization of the provided matrix
   * with super nodes.
   *
   * The 1x1 pivots selected in the factorization phase must satisfy
   *
   * |F[k, k]| >= ustab * max |F[k + 1:, k]|
   *
   * This ensures that the multipliers satsify
   *
   * |F[k + 1, :]| / |F[k, k]| <= 1 / ustab
   *
   * Similarly, the 2x2 pivots selected in the factorization phase satisfy a
   * criterion where delta is defined as
   *
   * 1 / delta =   || [F[k, k],     sym            ]^{-1} ||
   *               || [F[k + 1, k]  F[k + 1, k + 1]]      ||_{1}
   *
   * As a result
   *
   * delta = | det | / max(|F[k, k]| + |F[k+1, k]|, |F[k+1, k+1]| + |F[k+1, k]|)
   *
   * and then subsequently the criterion for acceptance is
   *
   * delta >= ustab * max |F[k + 2:, k]|
   * delta >= ustab * max |F[k + 2:, k + 1]|
   *
   * @param mat The CSR matrix (treated as symmetric)
   * @param solver_type The type of solver (Cholesky or LDL)
   * @param ustab Stability parameter for 1x1 and 2x2 pivots for the LDL
   * @param pivot_tol Pivot tolerance below which pivots are treated as zero
   * @param delay_growth Delayed pivot growth factor (used to estimate memory
   * requirements)
   */
  SparseLDL(std::shared_ptr<CSRMat<T>> mat,
            SolverType solver_type = SolverType::LDL, double ustab = 0.01,
            double pivot_tol = 1e-14, double delay_growth = 2.0,
            OrderingType order = OrderingType::NATURAL)
      : mat(mat),
        solver_type(solver_type),
        ustab(ustab),
        pivot_tol(pivot_tol),
        delay_growth(delay_growth),
        order(order) {
    // Bound the stability factor between [0, 0.5]
    if (ustab > 0.5) {
      ustab = 0.5;
    }
    if (ustab < 0.0) {
      ustab = 0.0;
    }

    // Bound the growth factor
    if (solver_type == SolverType::CHOLESKY) {
      delay_growth = 1.0;
    } else {
      if (delay_growth < 1.0) {
        delay_growth = 1.0;
      }
      if (delay_growth > 10.0) {
        delay_growth = 10.0;
      }
    }

    // Initialize data
    max_frontal_mat_dimension = 0;
    stack_int_estimate = 0;
    stack_nnz_estimate = 0;
    cholesky_int_nnz = 0;
    cholesky_factor_nnz = 0;
    num_snodes = 0;
    invperm = nullptr;
    snode_size = nullptr;
    snode_to_var = nullptr;
    num_children = nullptr;
    max_contrib = 0;
    contrib_ptr = nullptr;
    contrib_rows = nullptr;

    // Get the non-zero pattern
    int nrows;
    const int *rowp, *cols;
    mat->get_data(&nrows, nullptr, nullptr, &rowp, &cols, nullptr);

    // Perform the symbolic anallysis based on the input pattern
    symbolic_analysis(order, nrows, rowp, cols);
  }
  ~SparseLDL() {
    if (invperm) {
      delete[] invperm;
    }
    delete[] snode_size;
    delete[] snode_to_var;
    delete[] num_children;
    delete[] contrib_ptr;
    delete[] contrib_rows;
  }

  /**
   * @brief Perform an LDL^{T} factorization of the matrix
   *
   * @return int The return flag
   */
  int factor() {
    // Get the non-zero pattern
    int nrows, ncols, nnz;
    const int *rowp, *cols;
    const T* data;
    mat->get_data(&nrows, &ncols, &nnz, &rowp, &cols, &data);

    // Perform the numerical factorization
    int info = 0;
    if (solver_type == SolverType::CHOLESKY) {
      info = factor_numeric<SolverType::CHOLESKY>(nrows, rowp, cols, data);
    } else {
      info = factor_numeric<SolverType::LDL>(nrows, rowp, cols, data);
    }
    return info;
  }

  /**
   * @brief Compute the solution of the system of equations
   *
   * L * D * L^{T} * y = x
   *
   * where y <- x. The solution vector overwrites the right-hand-side.
   *
   * @param xvec
   */
  void solve(Vector<T>* xvec) const {
    if (solver_type == SolverType::CHOLESKY) {
      solve_cholesky(xvec);
    } else {  // solver_type == SolverType::LDL
      solve_ldl(xvec);
    }
  }

  /**
   * @brief Get the inertia of the matrix based on the factorization
   *
   * @param npos Number of positive eigenvalues
   * @param nneg Number of negative eigenvalues
   */
  void get_inertia(int* npos, int* nneg) const {
    *npos = 0;
    *nneg = 0;
    if (solver_type == SolverType::LDL) {
      for (int ks = 0; ks < num_snodes; ks++) {
        add_pivot_inertia(ks, npos, nneg);
      }
    }
  }

 private:
  /**
   * @brief The contribution stack object used for the factorization
   */
  class ContributionStack {
   public:
    ContributionStack(int max_idx, int max_work)
        : idx(max_idx), work(max_work) {
      top_idx = 0;
      top_work = 0;
    }
    ~ContributionStack() {}

    /**
     * @brief Add the delayed pivots to the list of indices/vars
     *
     * @param nchildren Number of chiledren to look back at
     * @param fully_summed Initial number of fully summed variables
     * @param front_indices Indices in the front matrix
     * @param front_vars Front variables
     * @return Number of fully summed delayed pivots
     */
    int add_delayed_pivots(int nchildren, int fully_summed, int front_indices[],
                           int front_vars[]) {
      // Peak at the nchildren top entries
      int tmp_top = top_idx;

      for (int k = 0; k < nchildren; k++) {
        int num_delayed_pivots = idx[tmp_top - 2];
        int contrib_size = idx[tmp_top - 1];
        int* vars = &idx[tmp_top - 2 - contrib_size];

        for (int j = 0; j < num_delayed_pivots; j++) {
          int delayed = vars[j];
          if (front_indices[delayed] == -1) {
            front_indices[delayed] = fully_summed;
            front_vars[fully_summed] = delayed;
            fully_summed++;
          }
        }

        tmp_top -= (2 + contrib_size);
      }

      return fully_summed;
    }

    /**
     * @brief Push a contribution block onto the stack
     *
     * The matrix is arranged like this:
     *
     * F = [ F11  F12 ]
     *     [ F21  C   ]
     *
     * F11 is size num_pivots x num_pivots
     * C is size (front_size - num_pivots)
     *
     * @param num_pivots Number of columns selected as pivots
     * @param num_delayed_pivots Number of delayed pivots
     * @param front_size Size of the front matrix F
     * @param vars Variables on the front matrix
     * @param F The front matrix values
     */
    void push(int num_pivots, int num_delayed_pivots, int front_size,
              const int vars[], const T F[]) {
      // Check the size of the integer storage on the stack
      int contrib_size = front_size - num_pivots;
      if (top_idx + 2 + contrib_size > int(idx.size())) {
        idx.resize(int(top_idx + 2 + contrib_size + 0.5 * idx.size()));
      }

      // Copy the values and set the values
      std::copy(vars + num_pivots, vars + front_size, &idx[top_idx]);
      top_idx += contrib_size;

      // Save the delayed pivots and size of the contribution block
      idx[top_idx] = num_delayed_pivots;
      idx[top_idx + 1] = contrib_size;
      top_idx += 2;

      // Check the size of the contribution block
      int block_size = contrib_size * (contrib_size + 1) / 2;
      if (top_work + block_size > int(work.size())) {
        work.resize(int(top_work + block_size + 0.5 * work.size()));
      }

      // Copy the values into the data array
      T* ptr = &work[top_work];
      for (int j = num_pivots; j < front_size; j++) {
        for (int i = j; i < front_size; i++, ptr++) {
          ptr[0] = F[i + front_size * j];
        }
      }
      top_work += block_size;
    }

    /**
     * @brief Pop a contribution block from the top of the stack
     *
     * @param num_delayed_pivots Number of delayed pivots
     * @param contrib_size Contribution block size
     * @param vars Indices for the contribution block
     * @param C The contribution block values
     */
    void pop(int* num_delayed_pivots, int* contrib_size, int* vars[], T* C[]) {
      *num_delayed_pivots = idx[top_idx - 2];
      int cb_size = idx[top_idx - 1];
      *contrib_size = cb_size;
      *vars = &idx[top_idx - 2 - cb_size];
      top_idx -= (2 + cb_size);

      int block_size = cb_size * (cb_size + 1) / 2;
      *C = &work[top_work - block_size];
      top_work -= block_size;
    }

   private:
    int top_idx;           // Top of the index stack
    std::vector<int> idx;  // Index/size values
    int top_work;          // Top of the entry stack
    std::vector<T> work;   // Entries in the matrix
  };

  /**
   * @brief Store the factored contributions from the matrix
   */
  class MatrixFactor {
   public:
    MatrixFactor() {
      num_snodes = 0;
      max_pivots = 0;
      max_delayed = 0;

      int_size = 0;
      factor_size = 0;
    }

    /**
     * @brief Allocate the space for the factored matrix
     *
     * The integer space consists of the number of pivots + delayed for each
     * super node. The number of non-zeros consists of the non-zeros in the
     * (L11, L21) combined pivot, delayed and contribution blocks. These sizes
     * are automatically re-allocated if the prediction is wrong.
     *
     * @param num_super_nodes Number of super nodes
     * @param int_nnz Number of expected integers
     * @param factor_nnz Number of
     */
    void allocate(int num_super_nodes, int int_nnz, int factor_nnz) {
      num_snodes = num_super_nodes;
      max_pivots = 0;
      max_delayed = 0;

      // Set all of the values to an empty node
      meta.assign(num_snodes, NodeMeta{});

      // Preallocation
      int_data.resize(int_nnz);
      factor_data.resize(factor_nnz);
    }

    /**
     * @brief Clear the data before factoring the matrix
     */
    void clear() {
      int_size = 0;
      factor_size = 0;
      int_data.clear();
      factor_data.clear();
    }

    /**
     * @brief Add contributions from the factors
     *
     * @param ks The supernode index
     * @param num_pivots The number of pivots for this node
     * @param pivots The pivot variable numbers
     * @param num_delayed The number of delayed pivots
     * @param delayed The delayed pivot indices
     * @param contrib_size The contribution block size
     * @param L The factor entries
     * @param num_ipiv Number of ipiv entries
     * @param ipiv Entries of ipiv (if any)
     */
    void add_factor(int ks, int num_pivots, const int pivots[], int num_delayed,
                    const int delayed[], int contrib_size, const T L[],
                    int num_ipiv = 0, const int ipiv[] = nullptr) {
      const int nrows = num_pivots + num_delayed + contrib_size;
      const int block_size = nrows * num_pivots;

      NodeMeta& m = meta[ks];
      m.num_pivots = num_pivots;
      m.num_delayed = num_delayed;
      m.num_ipiv = num_ipiv;

      if (num_delayed > max_delayed) {
        max_delayed = num_delayed;
      }
      if (num_pivots > max_pivots) {
        max_pivots = num_pivots;
      }

      m.int_offset = int_size;
      m.factor_offset = factor_size;

      // Check if we need to resize the vector
      int new_int_size = num_pivots + num_delayed + num_ipiv;
      if (int_size + new_int_size > int(int_data.size())) {
        int_data.resize(int(int_size + new_int_size + 0.5 * int_data.size()));
      }

      // Insert the data into the stored factorization
      int* iptr = int_data.data();
      std::copy(pivots, pivots + num_pivots, iptr + int_size);
      int_size += num_pivots;
      std::copy(delayed, delayed + num_delayed, iptr + int_size);
      int_size += num_delayed;
      std::copy(ipiv, ipiv + num_ipiv, iptr + int_size);
      int_size += num_ipiv;

      // Check if we need to to resize the factor data vector
      if (factor_size + block_size > int(factor_data.size())) {
        factor_data.resize(
            int(factor_size + block_size + 0.5 * factor_data.size()));
      }

      // Insert the data into the stored factorization
      T* ptr = factor_data.data();
      std::copy(L, L + block_size, ptr + factor_size);
      factor_size += block_size;
    }

    /**
     * @brief For the root factorization, get the raw pointers to reserved space
     * in the factorization.
     *
     * This data must be filled in by the factorization code!
     *
     * @param ks The supernode index
     * @param num_pivots The number of pivots for this node
     * @param pivots The pivot variable numbers
     * @param L The factor entries
     * @param ipiv Entries of ipiv
     */
    void reserve_factor_root(int ks, int num_pivots, int* pivots[], T* L[],
                             int* ipiv[]) {
      int block_size = num_pivots * num_pivots;

      NodeMeta& m = meta[ks];
      m.num_pivots = num_pivots;
      m.num_delayed = 0;
      m.num_ipiv = num_pivots;

      if (num_pivots > max_pivots) {
        max_pivots = num_pivots;
      }

      m.int_offset = int_size;
      m.factor_offset = factor_size;

      // Check the integer space needed
      int new_int_size = 2 * num_pivots;
      if (int_size + new_int_size > int(int_data.size())) {
        int_data.resize(int(int_size + new_int_size));
      }

      // Check if we need to to resize the factor data vector
      if (factor_size + block_size > int(factor_data.size())) {
        factor_data.resize(int(factor_size + block_size));
      }

      // Increase the size of the offsets
      factor_size += block_size;
      int_size += 2 * num_pivots;

      int* iptr = int_data.data();
      if (pivots) {
        *pivots = &iptr[m.int_offset];
      }
      if (ipiv) {
        *ipiv = &iptr[m.int_offset + num_pivots];
      }

      T* ptr = factor_data.data();
      if (L) {
        *L = &ptr[m.factor_offset];
      }
    }

    /**
     * @brief Get the factor for the specified super node
     *
     * @param ks The supernode index
     * @param num_pivots The number of pivots for this node
     * @param pivots The pivot variable numbers
     * @param num_delayed The number of delayed pivots
     * @param delayed The delayed pivot indices
     * @param L The factor entries
     */
    void get_factor(int ks, int* num_pivots, const int* pivots[],
                    int* num_delayed, const int* delayed[], const T* L[],
                    int* num_ipiv = nullptr,
                    const int* ipiv[] = nullptr) const {
      const NodeMeta& m = meta[ks];
      if (num_pivots) {
        *num_pivots = m.num_pivots;
      }
      if (num_delayed) {
        *num_delayed = m.num_delayed;
      }
      if (num_ipiv) {
        *num_ipiv = m.num_ipiv;
      }

      // Pivots must always be defined
      if (pivots) {
        *pivots = &int_data[m.int_offset];
      }

      // Delayed pivots may or may not be defined
      if (delayed) {
        if (m.num_delayed == 0) {
          *delayed = nullptr;
        } else {
          *delayed = &int_data[m.int_offset + m.num_pivots];
        }
      }

      // L must always be defined
      if (L) {
        *L = &factor_data[m.factor_offset];
      }

      // ipiv may not be defined
      if (ipiv) {
        if (m.num_ipiv == 0) {
          *ipiv = nullptr;
        } else {
          *ipiv = &int_data[m.int_offset + m.num_pivots + m.num_delayed];
        }
      }
    }

    /**
     * @brief Get the max pivots for any super node
     *
     * @return int
     */
    int get_max_pivots() const { return max_pivots; }

    /**
     * @brief Get the max delayed pivots for any super node
     *
     * @return int
     */
    int get_max_delayed() const { return max_delayed; }

    /**
     * @brief Get the number of nonzeros in the factor
     *
     * @param int_nnz The size of the integer vector
     * @param factor_nnz The size of the factor vector
     */
    void get_num_nonzeros(int* int_nnz, int* factor_nnz) const {
      if (int_nnz) {
        *int_nnz = int_size;
      }
      if (factor_nnz) {
        *factor_nnz = factor_size;
      }
    }

   private:
    struct NodeMeta {
      int num_pivots;
      int num_delayed;
      int num_ipiv;
      int int_offset;
      int factor_offset;
    };

    int num_snodes;
    int max_pivots;
    int max_delayed;

    std::vector<NodeMeta> meta;
    int int_size;
    std::vector<int> int_data;  // pivots and delayed indices
    int factor_size;
    std::vector<T> factor_data;  // all L blocks
  };

  /**
   * @brief Perform the multifrontal factorization
   *
   * The overall algorithm is the following:
   *
   * for each supernode:
   *   Find the frontal variables from the children
   *   Assemble the frontal matrix from the children
   *   Factor the frontal matrix
   *   Store the factorization pieces
   *   Push the contribution onto the stack
   *
   * @tparam stype The type of factorization (LDL or Cholesky)
   * @param ncols The number of columns
   * @param colp Pointer into the column
   * @param rows Row indices for each column
   * @param data Values for the matrix entries
   * @return int Return flag (0 for success)
   */
  template <SolverType stype>
  int factor_numeric(const int ncols, const int colp[], const int rows[],
                     const T data[]) {
    // Clear any old factorization data
    fact.clear();

    // Allocate space for indices
    int* temp = new int[2 * ncols];
    std::fill(temp, temp + 2 * ncols, -1);
    int* front_indices = temp;       // Indices in the front matrix
    int* front_vars = &temp[ncols];  // Variables in the front

    // Compute proper size for the frontal matrix
    const int nblock = 64;  // TODO: Optimize this block size?
    int fdim = int(delay_growth * max_frontal_mat_dimension);
    std::vector<T> F(fdim * fdim);
    std::vector<T> W;
    if constexpr (stype == SolverType::LDL) {
      W.resize(fdim * nblock);
    }

    // Use estimates of the contribution stack sizes
    int int_estimate = int(delay_growth * stack_int_estimate);
    int nnz_estimate = int(delay_growth * stack_nnz_estimate);
    ContributionStack stack(int_estimate, nnz_estimate);

    // Info flag
    int info = 0;
    int ns = 0;
    for (int ks = 0, k = 0; ks < num_snodes; k += ns, ks++) {
      // Size of the super node
      ns = snode_size[ks];

      // Number of children for this super node
      int nchildren = num_children[ks];

      // Get the frontal variables
      int fully_summed = 0, front_size = 0;
      get_frontal_vars(ks, k, ns, nchildren, stack, front_indices, front_vars,
                       &fully_summed, &front_size);

      // Resize the frontal matrix if needed (this will only happen if stype ==
      // SolverType::LDL)
      if (front_size > fdim) {
        fdim = front_size;
        F.resize(fdim * fdim);
        W.resize(fdim * nblock);
      }

      // Get the underlying array
      T* Fptr = F.data();

      // Assemble the frontal matrix
      assemble_front_matrix(k, ns, front_size, front_indices, colp, rows, data,
                            nchildren, stack, Fptr);

      // Factor the frontal matrix and save the results
      int info = 0;
      if constexpr (stype == SolverType::CHOLESKY) {
        // The Cholesky code works for both frontal and root matrices
        info = factor_front_matrix_cholesky(ks, fully_summed, front_size,
                                            front_vars, Fptr, stack, fact);
      } else {
        if (fully_summed < front_size) {
          T* Wptr = W.data();
          info = factor_front_matrix_block(ks, fully_summed, front_size,
                                           front_vars, Fptr, nblock, Wptr,
                                           stack, fact);
        } else {
          info = factor_root_matrix(ks, front_size, front_vars, Fptr, fact);
        }
      }

      // Check the flag
      if (info != 0) {
        info += k;
        break;
      }

      // Reset the front indices back to -1
      for (int j = 0; j < front_size; j++) {
        int var = front_vars[j];
        if (var < 0) {
          var = -var - 1;
        }
        front_indices[var] = -1;
      }
    }

    // Clean up the data
    delete[] temp;

    return info;
  }

  /**
   * @brief Get the variables for this front
   *
   * @param ks The super nodal index ks
   * @param k The offset into the super node variable list
   * @param ns The size of the super node
   * @param nchildren the number of children for this super node
   * @param stack The stack of contribution blocks
   * @param front_indices The front indices
   * @param front_vars The variables on the front
   * @param fully_summed Number of fully summed variables
   * @param front_size The front size
   */
  void get_frontal_vars(const int ks, const int k, const int ns,
                        const int nchildren, ContributionStack& stack,
                        int front_indices[], int front_vars[],
                        int* fully_summed, int* front_size) {
    // Set the ordering of the degrees of freeom in the front
    // Number of fully summed contributions (supernode pivots + delayed
    // pivots)
    for (int j = 0; j < ns; j++) {
      int var = snode_to_var[k + j];
      front_indices[var] = j;
      front_vars[j] = var;
    }

    // Add the additional contributions from the delayed pivots
    int full_sum =
        stack.add_delayed_pivots(nchildren, ns, front_indices, front_vars);

    // Get the entries predicted from Cholesky
    int start = contrib_ptr[ks];
    int contrib_size = contrib_ptr[ks + 1] - start;
    for (int j = 0, *row = &contrib_rows[start]; j < contrib_size; j++, row++) {
      front_indices[*row] = full_sum + j;
      front_vars[full_sum + j] = *row;
    }

    // Get the size of the front
    *fully_summed = full_sum;
    *front_size = full_sum + contrib_size;
  }

  /**
   * @brief Assemble the frontal matrix associated with the delayed pivots and
   * super node entries
   *
   * @param k Offset into the super node list
   * @param ns Number of variables in this super node
   * @param front_size Front size
   * @param front_indices Front indices
   * @param colp Pointer into the column
   * @param rows Row indices
   * @param data Entries from the matrix
   * @param nchildren Number of children in the etree for this super node
   * @param stack Contribution stack
   * @param F The frontal matrix
   */
  void assemble_front_matrix(const int k, const int ns, int front_size,
                             const int front_indices[], const int colp[],
                             const int rows[], const T data[],
                             const int nchildren, ContributionStack& stack,
                             T F[]) {
    std::fill(F, F + front_size * front_size, 0.0);

    if (order == OrderingType::NATURAL) {
      // Assemble the contributions into F from the matrix. Since the ordering
      // is natural, it's straightforward to assemble only the entries below the
      // diagonal of F that are required
      for (int j = 0; j < ns; j++) {
        // Get the column variable associated with the snode
        int var = snode_to_var[k + j];
        T* Fj = &F[front_size * j];

        for (int ip = colp[var]; ip < colp[var + 1]; ip++) {
          // Get the original row index
          int i = rows[ip];
          if (i >= var) {
            // Get the front index
            int ifront = front_indices[i];

            // Add the contribution to the frontal matrix
            Fj[ifront] += data[ip];
          }
        }
      }
    } else {
      // Assemble the contributions into F from the matrix. Since the ordering
      // is permuted, we assemble the lower and upper part of F.
      for (int j = 0; j < ns; j++) {
        // Get the column variable associated with the snode
        int var = snode_to_var[k + j];
        int pj = invperm[var];
        T* Fj = &F[front_size * j];

        for (int ip = colp[var]; ip < colp[var + 1]; ip++) {
          // Get the original row index
          int i = rows[ip];
          int pi = invperm[i];

          // Get the front index
          int ifront = front_indices[i];
          if (ifront >= 0 && pi >= pj) {
            // Add the contribution to the frontal matrix
            Fj[ifront] += data[ip];
          }
        }
      }
    }

    // Add the contributions from the children
    for (int child = 0; child < nchildren; child++) {
      int num_delayed_pivots;
      int contrib_size;
      int* contrib_indices;
      T* C;
      stack.pop(&num_delayed_pivots, &contrib_size, &contrib_indices, &C);

      // Get the indices of the contribution indices, so we don't have to
      // reference these again
      for (int i = 0; i < contrib_size; i++) {
        contrib_indices[i] = front_indices[contrib_indices[i]];
      }

      // The delayed pivots may be ordered in any way because of the pivoting
      for (int j = 0; j < num_delayed_pivots; j++) {
        const int jfront = contrib_indices[j];

        // Set the offset into the contribution block. The diagonal C element
        // for column j is at index n * j - j * (j - 1)/2, but we subtract j
        // from this since the row index begins at i = j. This accounts for
        // indexing Cj using i directly.
        const int cjindex = j * contrib_size - j * (j + 1) / 2;
        const T* Cj = &C[cjindex];

        // Set the column and row pointers into the frontal matrix
        T* Fj = &F[front_size * jfront];
        T* Fi = &F[jfront];

        // Make sure that the contributions go into the lower part of F
        for (int i = j; i < contrib_size; i++) {
          const int ifront = contrib_indices[i];
          if (ifront >= jfront) {
            Fj[ifront] += Cj[i];
          } else {
            Fi[front_size * ifront] += Cj[i];
          }
        }
      }

      // The remaining contributions are sorted, so we don't have to check if
      // these contributions - they are from the lower block
      for (int j = num_delayed_pivots; j < contrib_size; j++) {
        const int jfront = contrib_indices[j];

        // Set the offset into the contribution block. This accounts for
        // indexing Cj using i directly
        const int cjindex = j * contrib_size - j * (j + 1) / 2;
        const T* Cj = &C[cjindex];

        // Set the column pointer into the frontal matrix
        T* Fj = &F[front_size * jfront];

        // Add contributions to the lower part of F
        for (int i = j; i < contrib_size; i++) {
          const int ifront = contrib_indices[i];
          Fj[ifront] += Cj[i];
        }
      }
    }
  }

  /**
   * @brief Perform the trailing update for the matrix
   *
   * C = C - A * B^{T}
   *
   * @param ndim Number of rows of C and A and number of columns of C and B
   * @param kdim Number of columns of A and number of rows of B
   * @param A Pointer to the A entries
   * @param lda Leading dimension of A
   * @param B Pointer to the B entries
   * @param ldb Leading dimension of B
   * @param C Pointer to the C entries
   * @param ldc leading dimension of C
   */
  void frontal_trailing_update(int ndim, int kdim, const T* A, int lda,
                               const T* B, int ldb, T* C, int ldc) const {
    // Apply the regular update
    if (ndim < 32) {
      blas_gemm<T>("N", "T", ndim, ndim, kdim, -1.0, A, lda, B, ldb, 1.0, C,
                   ldc);
    } else {
      // [ C11  0   ] -= [ A11 ] [ B11  B12 ]
      // [ C21  C22 ]    [ A21 ]

      // Try just splitting this into 3 blocks to see the performance
      int n1 = ndim / 2;
      int n2 = ndim - n1;

      // Compute C21 = C21 - A21 * B11^{T}
      blas_gemm<T>("N", "T", n2, n1, kdim, -1.0, &A[n1], lda, B, ldb, 1.0,
                   &C[n1], ldc);

      // Compute C11 = C11 - A11 * B11
      frontal_trailing_update(n1, kdim, A, lda, B, ldb, C, ldc);

      // Compute C22 = C22 - A21 * B12
      frontal_trailing_update(n2, kdim, &A[n1], lda, &B[n1], ldb,
                              &C[n1 * (ldc + 1)], ldc);
    }
  }

  /**
   * @brief Factor a root matrix using sytrf
   *
   * @param ks The index of the super node
   * @param front_size The front size
   * @param front_vars The front variables
   * @param F The final frontier matrix (or one of several)
   * @param factor The factor
   */
  int factor_root_matrix(const int ks, const int front_size,
                         const int front_vars[], T F[], MatrixFactor& factor) {
    int n = front_size;

    // Get space to fill in
    int *vars, *ipiv;
    T* L;
    factor.reserve_factor_root(ks, n, &vars, &L, &ipiv);

    // The work space size = matrix size
    T* work = L;
    int lwork = n * n;

    // Factor the matrix
    int info;
    lapack_sytrf<T>("L", n, F, n, ipiv, work, lwork, &info);

    // Copy the data into the factorization storage
    std::copy(front_vars, front_vars + n, vars);
    std::copy(F, F + n * n, L);

    return info;
  }

  /**
   * @brief Perform a symmetric swap of the entries within the frontal matrix.
   *
   * This code assumes k < c!
   *
   * This performs the swap operation on the variables and numerical entries in
   * F. The variables are swapped using
   *
   * vars[[k, c]] = vars[[c, k]]
   *
   * The numerical values in F are swapped using
   *
   * F[[k, c], :] = F[[c, k], :] and F[:, [k, c]] = F[:, [c, k]]
   *
   * @param vars The variable numbers
   * @param F The entries of the frontal matrix
   * @param ldf The leading dimension of the frontal factorization
   * @param k The current column in the frontal factorization
   * @param c The pivot to swap with
   */
  void symm_swap(int vars[], T F[], int ldf, int k, int c) const {
    // Swap the variable indices
    int t = vars[k];
    vars[k] = vars[c];
    vars[c] = t;

    // Swap entries F[k, 0:k] with F[c, 0:k]
    blas_swap<T>(k, &F[k], ldf, &F[c], ldf);

    // Swap diagonal entries
    T tmp = F[k + k * ldf];
    F[k + k * ldf] = F[c + c * ldf];
    F[c + c * ldf] = tmp;

    // Swap F[j, k] <-> F[c, j] for k < j < c
    if (c > k + 1) {
      blas_swap<T>(c - (k + 1), &F[k + 1 + k * ldf], 1, &F[c + (k + 1) * ldf],
                   ldf);
    }

    // Swap entries below both rows/cols: F[j, k] <-> F[j, c] for j > c
    blas_swap<T>(ldf - (c + 1), &F[c + 1 + k * ldf], 1, &F[c + 1 + c * ldf], 1);
  }

  /**
   * @brief Factor the frontal matrix and add the results to the stack and
   * matrix factor.
   *
   * Since this is a frontal matrix we will always have strictly that the number
   * of fully summed variables will be less than the front size. So that
   * fully_summed < front_size. (The root factorization takes care of any roots
   * in the elimination tree separately.)
   *
   * After factorization, the contribution blocks are pushed on the stack and we
   * store the pivots and factorization pieces
   *
   * @param ks The index of the super node
   * @param fully_summed The number of fully summed equations
   * @param front_size The front size
   * @param front_vars The front variables
   * @param F The frontal matrix itself
   * @param nblock The block size for the panel
   * @param W The work array dimension (front_size x nblock)
   * @param stack The stack for the contribution blocks
   * @param factor The factor contributions
   */
  int factor_front_matrix_block(const int ks, const int fully_summed,
                                const int front_size, int front_vars[], T F[],
                                const int nblock, T W[],
                                ContributionStack& stack,
                                MatrixFactor& factor) {
    const int ldf = front_size;
    const int ldw = front_size;
    const int n = front_size;
    int num_candidates = fully_summed;
    int contrib_size = front_size - fully_summed;

    for (int k = 0; k < num_candidates;) {
      int kstart = k;  // When we entered the panel factorization
      int kw = 0;      // Block index within the panel

      // Assemble a panel for factorization. We loop over the panel with
      // kw < nblock - 1 in case there are 2x2 pivots encountered
      for (; kw < nblock - 1 && k < num_candidates;) {
        int step = 0;
        int kpiv = -1;  // No pivot selected yet

        // Copy W[:, kw] = F[:, k]
        blas_copy<T>(n - k, &F[k * (ldf + 1)], 1, &W[k + kw * ldw], 1);

        // Compute W[k:, kw] -= F[k:, kstart:k] @ W[k, :kw]
        if (kw > 0) {
          blas_gemv<T>("N", n - k, kw, -1.0, &F[k + kstart * ldf], ldf, &W[k],
                       ldw, 1.0, &W[k + kw * ldw], 1);
        }

        // Get the absolute value of the diagonal element
        double abswkk = std::abs(W[k + kw * ldw]);

        // Find the row entry in column k with the maximum absolute value.
        // Distinguish between the maximum over max(|W[k+1:nc, kw]|) and
        // max(|W[nc:n, kw]|).
        const int nc = num_candidates;

        double abswik = 0.0;
        int imax = 0;
        if (k + 1 < nc) {
          // Get the location of the maximum absolute entry within the candidate
          // set of pivots
          imax = k + 1 + blas_imax<T>(nc - (k + 1), &W[k + 1 + kw * ldw], 1);

          // This will be the off-diagonal of a 2x2 pivot candidate for columns
          // k and k + 1 in W
          abswik = std::abs(W[imax + kw * ldw]);
        }

        // Number of fully summed candidate is always strictly less than n
        int ismax = nc + blas_imax<T>(n - nc, &W[nc + kw * ldw], 1);

        // Maximum absolute value of any entry in the column
        double colmax = std::max(abswik, std::abs(W[ismax + kw * ldw]));

        // Check if this pivot is large enough and satisfies the stability
        // criterion. If so, we don't need to do any more work
        if (abswkk >= pivot_tol && abswkk >= ustab * colmax) {
          step = 1;
          kpiv = k;
        } else if (k + 1 < nc) {
          // Note that we have to have k + 1 < nc since otherwise there are no
          // candidate pivots remaining

          // The initial pivot criterion failed. Try a different pivot. To
          // construct the candidate pivot, copy the column/row imax which
          // is the max absolute value below the diagonal but within the
          // candidate pivot block into the temporary column W[k:, kw + 1]

          // First part of the column W[k:imax, k+1] comes from
          // F[imax, k:imax]
          blas_copy<T>(imax - k, &F[imax + k * ldf], ldf,
                       &W[k + (kw + 1) * ldw], 1);

          // Second part of the column W[imax + 1:, k + 1] = F[imax + 1:, imax]
          blas_copy<T>(n - imax, &F[imax + imax * ldf], 1,
                       &W[imax + (kw + 1) * ldw], 1);

          // Compute the update for the column W[k:, k + 1]
          // Compute W[k:, kw + 1] -= F[k:, kstart:k] @ W[k, :kw]
          if (kw > 0) {
            blas_gemv<T>("N", n - k, kw, -1.0, &F[k + kstart * ldf], ldf,
                         &W[imax], ldw, 1.0, &W[k + (kw + 1) * ldw], 1);
          }

          // Compute the maximum entry in the candidate column (excluding the
          // pivot imax) Look at the max absolute value over W[k:imax, k + 1]
          int jmax = k + blas_imax<T>(imax - k, &W[k + (kw + 1) * ldw], 1);
          double rowmax = std::abs(W[jmax + (kw + 1) * ldw]);

          // Look at the max absolute value over W[imax+1:n, k + 1]
          if (imax + 1 < n) {
            jmax =
                imax + 1 +
                blas_imax<T>(n - (imax + 1), &W[imax + 1 + (kw + 1) * ldw], 1);
            rowmax = std::max(rowmax, std::abs(W[jmax + (kw + 1) * ldw]));
          }

          // The absolute value of the candidate pivot stored in the assembled
          // column W[k:, k + 1]
          double abswii = std::abs(W[imax + (kw + 1) * ldw]);

          // Check if we can use imax as a regular pivot
          if (abswii >= pivot_tol && abswii >= ustab * rowmax) {
            // Copy W[k:, kw] = W[k:, kw + 1]
            blas_copy<T>(n - k, &W[k + (kw + 1) * ldw], 1, &W[k + kw * ldw], 1);

            step = 1;
            kpiv = imax;
          } else {
            // A candidate 2x2 pivot is the matrix is
            // [ W[k, kw],                  ]
            // [ W[imax, kw], W[imax, imax] ]
            // Note that the column for imax is currently stored
            // in W[:, kw + 1]
            double wkk = W[k + kw * ldw];
            double wik = W[imax + kw * ldw];
            double wii = W[imax + (kw + 1) * ldw];
            double det = std::abs(wkk * wii - wik * wik);

            if (det < pivot_tol) {
              // We can't use this as a pivot because the determinant is too
              // small, delay factoring either column
              step = 0;
            } else {
              // Check if this works as a 2x2 pivot
              double delta = det / std::max(abswii + abswik, abswkk + abswik);

              // Use the 2x2 Duff pivot test
              if (delta >= ustab * colmax && delta >= ustab * rowmax) {
                // This 2x2 pivot passes the test
                step = 2;
                kpiv = imax;
              } else {
                // These steps have failed. We need to postpone this pivot
                step = 0;
              }
            }
          }
        }

        // No acceptable pivots. Delay this pivot candidate
        if (step == 0) {
          // Pivot k to the end
          if (k < num_candidates - 1) {
            symm_swap(front_vars, F, ldf, k, num_candidates - 1);

            // Swap the rows of the corresponding entries in W
            blas_swap<T>(kw, &W[k], ldw, &W[num_candidates - 1], ldw);
          }

          num_candidates -= 1;
          continue;
        }

        // We're taking a pivot from this k location
        int ksrc = k + step - 1;

        // Pivot to place the entries in the correct location
        if (kpiv != ksrc) {
          // Swap the rows and columns of F
          symm_swap(front_vars, F, ldf, ksrc, kpiv);

          // Swap the rows of the corresponding entries in W
          blas_swap<T>(kw + step, &W[ksrc], ldw, &W[kpiv], ldw);
        }

        // Now, since we've pivoted, the inverse and determinant are well
        // defined. Compute the update F[k + step:, k:k + step] =
        // W[k + step:, k: k + step] * D^{-1} and store F[k:step, k:step] = D.
        if (step == 1) {
          // 1x1 pivot
          // Copy F[k:, k] = W[k:, kw]
          blas_copy<T>(n - k, &W[k + kw * ldw], 1, &F[k * (ldf + 1)], 1);

          // Compute the inverse of the diagonal
          T dinv = 1.0 / F[k * (ldf + 1)];

          // Scale the column of F[k + 1:, k] *= dinv
          blas_scal<T>(n - k - 1, dinv, &F[1 + k * (ldf + 1)], 1);
        } else if (step == 2) {
          // Extract the entries of the D matrix
          T d11 = W[k + kw * ldw];
          T d21 = W[k + 1 + kw * ldw];
          T d22 = W[k + 1 + (kw + 1) * ldw];

          // Set the values into F
          F[k * (ldf + 1)] = d11;
          F[k * (ldf + 1) + 1] = d21;
          F[(k + 1) * (ldf + 1)] = d22;

          // Compute the inverse of the 2x2 entries
          T detinv = 1.0 / (d11 * d22 - d21 * d21);
          T Dinv[4];
          Dinv[0] = d22 * detinv;
          Dinv[1] = Dinv[2] = -d21 * detinv;
          Dinv[3] = d11 * detinv;

          blas_gemm<T>("N", "N", n - (k + 2), 2, 2, 1.0, &W[k + 2 + kw * ldw],
                       ldw, Dinv, 2, 0.0, &F[k + 2 + k * ldf], ldf);

          // Mark these variables as 2x2 pivots
          front_vars[k] = -(front_vars[k] + 1);
          front_vars[k + 1] = -(front_vars[k + 1] + 1);
        }

        k += step;
        kw += step;
      }

      // Apply the trailing update
      // F[k:, k:] -= F[k:, kstart:kstart + kdim] @ W[k:, :kdim]^{T}
      int ndim = front_size - k;
      int kdim = kw;
      frontal_trailing_update(ndim, kdim, &F[k + kstart * ldf], ldf, &W[k], ldw,
                              &F[k * (ldf + 1)], ldf);
    }

    // The candidates have now been reduced to the number of pivots
    int num_pivots = num_candidates;

    // The number of delayed pivots is the difference between the fully summed
    // candidates we started from and the pivots that were successful
    int num_delayed = fully_summed - num_pivots;

    // Push this onto the stack
    stack.push(num_pivots, num_delayed, front_size, front_vars, F);

    // Push the factored matrix onto the stack
    const int* pivots = front_vars;
    const int* delayed = &front_vars[num_pivots];
    factor.add_factor(ks, num_pivots, pivots, num_delayed, delayed,
                      contrib_size, F);

    return 0;
  }

  /**
   * @brief Solve the system of equations L * x = b
   *
   * On input x = b on output the solution is stored in x
   *
   * @param n Number of rows/columns in L
   * @param piv Pivots (used to detect 1x1 and 2x2 pivots)
   * @param L The entries in the matrix
   * @param ldl The leading dimension of L
   * @param x The right hand side on input and the solution output
   */
  void solve_pivot_lower(const int n, const int piv[], const T L[],
                         const int ldl, T x[]) const {
    for (int j = 0; j < n;) {
      if (piv[j] >= 0) {
        for (int i = j + 1; i < n; i++) {
          x[i] -= L[i + ldl * j] * x[j];
        }
        j += 1;
      } else {
        for (int i = j + 2; i < n; i++) {
          x[i] -= L[i + ldl * j] * x[j] + L[i + ldl * (j + 1)] * x[j + 1];
        }
        j += 2;
      }
    }
  }

  /**
   * @brief Solve the equations D * x = b where D is stored in L
   *
   * On input x = b on output the solution is stored in x
   *
   * @param n Number of rows/columns in L
   * @param piv Pivots (used to detect 1x1 and 2x2 pivots)
   * @param L The entries in the matrix
   * @param ldl The leading dimension of L
   * @param x The right hand side on input and the solution output
   */
  void solve_pivot_diagonal(const int n, const int piv[], const T L[],
                            const int ldl, T x[]) const {
    for (int i = 0; i < n;) {
      if (piv[i] >= 0) {
        x[i] /= L[i * (ldl + 1)];
        i++;
      } else {
        T d11 = L[i * (ldl + 1)];
        T d21 = L[i * (ldl + 1) + 1];
        T d22 = L[(i + 1) * (ldl + 1)];

        T inv = 1.0 / (d11 * d22 - d21 * d21);
        T x1 = x[i];
        T x2 = x[i + 1];
        x[i] = inv * (d22 * x1 - d21 * x2);
        x[i + 1] = inv * (d11 * x2 - d21 * x1);
        i += 2;
      }
    }
  }

  /**
   * @brief Solve the system of equations L^{T} * x = b
   *
   * On input x = b on output the solution is stored in x
   *
   * @param n Number of rows/columns in L
   * @param piv Pivots (used to detect 1x1 and 2x2 pivots)
   * @param L The entries in the matrix
   * @param ldl The leading dimension of L
   * @param x The right hand side on input and the solution output
   */
  void solve_pivot_transpose(const int n, const int piv[], const T L[],
                             const int ldl, T x[]) const {
    for (int i = n - 1; i >= 0;) {
      if (piv[i] >= 0) {
        for (int j = i + 1; j < n; j++) {
          x[i] -= L[j + ldl * i] * x[j];
        }
        i -= 1;
      } else {
        for (int j = i + 1; j < n; j++) {
          x[i - 1] -= L[j + ldl * (i - 1)] * x[j];
          x[i] -= L[j + ldl * i] * x[j];
        }
        i -= 2;
      }
    }
  }

  /**
   * @brief Solve the system of equations based on an LDL factorization
   *
   * @param xvec The right hand side and solution vector
   */
  void solve_ldl(Vector<T>* xvec) const {
    T* x = xvec->get_array();

    int max_pivots = fact.get_max_pivots();
    int max_delayed = fact.get_max_delayed();
    T* temp = new T[max_pivots + max_delayed + max_contrib];

    for (int ks = 0; ks < num_snodes; ks++) {
      // Get the pointers to the factor data
      int num_pivots, num_delayed, num_ipiv;
      const int* pivots = nullptr;
      const int* delayed = nullptr;
      const T* L;
      const int* ipiv = nullptr;

      // Get the factor L = (L11, L21) that constitute the factor data
      fact.get_factor(ks, &num_pivots, &pivots, &num_delayed, &delayed, &L,
                      &num_ipiv, &ipiv);
      int num_contrib = contrib_ptr[ks + 1] - contrib_ptr[ks];
      int ldl = num_pivots + num_delayed + num_contrib;

      // Extract the variables from x - 2x2 pivots have negative entries
      for (int j = 0; j < num_pivots; j++) {
        int var = pivots[j];
        if (var < 0) {
          var = -var - 1;
        }
        temp[j] = x[var];
      }

      if (ipiv) {
        int nrhs = 1;
        int info;
        lapack_sytrs("L", ldl, nrhs, L, ldl, ipiv, temp, ldl, &info);
      } else if (num_pivots > 0) {
        // Find the solution t1 = L11^{-1} * t1, overwriting temp
        // with the solution
        solve_pivot_lower(num_pivots, pivots, L, ldl, temp);

        // Compute the matrix-vector t2 = L21 * t1 and add the contributions
        // to the lower block
        int size = num_delayed + num_contrib;
        blas_gemv<T>("N", size, num_pivots, 1.0, &L[num_pivots], ldl, temp, 1,
                     0.0, &temp[num_pivots], 1);

        // Add the contributions
        for (int j = 0, jj = num_pivots; j < num_delayed; j++, jj++) {
          x[delayed[j]] -= temp[jj];
        }
        for (int jp = contrib_ptr[ks], jj = num_pivots + num_delayed;
             jp < contrib_ptr[ks + 1]; jp++, jj++) {
          x[contrib_rows[jp]] -= temp[jj];
        }

        // Compute x = D11^{1} * temp
        solve_pivot_diagonal(num_pivots, pivots, L, ldl, temp);
      }

      // Assign the entries back
      for (int j = 0; j < num_pivots; j++) {
        int var = pivots[j];
        if (var < 0) {
          var = -var - 1;
        }
        x[var] = temp[j];
      }
    }

    for (int ks = num_snodes - 1; ks >= 0; ks--) {
      // Get the pointers to the factor data
      int num_pivots, num_delayed, num_ipiv;
      const int* pivots = nullptr;
      const int* delayed = nullptr;
      const T* L;
      const int* ipiv;

      // Get the factor L = (L11, L21) that constitute the factor data
      fact.get_factor(ks, &num_pivots, &pivots, &num_delayed, &delayed, &L,
                      &num_ipiv, &ipiv);

      if (!ipiv && num_pivots > 0) {
        int num_contrib = contrib_ptr[ks + 1] - contrib_ptr[ks];
        int ldl = num_pivots + num_delayed + num_contrib;

        // Extract the variables from x
        for (int j = 0; j < num_pivots; j++) {
          int var = pivots[j];
          if (var < 0) {
            var = -var - 1;
          }
          temp[j] = x[var];
        }

        // Collect the values from the contributions
        for (int j = 0, jj = num_pivots; j < num_delayed; j++, jj++) {
          temp[jj] = x[delayed[j]];
        }
        for (int jp = contrib_ptr[ks], jj = num_pivots + num_delayed;
             jp < contrib_ptr[ks + 1]; jp++, jj++) {
          temp[jj] = x[contrib_rows[jp]];
        }

        // Compute the matrix-vector product
        int size = num_delayed + num_contrib;
        blas_gemv<T>("T", size, num_pivots, -1.0, &L[num_pivots], ldl,
                     &temp[num_pivots], 1, 1.0, temp, 1);

        // Solve L11^{T} * x = temp
        solve_pivot_transpose(num_pivots, pivots, L, ldl, temp);

        // Assign the entries back
        for (int j = 0; j < num_pivots; j++) {
          int var = pivots[j];
          if (var < 0) {
            var = -var - 1;
          }
          x[var] = temp[j];
        }
      }
    }

    delete[] temp;
  }

  /**
   * @brief Add the inertia from the pivot block associated with the supernode
   *
   * @param ks The super node index
   * @param npos Pointer where to add the positive number of eigenvalues
   * @param nneg Pointer where to add the negative number of eigenvalues
   */
  void add_pivot_inertia(int ks, int* npos, int* nneg) const {
    // Get the pointers to the factor data
    int num_pivots, num_delayed;
    const int* pivots = nullptr;
    const int* ipiv = nullptr;

    const T* L;

    // Get the factor L = (L11, L21) that constitute the factor data
    fact.get_factor(ks, &num_pivots, &pivots, &num_delayed, nullptr, &L,
                    nullptr, &ipiv);

    int num_contrib = contrib_ptr[ks + 1] - contrib_ptr[ks];
    int ldl = num_pivots + num_delayed + num_contrib;

    const int* piv = pivots;
    if (ipiv) {
      piv = ipiv;
    }

    int pos = 0, neg = 0;
    for (int i = 0; i < num_pivots;) {
      if (piv[i] >= 0) {
        if (L[i * (ldl + 1)] >= 0.0) {
          pos++;
        } else {
          neg++;
        }
        i++;
      } else {
        T d11 = L[i * (ldl + 1)];
        T d21 = L[i * (ldl + 1) + 1];
        T d22 = L[(i + 1) * (ldl + 1)];
        T det = d11 * d22 - d21 * d21;

        if (det >= 0.0) {
          if (d11 >= 0.0) {
            pos += 2;
          } else {
            neg += 2;
          }
        } else {  // det < 1
          pos++;
          neg++;
        }
        i += 2;
      }
    }

    // Set the output values
    *npos += pos;
    *nneg += neg;
  }

  /**
   * @brief Factor the frontal matrix now that it is assembled.
   *
   * This code works for the frontal and root matrices.
   *
   * Perform a pure Cholesky factorization of the matrix
   *
   * [L11   0][I  0  ][L11^{T} L21^{T}] = [F11  .sym]
   * [L21   I][0  F22][0             I]   [F21   F22]
   *
   * After factorization, push the contribution matrix to the stack and add
   * the factorization pieces
   *
   * @param ks The index of the super node
   * @param fully_summed The number of fully summed equations
   * @param front_size The front size
   * @param front_vars The front variables
   * @param F The frontal matrix itself
   * @param stack The stack for the contribution blocks
   * @param factor The factor contributions
   */
  int factor_front_matrix_cholesky(const int ks, const int fully_summed,
                                   const int front_size, const int front_vars[],
                                   T F[], ContributionStack& stack,
                                   MatrixFactor& factor) {
    // Cholesky implementation
    int num_delayed = 0;
    int num_pivots = fully_summed;
    int contrib_size = front_size - num_pivots;

    // Positive definite matrix factorization of L11 * L11^{T} = F11
    int ldf = front_size;
    int info;
    lapack_potrf<T>("L", num_pivots, F, ldf, &info);

    // Compute L21 = F21 * L^{-T}
    blas_trsm<T>("R", "L", "T", "N", contrib_size, num_pivots, 1.0, F, ldf,
                 &F[num_pivots], ldf);

    // Compute the trailing update for the lower part of the matrix
    // F22 = F22 - L21 * L21^{T}
    blas_syrk<T>("L", "N", contrib_size, num_pivots, -1.0, &F[num_pivots], ldf,
                 1.0, &F[num_pivots * (ldf + 1)], ldf);

    // Push the update to F22 onto the stack
    stack.push(num_pivots, num_delayed, front_size, front_vars, F);

    // Push the combined columns of L11 and L21 of the matrix onto the stack
    const int* pivots = front_vars;
    const int* delayed = &front_vars[num_pivots];
    factor.add_factor(ks, num_pivots, pivots, num_delayed, delayed,
                      front_size - fully_summed, F);

    return info;
  }

  /**
   * @brief Solve the system of equations based on a Cholesky factorization
   *
   * @param xvec The right hand side and solution vector
   */
  void solve_cholesky(Vector<T>* xvec) const {
    T* x = xvec->get_array();

    int max_pivots = fact.get_max_pivots();
    int max_delayed = fact.get_max_delayed();
    T* temp = new T[max_pivots + max_delayed + max_contrib];

    for (int ks = 0; ks < num_snodes; ks++) {
      // Get the pointers to the factor data
      int num_pivots, num_delayed;
      const int* pivots = nullptr;
      const int* delayed = nullptr;
      const T* L;

      // Get the factor L = (L11, L21) that constitute the factor data
      fact.get_factor(ks, &num_pivots, &pivots, &num_delayed, &delayed, &L);
      int num_contrib = contrib_ptr[ks + 1] - contrib_ptr[ks];
      int ldl = num_pivots + num_delayed + num_contrib;

      // Extract the variables from x corresponding to the pivots which are
      // the columns shared by L11 and L21
      for (int j = 0; j < num_pivots; j++) {
        temp[j] = x[pivots[j]];
      }

      // Compute the solution of L11 * t1 = t1
      int nrhs = 1;
      blas_trsm<T>("L", "L", "N", "N", num_pivots, nrhs, 1.0, L, ldl, temp,
                   num_pivots);

      // Assign the t1 entries back into the x vector
      for (int j = 0; j < num_pivots; j++) {
        x[pivots[j]] = temp[j];
      }

      // Compute the matrix-vector product t2 = L21 * t1
      int size = num_delayed + num_contrib;
      blas_gemv<T>("N", size, num_pivots, 1, &L[num_pivots], ldl, temp, 1, 0.0,
                   &temp[num_pivots], 1);

      // Add the contributions x -= t2
      for (int j = 0, jj = num_pivots; j < num_delayed; j++, jj++) {
        x[delayed[j]] -= temp[jj];
      }
      for (int jp = contrib_ptr[ks], jj = num_pivots + num_delayed;
           jp < contrib_ptr[ks + 1]; jp++, jj++) {
        x[contrib_rows[jp]] -= temp[jj];
      }
    }

    for (int ks = num_snodes - 1; ks >= 0; ks--) {
      // Get the pointers to the factor data
      int num_pivots, num_delayed;
      const int* pivots = nullptr;
      const int* delayed = nullptr;
      const T* L;

      // Get the factor L = (L11, L21) that constitute the factor data
      fact.get_factor(ks, &num_pivots, &pivots, &num_delayed, &delayed, &L);
      int num_contrib = contrib_ptr[ks + 1] - contrib_ptr[ks];
      int ldl = num_pivots + num_delayed + num_contrib;

      // Extract the variables from x
      for (int j = 0; j < num_pivots; j++) {
        temp[j] = x[pivots[j]];
      }

      // Collect the values from the contributions
      for (int j = 0, jj = num_pivots; j < num_delayed; j++, jj++) {
        temp[jj] = x[delayed[j]];
      }
      for (int jp = contrib_ptr[ks], jj = num_pivots + num_delayed;
           jp < contrib_ptr[ks + 1]; jp++, jj++) {
        temp[jj] = x[contrib_rows[jp]];
      }

      // Compute the matrix-vector product t1 = t1 - L21 * t2
      int size = num_delayed + num_contrib;
      blas_gemv<T>("T", size, num_pivots, -1.0, &L[num_pivots], ldl,
                   &temp[num_pivots], 1, 1.0, temp, 1);

      // Compute the solution x1 = L11^{-T} * t1
      int nrhs = 1;
      blas_trsm<T>("L", "L", "T", "N", num_pivots, nrhs, 1.0, L, ldl, temp,
                   num_pivots);

      // Assign the t1 entries back to x
      for (int j = 0; j < num_pivots; j++) {
        x[pivots[j]] = temp[j];
      }
    }
  }

  /**
   * @brief Perform the symbolic analysis phase on the non-zero matrix
   * pattern
   *
   * This performs a post-order of the elimination tree, identifies super
   * nodes based on the post-ordering and performs a count of the numbers of
   * non-zero entries in the matrices.
   *
   * @param order The ordering type to use
   * @param ncols Number of columns (equal to number of rows) in the matrix
   * @param colp Pointer into each column of the matrix
   * @param rows Row indices within each column of the matrix
   */
  void symbolic_analysis(OrderingType order, const int ncols, const int colp[],
                         const int rows[]) {
    // Compute the ordering
    int* perm = nullptr;
    int* iperm = nullptr;

    // If the ordering type is natural, we don't create a permutation of the
    // original matrix
    if (order != OrderingType::NATURAL) {
      int* colp_copy = nullptr;
      int* rows_copy = nullptr;
      OrderingUtils::copy_for_reorder(order, ncols, colp, rows, &colp_copy,
                                      &rows_copy);
      OrderingUtils::reorder(order, ncols, colp_copy, rows_copy, &perm, &iperm);

      // Free the copied matrix data
      delete[] colp_copy;
      delete[] rows_copy;
    }

    // Allocate storage that we'll need
    int* work = new int[4 * ncols];

    // Compute the elimination tree
    int* parent = new int[ncols];
    compute_etree(ncols, colp, rows, perm, iperm, parent, work);

    // Find the post-ordering for the elimination tree
    int* post = new int[ncols];
    post_order_etree(ncols, parent, post, work);

    // Count the column non-zeros in the post-ordering, including diagonals
    int* Lnz = new int[ncols];
    count_column_nonzeros(ncols, colp, rows, post, perm, iperm, parent, Lnz,
                          work);

    // Subtract 1, so that Lnz is the number of non-zeros in the strict lower
    // triangular part of the factorization
    for (int k = 0; k < ncols; k++) {
      Lnz[k] -= 1;
    }

    // Initialize the super nodes. snode_to_var points from the
    // supernode to the variables in the permuted order. After initializing the
    // the non-zero pattern in the matrix, we set snode_to_var so that it points
    // to the variables in their original order.
    int* var_to_snode = new int[ncols];
    snode_to_var = new int[ncols];
    num_snodes =
        init_super_nodes(ncols, post, parent, Lnz, var_to_snode, snode_to_var);

    // Count up the size of each snode
    snode_size = new int[num_snodes];
    std::fill(snode_size, snode_size + num_snodes, 0);
    for (int i = 0; i < ncols; i++) {
      snode_size[var_to_snode[i]]++;
    }

    // Count the children of supernodes within the post-ordered elimination
    // tree
    num_children = new int[num_snodes];
    count_super_node_children(ncols, parent, num_snodes, var_to_snode,
                              num_children, work);

    // Count up the sizes of the contribution blocks
    contrib_ptr = new int[num_snodes + 1];
    contrib_ptr[0] = 0;
    for (int is = 0, i = 0; is < num_snodes; i += snode_size[is], is++) {
      int var = snode_to_var[i + snode_size[is] - 1];
      contrib_ptr[is + 1] = Lnz[var];
    }

    // Count up the contribution block pointer
    for (int i = 0; i < num_snodes; i++) {
      contrib_ptr[i + 1] += contrib_ptr[i];
    }

    // Find the max contribution size
    max_contrib = 0;
    for (int i = 0; i < num_snodes; i++) {
      if (contrib_ptr[i + 1] - contrib_ptr[i] > max_contrib) {
        max_contrib = contrib_ptr[i + 1] - contrib_ptr[i];
      }
    }

    // Fill in the rows in the contribution blocks
    contrib_rows = new int[contrib_ptr[num_snodes]];
    build_nonzero_pattern(ncols, colp, rows, perm, iperm, parent, num_snodes,
                          snode_size, var_to_snode, snode_to_var, contrib_ptr,
                          contrib_rows, work);

    // Permute the snode_to_var variables so that each snode_to_var points
    // to the the original matrix ordering
    if (perm) {
      for (int i = 0; i < ncols; i++) {
        snode_to_var[i] = perm[snode_to_var[i]];
      }
    }

    estimate_cholesky_nonzeros(work);

    // Allocate the arrays within the factorization
    int int_nnz = int(delay_growth * cholesky_int_nnz);
    int factor_nnz = int(delay_growth * cholesky_factor_nnz);
    fact.allocate(num_snodes, int_nnz, factor_nnz);

    delete[] work;
    delete[] parent;
    delete[] post;
    delete[] Lnz;
    delete[] var_to_snode;

    if (perm) {
      delete[] perm;
    }

    // Set the inverse permutation
    invperm = iperm;
  }

  /**
   * @brief Compute the elimination tree with the permuted ordering
   *
   * parent[i] = j
   *
   * Variable i (in the permuted ordering) is a child of j.
   *
   * @param ncols Number of columns
   * @param colp Pointer into each column
   * @param rows Row indices in each column
   * @param perm The permutation array (or nullptr)
   * @param iperm The inverse permultation array (or nullptr)
   * @param parent The etree parent child array
   * @param ancestor Largest ancestor of each node
   */
  void compute_etree(const int ncols, const int colp[], const int rows[],
                     const int perm[], const int iperm[], int parent[],
                     int ancestor[]) {
    // Initialize the parent and ancestor arrays
    std::fill(parent, parent + ncols, -1);
    std::fill(ancestor, ancestor + ncols, -1);

    if (perm && iperm) {
      for (int k = 0; k < ncols; k++) {
        // Get the original column index for the matrix
        const int ka = perm[k];

        // Loop over the column of k
        const int start = colp[ka];
        const int end = colp[ka + 1];
        for (int ip = start; ip < end; ip++) {
          // Get the original row index
          const int ia = rows[ip];

          // Get the new row index
          int i = iperm[ia];

          while (i < k) {
            int tmp = ancestor[i];

            // Update the largest ancestor of i
            ancestor[i] = k;

            // We've reached the root of the previous tree,
            // set the parent of i to k
            if (tmp == -1) {
              parent[i] = k;
              break;
            }

            i = tmp;
          }
        }
      }
    } else {
      for (int k = 0; k < ncols; k++) {
        // Loop over the column of k
        const int start = colp[k];
        const int end = colp[k + 1];
        for (int ip = start; ip < end; ip++) {
          int i = rows[ip];

          while (i < k) {
            int tmp = ancestor[i];

            // Update the largest ancestor of i
            ancestor[i] = k;

            // We've reached the root of the previous tree,
            // set the parent of i to k
            if (tmp == -1) {
              parent[i] = k;
              break;
            }

            i = tmp;
          }
        }
      }
    }
  }

  /**
   * @brief Post-order the elimination tree
   *
   * post[i] = j
   *
   * means that node j of the original tree is the i-th node of the
   * post-ordered tree
   *
   * @param ncols Number of columns
   * @param parent The etree parent child array
   * @param post The computed post order
   * @param work Work array of size 3 * ncols
   */
  void post_order_etree(const int ncols, const int parent[], int post[],
                        int work[]) {
    int* head = work;
    int* next = &work[ncols];
    int* stack = &work[2 * ncols];

    std::fill(head, head + ncols, -1);
    std::fill(next, next + ncols, -1);

    // Initialize the heads of each linked list
    for (int j = ncols - 1; j >= 0; j--) {
      if (parent[j] != -1) {
        next[j] = head[parent[j]];
        head[parent[j]] = j;
      }
    }

    for (int j = 0, k = 0; j < ncols; j++) {
      if (parent[j] == -1) {
        // Perform a depth first search starting from j which is a root
        // in the etree
        k = depth_first_search(j, k, head, next, post, stack);
      }
    }
  }

  /**
   * @brief Perform a depth first search from node j
   *
   * @param j The root node to start from
   * @param k The post-order index
   * @param head The head of each linked list
   * @param next The next child in the linked lists
   * @param post The post order post[post node i] = origin node j
   * @param stack The stack for the depth first search
   * @return int The final post-order index
   */
  int depth_first_search(int j, int k, int head[], const int next[], int post[],
                         int stack[]) {
    int last = 0;     // Last position on the tack
    stack[last] = j;  // Put node j on the stack

    while (last >= 0) {
      // Look at the top of the stack and find the top node p and
      // its child i
      const int p = stack[last];
      int i = head[p];

      if (i == -1) {
        // No unordered children of p left in the list
        post[k] = p;
        k++;
        last--;
      } else {
        // Remove i from the children of p and add i to the
        // stack to continue the depth first search
        head[p] = next[i];
        last++;
        stack[last] = i;
      }
    }

    return k;
  }

  /**
   * @brief Find the root of the row etree
   *
   * @param p The given node
   * @param ancestor Array of ancestors
   * @return int Root found
   */
  int find_root(int p, int ancestor[]) const {
    int r = p;
    while (ancestor[r] != r) {
      r = ancestor[r];
    }

    // Path compression
    while (ancestor[p] != p) {
      int next = ancestor[p];
      ancestor[p] = r;
      p = next;
    }

    return r;
  }

  /**
   * @brief Initialize the data required for the column count algorithm
   *
   * @param ncols The number of columns in the matrix
   * @param parent The elimination tree/forest
   * @param post Post-ordering of the etree
   * @param first first[k] is the first descendant of node k
   * @param delta The node weight
   */
  void init_column_count_data(const int ncols, const int parent[],
                              const int post[], int first[],
                              int delta[]) const {
    std::fill(first, first + ncols, -1);
    std::fill(delta, delta + ncols, 0);

    // Propagate minimum postorder index upward through the tree
    for (int k = 0; k < ncols; k++) {
      int j = post[k];
      if (first[j] == -1) {
        delta[j] = 1;
      }
      while (j != -1 && first[j] == -1) {
        first[j] = k;
        j = parent[j];
      }
    }
  }

  /**
   * @brief Count the number of non-zeros in the colums
   *
   * @param ncols The number of columns in the matrix
   * @param colp The pointer into each column
   * @param rows The row indices for each matrix entry
   * @param post Post-ordering of the matrix
   * @param perm The permutation array
   * @param iperm The inverse permutation array
   * @param parent The elimination tree/forest
   * @param colcount The number of non-zeros
   * @param work Work array with size at least 4 times ncols
   */
  void count_column_nonzeros(const int ncols, const int colp[],
                             const int rows[], const int post[],
                             const int perm[], const int iperm[],
                             const int parent[], int colcount[],
                             int work[]) const {
    int* first = &work[0];
    init_column_count_data(ncols, parent, post, first, colcount);

    // Initialize the remaining data
    int* prevleaf = &work[ncols];
    int* maxfirst = &work[2 * ncols];
    int* ancestor = &work[3 * ncols];

    std::fill(prevleaf, prevleaf + ncols, -1);
    std::fill(maxfirst, maxfirst + ncols, -1);
    std::iota(ancestor, ancestor + ncols, 0);

    if (perm && iperm) {
      for (int k = 0; k < ncols; k++) {
        const int j = post[k];

        // Get the original column in the matrix
        const int ja = perm[j];

        // If j is not a root node
        if (parent[j] != -1) {
          colcount[parent[j]]--;
        }

        const int start = colp[ja];
        const int end = colp[ja + 1];
        for (int ip = start; ip < end; ip++) {
          // Row in the original matrix
          const int ia = rows[ip];

          // Row in the new matrix
          int i = iperm[ia];

          // Only use the sub diagonal part A[i, j], i > j.
          if (i <= j) {
            continue;
          }

          // If this column's subtree has already been accounted for in row i,
          // then j is not a new leaf of row i's pruned row subtree.
          if (first[j] <= maxfirst[i]) {
            continue;
          }

          // Increment the weight by 1
          colcount[j]++;
          maxfirst[i] = first[j];
          const int jprev = prevleaf[i];
          prevleaf[i] = j;

          if (jprev != -1) {
            // Subsequent leaf: subtract overlap above the LCA.
            int q = find_root(jprev, ancestor);
            colcount[q]--;
          }
        }

        if (parent[j] != -1) {
          ancestor[j] = parent[j];
        }
      }
    } else {
      for (int k = 0; k < ncols; k++) {
        const int j = post[k];

        // If j is not a root node
        if (parent[j] != -1) {
          colcount[parent[j]]--;
        }

        const int start = colp[j];
        const int end = colp[j + 1];
        for (int ip = start; ip < end; ip++) {
          const int i = rows[ip];

          // Only use the sub diagonal part A[i, j], i > j.
          if (i <= j) {
            continue;
          }

          // If this column's subtree has already been accounted for in row i,
          // then j is not a new leaf of row i's pruned row subtree.
          if (first[j] <= maxfirst[i]) {
            continue;
          }

          // Increment the weight by 1
          colcount[j]++;
          maxfirst[i] = first[j];
          const int jprev = prevleaf[i];
          prevleaf[i] = j;

          if (jprev != -1) {
            // Subsequent leaf: subtract overlap above the LCA.
            int q = find_root(jprev, ancestor);
            colcount[q]--;
          }
        }

        if (parent[j] != -1) {
          ancestor[j] = parent[j];
        }
      }
    }

    // Accumulate delta up the elimination tree.
    for (int k = 0; k < ncols; k++) {
      const int j = post[k];
      if (parent[j] != -1) {
        colcount[parent[j]] += colcount[j];
      }
    }
  }

  /**
   * @brief Initialize the supernodes in the matrix
   *
   * The supernodes share the same column non-zero pattern
   *
   * @param ncols The number of columns in the matrix
   * @param post The etree post ordering
   * @param parent The etree parents
   * @param Lnz The number of non-zeros per variable
   * @param vtosn Variable to super node
   * @param sntov Super node to variable
   * @return int The number of super nodes
   */
  int init_super_nodes(int ncols, const int post[], const int parent[],
                       const int Lnz[], int vtosn[], int sntov[]) {
    // First find the supernodes
    int snode = 0;

    // Loop over subsequent numbers in the post-ordering of the permuted matrix
    for (int i = 0; i < ncols; snode++) {
      int var = post[i];

      // Set the super node
      vtosn[var] = snode;
      sntov[i] = var;
      i++;

      int next_var = ncols;
      if (i < ncols) {
        next_var = post[i];
      }
      while (i < ncols && parent[var] == next_var &&
             (Lnz[next_var] == Lnz[var] - 1)) {
        var = next_var;
        vtosn[var] = snode;
        sntov[i] = var;
        i++;
        if (i < ncols) {
          next_var = post[i];
        }
      }
    }

    return snode;
  }

  /**
   * @brief Count up the number of children for each super node
   *
   * @param ncols Number of columns
   * @param parent Parent pointer for the elimination tree
   * @param ns Number of super nodes
   * @param vtosn Variable to super node array
   * @param nchild Number of children (output)
   * @param snode_parent Super node parent
   */
  void count_super_node_children(const int ncols, const int parent[],
                                 const int ns, const int vtosn[], int nchild[],
                                 int snode_parent[]) {
    std::fill(nchild, nchild + ns, 0);
    std::fill(snode_parent, snode_parent + ns, -1);

    // Set up the snode parents first
    for (int j = 0; j < ncols; j++) {
      int pj = parent[j];

      if (pj != -1) {
        int js = vtosn[j];
        int pjs = vtosn[pj];

        if (pjs != js) {
          snode_parent[js] = pjs;
        }
      }
    }

    // Count up the children within the post-ordered elmination tree
    for (int i = 0; i < ns; i++) {
      if (snode_parent[i] != -1) {
        nchild[snode_parent[i]]++;
      }
    }
  }

  /**
   * @brief Find the non-zero rows in the post-ordered column using the
   * elimination tree - this does not included delayed pivots
   *
   * Find the non-zero rows below the diagonal in a column of L.
   *
   * @param ncols
   * @param colp Column pointer
   * @param rows Rows for each column
   * @param perm Permutation array
   * @param iperm Inverse permutation
   * @param parent Parent array encoding the etree
   * @param ns Number of super nodes
   * @param snsize Size of each super node
   * @param vtosn Supernode index for each variable
   * @param sntov Variable indices for each super node
   * @param cptr Pointer into the row indices for each column
   * @param cvars Row indices for each column
   * @param work Work array - size 4 * ns
   */
  void build_nonzero_pattern(const int ncols, const int colp[],
                             const int rows[], const int perm[],
                             const int iperm[], const int parent[], int ns,
                             int snsize[], const int vtosn[], const int sntov[],
                             const int cptr[], int cvars[], int work[]) {
    int* Lnz = work;
    int* flag = &work[ns];
    int* snvar = &work[2 * ns];
    int* snparent = &work[3 * ns];

    std::fill(Lnz, Lnz + ns, 0);
    std::fill(flag, flag + ns, -1);

    // Find the last variable in each super node
    for (int ks = 0, k = 0; ks < ns; k += snsize[ks], ks++) {
      snvar[ks] = sntov[k + snsize[ks] - 1];
    }

    // Find the supernode parents
    for (int is = 0; is < ns; is++) {
      int last = snvar[is];
      int p = parent[last];
      if (p >= 0) {
        snparent[is] = vtosn[p];
      } else {
        snparent[is] = -1;
      }
    }

    if (perm && iperm) {
      // Loop over the rows of the permuted matrix
      for (int k = 0; k < ncols; k++) {
        // Find the super node associated with ks
        int ks = vtosn[k];
        int ka = perm[k];  // Original column in A

        const int start = colp[ka];
        const int end = colp[ka + 1];
        for (int ip = start; ip < end; ip++) {
          // Get the corresponding row in the original matrix
          int ia = rows[ip];

          // Get the row in the permuted matrix
          int i = iperm[ia];

          if (i < k) {
            // Loop over the supernode parents
            for (int is = vtosn[i]; is >= 0 && is != ks; is = snparent[is]) {
              if (flag[is] == k) {
                break;
              }

              flag[is] = k;
              cvars[cptr[is] + Lnz[is]] = ka;
              Lnz[is]++;
            }
          }
        }
      }
    } else {
      // Loop over the rows of the original matrix
      for (int k = 0; k < ncols; k++) {
        // Find the super node associated with ks
        int ks = vtosn[k];

        const int start = colp[k];
        const int end = colp[k + 1];
        for (int ip = start; ip < end; ip++) {
          // Get the corresponding row
          int i = rows[ip];

          if (i < k) {
            // Loop over the supernode parents
            for (int is = vtosn[i]; is >= 0 && is != ks; is = snparent[is]) {
              if (flag[is] == k) {
                break;
              }

              flag[is] = k;
              cvars[cptr[is] + Lnz[is]] = k;
              Lnz[is]++;
            }
          }
        }
      }
    }
  }

  /**
   * @brief Estimate the non-zeros in the factor and stack and max frontal
   * matrix size. This serves as an estimate of the
   *
   * @param work Work array of at least size 2 * num_snodes
   */
  void estimate_cholesky_nonzeros(int work[]) {
    // Maximum frontal matrix size
    max_frontal_mat_dimension = 0;

    // Estimate the size of the stack
    stack_int_estimate = 0;
    stack_nnz_estimate = 0;

    int top = 0;
    for (int ks = 0; ks < num_snodes; ks++) {
      int ns = snode_size[ks];

      // Find the total stack size at this point
      int int_size = 0;
      int nnz_size = 0;
      for (int tmp_top = top; tmp_top >= 0; tmp_top -= 2) {
        int_size += work[tmp_top];
        nnz_size += work[tmp_top + 1];
      }

      // Now pop the children off the stack
      for (int k = 0; k < num_children[ks]; k++) {
        top -= 2;
      }

      if (int_size > stack_int_estimate) {
        stack_int_estimate = int_size;
      }
      if (nnz_size > stack_nnz_estimate) {
        stack_nnz_estimate = nnz_size;
      }

      int contrib_size = contrib_ptr[ks + 1] - contrib_ptr[ks];

      if (max_frontal_mat_dimension > ns + contrib_size) {
        max_frontal_mat_dimension = ns + contrib_size;
      }

      // Push this contribution on to the stack
      work[top] = 2 + contrib_size;
      work[top + 1] = contrib_size * (contrib_size + 1) / 2;
      top += 2;
    }

    // Count up the size for the cholesky factorization
    cholesky_int_nnz = 0;
    cholesky_factor_nnz = 0;
    for (int is = 0; is < num_snodes; is++) {
      int ns = snode_size[is];
      int contrib_size = contrib_ptr[is + 1] - contrib_ptr[is];
      int ldf = contrib_size + ns;
      cholesky_int_nnz += ns;           // Space to store pivots
      cholesky_factor_nnz += ns * ldf;  // Space to store factorization
    }

    // Add the space for the root ipiv array
    cholesky_int_nnz += max_frontal_mat_dimension;
  }

  // The matrix
  std::shared_ptr<CSRMat<T>> mat;

  // Solver type to use
  SolverType solver_type;

  // Stability factor
  double ustab;

  // Pivot tolerance - tolerance below which pivots are declared singular
  double pivot_tol;

  // Estimated growth factor for delayed pivots
  double delay_growth;

  // Type of ordering used for the matrix
  OrderingType order;

  // Space required for the frontal matrix for a Cholesky
  int max_frontal_mat_dimension;

  // Space required for the stack and the max front size
  int stack_int_estimate;
  int stack_nnz_estimate;

  // Space required for a Choleksy factorization
  int cholesky_int_nnz;
  int cholesky_factor_nnz;

  // Permutation array defined - nullptr if order == NATURAL
  int* invperm;

  // Number of super nodes in the matrix
  int num_snodes;

  // Size of the super nodes
  int* snode_size;

  // Go from var to super node or super node to variable
  int* snode_to_var;

  // Number of children for each super node
  int* num_children;

  // The contribution blocks sizes (without delayed pivots)
  int max_contrib;    // Max size
  int* contrib_ptr;   // Pointer into the rows
  int* contrib_rows;  // Contribution rows

  // Storage for the matrix factorization
  MatrixFactor fact;
};

}  // namespace amigo

#endif  // AMIGO_SPARSE_LDL_H