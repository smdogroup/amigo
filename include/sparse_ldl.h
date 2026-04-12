#ifndef AMIGO_SPARSE_LDL_H
#define AMIGO_SPARSE_LDL_H

#include "blas_interface.h"
#include "csr_matrix.h"

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
   * @param mat The CSR matrix (treated as symmetric)
   * @param solver_type The type of solver (Cholesky or LDL)
   * @param ustab Stability parameter for 1x1 and 2x2 pivots for the LDL
   * @param pivot_tol Pivot tolerance below which pivots are treated as zero
   * @param delay_growth Delayed pivot growth factor (used to estimate memory
   * requirements)
   */
  SparseLDL(std::shared_ptr<CSRMat<T>> mat,
            SolverType solver_type = SolverType::LDL, double ustab = 0.01,
            double pivot_tol = 1e-14, double delay_growth = 2.0)
      : mat(mat),
        solver_type(solver_type),
        ustab(ustab),
        pivot_tol(pivot_tol),
        delay_growth(delay_growth) {
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
    snode_size = nullptr;
    var_to_snode = nullptr;
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
    symbolic_analysis(nrows, rowp, cols);
  }
  ~SparseLDL() {
    delete[] snode_size;
    delete[] var_to_snode;
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
        int delayed_pivots = idx[tmp_top - 2];
        int contrib_size = idx[tmp_top - 1];
        int* vars = &idx[tmp_top - 2 - contrib_size];

        for (int j = 0; j < delayed_pivots; j++) {
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
      if (top_idx + 2 + contrib_size > idx.size()) {
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
      int block_size = contrib_size * contrib_size;
      if (top_work + block_size > work.size()) {
        work.resize(int(top_work + block_size + 0.5 * work.size()));
      }

      // Copy the values into the data array
      T* ptr = &work[top_work];
      for (int j = num_pivots, k = 0; j < front_size; j++) {
        for (int i = num_pivots; i < front_size; i++, k++) {
          ptr[k] = F[i + front_size * j];
        }
      }
      top_work += block_size;
    }

    /**
     * @brief Pop a contribution block from the top of the stack
     *
     * @param delayed_pivots Number of delayed pivots
     * @param contrib_size Contribution block size
     * @param vars Indices for the contribution block
     * @param C The contribution block values
     */
    void pop(int* delayed_pivots, int* contrib_size, const int* vars[],
             const T* C[]) {
      *delayed_pivots = idx[top_idx - 2];
      int cb_size = idx[top_idx - 1];
      *contrib_size = cb_size;
      *vars = &idx[top_idx - 2 - cb_size];
      top_idx -= (2 + cb_size);

      *C = &work[top_work - cb_size * cb_size];
      top_work -= cb_size * cb_size;
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
      if (int_size + new_int_size > int_data.size()) {
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
      if (factor_size + block_size > factor_data.size()) {
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
      if (int_size + new_int_size > int_data.size()) {
        int_data.resize(int(int_size + new_int_size + 0.5 * int_data.size()));
      }

      // Check if we need to to resize the factor data vector
      if (factor_size + block_size > factor_data.size()) {
        factor_data.resize(
            int(factor_size + block_size + 0.5 * factor_data.size()));
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
    int fdim = int(delay_growth * max_frontal_mat_dimension);
    std::vector<T> F(fdim * fdim);

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

      // Resize the frontal matrix if needed
      if (front_size > fdim) {
        fdim = front_size;
        F.resize(fdim * fdim);
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
          info = factor_front_matrix(ks, fully_summed, front_size, front_vars,
                                     Fptr, stack, fact);
        } else {  // fully_summed = front_size
          info =
              factor_root_matrix(ks, front_size, front_vars, Fptr, stack, fact);
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
    int f_summed =
        stack.add_delayed_pivots(nchildren, ns, front_indices, front_vars);

    // Get the entries predicted from Cholesky
    int start = contrib_ptr[ks];
    int cbsize = contrib_ptr[ks + 1] - start;
    for (int j = 0, *row = &contrib_rows[start]; j < cbsize; j++, row++) {
      front_indices[*row] = f_summed + j;
      front_vars[f_summed + j] = *row;
    }

    // Get the size of the front
    *fully_summed = f_summed;
    *front_size = f_summed + cbsize;
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

    // Assemble the contributions into F from the matrix
    for (int j = 0; j < ns; j++) {
      // Get the column variable associated with the snode
      int var = snode_to_var[k + j];

      for (int ip = colp[var]; ip < colp[var + 1]; ip++) {
        // Get the
        int i = rows[ip];

        // Get the front index
        int ifront = front_indices[i];

        // Add the contribution to the frontal matrix
        if (ifront >= 0) {
          F[ifront + front_size * j] += data[ip];
        }
      }
    }

    // Add the contributions
    for (int child = 0; child < nchildren; child++) {
      int delayed_pivots;
      int contrib_size;
      const int* contrib_indices;
      const T* C;
      stack.pop(&delayed_pivots, &contrib_size, &contrib_indices, &C);

      // Add the contribution blocks
      for (int i = 0; i < contrib_size; i++) {
        int ifront = front_indices[contrib_indices[i]];

        for (int j = 0; j < contrib_size; j++) {
          int jfront = front_indices[contrib_indices[j]];

          F[ifront + front_size * jfront] += C[i + contrib_size * j];
        }
      }
    }
  }

  /**
   * @brief Get the absolute max value in the column of F and within the fully
   * summed rows
   *
   * Note that k < npc < ldf must hold on input
   *
   * @param F The entries of the frontal matrix
   * @param ldf The leading dimension of the frontal matrix
   * @param npc The number of pivot candidates remaining
   * @param k The current column of the matrix
   * @param amax The maximum absolute value of the sub-diagonal elements
   * @param r Row index for the max absolute pivot candidate
   * @param i_pc The index of the maximum pivot candidate
   */
  void get_max_subdiagonal(const T F[], int ldf, int npc, int k, double* amax,
                           double* amaxr, int* r) const {
    // Dimension of the remaining pivot candidates
    int nsub_pc = npc - (k + 1);

    // Find the max value below the diagonal but before npc
    double rval = 0.0;
    int row = -1;
    if (nsub_pc > 0) {
      int one = 1;
      row = k + 1 + blas_imax<T>(&nsub_pc, &F[k + 1 + ldf * k], &one);
      rval = std::abs(F[row + ldf * k]);
    }
    if (r) {
      *r = row;
    }
    if (amaxr) {
      *amaxr = rval;
    }

    // Find the max value below npc
    int nsub = ldf - npc;
    double sval = 0.0;
    if (nsub > 0) {
      int one = 1;
      int subrow = npc + blas_imax<T>(&nsub, &F[npc + ldf * k], &one);
      sval = std::abs(F[subrow + ldf * k]);
    }

    if (sval > rval) {
      *amax = sval;
    } else {
      *amax = rval;
    }
  }

  /**
   * @brief Check if the k-th column of the frontal matrix satisfies the
   * 1x1 pivot stability condition
   *
   * @param F The frontal matrix
   * @param ldf The leading dimension of the frontal matrix
   * @param npc The current number of pivot candidates
   * @param k The current column in the frontal factorization
   * @param r The row index for the max absolute value from pivot candidates
   * @return true The pivot is acceptable
   * @return false The pivot is not acceptable
   */
  bool check_1x1_column(const T F[], int ldf, int npc, int k, int* r) const {
    // fkk = abs(F[k, k])
    double fkk = std::abs(F[k * (ldf + 1)]);

    // Find the max sub-diagonal entry
    double frk;
    get_max_subdiagonal(F, ldf, npc, k, &frk, nullptr, r);

    // The diagonal pivot is below the tolerance, we can't use it
    if (fkk < pivot_tol) {
      return false;
    }

    // Perform the pivot stability check
    if (fkk >= ustab * frk) {
      return true;
    }

    return false;
  }

  /**
   * @brief Get the max absolute value in a candidate row/column swap in the new
   * hypothetical column that lies below the diagonal
   *
   * Note that on input k < c < ldf
   *
   *          k     c
   *   [*                     ]
   *   [   *                  ]
   * k [      *               ]
   *   [         *            ]
   * c [      .  .  *         ]
   *   [            .  *      ]
   *   [            .     *   ]
   *   [            .        *]
   *
   * Dots indicate candidate row/column entries that would be below the new
   * diagonal. The F[c, c] entry would be the new diagonal entry.
   *
   * @param F The entries in the front matrix
   * @param ldf The row/column dimension of F
   * @param k The current column index that is factored in F
   * @param c The candidate row/column to swap
   * @param amax The computed max absolute value
   * @param p The row containing the absolute max from the pivot entries
   * @return int The row index corresponding to the max value
   */
  void get_max_candidate(const T F[], int ldf, int npc, int k, int c,
                         double* amax, int* p) const {
    // Search the candidate row c starting from k
    int ncols = c - k;
    double p1val = 0.0;
    int p1 = -1;
    if (ncols > 0) {
      int p1 = k + blas_imax<T>(&ncols, &F[c + ldf * k], &ldf);
      p1val = std::abs(F[p1 + ldf * k]);
    }

    // Test the sub diagonal from c
    double sval, p2val;
    int p2;
    get_max_subdiagonal(F, ldf, npc, c, &sval, &p2val, &p2);

    if (p1val > sval) {
      *p = p1;
      *amax = p1val;
    } else {
      *p = p2;
      *amax = sval;
    }
  }

  /**
   * @brief Check whether the pivot candidate will satisfy the 1x1 pivot test
   *
   * @param F The frontal matrix
   * @param ldf The leading dimension of the frontal matrix
   * @param npc The current number of pivot candidates
   * @param k The current column in the frontal factorization
   * @param c The candidate pivot
   * @param p The index of the max entry in the pivot row
   * @return true The pivot is acceptable
   * @return false The pivot is not acceptable
   */
  bool check_1x1_candidate(const T F[], int ldf, int npc, int k, int c,
                           int* p) const {
    // Check if fcc is
    double fcc = std::abs(F[c * (ldf + 1)]);

    // Find the max sub-diagonal entry
    double frc;
    get_max_candidate(F, ldf, npc, k, c, &frc, p);

    // The diagonal pivot is below the tolerance, we can't use it
    if (fcc < pivot_tol) {
      return false;
    }

    // Perform the pivot stability check
    if (fcc >= ustab * frc) {
      return true;
    }

    return false;
  }

  /**
   * @brief Check if the candidate obtained by the candidate satisfies the
   * stability test
   *
   * @param F The frontal matrix
   * @param ldf The leading dimension of the frontal matrix
   * @param npc The current number of pivot candidates
   * @param k The current column in the frontal factorization
   * @param c The candidate pivot
   * @param p The index of the max entry in the pivot row
   * @return true The 2x2 pivot is acceptable
   * @return false The 2x2 pivot is not acceptable
   */
  bool check_2x2_candidate(const T F[], int ldf, int npc, int k, int c,
                           int p) const {
    return false;
  }

  /**
   * @brief Perform a symmetric swap of the entries within the frontal matrix.
   *
   * This code assumes k != c.
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

    T tmp;

    // Swap entries [k, 0:k - 1] with [c, 0:k - 1]
    for (int j = 0; j < k; j++) {
      tmp = F[k + j * ldf];
      F[k + j * ldf] = F[c + j * ldf];
      F[c + j * ldf] = tmp;
    }

    // Swap diagonal entries
    tmp = F[k + k * ldf];
    F[k + k * ldf] = F[c + c * ldf];
    F[c + c * ldf] = tmp;

    // Swap [j, k] <-> [c, j] for k < j < c
    for (int j = k + 1; j < c; j++) {
      tmp = F[j + k * ldf];
      F[j + k * ldf] = F[c + j * ldf];
      F[c + j * ldf] = tmp;
    }

    // Swap entries below both rows/cols: [j, k] <-> [j, c] for j > c
    for (int j = c + 1; j < ldf; j++) {
      tmp = F[j + k * ldf];
      F[j + k * ldf] = F[j + c * ldf];
      F[j + c * ldf] = tmp;
    }
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
   * @param stack The stack for the contribution blocks
   * @param factor The factor contributions
   */
  int factor_front_matrix(const int ks, const int fully_summed,
                          const int front_size, int front_vars[], T F[],
                          ContributionStack& stack, MatrixFactor& factor) {
    int ldf = front_size;
    int num_candidates = fully_summed;
    int contrib_size = front_size - fully_summed;

    // Loop until there are no candidates or k == num_candidates
    for (int k = 0; k < num_candidates;) {
      // Default to pivot not accepted (s = 0)
      int s = 0;

      // The index and maximum value of the pivot candidate
      int r = 0, p = 0;
      if (check_1x1_column(F, ldf, num_candidates, k, &r)) {
        s = 1;  // This column is okay, and no swapping required
      } else if (k < num_candidates - 1 &&
                 check_1x1_candidate(F, ldf, num_candidates, k, r, &p)) {
        symm_swap(front_vars, F, ldf, k, r);
        s = 1;
      } else if (k < num_candidates - 2 &&
                 check_2x2_candidate(F, ldf, num_candidates, k, r, p)) {
        symm_swap(front_vars, F, ldf, k, r);
        if (k + 1 != p) {
          symm_swap(front_vars, F, ldf, k + 1, p);
        }

        // Indicate the 2x2 swap
        front_vars[k] = -(front_vars[k] + 1);
        front_vars[k + 1] = -(front_vars[k + 1] + 1);
        s = 2;
      }

      // No acceptable pivots found, we have to delay this variable
      if (s == 0) {
        symm_swap(front_vars, F, ldf, k, num_candidates - 1);
        num_candidates -= 1;

        // No need to continue this iteration of the loop.
        // k remains the same the next loop, but the number of candidates
        // has decreased.
        continue;
      }

      // Apply the update
      if (s == 1) {
        // Copy the column of F[k + 1:, k] to the row F[k, k + 1:] for
        // later usage
        int n = ldf - (k + 1);
        int one = 1;
        blas_copy<T>(&n, &F[(ldf + 1) * k + 1], &one, &F[k + (k + 1) * ldf],
                     &ldf);

        // Find the inverse of the 1x1 pivot
        T d11inv = 1.0 / F[k * (ldf + 1)];

        // Update the column below the diagonal
        T* f = &F[k * (ldf + 1) + 1];
        for (int j = k + 1; j < ldf; j++, f++) {
          f[0] *= d11inv;
        }
      } else if (s == 2) {
        // Copy the columns F[k + 2:, k:k + 2] -> F[k:k + 2, k + 2:].T
        int n = ldf - (k + 2);
        int one = 1;
        blas_copy<T>(&n, &F[k + 2 + ldf * k], &one, &F[k + (k + 2) * ldf],
                     &ldf);
        blas_copy<T>(&n, &F[k + 2 + ldf * (k + 1)], &one,
                     &F[k + 1 + (k + 2) * ldf], &ldf);

        // Find the inverse of the 2x2 pivot
        T d11 = F[k * (ldf + 1)];
        T d21 = F[k * (ldf + 1) + 1];
        T d22 = F[(k + 1) * (ldf + 1)];

        T inv = 1.0 / (d11 * d22 - d21 * d21);
        T d11inv = d22 * inv;
        T d21inv = -d21 * inv;
        T d22inv = d11 * inv;

        // Update the column below the block diagonal
        T* f = &F[k * (ldf + 1) + 2];
        for (int j = k + 2; j < ldf; j++, f++) {
          T f1 = f[0];
          T f2 = f[ldf];

          f[0] = d11inv * f1 + d21inv * f2;
          f[ldf] = d21inv * f1 + d22inv * f2;
        }
      }

      // Apply the update to the remaining candidate columns
      if (s > 0) {
        int mdim = (ldf - (k + s));             // number of rows
        int ndim = (num_candidates - (k + s));  // number of columns
        int kdim = s;
        T alpha = -1.0;
        T beta = 1.0;

        blas_gemm<T>("N", "N", &mdim, &ndim, &kdim, &alpha,
                     &F[(k + s) + k * ldf], &ldf, &F[k + (k + s) * ldf], &ldf,
                     &beta, &F[(k + s) + (k + s) * ldf], &ldf);
      }

      // Increment the
      k += s;
    }

    // The candidates have now been reduced to the number of pivots
    int num_pivots = num_candidates;

    // Apply the trailing update (where npiv = num_pivots)
    // F[npiv:, npiv:] -= F[npiv:, :npiv] @ F[:npiv, npiv:]
    int ndim = front_size - num_pivots;
    int kdim = num_pivots;
    frontal_trailing_update(ndim, kdim, &F[num_pivots], ldf,
                            &F[ldf * num_pivots], ldf,
                            &F[(ldf + 1) * num_pivots], ldf);

    // The number of delayed pivots is the difference between the fully summed
    // candidates we started from and the pivots that were successful
    int num_delayed = fully_summed - num_pivots;

    // Push this onto the stack
    stack.push(num_pivots, num_delayed, front_size, front_vars, F);

    // Push the factored matrix onto the stack
    const int* pivots = front_vars;
    const int* delayed = &front_vars[num_pivots];
    factor.add_factor(ks, num_pivots, pivots, num_delayed, delayed,
                      front_size - fully_summed, F);

    return 0;
  }

  /**
   * @brief Perform the trailing update for the matrix
   *
   * C = C - A * B
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
    T alpha = -1.0;
    T beta = 1.0;

    // Apply the regular update
    if (ndim < 128) {
      blas_gemm<T>("N", "N", &ndim, &ndim, &kdim, &alpha, A, &lda, B, &ldb,
                   &beta, C, &ldc);
    } else {
      //
      // [ C11  0   ] -= [ A11 ] [ B11  B12 ]
      // [ C21  C22 ]    [ A21 ]

      // Try just splitting this into 3 blocks to see the performance
      int n1 = ndim / 2;
      int n2 = ndim - n1;

      // Compute C21 = C21 - A21 * B11
      blas_gemm<T>("N", "N", &n2, &n1, &kdim, &alpha, &A[n1], &lda, B, &ldb,
                   &beta, &C[n1], &ldc);

      // Compute C11 = C11 - A11 * B11
      frontal_trailing_update(n1, kdim, A, lda, B, ldb, C, ldc);

      // Compute C22 = C22 - A21 * B12
      frontal_trailing_update(n2, kdim, &A[n1], lda, &B[n1 * ldb], ldb,
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
   * @param stack Stack here for the memories
   * @param factor The factor
   */
  int factor_root_matrix(const int ks, const int front_size,
                         const int front_vars[], T F[],
                         ContributionStack& stack, MatrixFactor& factor) {
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
    lapack_sytrf<T>("L", &n, F, &n, ipiv, L, &lwork, &info);

    // Copy the data into the factorization storage
    std::copy(front_vars, front_vars + n, vars);
    std::copy(F, F + n * n, L);

    return info;
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
        for (int j = i + 2; j < n; j++) {
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

    int ns = 0;
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
        lapack_sytrs("L", &ldl, &nrhs, L, &ldl, ipiv, temp, &ldl, &info);
      } else {
        // Find the solution t1 = L11^{-1} * t1, overwriting temp
        // with the solution
        solve_pivot_lower(num_pivots, pivots, L, ldl, temp);

        // Compute the matrix-vector t2 = L21 * t1 and add the contributions
        // to the lower block
        int size = num_delayed + num_contrib;
        T alpha = 1.0, beta = 0.0;
        int inc = 1;
        blas_gemv<T>("N", &size, &num_pivots, &alpha, &L[num_pivots], &ldl,
                     temp, &inc, &beta, &temp[num_pivots], &inc);

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

      if (!ipiv) {
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
        T alpha = -1.0, beta = 1.0;
        int inc = 1;
        blas_gemv<T>("T", &size, &num_pivots, &alpha, &L[num_pivots], &ldl,
                     &temp[num_pivots], &inc, &beta, temp, &inc);

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
    lapack_potrf<T>("L", &num_pivots, F, &ldf, &info);

    // Compute L21 = F21 * L^{-T}
    T alpha = 1.0;
    blas_trsm<T>("R", "L", "T", "N", &contrib_size, &num_pivots, &alpha, F,
                 &ldf, &F[num_pivots], &ldf);

    // Compute the trailing update for the lower part of the matrix
    // F22 = F22 - L21 * L21^{T}
    alpha = -1.0;
    T beta = 1.0;
    blas_syrk<T>("L", "N", &contrib_size, &num_pivots, &alpha, &F[num_pivots],
                 &ldf, &beta, &F[num_pivots * (ldf + 1)], &ldf);

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

    int ns = 0;
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
      T alph = 1.0;
      blas_trsm<T>("L", "L", "N", "N", &num_pivots, &nrhs, &alph, L, &ldl, temp,
                   &num_pivots);

      // Assign the t1 entries back into the x vector
      for (int j = 0; j < num_pivots; j++) {
        x[pivots[j]] = temp[j];
      }

      // Compute the matrix-vector product t2 = L21 * t1
      int size = num_delayed + num_contrib;
      T alpha = 1.0, beta = 0.0;
      int inc = 1;
      blas_gemv<T>("N", &size, &num_pivots, &alpha, &L[num_pivots], &ldl, temp,
                   &inc, &beta, &temp[num_pivots], &inc);

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
      T alpha = -1.0, beta = 1.0;
      int inc = 1;
      blas_gemv<T>("T", &size, &num_pivots, &alpha, &L[num_pivots], &ldl,
                   &temp[num_pivots], &inc, &beta, temp, &inc);

      // Compute the solution x1 = L11^{-T} * t1
      int nrhs = 1;
      T alph = 1.0;
      blas_trsm<T>("L", "L", "T", "N", &num_pivots, &nrhs, &alph, L, &ldl, temp,
                   &num_pivots);

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
   * @param ncols Number of columns (equal to number of rows) in the matrix
   * @param colp Pointer into each column of the matrix
   * @param rows Row indices within each column of the matrix
   */
  void symbolic_analysis(const int ncols, const int colp[], const int rows[]) {
    // Allocate storage that we'll need
    int* work = new int[3 * ncols];

    // Compute the elimination tree
    int* parent = new int[ncols];
    compute_etree(ncols, colp, rows, parent, work);

    // Find the post-ordering for the elimination tree
    int* ipost = new int[ncols];
    post_order_etree(ncols, colp, rows, parent, ipost, work);

    // Count the column non-zeros in the post-ordering
    int* Lnz = new int[ncols];
    count_column_nonzeros(ncols, colp, rows, ipost, parent, Lnz, work);

    // Use the work array as a temporary here
    int* post = work;
    for (int i = 0; i < ncols; i++) {
      post[ipost[i]] = i;
    }

    // Initialize the super nodes
    var_to_snode = new int[ncols];
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
    build_nonzero_pattern(ncols, colp, rows, parent, num_snodes, snode_size,
                          var_to_snode, snode_to_var, contrib_ptr, contrib_rows,
                          work);

    estimate_cholesky_nonzeros(work);

    // Allocate the arrays within the factorization
    int int_nnz = int(delay_growth * cholesky_int_nnz);
    int factor_nnz = int(delay_growth * cholesky_factor_nnz);
    fact.allocate(num_snodes, int_nnz, factor_nnz);

    delete[] work;
    delete[] parent;
    delete[] ipost;
    delete[] Lnz;
  }

  /**
   * @brief Compute the elimination tree
   *
   * @param ncols Number of columns
   * @param colp Pointer into each column
   * @param rows Row indices in each column
   * @param parent The etree parent child array
   * @param ancestor Largest ancestor of each node
   */
  void compute_etree(const int ncols, const int colp[], const int rows[],
                     int parent[], int ancestor[]) {
    // Initialize the parent and ancestor arrays
    std::fill(parent, parent + ncols, -1);
    std::fill(ancestor, ancestor + ncols, -1);

    for (int k = 0; k < ncols; k++) {
      // Loop over the column of k
      int start = colp[k];
      int end = colp[k + 1];
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

  /**
   * @brief Post-order the elimination tree
   *
   * ipost[i] = j
   *
   * means that node i of the original tree is the j-th node of the
   * post-ordered tree
   *
   * @param ncols Number of columns
   * @param colp Pointer into each column
   * @param rows Row indices in each column
   * @param parent The etree parent child array
   * @param ipost The computed post order
   * @param work Work array of size 3 * ncols
   */
  void post_order_etree(const int ncols, const int colp[], const int rows[],
                        const int parent[], int ipost[], int work[]) {
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
        k = depth_first_search(j, k, head, next, ipost, stack);
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
   * @param ipost The post order ipost[origin node i] = post node j
   * @param stack The stack for the depth first search
   * @return int The final post-order index
   */
  int depth_first_search(int j, int k, int head[], const int next[],
                         int ipost[], int stack[]) {
    int last = 0;     // Last position on the tack
    stack[last] = j;  // Put node j on the stack

    while (last >= 0) {
      // Look at the top of the stack and find the top node p and
      // its child i
      int p = stack[last];
      int i = head[p];

      if (i == -1) {
        // No unordered children of p left in the list
        ipost[p] = k;
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
   * @brief Build the elimination tree and compute the number of non-zeros in
   * each column.
   *
   * @param ncols The number of columns in the matrix
   * @param colp The pointer into each column
   * @param rows The row indices for each matrix entry
   * @param parent The elimination tree/forest
   * @param Lnz The number of non-zeros in each column
   * @param work Work array of size ncols
   */
  void count_column_nonzeros(const int ncols, const int colp[],
                             const int rows[], const int ipost[],
                             const int parent[], int Lnz[], int work[]) {
    int* flag = work;
    std::fill(Lnz, Lnz + ncols, 0);
    std::fill(flag, flag + ncols, -1);

    // Loop over the original ordering of the matrix
    for (int k = 0; k < ncols; k++) {
      flag[k] = k;

      // Loop over the k-th column of the original matrix
      int ip_end = colp[k + 1];
      for (int ip = colp[k]; ip < ip_end; ip++) {
        int i = rows[ip];

        if (i < k) {
          // Scan up the etree
          while (flag[i] != k) {
            Lnz[i]++;
            flag[i] = k;

            // Set the next parent
            i = parent[i];
          }
        }
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

    // Loop over subsequent numbers in the post-ordering
    for (int i = 0; i < ncols;) {
      int var = post[i];  // Get the original variable number

      // Set the super node
      vtosn[var] = snode;
      sntov[i] = var;
      i++;

      int next_var = post[i];
      while (i < ncols && parent[var] == next_var &&
             (Lnz[next_var] == Lnz[var] - 1)) {
        vtosn[next_var] = snode;
        var = next_var;
        sntov[i] = var;
        i++;
        if (i < ncols) {
          next_var = post[i];
        }
      }

      snode++;
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
   * @param work Work array - at least number of super nodes
   */
  void count_super_node_children(const int ncols, const int parent[],
                                 const int ns, const int vtosn[], int nchild[],
                                 int work[]) {
    int* snode_parent = work;
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
   * This utilizes the elimination tree
   *
   * @param colp Column pointer
   * @param rows Rows for each column
   * @param parent Parent in the elimination tree
   * @param row_count The number of indices found so far
   * @param row_indices The array of row indices
   * @param tag Tag for the visited nodes
   * @param flag Array of flags (tag must not be contained in flag initially)
   */
  void build_nonzero_pattern(const int ncols, const int colp[],
                             const int rows[], const int parent[], int sn,
                             int snsize[], const int vtosn[], const int sntov[],
                             const int cptr[], int cvars[], int work[]) {
    int* Lnz = work;
    int* flag = &work[ncols];
    int* snvar = &work[2 * ncols];

    std::fill(Lnz, Lnz + ncols, 0);
    std::fill(flag, flag + ncols, -1);

    // Find the last variable in each super node
    for (int ks = 0, k = 0; ks < sn; k += snsize[ks], ks++) {
      snvar[ks] = sntov[k + snsize[ks] - 1];
    }

    // Loop over the original ordering of the matrix
    for (int k = 0; k < ncols; k++) {
      flag[k] = k;

      // Loop over the k-th column of the original matrix
      int iptr_end = colp[k + 1];
      for (int iptr = colp[k]; iptr < iptr_end; iptr++) {
        int i = rows[iptr];

        if (i < k) {
          // Scan up the etree
          while (flag[i] != k) {
            // Get the super node from the variable
            int is = vtosn[i];

            // Find the last variable in this super node
            int ivar = snvar[is];

            // If this is the last variable, add this to the row
            if (ivar == i) {
              cvars[cptr[is] + Lnz[is]] = k;
              Lnz[is]++;
            }

            // Flag the node
            flag[i] = k;

            // Set the next parent
            i = parent[i];
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
      work[top + 1] = contrib_size * contrib_size;
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

  // Space required for the frontal matrix for a Cholesky
  int max_frontal_mat_dimension;

  // Space required for the stack and the max front size
  int stack_int_estimate;
  int stack_nnz_estimate;

  // Space required for a Choleksy factorization
  int cholesky_int_nnz;
  int cholesky_factor_nnz;

  // Number of super nodes in the matrix
  int num_snodes;

  // Size of the super nodes
  int* snode_size;

  // Go from var to super node or super node to variable
  int* var_to_snode;
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