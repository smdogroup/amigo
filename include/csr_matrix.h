#ifndef AMIGO_CSR_MATRIX_H
#define AMIGO_CSR_MATRIX_H

#include "node_owners.h"
#include "ordering_utils.h"
#include "vector.h"

namespace amigo {

template <typename T>
class CSRMat {
 public:
  /**
   * Create an empty CSRMat
   */
  CSRMat()
      : nrows(0),
        ncols(0),
        nnz(0),
        diag(nullptr),
        rowp(nullptr),
        cols(nullptr),
        data(nullptr),
        sqdef_index(-1) {}
  ~CSRMat() {
    if (rowp) {
      delete[] rowp;
    }
    if (cols) {
      delete[] cols;
    }
    if (diag) {
      delete[] diag;
    }
    if (data) {
      delete[] data;
    }
  }

  /**
   * @brief Construct a new CSRMat object
   *
   * Build a general CSR-based data structure directly from the non-zero
   * pattern. Since we assume that the rows are sorted for assembly, we sort
   * them here.
   *
   * @param nrows Number of rows
   * @param ncols Number of columns
   * @param nnz Number of non-zeros
   * @param rowp Pointer into the rows of the matrix
   * @param cols Column indices
   * @param row_owners Distribution of the rows
   * @param col_owners Distribution of the columns
   * @param sqdef_index Index at which 2x2 negative definite block matrix begins
   */
  template <class ArrayType>
  static std::shared_ptr<CSRMat<T>> create_from_csr_data(
      int nrows, int ncols, int nnz, const ArrayType rowp_,
      const ArrayType cols_, std::shared_ptr<NodeOwners> row_owners,
      std::shared_ptr<NodeOwners> col_owners, int sqdef_index = -1) {
    int* rowp = new int[nrows + 1];
    int* cols = new int[nnz];
    std::copy(rowp_, rowp_ + (nrows + 1), rowp);
    std::copy(cols_, cols_ + nnz, cols);

    return std::make_shared<CSRMat<T>>(nrows, ncols, nnz, rowp, cols,
                                       row_owners, col_owners, sqdef_index);
  }

  /**
   * @brief Construct a new CSRMat object from element -> node connectivity data
   *
   * This constructor assumes that nrows >= ncols.
   *
   * Note that the degrees of freedom must satisfy the relationship
   *
   * 0 <= element_nodes[nodes_per_element * i + j] < ncols
   *
   * The non-zero pattern for rows that have indices greater than nrows are not
   * computed.
   *
   * @tparam Functor Class type for the functor
   * @param nrows Number of rows
   * @param ncols Number of columns
   * @param nelems Number of elements in the connectivity matrix
   * @param element_nodes Functor returning the number of nodes and node numbers
   * @param row_owners Distribution of the rows
   * @param col_owners Distribution of the columns
   * @param sqdef_index Index at which 2x2 negative definite block matrix begins
   */
  template <class Functor>
  static std::shared_ptr<CSRMat<T>> create_from_element_conn(
      int nrows, int ncols, int nelems, const Functor& element_nodes,
      std::shared_ptr<NodeOwners> row_owners,
      std::shared_ptr<NodeOwners> col_owners, int sqdef_index = -1) {
    // Create the CSR structure
    bool include_diagonal = true;
    bool sort_columns = true;
    int *rowp, *cols;
    OrderingUtils::create_csr_from_element_conn(nrows, ncols, nelems,
                                                element_nodes, include_diagonal,
                                                sort_columns, &rowp, &cols);

    int nnz = rowp[nrows];
    return std::make_shared<CSRMat<T>>(nrows, ncols, nnz, rowp, cols,
                                       row_owners, col_owners, sqdef_index);
  }

  /**
   * @brief Create a CSR structure from the input/output for each element
   *
   * @tparam Functor Class type for the functor
   * @param nrows Number of rows
   * @param ncols Number of columns
   * @param nelems Number of elements in the connectivity matrix
   * @param elements Functor returning the number of nodes and node numbers
   */
  template <class Functor>
  static std::shared_ptr<CSRMat<T>> create_from_output_data(
      int nrows, int ncols, int nelems, const Functor& elements,
      std::shared_ptr<NodeOwners> row_owners,
      std::shared_ptr<NodeOwners> col_owners) {
    int *rowp, *cols;
    OrderingUtils::create_csr_from_output_data(nrows, ncols, nelems, elements,
                                               &rowp, &cols);
    int nnz = rowp[nrows];
    int sqdef_index = -1;
    return std::make_shared<CSRMat<T>>(nrows, ncols, nnz, rowp, cols,
                                       row_owners, col_owners, sqdef_index);
  }

  /**
   * @brief Duplicate the non-zero structure of the matrix
   *
   * @return std::shared_ptr<CSRMat<T>>
   */
  std::shared_ptr<CSRMat<T>> duplicate() const {
    int* dup_rowp = new int[nrows + 1];
    int* dup_cols = new int[nnz];
    std::copy(rowp, rowp + (nrows + 1), dup_rowp);
    std::copy(cols, cols + nnz, dup_cols);

    return std::make_shared<CSRMat<T>>(nrows, ncols, nnz, dup_rowp, dup_cols,
                                       row_owners, col_owners, sqdef_index);
  }

  /**
   * @brief Zero the numerical values
   */
  void zero() { std::fill(data, data + nnz, 0.0); }

  /**
   * @brief Extract a submatrix from the CSRMat class given the row and column
   * indices
   *
   * @param nsubrows Number of rows in the extracted matrix
   * @param subrows Rows of the submatrix (must be unique)
   * @param nsubcols Number of columns in the extracted matrix
   * @param subcols Columns of the submatrix (must be unique)
   * @return The submatrix with its numerical values
   */
  std::shared_ptr<CSRMat<T>> extract_submatrix(int nsubrows,
                                               const int subrows[],
                                               int nsubcols,
                                               const int subcols[]) const {
    int* subcolptr = new int[ncols];
    std::fill(subcolptr, subcolptr + nrows, -1);
    for (int i = 0; i < nsubcols; i++) {
      subcolptr[subcols[i]] = i;
    }

    // Count up the space for things
    std::shared_ptr<CSRMat<T>> mat = std::make_shared<CSRMat<T>>();

    mat->nrows = nsubrows;
    mat->ncols = nsubcols;
    mat->rowp = new int[nsubrows + 1];
    mat->rowp[0] = 0;

    for (int isub = 0; isub < nsubrows; isub++) {
      int i = subrows[isub];

      int count = 0;
      for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
        int j = cols[jp];
        int jsub = subcolptr[j];
        if (jsub >= 0) {
          count++;
        }
      }

      mat->rowp[isub + 1] = mat->rowp[isub] + count;
    }

    mat->nnz = mat->rowp[mat->nrows];
    mat->diag = new int[mat->nrows];
    mat->cols = new int[mat->nnz];

    for (int isub = 0; isub < nsubrows; isub++) {
      int i = subrows[isub];

      int ptr = mat->rowp[isub];
      for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
        int j = cols[jp];
        int jsub = subcolptr[j];
        if (jsub >= 0) {
          mat->cols[ptr] = jsub;
          ptr++;
        }
      }

      // Sort the column indices
      int size = mat->rowp[isub + 1] - mat->rowp[isub];
      int* start = &mat->cols[mat->rowp[isub]];
      int* end = start + size;
      std::sort(start, end);

      // Set the diagonal elements of the matrix
      auto* it = std::lower_bound(start, end, isub);
      if (it != end && *it == isub) {
        mat->diag[i] = it - mat->cols;
      } else {
        mat->diag[i] = -1;
      }
    }

    mat->data = new T[mat->nnz];
    extract_submatrix_values(nsubrows, subrows, nsubcols, subcols, mat,
                             subcolptr);

    delete[] subcolptr;
    return mat;
  }

  /**
   * @brief Extract the submatrix values
   *
   * @param nsubrows Number of rows in the extracted matrix
   * @param subrows Rows of the submatrix (must be unique)
   * @param nsubcols Number of columns in the extracted matrix
   * @param subcols Columns of the submatrix (must be unique)
   * @param subcolptr (optional) subcolptr pointer from columns to sub matrix
   * colums
   * @return The submatrix with its numerical values
   */
  void extract_submatrix_values(int nsubrows, const int subrows[], int nsubcols,
                                const int subcols[],
                                std::shared_ptr<CSRMat<T>> mat,
                                const int* _subcolptr = nullptr) const {
    int* space = nullptr;
    const int* subcolptr = nullptr;
    if (_subcolptr) {
      subcolptr = _subcolptr;
    } else {
      space = new int[ncols];
      std::fill(space, space + nrows, -1);
      for (int i = 0; i < nsubcols; i++) {
        space[subcols[i]] = i;
      }
      subcolptr = space;
    }

    std::fill(mat->data, mat->data + mat->nnz, T(0.0));
    T* temp = new T[mat->ncols];
    std::fill(temp, temp + mat->ncols, T(0.0));

    for (int isub = 0; isub < nsubrows; isub++) {
      int i = subrows[isub];

      // Assign values to a row of the submatrix
      for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
        int j = cols[jp];
        int jsub = subcolptr[j];
        if (jsub >= 0) {
          temp[jsub] = data[jp];
        }
      }

      // Extract the values from the submatrix row
      for (int jp = mat->rowp[isub]; jp < mat->rowp[isub + 1]; jp++) {
        int jsub = mat->cols[jp];
        data[jp] = temp[jsub];
      }
    }

    delete[] temp;
    if (space) {
      delete[] space;
    }
  }

  /**
   * @brief Perform an iteration of Gauss Seidel
   *
   * @param y The right-hand-side
   * @param x The solution vector
   */
  void gauss_seidel(const std::shared_ptr<Vector<T>>& y,
                    std::shared_ptr<Vector<T>>& x) const {
    const T* y_array = y->get_array();
    T* x_array = x->get_array();

    for (int i = 0; i < nrows; i++) {
      T val = y_array[i];
      for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
        val -= data[jp] * x_array[cols[jp]];
      }

      x_array[i] = x_array[i] + val / data[diag[i]];
    }
  }

  /**
   * @brief Add diagonal entries to the matrix
   *
   * @param x The vector of diagonal elements
   */
  void add_diagonal(const std::shared_ptr<Vector<T>>& x) {
    int size = nrows;
    if (row_owners) {
      size = row_owners->get_local_size();
    }
    if (nrows < size) {
      size = nrows;
    }
    const T* x_array = x->get_array();
    for (int row = 0; row < size; row++) {
      if (diag[row] != -1) {
        data[diag[row]] += x_array[row];
      }
    }
  }

  /**
   * @brief Add a row to the matrix
   *
   * @tparam ArrayType Type of the array that stores the data
   * @param row Row index
   * @param nvalues Number of values to add
   * @param indices Indices of length nvalues
   * @param values Numerical values to add to the matrix
   */
  template <class ArrayType>
  void add_row(int row, int nvalues, const int indices[],
               const ArrayType values) {
    int size = rowp[row + 1] - rowp[row];
    int* start = &cols[rowp[row]];
    int* end = start + size;
    for (int i = 0; i < nvalues; i++) {
      auto* it = std::lower_bound(start, end, indices[i]);

      if (it != end && *it == indices[i]) {
        data[it - cols] += values[i];
      }
    }
  }

  /**
   * @brief Add a sorted row to the matrix
   *
   * @tparam ArrayType
   * @param row
   * @param nvalues
   * @param indices
   * @param values
   */
  template <class ArrayType>
  void add_row_sorted(int row, int nvalues, const int indices[],
                      const ArrayType values) {
    int size = rowp[row + 1] - rowp[row];
    int* col_ptr = &cols[rowp[row]];
    T* data_ptr = &data[rowp[row]];

    int i = 0;  // index into input indices
    int j = 0;  // index into col_ptr

    while (i < nvalues && j < size) {
      if (indices[i] < col_ptr[j]) {
        i++;
      } else if (indices[i] > col_ptr[j]) {
        j++;
      } else {
        data_ptr[j] += values[i];
        i++;
        j++;
      }
    }
  }

  /**
   * @brief Compute a matrix-vector product y = A @ x
   *
   * @param x The input vector
   * @param y The matrix-vector product value
   */
  void mult(const std::shared_ptr<Vector<T>>& x,
            std::shared_ptr<Vector<T>>& y) const {
    const T* x_array = x->get_array();
    T* y_array = y->get_array();

    for (int i = 0; i < nrows; i++) {
      y_array[i] = 0.0;
      for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
        y_array[i] += data[jp] * x_array[cols[jp]];
      }
    }
  }

  /**
   * @brief Get the data from the CSR object
   *
   * @param nrows Number of rows
   * @param ncols Number of columns
   * @param nnz Number of nonzero entries
   * @param rowp Pointer into each row
   * @param cols Column indices
   * @param data Numerical entries
   */
  void get_data(int* nrows_, int* ncols_, int* nnz_, const int* rowp_[],
                const int* cols_[], T* data_[]) {
    if (nrows_) {
      *nrows_ = nrows;
    }
    if (ncols_) {
      *ncols_ = ncols;
    }
    if (nnz_) {
      *nnz_ = nnz;
    }
    if (rowp_) {
      *rowp_ = rowp;
    }
    if (cols_) {
      *cols_ = cols;
    }
    if (data_) {
      *data_ = data;
    }
  }

  /**
   * @brief Get the index of the first sqdef element
   *
   */
  int get_sqdef_index() const { return sqdef_index; }

  /**
   * @brief Get the row owners object
   *
   * @return std::shared_ptr<NodeOwners>
   */
  std::shared_ptr<NodeOwners> get_row_owners() { return row_owners; }

  /**
   * @brief Get the column owners object
   *
   * @return std::shared_ptr<NodeOwners>
   */
  std::shared_ptr<NodeOwners> get_column_owners() { return col_owners; }

  CSRMat(int nrows, int ncols, int nnz, int* rowp, int* cols,
         std::shared_ptr<NodeOwners> row_owners,
         std::shared_ptr<NodeOwners> col_owners, int sqdef_index = -1)
      : nrows(nrows),
        ncols(ncols),
        nnz(nnz),
        rowp(rowp),
        cols(cols),
        row_owners(row_owners),
        col_owners(col_owners),
        sqdef_index(sqdef_index) {
    diag = new int[nrows];

    // Set the diagonal based on the
    int rank;
    MPI_Comm_rank(row_owners->get_mpi_comm(), &rank);
    const int* row_ranges = row_owners->get_range();
    int nrows_local = row_owners->get_local_size();

    for (int i = 0; i < nrows_local; i++) {
      int size = rowp[i + 1] - rowp[i];
      int* start = &cols[rowp[i]];
      int* end = start + size;
      int row = row_ranges[rank] + i;

      // Set the diagonal elements of the matrix
      auto* it = std::lower_bound(start, end, row);
      if (it != end && *it == i) {
        diag[i] = it - cols;
      } else {
        diag[i] = -1;
      }
    }
    for (int i = nrows_local; i < nrows; i++) {
      diag[i] = -1;
    }

    // Don't forget to allocate the space
    data = new T[nnz];
    std::fill(data, data + nnz, 0.0);
  }

 private:
  int nrows;  // Number of rows in the matrix
  int ncols;  // Number of columns in the matrix
  int nnz;    // Number of non-zeros in the matrix
  int* diag;  // The diagonal entry
  int* rowp;  // Pointer into the column array
  int* cols;  // Column indices
  T* data;    // Matrix values

  // For parallel matrices
  std::shared_ptr<NodeOwners> row_owners;
  std::shared_ptr<NodeOwners> col_owners;

  // Specific to symmetric quasi-definite matrices. Which index is where
  // -C is located?
  // A = [ A   B^{T} ]
  //     [ B   -C    ]
  int sqdef_index;  // Index at which C starts. Negative value indicates
  // that the matrix is not SQD.
};

}  // namespace amigo

#endif  // AMIGO_CSR_MATRIX_H
