#ifndef AMIGO_CSR_MATRIX_H
#define AMIGO_CSR_MATRIX_H

#include "ordering_utils.h"
#include "vector.h"

namespace amigo {

template <typename T>
class CSRMat {
 public:
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
   * @param sqdef_index Index at which 2x2 negative definite block matrix begins
   */
  template <class ArrayType>
  CSRMat(int nrows, int ncols, int nnz, const ArrayType rowp_,
         const ArrayType cols_, int sqdef_index = -1)
      : nrows(nrows), ncols(ncols), nnz(nnz), sqdef_index(sqdef_index) {
    rowp = new int[nrows + 1];
    cols = new int[nnz];
    std::copy(rowp_, rowp_ + (nrows + 1), rowp);
    std::copy(cols_, cols_ + nnz, cols);

    // Sort the column indices for later use
    for (int i = 0; i < nrows; i++) {
      std::sort(&cols[rowp[i]], &cols[rowp[i + 1]]);
    }

    data = new T[nnz];
    std::fill(data, data + nnz, 0.0);
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
   * @param sqdef_index Index at which 2x2 negative definite block matrix begins
   */
  template <class Functor>
  CSRMat(int nrows, int ncols, int nelems, const Functor& element_nodes,
         int sqdef_index = -1)
      : nrows(nrows), ncols(ncols), sqdef_index(sqdef_index) {
    // Create the CSR structure
    bool include_diagonal = true;
    bool sort_columns = true;
    OrderingUtils::create_csr_from_elements(nrows, ncols, nelems, element_nodes,
                                            include_diagonal, sort_columns,
                                            &rowp, &cols);

    // Compute the number of non-zeros
    nnz = rowp[nrows];

    // Don't forget to allocate the space
    data = new T[nnz];
    std::fill(data, data + nnz, 0.0);
  }
  ~CSRMat() {
    delete[] rowp;
    delete[] cols;
    delete[] data;
  }

  void zero() { std::fill(data, data + nnz, 0.0); }

  /**
   * @brief Add diagonal entries to the matrix
   *
   * @param x The vector of diagonal elements
   */
  void add_diagonal(const std::shared_ptr<Vector<T>>& x) {
    const T* x_array = x->get_array();
    for (int row = 0; row < nrows; row++) {
      int size = rowp[row + 1] - rowp[row];
      int* start = &cols[rowp[row]];
      int* end = start + size;
      auto* it = std::lower_bound(start, end, row);
      if (it != end && *it == row) {
        data[it - cols] += x_array[row];
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

  int nrows;  // Number of rows in the matrix
  int ncols;  // Number of columns in the matrix
  int nnz;    // Number of non-zeros in the matrix
  int* rowp;  // Pointer into the column array
  int* cols;  // Column indices
  T* data;    // Matrix values

  // Specific to symmetric quasi-definite matrices. Which index is where
  // -C is located?
  // A = [ A   B^{T} ]
  //     [ B   -C    ]
  int sqdef_index;  // Index at which C starts. Negative value indicates
                    // that the matrix is not SQD.
};

}  // namespace amigo

#endif  // AMIGO_CSR_MATRIX_H
