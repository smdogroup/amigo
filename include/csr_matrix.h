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
   */
  template <class ArrayType>
  CSRMat(int nrows, int ncols, int nnz, const ArrayType rowp_,
         const ArrayType cols_)
      : nrows(nrows), ncols(ncols), nnz(nnz) {
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
   */
  template <class Functor>
  CSRMat(int nrows, int ncols, int nelems, const Functor& element_nodes)
      : nrows(nrows), ncols(ncols) {
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

  int nrows;
  int ncols;
  int nnz;
  int* rowp;
  int* cols;
  T* data;
};

}  // namespace amigo

#endif  // AMIGO_CSR_MATRIX_H
