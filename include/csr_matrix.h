#ifndef AMIGO_CSR_MATRIX_H
#define AMIGO_CSR_MATRIX_H

#include "vector.h"

namespace amigo {

template <typename T>
class CSRMat {
 public:
  template <class ArrayType>
  CSRMat(int nrows, int ncols, int nnz, const ArrayType rowp_,
         const ArrayType cols_)
      : nrows(nrows), ncols(ncols), nnz(nnz) {
    rowp = new int[nrows + 1];
    cols = new int[nnz];
    data = new T[nnz];

    std::copy(rowp_, rowp_ + (nrows + 1), rowp);
    std::copy(cols_, cols_ + nnz, cols);
    std::fill(data, data + nnz, 0.0);
  }

  /**
   * @brief Construct a new CSRMat object
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
   * @param nrows Number of block rows
   * @param ncols Number of block columns
   * @param nelems Number of elements in the connectivity matrix
   * @param element_nodes Functor returning the number of nodes and node numbers
   */
  template <class Functor>
  CSRMat(int nrows, int ncols, int nelems, const Functor& element_nodes)
      : nrows(nrows), ncols(ncols) {
    int* node_to_elem_ptr = nullptr;
    int* node_to_elem = nullptr;
    compute_node_to_element_ptr(ncols, nelems, element_nodes, &node_to_elem_ptr,
                                &node_to_elem);

    // Set up the CSR data structure
    rowp = new int[nrows + 1];
    int* counter = new int[ncols];

    // Initialize the counter
    for (int i = 0; i < ncols; i++) {
      counter[i] = -1;
    }

    // Count up the number of non-zero entries
    rowp[0] = 0;
    int nnz_count = 0;
    for (int i = 0; i < nrows; i++) {
      // Count the diagonal first
      counter[i] = i;
      nnz_count++;

      // Count the off-diagonals
      for (int j = node_to_elem_ptr[i]; j < node_to_elem_ptr[i + 1]; j++) {
        int elem = node_to_elem[j];

        const int* ptr;
        int nodes_per_element = element_nodes(elem, &ptr);
        for (int k = 0; k < nodes_per_element; k++, ptr++) {
          int node = ptr[0];
          if (counter[node] < i) {
            counter[node] = i;
            nnz_count++;
          }
        }
      }
      rowp[i + 1] = nnz_count;
    }

    // Set the number of non-zeros
    nnz = nnz_count;

    // Allocate the column indices
    cols = new int[nnz];

    // Reset the counter
    for (int i = 0; i < ncols; i++) {
      counter[i] = -1;
    }

    // Count up the number of non-zero entries
    nnz_count = 0;
    for (int i = 0; i < nrows; i++) {
      // Add the diagonal
      counter[i] = i;
      cols[nnz_count] = i;
      nnz_count++;

      // Add the off-diagonals
      for (int j = node_to_elem_ptr[i]; j < node_to_elem_ptr[i + 1]; j++) {
        int elem = node_to_elem[j];

        const int* ptr;
        int nodes_per_element = element_nodes(elem, &ptr);
        for (int k = 0; k < nodes_per_element; k++, ptr++) {
          int node = ptr[0];
          if (counter[node] < i) {
            counter[node] = i;
            cols[nnz_count] = node;
            nnz_count++;
          }
        }
      }
      std::sort(&cols[rowp[i]], &cols[rowp[i + 1]]);
    }

    // Free unused data
    delete[] node_to_elem_ptr;
    delete[] node_to_elem;
    delete[] counter;

    // Don't forget to allocate the space
    data = new T[nnz];
    std::fill(data, data + nnz, 0.0);
  }
  ~CSRMat() {
    delete[] rowp;
    delete[] cols;
    delete[] data;
  }

  /**
   * @brief Given the element to nodal information, compute a way to quickly
   * find the elements that touch a given node
   *
   * @tparam Functor Class type for the functor
   * @param nnodes Number of nodes in the mesh
   * @param nelems Number of elements
   * @param element_nodes Functor returning the number of nodes and node numbers
   * @param node_to_elem_ptr_ The node to element pointer
   * @param node_to_elem_ The node to element data
   */
  template <class Functor>
  void compute_node_to_element_ptr(int nnodes, int nelems,
                                   const Functor& element_nodes,
                                   int** node_to_elem_ptr_,
                                   int** node_to_elem_) {
    // Create data to store node -> element connectivity
    int* node_to_elem_ptr = new int[nnodes + 1];
    for (int i = 0; i < nnodes + 1; i++) {
      node_to_elem_ptr[i] = 0;
    }

    for (int i = 0; i < nelems; i++) {
      const int* ptr;
      int nodes_per_element = element_nodes(i, &ptr);
      for (int j = 0; j < nodes_per_element; j++, ptr++) {
        node_to_elem_ptr[ptr[0] + 1]++;
      }
    }

    for (int i = 0; i < nnodes; i++) {
      node_to_elem_ptr[i + 1] += node_to_elem_ptr[i];
    }

    // Set up the node to element data
    int* node_to_elem = new int[node_to_elem_ptr[nnodes]];
    for (int i = 0; i < nelems; i++) {
      const int* ptr;
      int nodes_per_element = element_nodes(i, &ptr);
      for (int j = 0; j < nodes_per_element; j++, ptr++) {
        int node = ptr[0];
        node_to_elem[node_to_elem_ptr[node]] = i;
        node_to_elem_ptr[node]++;
      }
    }

    for (int i = nnodes; i > 0; i--) {
      node_to_elem_ptr[i] = node_to_elem_ptr[i - 1];
    }
    node_to_elem_ptr[0] = 0;

    // Set the outputs
    *node_to_elem_ptr_ = node_to_elem_ptr;
    *node_to_elem_ = node_to_elem;
  }

  void zero() { std::fill(data, data + nnz, 0.0); }

  template <class ArrayType>
  void add_row(int row, int nvalues, const int indices[],
               const ArrayType values) {
    int start = rowp[row];
    int end = rowp[row + 1];
    for (int i = 0; i < nvalues; i++) {
      for (int j = start; j < end; j++) {
        if (cols[j] == indices[i]) {
          data[j] += values[i];
          break;
        }
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