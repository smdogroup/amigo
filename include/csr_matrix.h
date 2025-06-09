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
  ~CSRMat() {
    delete[] rowp;
    delete[] cols;
    delete[] data;
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
    const T* x_array = x->get_host_array();
    T* y_array = y->get_host_array();

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