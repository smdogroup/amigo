#ifndef AMIGO_LAYOUT_H
#define AMIGO_LAYOUT_H

namespace amigo {

/**
 * @brief Index layout object. This is for a general layout of variables with
 * arbitrary indices
 *
 * @tparam Component
 */
template <int ncomp>
class IndexLayout {
 public:
  IndexLayout() {}
  IndexLayout(Vector<int> &indices_) : indices(indices_) {}
  ~IndexLayout() {}

  int get_length() const { return indices.get_size() / ncomp; }

  template <typename T, int dim, class ArrayType>
  void get_values(int index, const Vector<T, dim> &vec,
                  ArrayType &values) const {
    int *indx = indices.get_host_array();
    const T *vec_values = vec.get_host_array();
    for (int i = 0; i < ncomp; i++) {
      values[i] = vec_values[indx[ncomp * index + i]];
    }
  }

  template <typename T, int dim, class ArrayType>
  void add_values(int index, const ArrayType &values,
                  Vector<T, dim> &vec) const {
    int *indx = indices.get_host_array();
    T *vec_values = vec.get_host_array();
    for (int i = 0; i < ncomp; i++) {
      vec_values[indx[ncomp * index + i]] += values[i];
    }
  }

  template <class ArrayType>
  void get_indices(int index, ArrayType &idx) const {
    int *indx = indices.get_host_array();
    for (int i = 0; i < ncomp; i++) {
      idx[i] = indx[ncomp * index + i];
    }
  }

 private:
  Vector<int> indices;
};

}  // namespace amigo

#endif  // AMIGO_LAYOUT_H