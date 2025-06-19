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
  IndexLayout(std::shared_ptr<Vector<int>> indices) : indices(indices) {}
  ~IndexLayout() {}

  int get_length() const { return indices->get_size() / ncomp; }

  template <typename T, class ArrayType>
  void get_values(int index, const Vector<T> &vec, ArrayType &values) const {
    const int *indx = indices->get_array();
    const T *vec_values = vec.get_array();
    for (int i = 0; i < ncomp; i++) {
      values[i] = vec_values[indx[ncomp * index + i]];
    }
  }

  template <typename T, class ArrayType>
  void add_values(int index, const ArrayType &values, Vector<T> &vec) const {
    const int *indx = indices->get_array();
    T *vec_values = vec.get_array();
    for (int i = 0; i < ncomp; i++) {
      vec_values[indx[ncomp * index + i]] += values[i];
    }
  }

  template <class ArrayType>
  void get_indices(int index, ArrayType &idx) const {
    const int *indx = indices->get_array();
    for (int i = 0; i < ncomp; i++) {
      idx[i] = indx[ncomp * index + i];
    }
  }

  void get_data(int *length_, int *ncomp_, const int **array_) const {
    *length_ = indices->get_size() / ncomp;
    *ncomp_ = ncomp;
    *array_ = indices->get_array();
  }

  /**
   * @brief Reorder the indices of this layout so that it is consistent with the
   * coloring.
   *
   * The elements (or components) are re-ordered but not the variables.
   *
   * The elem_by_color array is defined such that old_elem_idx =
   * elem_by_color[new_elem_idx]
   *
   * @param elem_by_color Order of the elements by color
   */
  void reorder(const int *elem_by_color) {
    // Make a new vector that will contain the re-ordered elements
    std::shared_ptr<Vector<int>> new_indices =
        std::make_shared<Vector<int>>(indices->get_size());

    int *new_idx = new_indices->get_array();
    int *old_idx = indices->get_array();

    // Set the values
    int length = get_length();
    for (int i = 0; i < length; i++) {
      int elem = elem_by_color[i];

      for (int j = 0; j < ncomp; j++) {
        new_idx[ncomp * i + j] = old_idx[ncomp * elem + j];
      }
    }

    indices = new_indices;
  }

 private:
  std::shared_ptr<Vector<int>> indices;
};

}  // namespace amigo

#endif  // AMIGO_LAYOUT_H
