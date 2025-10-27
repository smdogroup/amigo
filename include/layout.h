#ifndef AMIGO_LAYOUT_H
#define AMIGO_LAYOUT_H

namespace amigo {

/**
 * @brief Index layout object. This is for a general layout of variables with
 * arbitrary indices
 *
 * @tparam Component
 */
template <int nodes_per_elem>
class IndexLayout {
 public:
  IndexLayout(int num_elements, std::shared_ptr<Vector<int>> indices)
      : num_elements(num_elements), indices(indices) {}
  ~IndexLayout() {}

  int get_num_elements() const { return num_elements; }

  template <typename T, class ArrayType>
  void get_values(int index, const Vector<T>& vec, ArrayType& values) const {
    const int* indx = indices->get_array();
    const T* vec_values = vec.get_array();
    for (int i = 0; i < nodes_per_elem; i++) {
      values[i] = vec_values[indx[nodes_per_elem * index + i]];
    }
  }

  template <typename T, class ArrayType>
  void add_values(int index, const ArrayType& values, Vector<T>& vec) const {
    const int* indx = indices->get_array();
    T* vec_values = vec.get_array();
    for (int i = 0; i < nodes_per_elem; i++) {
      vec_values[indx[nodes_per_elem * index + i]] += values[i];
    }
  }

  template <class ArrayType>
  void get_indices(int index, ArrayType& idx) const {
    const int* indx = indices->get_array();
    for (int i = 0; i < nodes_per_elem; i++) {
      idx[i] = indx[nodes_per_elem * index + i];
    }
  }

  void get_data(int* num_elements_, int* nodes_per_elem_,
                const int** array_) const {
    *num_elements_ = num_elements;
    *nodes_per_elem_ = nodes_per_elem;
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
  void reorder(const int* elem_by_color) {
    // Make a new vector that will contain the re-ordered elements
    std::shared_ptr<Vector<int>> new_indices =
        std::make_shared<Vector<int>>(indices->get_size());

    int* new_idx = new_indices->get_array();
    int* old_idx = indices->get_array();

    // Set the values
    for (int i = 0; i < num_elements; i++) {
      int elem = elem_by_color[i];

      for (int j = 0; j < nodes_per_elem; j++) {
        new_idx[nodes_per_elem * i + j] = old_idx[nodes_per_elem * elem + j];
      }
    }

    indices = new_indices;
  }

 private:
  int num_elements;
  std::shared_ptr<Vector<int>> indices;
};

}  // namespace amigo

#endif  // AMIGO_LAYOUT_H
