#ifndef AMIGO_OPTIMIZATION_PROBLEM_H
#define AMIGO_OPTIMIZATION_PROBLEM_H

#include "component_group_base.h"

namespace amigo {

template <typename T>
class OptimizationProblem {
 public:
  using Vec = std::shared_ptr<Vector<T>>;
  using Mat = std::shared_ptr<CSRMat<T>>;

  OptimizationProblem(int data_size, int num_variables,
                      std::vector<std::shared_ptr<ComponentGroupBase<T>>> comps)
      : data_size(data_size), num_variables(num_variables), comps(comps) {
    data_vec = std::make_shared<Vector<T>>(data_size);
  }

  int get_num_variables() const { return num_variables; }
  Vec create_vector() const {
    return std::make_shared<Vector<T>>(num_variables);
  }
  Vec get_data_vector() { return data_vec; }

  T lagrangian(Vec x) const {
    T lagrange = 0.0;
    for (size_t i = 0; i < comps.size(); i++) {
      lagrange += comps[i]->lagrangian(*data_vec, *x);
    }
    return lagrange;
  }

  void gradient(const Vec x, Vec g) const {
    g->zero();
    for (size_t i = 0; i < comps.size(); i++) {
      comps[i]->add_gradient(*data_vec, *x, *g);
    }
  }

  void hessian_product(const Vec x, const Vec p, Vec h) const {
    h->zero();
    for (size_t i = 0; i < comps.size(); i++) {
      comps[i]->add_hessian_product(*data_vec, *x, *p, *h);
    }
  }

  void hessian(const Vec x, Mat mat) const {
    mat->zero();
    for (size_t i = 0; i < comps.size(); i++) {
      comps[i]->add_hessian(*data_vec, *x, *mat);
    }
  }

  Mat create_csr_matrix() const {
    std::vector<int> intervals(comps.size() + 1);
    intervals[0] = 0;
    for (size_t i = 0; i < comps.size(); i++) {
      int length, ncomp;
      const int* data;
      comps[i]->get_layout_data(&length, &ncomp, &data);
      intervals[i + 1] = intervals[i] + length;
    }

    auto element_nodes = [&](int element, const int** ptr) {
      // upper_bound finds the first index i such that intervals[i] >
      // element
      auto it = std::upper_bound(intervals.begin(), intervals.end(), element);

      // Decrement to get the interval where element fits: intervals[idx]
      // <= element < intervals[idx+1]
      int idx = static_cast<int>(it - intervals.begin()) - 1;

      int length, ncomp;
      const int* data;
      comps[idx]->get_layout_data(&length, &ncomp, &data);

      int elem = element - intervals[idx];
      *ptr = &data[ncomp * elem];
      return ncomp;
    };

    return std::make_shared<CSRMat<T>>(num_variables, num_variables,
                                       intervals[comps.size()], element_nodes);
  }

 private:
  int data_size;
  int num_variables;
  std::vector<std::shared_ptr<ComponentGroupBase<T>>> comps;
  Vec data_vec;
};

}  // namespace amigo

#endif  // AMIGO_OPTIMIZATION_PROBLEM_H