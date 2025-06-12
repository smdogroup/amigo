#ifndef AMIGO_OPTIMIZATION_PROBLEM_H
#define AMIGO_OPTIMIZATION_PROBLEM_H

#include "component_group_base.h"

namespace amigo {

template <typename T>
class OptimizationProblem {
 public:
  using Vec = std::shared_ptr<Vector<T>>;
  using Mat = std::shared_ptr<CSRMat<T>>;

  OptimizationProblem(int ndof,
                      std::vector<std::shared_ptr<ComponentGroupBase<T>>> comps)
      : ndof(ndof), comps(comps) {}

  int get_num_dof() const { return ndof; }
  Vec create_vector() const { return std::make_shared<Vector<T>>(ndof); }

  T lagrangian(Vec& x) const {
    T lagrange = 0.0;
    for (size_t i = 0; i < comps.size(); i++) {
      lagrange += comps[i]->lagrangian(*x);
    }
    return lagrange;
  }

  void gradient(const Vec& x, Vec& g) const {
    g->zero();
    for (size_t i = 0; i < comps.size(); i++) {
      comps[i]->add_gradient(*x, *g);
    }
  }

  void hessian_product(const Vec& x, const Vec& p, Vec& h) const {
    h->zero();
    for (size_t i = 0; i < comps.size(); i++) {
      comps[i]->add_hessian_product(*x, *p, *h);
    }
  }

  void hessian(const Vec& x, Mat& mat) const {
    mat->zero();
    for (size_t i = 0; i < comps.size(); i++) {
      comps[i]->add_hessian(*x, *mat);
    }
  }

  std::shared_ptr<CSRMat<T>> create_csr_matrix() const {
    std::set<std::pair<int, int>> node_set;
    for (size_t i = 0; i < comps.size(); i++) {
      comps[i]->add_nonzero_pattern(node_set);
    }

    std::vector<int> rowp(ndof + 1);
    for (auto it = node_set.begin(); it != node_set.end(); it++) {
      rowp[it->first + 1] += 1;
    }

    // Set the pointer into the rows
    rowp[0] = 0;
    for (int i = 0; i < ndof; i++) {
      rowp[i + 1] += rowp[i];
    }

    int nnz = rowp[ndof];
    std::vector<int> cols(nnz);

    for (auto it = node_set.begin(); it != node_set.end(); it++) {
      cols[rowp[it->first]] = it->second;
      rowp[it->first]++;
    }

    // Reset the pointer into the nodes
    for (int i = ndof; i > 0; i--) {
      rowp[i] = rowp[i - 1];
    }
    rowp[0] = 0;

    return std::make_shared<CSRMat<T>>(ndof, ndof, nnz, rowp.data(),
                                       cols.data());
  }

 private:
  int ndof;
  std::vector<std::shared_ptr<ComponentGroupBase<T>>> comps;
};

}  // namespace amigo

#endif  // AMIGO_OPTIMIZATION_PROBLEM_H