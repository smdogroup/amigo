#ifndef AMIGO_COMPONENT_GROUP_BASE_H
#define AMIGO_COMPONENT_GROUP_BASE_H

#include <set>

#include "csr_matrix.h"
#include "vector.h"

namespace amigo {

template <typename T>
class ComponentGroupBase {
 public:
  virtual ~ComponentGroupBase() {}
  virtual int get_max_dof() const { return 0; }
  virtual T lagrangian(const Vector<T>& x) const { return T(0.0); }
  virtual void add_gradient(const Vector<T>& x, Vector<T>& g) const {}
  virtual void add_hessian_product(const Vector<T>& x, const Vector<T>& p,
                                   Vector<T>& h) const {}
  virtual void add_nonzero_pattern(std::set<std::pair<int, int>>& s) const {}
  virtual void add_hessian(const Vector<T>& x, CSRMat<T>& mat) const {}
};

}  // namespace amigo

#endif  // AMIGO_COMPONENT_GROUP_BASE_H
