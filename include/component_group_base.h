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

  virtual T lagrangian(const Vector<T>& data, const Vector<T>& x) const {
    return T(0.0);
  }
  virtual void add_gradient(const Vector<T>& data, const Vector<T>& x,
                            Vector<T>& g) const {}
  virtual void add_hessian_product(const Vector<T>& data, const Vector<T>& x,
                                   const Vector<T>& p, Vector<T>& h) const {}
  virtual void add_hessian(const Vector<T>& data, const Vector<T>& x,
                           CSRMat<T>& mat) const {}

  virtual void get_layout_data(int* length_, int* ncomp_,
                               const int** array_) const {}
};

}  // namespace amigo

#endif  // AMIGO_COMPONENT_GROUP_BASE_H
