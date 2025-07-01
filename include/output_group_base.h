#ifndef AMIGO_OUTPUT_GROUP_BASE_H
#define AMIGO_OUTPUT_GROUP_BASE_H

#include "csr_matrix.h"
#include "vector.h"

namespace amigo {

template <class T>
class OutputGroupBase {
 public:
  virtual ~OutputGroupBase() {}

  virtual void add_outputs(const Vector<T>& data, const Vector<T>& x,
                           Vector<T>& out) const {}
  virtual void add_jacobian(const Vector<T>& data, const Vector<T>& x,
                            CSRMat<T>& mat) const {}

  virtual void get_layout_data(int* length_, int* nout_, int* ncomp_,
                               const int** outputs_,
                               const int** inputs_) const {}
};

}  // namespace amigo

#endif  // AMIGO_OUTPUT_GROUP_BASE_H