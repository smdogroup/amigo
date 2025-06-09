
#ifndef MDGO_CONSTRAINT_COMPONENT_H
#define MDGO_CONSTRAINT_COMPONENT_H

#include "a2dcore.h"
#include "vector.h"

namespace mdgo {

template <typename T, int N>
class ConstraintComponent {
 public:
  // Define the input: The state and design variables and the Lagrange
  // multipliers
  template <typename T1>
  using Input = A2D::VarTuple<T1, A2D::Vec<T1, N>, A2D::Vec<T1, N>>;

  // The total number of variables (components in the input)
  static constexpr int ncomp = Input<T>::ncomp;

  /**
   * @brief Get the number of variables associated with this component
   *
   * @return std::map<std::string, int> Mapping of the variables
   */
  static std::map<std::string, int> get_variable_names() {
    std::map<std::string, int> map = {{"states", N}, {"lambda_con", N}};
    return map;
  }

  ConstraintComponent() {}
  ConstraintComponent(A2D::Vec<T, N>& values) : values(values) {}

  template <typename T1>
  T1 lagrange(const Input<T1>& input) const {
    const A2D::Vec<T1, N>& q = A2D::get<0>(input);
    const A2D::Vec<T1, N>& lam = A2D::get<1>(input);

    T1 dot1, dot2, result;
    A2D::VecDot(lam, q, dot1);
    A2D::VecDot(lam, values, dot2);
    result = dot1 - dot2;

    return result;
  }

  template <typename T1>
  void gradient(Input<T1>& input, Input<T1>& grad) const {
    A2D::ADObj<A2D::Vec<T1, N>&> q(A2D::get<0>(input), A2D::get<0>(grad));
    A2D::ADObj<A2D::Vec<T1, N>&> lam(A2D::get<1>(input), A2D::get<1>(grad));

    A2D::ADObj<T1> dot1, dot2, result;

    auto stack =
        MakeStack(A2D::VecDot(lam, q, dot1), A2D::VecDot(lam, values, dot2),
                  A2D::Eval(dot1 - dot2, result));

    result.bvalue() = 1.0;
    stack.reverse();
  }

  template <typename T1>
  void hessian_product(Input<T1>& input, Input<T1>& dir, Input<T1>& grad,
                       Input<T1>& prod) const {
    A2D::A2DObj<A2D::Vec<T1, N>&> q(A2D::get<0>(input), A2D::get<0>(grad),
                                    A2D::get<0>(dir), A2D::get<0>(prod));
    A2D::A2DObj<A2D::Vec<T1, N>&> lam(A2D::get<1>(input), A2D::get<1>(grad),
                                      A2D::get<1>(dir), A2D::get<1>(prod));

    A2D::A2DObj<T1> dot1, dot2, result;

    auto stack =
        MakeStack(A2D::VecDot(lam, q, dot1), A2D::VecDot(lam, values, dot2),
                  A2D::Eval(dot1 - dot2, result));

    // Set the seed value and compute the forward derivatives
    result.bvalue() = 1.0;
    stack.reverse();

    // Set the seed and compute the Hessian-vector product
    stack.hforward();
    stack.hreverse();
  }

  A2D::Vec<T, N> values;
};

}  // namespace mdgo

#endif  // MDGO_CONSTRAINT_COMPONENT_H
