#ifndef MDGO_TRAPEZOID_COMPONENT_H
#define MDGO_TRAPEZOID_COMPONENT_H

#include <map>

#include "a2dcore.h"

namespace mdgo {

template <typename T>
class TrapezoidComponent {
 public:
  // The number of states
  template <typename T1>
  using Input = A2D::VarTuple<T1, T1, T1, T1, T1, T1, T1>;

  // The total number of variables (components in the input)
  static constexpr int ncomp = Input<T>::ncomp;

  /**
   * @brief Get the number of variables associated with this component
   *
   * @return std::map<std::string, int> Mapping of the variables
   */
  static std::map<std::string, int> get_variable_names() {
    std::map<std::string, int> map = {{"dt", 1},     {"state1", 1},
                                      {"state2", 1}, {"rate1", 1},
                                      {"rate2", 1},  {"lambda_trapezoid", 1}};
    return map;
  }

  TrapezoidComponent() {}

  /**
   * @brief Evaluate the contributions to the Lagrangian from this function
   *
   * @tparam T1 The scalar type for the input
   * @param input The intput variable values and Lagrange multipliers
   * @return T1 The contribution to the Lagrangian
   */
  template <typename T1>
  T1 lagrange(const Input<T1>& input) const {
    const T1& dt = A2D::get<0>(input);
    const T1& q1 = A2D::get<1>(input);
    const T1& q2 = A2D::get<2>(input);
    const T1& q1dot = A2D::get<3>(input);
    const T1& q2dot = A2D::get<4>(input);
    const T1& lam = A2D::get<5>(input);

    T1 result = lam * (q2 - q1 + 0.5 * dt * (q1dot + q2dot));

    return result;
  }

  /**
   * @brief Evaluate the gradient of the Lagrangian
   *
   * @tparam T1 The scalar type for the input and gradient
   * @param input The intput variable values and Lagrange multipliers
   * @param grad The gradient of the Lagrangian
   */
  template <typename T1>
  void gradient(Input<T1>& input, Input<T1>& grad) const {
    A2D::ADObj<T1&> dt(A2D::get<0>(input), A2D::get<0>(grad));
    A2D::ADObj<T1&> q1(A2D::get<1>(input), A2D::get<1>(grad));
    A2D::ADObj<T1&> q2(A2D::get<2>(input), A2D::get<2>(grad));
    A2D::ADObj<T1&> q1dot(A2D::get<3>(input), A2D::get<3>(grad));
    A2D::ADObj<T1&> q2dot(A2D::get<4>(input), A2D::get<4>(grad));
    A2D::ADObj<T1&> lam(A2D::get<5>(input), A2D::get<5>(grad));

    A2D::ADObj<T1> result;

    auto stack = A2D::MakeStack(
        A2D::Eval(lam * (q2 - q1 + 0.5 * dt * (q1dot + q2dot)), result));

    result.bvalue() = 1.0;
    stack.reverse();
  }

  /**
   * @brief Compute a Hessian vector product
   *
   * @tparam T1 The type for the computation
   * @param input Input variables
   * @param dir The input direction vector
   * @param grad The gradient at the point
   * @param prod The Hessian-vector product
   */
  template <typename T1>
  void hessian_product(Input<T1>& input, Input<T1>& dir, Input<T1>& grad,
                       Input<T1>& prod) const {
    A2D::A2DObj<T1&> dt(A2D::get<0>(input), A2D::get<0>(grad), A2D::get<0>(dir),
                        A2D::get<0>(prod));
    A2D::A2DObj<T1&> q1(A2D::get<1>(input), A2D::get<1>(grad), A2D::get<1>(dir),
                        A2D::get<1>(prod));
    A2D::A2DObj<T1&> q2(A2D::get<2>(input), A2D::get<2>(grad), A2D::get<2>(dir),
                        A2D::get<2>(prod));
    A2D::A2DObj<T1&> q1dot(A2D::get<3>(input), A2D::get<3>(grad),
                           A2D::get<3>(dir), A2D::get<3>(prod));
    A2D::A2DObj<T1&> q2dot(A2D::get<4>(input), A2D::get<4>(grad),
                           A2D::get<4>(dir), A2D::get<4>(prod));
    A2D::A2DObj<T1&> lam(A2D::get<5>(input), A2D::get<5>(grad),
                         A2D::get<5>(dir), A2D::get<5>(prod));

    A2D::A2DObj<T1> result;

    auto stack = A2D::MakeStack(
        A2D::Eval(lam * (q2 - q1 + 0.5 * dt * (q1dot + q2dot)), result));

    result.bvalue() = 1.0;
    stack.reverse();

    stack.hforward();
    stack.hreverse();
  }
};

}  // namespace mdgo

#endif  // MDGO_TRAPEZOID_COMPONENT_H