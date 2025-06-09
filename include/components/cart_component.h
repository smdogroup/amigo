#ifndef AMIGO_CART_COMPONENT_H
#define AMIGO_CART_COMPONENT_H

#include <map>

#include "a2dcore.h"

namespace amigo {

template <typename T>
class CartPoleComponent {
 public:
  // Number of equations = number of states for the cart-pole system
  static constexpr int num_states = 4;

  // Define the input: The state and design variables and the Lagrange
  // multipliers
  template <typename T1>
  using Input =
      A2D::VarTuple<T1, A2D::Vec<T1, num_states>, A2D::Vec<T1, num_states>, T1,
                    A2D::Vec<T1, num_states>>;

  // The total number of variables (components in the input)
  static constexpr int ncomp = Input<T>::ncomp;

  /**
   * @brief Get the number of variables associated with this component
   *
   * @return std::map<std::string, int> Mapping of the variables
   */
  static std::map<std::string, int> get_variable_names() {
    std::map<std::string, int> map = {{"states", num_states},
                                      {"rates", num_states},
                                      {"control", 1},
                                      {"lambda_rates", num_states}};
    return map;
  }

  /**
   * @brief Construct a new Cart Pole Component object
   *
   * @param g Acceleration due to gravity
   * @param L Length of the pole
   * @param m1 Mass of the cart
   * @param m2 Mass at the end of the pole
   */
  CartPoleComponent(T g, T L, T m1, T m2) : g(g), L(L), m1(m1), m2(m2) {}

  /**
   * @brief Evaluate the contributions to the Lagrangian from this function
   *
   * @tparam T1 The scalar type for the input
   * @param input The intput variable values and Lagrange multipliers
   * @return T1 The contribution to the Lagrangian
   */
  template <typename T1>
  T1 lagrange(const Input<T1>& input) const {
    const A2D::Vec<T1, num_states>& q = A2D::get<0>(input);
    const A2D::Vec<T1, num_states>& qdot = A2D::get<1>(input);
    const T1& x = A2D::get<2>(input);
    const A2D::Vec<T1, num_states>& lam = A2D::get<3>(input);

    // Pre-compute sin/cos of the pole angle
    T1 cost = A2D::cos(q[1]);
    T1 sint = A2D::sin(q[1]);

    // Compute the contribution to the Lagrangian
    T1 result =
        lam[0] * (q[2] - qdot[0]) + lam[1] * (q[3] - qdot[1]) +
        lam[2] * ((m1 + m2 * (1.0 - cost * cost)) * qdot[2] -
                  (L * m2 * sint * q[3] * q[3] * x + m2 * g * cost * sint)) +
        lam[3] * (L * (m1 + m2 * (1.0 - cost * cost)) * qdot[3] +
                  (L * m2 * cost * sint * q[3] * q[3] + x * cost +
                   (m1 + m2) * g * sint));

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
    A2D::ADObj<A2D::Vec<T1, num_states>&> q(A2D::get<0>(input),
                                            A2D::get<0>(grad));
    A2D::ADObj<A2D::Vec<T1, num_states>&> qdot(A2D::get<1>(input),
                                               A2D::get<1>(grad));
    A2D::ADObj<T1&> x(A2D::get<2>(input), A2D::get<2>(grad));
    A2D::ADObj<A2D::Vec<T1, num_states>&> lam(A2D::get<3>(input),
                                              A2D::get<3>(grad));
    A2D::ADObj<T1> cost, sint, result;

    auto stack = A2D::MakeStack(
        A2D::Eval(A2D::cos(q[1]), cost),  // cost = cos(q[1])
        A2D::Eval(A2D::sin(q[1]), sint),  // sint = sin(q[1])
        A2D::Eval((lam[0] * (q[2] - qdot[0]) + lam[1] * (q[3] - qdot[1]) +
                   lam[2] * ((m1 + m2 * (1.0 - cost * cost)) * qdot[2] -
                             (L * m2 * sint * q[3] * q[3] * x +
                              m2 * g * cost * sint)) +
                   lam[3] * (L * (m1 + m2 * (1.0 - cost * cost)) * qdot[3] +
                             (L * m2 * cost * sint * q[3] * q[3] + x * cost +
                              (m1 + m2) * g * sint))),
                  result));

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
    A2D::A2DObj<A2D::Vec<T1, num_states>&> q(
        A2D::get<0>(input), A2D::get<0>(grad), A2D::get<0>(dir),
        A2D::get<0>(prod));
    A2D::A2DObj<A2D::Vec<T1, num_states>&> qdot(
        A2D::get<1>(input), A2D::get<1>(grad), A2D::get<1>(dir),
        A2D::get<1>(prod));
    A2D::A2DObj<T1&> x(A2D::get<2>(input), A2D::get<2>(grad), A2D::get<2>(dir),
                       A2D::get<2>(prod));
    A2D::A2DObj<A2D::Vec<T1, num_states>&> lam(
        A2D::get<3>(input), A2D::get<3>(grad), A2D::get<3>(dir),
        A2D::get<3>(prod));
    A2D::A2DObj<T1> cost, sint, result;

    auto stack = A2D::MakeStack(
        A2D::Eval(A2D::cos(q[1]), cost),        // cost = cos(q[1])
        A2D::Eval(A2D::sin(q[1]), sint),        // sint = sin(q[1])
        A2D::Eval((lam[0] * (q[2] - qdot[0]) +  // First equation
                   lam[1] * (q[3] - qdot[1]) +  // Secont equation
                   lam[2] * ((m1 + m2 * (1.0 - cost * cost)) * qdot[2] -
                             (L * m2 * sint * q[3] * q[3] * x +
                              m2 * g * cost * sint)) +
                   lam[3] * (L * (m1 + m2 * (1.0 - cost * cost)) * qdot[3] +
                             (L * m2 * cost * sint * q[3] * q[3] + x * cost +
                              (m1 + m2) * g * sint))),
                  result));

    // Set the seed value and compute the forward derivatives
    result.bvalue() = 1.0;
    stack.reverse();

    // Set the seed and compute the Hessian-vector product
    stack.hforward();
    stack.hreverse();
  }

 private:
  T g;   // Acceleration due to gravity
  T L;   // Length of the pendulum
  T m1;  // Mass of the cart
  T m2;  // Mass attached to the end of the pole
};

}  // namespace amigo

#endif  // AMIGO_CART_COMPONENT_H