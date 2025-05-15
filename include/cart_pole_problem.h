#ifndef MDGO_CART_POLE_PROBLEM_H
#define MDGO_CART_POLE_PROBLEM_H

#include "cart_component.h"
#include "component_group.h"
#include "constraint_component.h"
#include "csr_matrix.h"
#include "layout.h"
#include "trapezoid_component.h"
#include "vector.h"

namespace mdgo {

template <typename T>
class CartPoleProblem {
 public:
  using CartComponent = CartPoleComponent<T>;
  using CartLayout = IndexLayout<CartComponent>;

  static constexpr int num_states = CartComponent::num_states;

  using TrapComponent = TrapezoidComponent<T>;
  using TrapLayout = IndexLayout<TrapComponent>;

  using ConComponent = ConstraintComponent<T, 2 * num_states>;
  using ConLayout = IndexLayout<ConComponent>;

  using Vec = std::shared_ptr<Vector<T>>;

  CartPoleProblem(int N = 201, T g = 9.81, T L = 0.5, T m1 = 1.0, T m2 = 0.5)
      : N(N),
        cart(g, L, m1, m2),
        cart_indices(N),
        cart_layout(cart_indices),
        cart_collect(cart, cart_layout),
        trap_indices(num_states * (N - 1)),
        trap_layout(trap_indices),
        trap_collect(trap, trap_layout),
        con_indices(1),
        con_layout(con_indices),
        con_collect(con, con_layout) {
    ndof = 0;

    // Set the variables for the cart-pole equations
    int* cart_array = cart_indices.get_host_array();
    for (int i = 0; i < cart_indices.get_size(); i++, ndof++) {
      cart_array[i] = ndof;
    }

    time_dof = ndof;
    ndof++;

    int* trap_array = trap_indices.get_host_array();

    // Set up the variables for the trapezoid rule
    for (int i = 0; i < N - 1; i++) {
      for (int j = 0; j < num_states; j++, trap_array += TrapComponent::ncomp) {
        trap_array[0] = time_dof;

        // The state variables
        trap_array[1] = cart_array[i * CartComponent::ncomp + j];
        trap_array[2] = cart_array[(i + 1) * CartComponent::ncomp + j];

        // The rate variables
        trap_array[3] = cart_array[i * CartComponent::ncomp + num_states + j];
        trap_array[4] =
            cart_array[(i + 1) * CartComponent::ncomp + num_states + j];

        // The multiplier
        trap_array[5] = ndof;
        ndof++;
      }
    }

    // Set the initial values
    con.values[0] = 0.0;
    con.values[1] = 0.0;
    con.values[2] = 0.0;
    con.values[3] = 0.0;

    // Set the final values
    con.values[4] = 2.0;
    con.values[5] = M_PI;
    con.values[6] = 0.0;
    con.values[7] = 0.0;

    // Set up the variables for the initial/final constraint values
    int* con_array = con_indices.get_host_array();
    for (int i = 0; i < num_states; i++) {
      con_array[i] = cart_array[i];
      con_array[2 * num_states + i] = ndof;
      ndof++;
    }

    for (int i = 0; i < num_states; i++) {
      con_array[num_states + i] =
          cart_array[i + CartComponent::ncomp * (N - 1)];
      con_array[2 * num_states + num_states + i] = ndof;
      ndof++;
    }
  }

  Vec create_vector() const { return std::make_shared<Vector<T>>(ndof); }

  T lagrangian(Vec& x) const {
    return cart_collect.lagrangian(*x) + trap_collect.lagrangian(*x) +
           con_collect.lagrangian(*x);
  }

  void gradient(const Vec& x, Vec& g) const {
    cart_collect.add_gradient(*x, *g);
    trap_collect.add_gradient(*x, *g);
    con_collect.add_gradient(*x, *g);
  }

  void hessian_product(const Vec& x, const Vec& p, Vec& h) const {
    cart_collect.add_hessian_product(*x, *p, *h);
    trap_collect.add_hessian_product(*x, *p, *h);
    con_collect.add_hessian_product(*x, *p, *h);
  }

  void hessian(const Vec& x, std::shared_ptr<CSRMat<T>>& mat) const {
    cart_collect.add_hessian(*x, *mat);
    trap_collect.add_hessian(*x, *mat);
    con_collect.add_hessian(*x, *mat);
  }

  std::shared_ptr<CSRMat<T>> create_csr_matrix() const {
    std::set<std::pair<int, int>> node_set;

    cart_collect.add_nonzero_pattern(node_set);
    trap_collect.add_nonzero_pattern(node_set);
    con_collect.add_nonzero_pattern(node_set);

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
  int N;  // Number of time levels = num_time_steps + 1

  // Cart component objects
  CartComponent cart;
  Vector<int, CartComponent::ncomp> cart_indices;
  IndexLayout<CartComponent> cart_layout;
  SerialCollection<T, CartComponent, CartLayout> cart_collect;

  // Trapezoid rule component objects
  TrapComponent trap;
  Vector<int, TrapComponent::ncomp> trap_indices;
  IndexLayout<TrapComponent> trap_layout;
  SerialCollection<T, TrapComponent, TrapLayout> trap_collect;

  // Trapezoid rule component objects
  ConComponent con;
  Vector<int, ConComponent::ncomp> con_indices;
  IndexLayout<ConComponent> con_layout;
  SerialCollection<T, ConComponent, ConLayout> con_collect;

  int ndof;      // Number of degrees of freedom
  int time_dof;  // Time degree of freedom;
};

}  // namespace mdgo

#endif  // MDGO_CART_POLE_PROBLEM_H