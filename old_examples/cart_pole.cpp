#include "cart_component.h"
#include "cart_pole_problem.h"
#include "component_group.h"
#include "constraint_component.h"
#include "csr_matrix.h"
#include "layout.h"
#include "trapezoid_component.h"
#include "vector.h"

int main(int argc, char* argv[]) {
  using T = double;

  using Vec = std::shared_ptr<mdgo::Vector<T>>;
  using Mat = std::shared_ptr<mdgo::CSRMat<T>>;

  int N = 201;
  mdgo::CartPoleProblem<T> cart(N);

  Vec x = cart.create_vector();
  Vec g1 = cart.create_vector();
  Vec g2 = cart.create_vector();
  Vec p = cart.create_vector();
  Vec h = cart.create_vector();

  x->set_random();
  p->set_random();
  g1->zero();
  g2->zero();
  h->zero();

  T L1 = cart.lagrangian(x);
  cart.gradient(x, g1);
  cart.hessian_product(x, p, h);

  Mat jac = cart.create_csr_matrix();
  cart.hessian(x, jac);

  double dh = 1e-7;
  x->axpy(dh, *p);
  T L2 = cart.lagrangian(x);
  cart.gradient(x, g2);

  T fd = (L2 - L1) / dh;
  T ans = g1->dot(*p);
  std::printf("Gradient error: \n");
  std::printf("%12.4e   %12.4e    %12.4e\n", ans, fd, (ans - fd) / fd);

  g2->axpy(-1.0, *g1);
  g2->scale(1.0 / dh);

  T* fd_array = g2->get_host_array();
  T* ans_array = h->get_host_array();
  T max_err = 0.0;
  int max_component = 0;
  for (int i = 0; i < h->get_size(); i++) {
    T rel_err = (ans_array[i] - fd_array[i]) / fd_array[i];
    if (std::fabs(rel_err) > max_err) {
      max_err = rel_err;
      max_component = i;
    }
  }
  std::printf("Max Hessian product error: \n");
  std::printf("%12.4e   %12.4e    %12.4e\n", ans_array[max_component],
              fd_array[max_component],
              (ans_array[max_component] - fd_array[max_component]) /
                  fd_array[max_component]);

  jac->mult(p, g2);

  max_err = 0.0;
  max_component = 0;
  for (int i = 0; i < h->get_size(); i++) {
    T rel_err = (ans_array[i] - fd_array[i]) / fd_array[i];
    if (std::fabs(rel_err) > max_err) {
      max_err = rel_err;
      max_component = i;
    }
  }

  std::printf("Max Hessian consistency error: \n");
  std::printf("%12.4e   %12.4e    %12.4e\n", ans_array[max_component],
              fd_array[max_component],
              (ans_array[max_component] - fd_array[max_component]) /
                  fd_array[max_component]);

  return 0;
}