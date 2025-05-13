#include "cart_component.h"
#include "component_group.h"
#include "layout.h"
#include "vector.h"

int main(int argc, char* argv[]) {
  using T = double;

  using Component = mdgo::CartPoleComponent<T>;
  using Input = Component::template Input<T>;
  using Layout = mdgo::IndexLayout<Component>;

  T g = 9.81;
  T L = 0.5;
  T m1 = 1.0;
  T m2 = 0.5;

  mdgo::CartPoleComponent<T> cart(g, L, m1, m2);

  int N = 201;  // Number of time levels
  mdgo::Vector<int, Component::ncomp> indices(N);

  int* array = indices.get_host_array();
  for (int i = 0; i < indices.get_size(); i++) {
    array[i] = i;
  }

  int ndof = indices.get_size();

  mdgo::IndexLayout<mdgo::CartPoleComponent<T>> layout(indices);

  mdgo::SerialCollection<T, Component, Layout> collect(cart, layout);

  mdgo::Vector<T> x(ndof);
  mdgo::Vector<T> grad1(ndof);
  mdgo::Vector<T> grad2(ndof);
  mdgo::Vector<T> p(ndof);
  mdgo::Vector<T> hprod(ndof);

  x.set_random();
  p.set_random();
  grad1.zero();
  grad2.zero();
  hprod.zero();

  T L1 = collect.lagrangian(x);
  collect.add_gradient(x, grad1);
  collect.add_hessian_product(x, p, hprod);

  double dh = 1e-6;
  x.axpy(dh, p);
  T L2 = collect.lagrangian(x);
  collect.add_gradient(x, grad2);

  T fd = (L2 - L1) / dh;
  T ans = grad1.dot(p);
  std::printf("%12.4e   %12.4e    %12.4e\n", ans, fd, (ans - fd) / fd);

  grad2.axpy(-1.0, grad1);
  grad2.scale(1.0 / dh);

  T* fd_array = grad2.get_host_array();
  T* ans_array = hprod.get_host_array();
  for (int i = 0; i < hprod.get_size(); i++) {
    std::printf("%12.4e   %12.4e    %12.4e\n", ans_array[i], fd_array[i],
                (ans_array[i] - fd_array[i]) / fd_array[i]);
  }

  return 0;
}