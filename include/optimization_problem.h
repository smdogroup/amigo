#ifndef AMIGO_OPTIMIZATION_PROBLEM_H
#define AMIGO_OPTIMIZATION_PROBLEM_H

#include "component_group_base.h"
#include "output_group_base.h"

namespace amigo {

template <typename T>
class OptimizationProblem {
 public:
  template <class ArrayType>
  OptimizationProblem(
      int data_size, int num_variables, int num_constraints,
      const ArrayType constraint_index,
      const std::vector<std::shared_ptr<ComponentGroupBase<T>>>& comps,
      int num_outputs = 0,
      const std::vector<std::shared_ptr<OutputGroupBase<T>>>& out_comps =
          std::vector<std::shared_ptr<OutputGroupBase<T>>>())
      : data_size(data_size),
        num_variables(num_variables),
        num_constraints(num_constraints),
        comps(comps),
        num_outputs(num_outputs),
        out_comps(out_comps) {
    is_multiplier = std::make_shared<Vector<int>>(num_variables);
    is_multiplier->zero();

    Vector<int>& is_mult = *is_multiplier;

    // Check whether this numbering of the problem is set up for block 2x2
    order_for_block = true;
    int sqdef_index = num_variables - num_constraints;
    for (int i = 0; i < num_constraints; i++) {
      if (constraint_index[i] < sqdef_index) {
        order_for_block = false;
      }

      if (is_mult[constraint_index[i]] == 0) {
        is_mult[constraint_index[i]] = 1;
      } else {
        throw std::runtime_error("Cannot use duplicate constraint indices");
      }
    }

    data_vec = std::make_shared<Vector<T>>(data_size);
  }

  int get_num_variables() const { return num_variables; }
  int get_num_constraints() const { return num_constraints; }

  std::shared_ptr<Vector<T>> create_vector() const {
    return std::make_shared<Vector<T>>(num_variables);
  }
  std::shared_ptr<Vector<T>> create_output_vector() const {
    return std::make_shared<Vector<T>>(num_outputs);
  }
  std::shared_ptr<Vector<T>> get_data_vector() { return data_vec; }
  const std::shared_ptr<Vector<int>> get_multiplier_indicator() const {
    return is_multiplier;
  }

  T lagrangian(std::shared_ptr<Vector<T>> x) const {
    T lagrange = 0.0;
    for (size_t i = 0; i < comps.size(); i++) {
      lagrange += comps[i]->lagrangian(*data_vec, *x);
    }
    return lagrange;
  }

  void gradient(const std::shared_ptr<Vector<T>> x,
                std::shared_ptr<Vector<T>> g) const {
    g->zero();
    for (size_t i = 0; i < comps.size(); i++) {
      comps[i]->add_gradient(*data_vec, *x, *g);
    }
  }

  void hessian_product(const std::shared_ptr<Vector<T>> x,
                       const std::shared_ptr<Vector<T>> p,
                       std::shared_ptr<Vector<T>> h) const {
    h->zero();
    for (size_t i = 0; i < comps.size(); i++) {
      comps[i]->add_hessian_product(*data_vec, *x, *p, *h);
    }
  }

  void hessian(const std::shared_ptr<Vector<T>> x,
               std::shared_ptr<CSRMat<T>> mat) const {
    mat->zero();
    for (size_t i = 0; i < comps.size(); i++) {
      comps[i]->add_hessian(*data_vec, *x, *mat);
    }
  }

  std::shared_ptr<CSRMat<T>> create_csr_matrix() const {
    std::vector<int> intervals(comps.size() + 1);
    intervals[0] = 0;
    for (size_t i = 0; i < comps.size(); i++) {
      int length, ncomp;
      const int* data;
      comps[i]->get_layout_data(&length, &ncomp, &data);
      intervals[i + 1] = intervals[i] + length;
    }

    auto element_nodes = [&](int element, const int** ptr) {
      // upper_bound finds the first index i such that intervals[i] >
      // element
      auto it = std::upper_bound(intervals.begin(), intervals.end(), element);

      // Decrement to get the interval where element fits: intervals[idx]
      // <= element < intervals[idx+1]
      int idx = static_cast<int>(it - intervals.begin()) - 1;

      int length, ncomp;
      const int* data;
      comps[idx]->get_layout_data(&length, &ncomp, &data);

      int elem = element - intervals[idx];
      *ptr = &data[ncomp * elem];
      return ncomp;
    };

    int sqdef_index = -1;
    if (order_for_block) {
      sqdef_index = num_variables - num_constraints;
    }

    return CSRMat<T>::create_from_element_conn(num_variables, num_variables,
                                               intervals[comps.size()],
                                               element_nodes, sqdef_index);
  }

  void analyze(const std::shared_ptr<Vector<T>> x,
               std::shared_ptr<Vector<T>> outputs) const {
    outputs->zero();
    for (size_t i = 0; i < out_comps.size(); i++) {
      out_comps[i]->add_outputs(*data_vec, *x, *outputs);
    }
  }

  void analyze_jacobian(const std::shared_ptr<Vector<T>> x,
                        std::shared_ptr<CSRMat<T>> jac) const {
    jac->zero();
    for (size_t i = 0; i < out_comps.size(); i++) {
      out_comps[i]->add_jacobian(*data_vec, *x, *jac);
    }
  }

  std::shared_ptr<CSRMat<T>> create_output_csr_matrix() const {
    std::vector<int> intervals(out_comps.size() + 1);
    intervals[0] = 0;
    for (size_t i = 0; i < out_comps.size(); i++) {
      int length, noutputs, ninputs;
      const int *outputs, *inputs;
      out_comps[i]->get_layout_data(&length, &noutputs, &ninputs, &outputs,
                                    &inputs);
      intervals[i + 1] = intervals[i] + length;
    }

    auto element_nodes = [&](int element, int* nrow, int* ncol,
                             const int** rows, const int** cols) {
      // upper_bound finds the first index i such that intervals[i] >
      // element
      auto it = std::upper_bound(intervals.begin(), intervals.end(), element);

      // Decrement to get the interval where element fits: intervals[idx]
      // <= element < intervals[idx+1]
      int idx = static_cast<int>(it - intervals.begin()) - 1;

      int length;
      const int *out, *in;
      comps[idx]->get_layout_data(&length, nrow, ncol, &out, &in);

      int elem = element - intervals[idx];
      *rows = &out[(*nrow) * elem];
      *cols = &in[(*ncol) * elem];
    };

    int sqdef_index = -1;
    return std::make_shared<CSRMat<T>>(num_outputs, num_variables,
                                       intervals[out_comps.size()],
                                       element_nodes, sqdef_index);
  }

 private:
  int data_size;        // Size of the data vector
  int num_variables;    // Number of variables
  int num_constraints;  // Number of constraints

  // Component groups for the optimization problem
  std::vector<std::shared_ptr<ComponentGroupBase<T>>> comps;

  int num_outputs;  // Number of outputs

  // Component output groups for the analysis
  std::vector<std::shared_ptr<OutputGroupBase<T>>> out_comps;

  // Is the optimization problem ordered for a 2x2 block structure
  bool order_for_block;

  // The shared data vector
  std::shared_ptr<Vector<T>> data_vec;

  // Is this variable a multiplier?
  std::shared_ptr<Vector<int>> is_multiplier;
};

}  // namespace amigo

#endif  // AMIGO_OPTIMIZATION_PROBLEM_H