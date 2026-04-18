#ifndef A2D_STACK_H
#define A2D_STACK_H

#include "../a2ddefs.h"
#include "../a2dtuple.h"
#include "a2dobj.h"
#include "a2dtuple.h"

namespace A2D {

template <class... Operations>
class OperationStack {
 public:
  using StackTuple = a2d_tuple<Operations...>;
  static constexpr index_t num_ops = sizeof...(Operations);

  A2D_FUNCTION OperationStack(Operations &&...s)
      : stack(a2d_forward<Operations>(s)...) {
    eval_<0>();
  }

  // First-order AD
  A2D_FUNCTION void bzero() { bzero_<0>(); }
  A2D_FUNCTION void forward() { forward_<0>(); }
  A2D_FUNCTION void reverse() { reverse_<num_ops - 1>(); }

  // Second-order AD
  A2D_FUNCTION void hzero() { hzero_<0>(); }
  A2D_FUNCTION void hforward() { hforward_<0>(); }
  A2D_FUNCTION void hreverse() { hreverse_<num_ops - 1>(); }

  // Perform a Hessian-vector product
  A2D_FUNCTION void hproduct() {
    reverse();
    hforward();
    hreverse();
  }

  // Apply Hessian-vector products to extract derivatives
  template <class Input, class Output, class Jacobian>
  A2D_FUNCTION void hextract(Input &p, Output &Jp, Jacobian &jac) {
    reverse();

    for (index_t i = 0; i < Input::ncomp; i++) {
      // Zero all the intermeidate values. This inter object must include the
      // input values, and all values included.
      p.zero();
      Jp.zero();
      hzero();

      p[i] = 1.0;

      // Forward sweep
      hforward();

      // Reverse sweep
      hreverse();

      // Extract the number of columns
      for (index_t j = 0; j < Output::ncomp; j++) {
        jac(j, i) = Jp[j];
      }
    }
  }

 private:
  StackTuple stack;

  template <index_t index>
  A2D_FUNCTION void eval_() {
    a2d_get<index>(stack).eval();
    if constexpr (index < num_ops - 1) {
      eval_<index + 1>();
    }
  }

  template <index_t index>
  A2D_FUNCTION void bzero_() {
    a2d_get<index>(stack).bzero();
    if constexpr (index < num_ops - 1) {
      bzero_<index + 1>();
    }
  }

  template <index_t index>
  A2D_FUNCTION void forward_() {
    a2d_get<index>(stack).template forward<ADorder::FIRST>();
    if constexpr (index < num_ops - 1) {
      forward_<index + 1>();
    }
  }

  template <index_t index>
  A2D_FUNCTION void reverse_() {
    a2d_get<index>(stack).reverse();
    if constexpr (index) {
      reverse_<index - 1>();
    }
  }

  template <index_t index>
  A2D_FUNCTION void hzero_() {
    a2d_get<index>(stack).hzero();
    if constexpr (index < num_ops - 1) {
      hzero_<index + 1>();
    }
  }

  template <index_t index>
  A2D_FUNCTION void hforward_() {
    a2d_get<index>(stack).template forward<ADorder::SECOND>();
    if constexpr (index < num_ops - 1) {
      hforward_<index + 1>();
    }
  }

  template <index_t index>
  A2D_FUNCTION void hreverse_() {
    a2d_get<index>(stack).hreverse();
    if constexpr (index) {
      hreverse_<index - 1>();
    }
  }
};

/**
 * @brief Make an operations stack for automatic differentiation
 *
 * Operations are evaluated immediately on construction
 *
 * @tparam Operations Template parameter list deduced from context
 * @param s The operator objects
 * @return The list of operations
 */
template <class... Operations>
A2D_FUNCTION auto MakeStack(Operations &&...s) {
  return OperationStack<Operations...>(a2d_forward<Operations>(s)...);
}

/**
 * @brief Compute the Jacobian-vector product depending on the input/output
 * states
 *
 * @tparam of Residual type
 * @tparam wrt Derivative type
 * @tparam Data Deduced data space type
 * @tparam Geo Deduced geometry space type
 * @tparam State Deduced state space type
 * @tparam Operations variadic template of operations
 * @param stack Stack of operations
 * @param data Data object
 * @param geo Geometry object
 * @param state State space object
 * @param p Direction vector - same type as wrt
 * @param res Result vector - same type as of
 */
template <FEVarType of, FEVarType wrt, class Data, class Geo, class State,
          class PType, class RType, class... Operations>
A2D_FUNCTION void JacobianProduct(OperationStack<Operations...> &stack,
                                  A2DObj<Data> &data, A2DObj<Geo> &geo,
                                  A2DObj<State> &state, PType &p, RType &res) {
  if constexpr (wrt == FEVarType::DATA) {
    data.pvalue().copy(p);
  } else if constexpr (wrt == FEVarType::GEOMETRY) {
    geo.pvalue().copy(p);
  } else if constexpr (wrt == FEVarType::STATE) {
    state.pvalue().copy(p);
  }

  stack.hproduct();

  if constexpr (of == FEVarType::DATA) {
    res.copy(data.hvalue());
  } else if constexpr (of == FEVarType::GEOMETRY) {
    res.copy(geo.hvalue());
  } else if constexpr (of == FEVarType::STATE) {
    res.copy(state.hvalue());
  }
}

/**
 * @brief Extract the Jacobian matrix using a series of vector-products
 * depending on the input/output state
 *
 * @tparam of Residual type
 * @tparam wrt Derivative type
 * @tparam Data Deduced data space type
 * @tparam Geo Deduced geometry space type
 * @tparam State Deduced state space type
 * @tparam MatType Deduced Jacobian matrix type
 * @tparam Operations variadic template of operations
 * @param stack Stack of operations
 * @param data Data object
 * @param geo Geometry object
 * @param state State space object
 * @param jac Output Jacobian matrix
 */
template <FEVarType of, FEVarType wrt, class Data, class Geo, class State,
          class MatType, class... Operations>
A2D_FUNCTION void ExtractJacobian(OperationStack<Operations...> &stack,
                                  A2DObj<Data> &data, A2DObj<Geo> &geo,
                                  A2DObj<State> &state, MatType &jac) {
  if constexpr (of == FEVarType::DATA) {
    if constexpr (wrt == FEVarType::DATA) {
      stack.hextract(data.pvalue(), data.hvalue(), jac);
    } else if constexpr (wrt == FEVarType::GEOMETRY) {
      stack.hextract(geo.pvalue(), data.hvalue(), jac);
    } else if constexpr (wrt == FEVarType::STATE) {
      stack.hextract(state.pvalue(), data.hvalue(), jac);
    }
  } else if constexpr (of == FEVarType::GEOMETRY) {
    if constexpr (wrt == FEVarType::DATA) {
      stack.hextract(data.pvalue(), geo.hvalue(), jac);
    } else if constexpr (wrt == FEVarType::GEOMETRY) {
      stack.hextract(geo.pvalue(), geo.hvalue(), jac);
    } else if constexpr (wrt == FEVarType::STATE) {
      stack.hextract(state.pvalue(), geo.hvalue(), jac);
    }
  } else if constexpr (of == FEVarType::STATE) {
    if constexpr (wrt == FEVarType::DATA) {
      stack.hextract(data.pvalue(), state.hvalue(), jac);
    } else if constexpr (wrt == FEVarType::GEOMETRY) {
      stack.hextract(geo.pvalue(), state.hvalue(), jac);
    } else if constexpr (wrt == FEVarType::STATE) {
      stack.hextract(state.pvalue(), state.hvalue(), jac);
    }
  }
}

}  // namespace A2D

#endif  // A2D_STACK_H
