#ifndef AMIGO_SLACK_COUPLING_H
#define AMIGO_SLACK_COUPLING_H

#include "component_group_base.h"
#include "csr_matrix.h"
#include "vector.h"

namespace amigo {

/**
 * Slack component generated from the following python class:
 *
 * class SlackComponent(am.Component):
 *     def __init__(self):
 *         super().__init__()
 *         self.add_input("s")
 *         self.add_constraint("res")
 *
 *     def compute(self):
 *         self.constraints["res"] = -self.inputs["s"]
 */
template <typename T__>
class SlackComponent__ {
 public:
  template <typename R__>
  using Input = A2D::VarTuple<R__, R__, R__>;
  static constexpr int ncomp = Input<T__>::ncomp;
  static constexpr int nconstraints = 1;
  static constexpr bool is_linear = false;
  template <typename R__>
  using Data = typename A2D::VarTuple<R__, R__>;
  static constexpr int ndata = 0;
  static constexpr bool is_compute_empty = false;
  static constexpr bool is_continuation_component = false;
  static constexpr bool is_output_empty = true;
  template <typename R__>
  using Output = typename A2D::VarTuple<R__, R__>;
  static constexpr int noutputs = 0;
  template <typename R__>
  AMIGO_HOST_DEVICE static R__ lagrange(R__ alpha__, Data<R__>& data__,
                                        Input<R__>& input__) {
    R__& s = A2D::get<0>(input__);
    R__& lam_res__ = A2D::get<1>(input__);
    R__ lagrangian__;
    lagrangian__ = (-(s)*lam_res__);
    return lagrangian__;
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void gradient(R__ alpha__, Data<R__>& data__,
                                         Input<R__>& input__,
                                         Input<R__>& boutput__) {
    A2D::ADObj<R__&> s(A2D::get<0>(input__), A2D::get<0>(boutput__));
    A2D::ADObj<R__&> lam_res__(A2D::get<1>(input__), A2D::get<1>(boutput__));
    A2D::ADObj<R__> lagrangian__;
    auto stack__ = A2D::MakeStack(A2D::Eval((-(s)*lam_res__), lagrangian__));
    lagrangian__.bvalue() = 1.0;
    stack__.reverse();
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void hessian(R__ alpha__, Data<R__>& data__,
                                        Input<R__>& input__,
                                        Input<R__>& pinput__,
                                        Input<R__>& boutput__,
                                        Input<R__>& houtput__) {
    A2D::A2DObj<R__&> s(A2D::get<0>(input__), A2D::get<0>(boutput__),
                        A2D::get<0>(pinput__), A2D::get<0>(houtput__));
    A2D::A2DObj<R__&> lam_res__(A2D::get<1>(input__), A2D::get<1>(boutput__),
                                A2D::get<1>(pinput__), A2D::get<1>(houtput__));
    A2D::A2DObj<R__> lagrangian__;
    auto stack__ = A2D::MakeStack(A2D::Eval((-(s)*lam_res__), lagrangian__));
    lagrangian__.bvalue() = 1.0;
    stack__.reverse();
    stack__.hforward();
    stack__.hreverse();
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void compute_output(Data<R__>& data__,
                                               Input<R__>& input__,
                                               Output<R__>& output__) {}
};

/*
  Couples slack variables to inequality constraints.

  Each inequality  lbc <= c(x) <= ubc  is reformulated as the equality
  c(x) - s = 0  with  lbc <= s <= ubc.  This component declares the
  Jacobian sparsity (-I coupling), adds gradient contributions
  (dL/ds = -lambda, primal feasibility c(x) - s), and writes the
  constant -1 entries into the KKT matrix.
*/
template <typename T, ExecPolicy policy>
class SlackCouplingGroup : public ComponentGroupBase<T, policy> {
 public:
  SlackCouplingGroup(int num_slacks, const int slack_indices[],
                     const int ineq_indices[])
      : n_(num_slacks) {
    si_ = std::make_shared<Vector<int>>(n_);
    ci_ = std::make_shared<Vector<int>>(n_);
    si_->copy(slack_indices);
    ci_->copy(ineq_indices);

    // Jacobian CSR: identity pattern in local indexing.
    // Row k has one entry at column k, mapping to (ineq_k, slack_k).
    jac_rowp_ = std::make_shared<Vector<int>>(n_ + 1);
    jac_cols_ = std::make_shared<Vector<int>>(n_);
    int* rp = jac_rowp_->get_array();
    int* cl = jac_cols_->get_array();
    for (int k = 0; k < n_; k++) {
      rp[k] = k;
      cl[k] = k;
    }
    rp[n_] = n_;

    // Data positions in the CSR matrix, filled once during
    // initialize_hessian_pattern
    loc_si_ = std::make_shared<Vector<int>>(n_);
    loc_ci_ = std::make_shared<Vector<int>>(n_);
  }

  std::shared_ptr<ComponentGroupBase<T, policy>> clone(
      int, std::shared_ptr<Vector<int>>, std::shared_ptr<Vector<int>>,
      std::shared_ptr<Vector<int>>) const override {
    return nullptr;
  }

  // Called once after create_matrix(). Finds the data[] positions of the
  // -1 entries so add_hessian is O(n) with direct array writes.
  void initialize_hessian_pattern(const NodeOwners& owners,
                                  CSRMat<T>& mat) override {
    const int* si = si_->get_array();
    const int* ci = ci_->get_array();
    int* lsi = loc_si_->get_array();
    int* lci = loc_ci_->get_array();
    for (int k = 0; k < n_; k++) {
      mat.get_sorted_locations(si[k], 1, &ci[k], &lsi[k]);
      mat.get_sorted_locations(ci[k], 1, &si[k], &lci[k]);
    }
  }

  // grad[slack_k] += -lambda_k   (slack stationarity: dL/ds = -lambda)
  // grad[ineq_k]  += -s_k        (primal feasibility: c(x) - s)
  void add_gradient(T alpha, const Vector<T>& data, const Vector<T>& x,
                    Vector<T>& g) const override {
    const int* si = si_->template get_array<policy>();
    const int* ci = ci_->template get_array<policy>();
    const T* xlam = x.template get_array<policy>();
    T* grad = g.template get_array<policy>();

    for (int k = 0; k < n_; k++) {
      grad[si[k]] += -xlam[ci[k]];
      grad[ci[k]] += -xlam[si[k]];
    }
  }

  // Write -1 at precomputed positions: H[slack_k, ineq_k] and H[ineq_k,
  // slack_k].
  void add_hessian(T alpha, const Vector<T>& data, const Vector<T>& x,
                   const NodeOwners& owners, CSRMat<T>& mat) const override {
    const int* lsi = loc_si_->get_array();
    const int* lci = loc_ci_->get_array();
    T* d = mat.get_data_ptr();
    for (int k = 0; k < n_; k++) {
      d[lsi[k]] += T(-1);
      d[lci[k]] += T(-1);
    }
  }

  void get_csr_data(int* nvars, const int* vars[], int* ncon, const int* cons[],
                    const int* jac_rowp[], const int* jac_cols[],
                    const int* hess_rowp[],
                    const int* hess_cols[]) const override {
    if (nvars) *nvars = n_;
    if (vars) *vars = si_->get_array();
    if (ncon) *ncon = n_;
    if (cons) *cons = ci_->get_array();
    if (jac_rowp) *jac_rowp = jac_rowp_->get_array();
    if (jac_cols) *jac_cols = jac_cols_->get_array();
    if (hess_rowp) *hess_rowp = nullptr;
    if (hess_cols) *hess_cols = nullptr;
  }

 private:
  int n_;
  std::shared_ptr<Vector<int>> si_, ci_;
  std::shared_ptr<Vector<int>> jac_rowp_, jac_cols_;
  std::shared_ptr<Vector<int>> loc_si_, loc_ci_;
};

}  // namespace amigo

#endif  // AMIGO_SLACK_COUPLING_H
