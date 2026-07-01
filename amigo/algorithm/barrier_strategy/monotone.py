"""Monotone barrier strategy"""

from .base import BarrierStrategy, BarrierInfo


class MonotoneBarrierStrategy(BarrierStrategy):
    def __init__(self, options, problem, optimizer):
        super().__init__(options)
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

    def update_barrier(self, evaluator, state):
        info = BarrierInfo()
        kappa_eps = self.options["barrier_progress_tol"]

        # Reduce mu once the subproblem error is below kappa_eps*mu.
        if state.kkt_error < kappa_eps * state.mu:
            # Superlinear decrease, floored at the solver tolerance.
            kappa_mu = self.options["mu_linear_decrease_factor"]
            theta_mu = self.options["mu_superlinear_decrease_power"]
            floor = (
                min(self.options["convergence_tolerance"], self.options["compl_inf_tol"])
                / (kappa_eps + 1.0)
            )
            mu_new = min(kappa_mu * state.mu, state.mu**theta_mu)
            mu_new = max(mu_new, floor, self.options["mu_min"])

            info.new_barrier = True
            info.mu_old = state.mu
            info.mu_new = mu_new

            self.set_mu(state, mu_new)
            state.invalidate(grad=False, hess=False)

        return info
