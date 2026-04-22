"""Scaling factors for the KKT error and the scaled barrier error.

Follows the Ipopt scaling rule for convergence tests and barrier
decisions:

    s_c = max(s_max, sum(|z|) / n_bounds) / s_max
    s_d = max(s_max, (sum(|y|) + sum(|z|)) / n_all) / s_max

with s_max = 100.  The scaled barrier error aggregates dual, primal,
and complementarity infeasibility in the infinity norm; the adaptive
mu strategy uses it to decide when the barrier subproblem is solved.
"""

import numpy as np


class OptimalityScaling:
    """Scaling factors for the KKT error and the scaled barrier error."""

    def _compute_optimality_scaling(self, mult_ind):
        """Compute optimality error scaling factors (s_d, s_c)."""
        # TODO: move to backend: backend.optimality_scaling() that returns (s_d, s_c) without reading zl/zu/bounds from Python.
        s_max = 100.0

        # Bound multiplier sums: |z_L| + |z_U|
        zl = np.array(self.vars.get_zl())
        zu = np.array(self.vars.get_zu())
        z_asum = float(np.sum(np.abs(zl)) + np.sum(np.abs(zu)))
        n_bounds = int(
            np.sum(np.isfinite(np.array(self.optimizer.get_lbx())))
            + np.sum(np.isfinite(np.array(self.optimizer.get_ubx())))
        )

        # Constraint multiplier sum: |y_c| + |y_d|
        xlam = self.vars.get_solution().get_array()
        y_asum = float(np.sum(np.abs(xlam[mult_ind])))
        n_constraints = int(np.sum(mult_ind))

        # s_c: bound multiplier scaling
        if n_bounds == 0:
            s_c = 1.0
        else:
            s_c = max(s_max, z_asum / n_bounds) / s_max

        # s_d: dual (stationarity) scaling
        n_all = n_constraints + n_bounds
        if n_all == 0:
            s_d = 1.0
        else:
            s_d = max(s_max, (y_asum + z_asum) / n_all) / s_max

        return s_d, s_c

    def _compute_scaled_barrier_error(self, mu, mult_ind):
        """Scaled barrier KKT error.

        max(dual_inf/s_d, primal_inf, complementarity(mu)/s_c)
        All components use infinity norm.
        """
        d_inf, p_inf, c_inf = self.optimizer.compute_kkt_error_mu(
            mu, self.vars, self.grad
        )
        s_d, s_c = self._compute_optimality_scaling(mult_ind)
        return max(d_inf / s_d, p_inf, c_inf / s_c)
