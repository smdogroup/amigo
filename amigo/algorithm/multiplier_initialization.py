"""Strategies for initializing constraint and bound multipliers.

The least-squares initializer solves the normal-equation system
[I, A^T; A, 0] to obtain the y that minimizes the dual infeasibility
norm, and falls back to y = 0 if the computed multipliers exceed
lambda_max in magnitude.  The affine-scaling initializer solves the
KKT system at mu = 0 and updates y and z from the resulting affine
step.  The third option sets all multipliers to zero.
"""

import numpy as np


class MultiplierInitialization:
    """Multiplier initialization strategies (least-squares, affine, zero)."""

    def _compute_least_squares_multipliers(self, lambda_max=1e3):
        """Least-squares constraint multiplier initialization (Section 3.6).

        Solves for lambda that minimizes the dual infeasibility norm:

            [ I  A^T ] [ w      ]   [ -(grad_f - zl + zu) ]
            [ A   0  ] [ lambda ] = [         0           ]

        The (1,1) block is I (no Hessian), making this a least-squares
        normal equation. w is discarded; lambda is the multiplier estimate.

        Safeguard: if ||lambda||_inf > lambda_max, discard and set to 0.
        """
        x = self.vars.get_solution()

        # Build RHS: -(grad_f - zl + zu) for primals, 0 for constraints
        self.optimizer.compute_dual_residual_vector(self.vars, self.grad, self.res)
        self.res.get_array()[:] *= -1.0
        self.res.copy_host_to_device()

        # Factor [I, A^T; A, 0]: W_factor=0 (no Hessian), diag=I on primals
        self.diag.zero()
        self.optimizer.set_design_vars_value(1.0, self.diag)
        self.diag.copy_host_to_device()
        self.solver.factor(0.0, x, self.diag, post_hessian=self._hessian_scaling_fn)
        self.solver.solve(self.res, self.px)

        # Safeguard: discard if multipliers are too large
        self.px.copy_device_to_host()
        px_arr = self.px.get_array()
        problem_ref = self.mpi_problem if self.distribute else self.problem
        mult_ind = np.array(problem_ref.get_multiplier_indicator(), dtype=bool)
        lam_vals = px_arr[mult_ind]
        if len(lam_vals) > 0 and np.max(np.abs(lam_vals)) > lambda_max:
            self.optimizer.set_multipliers_value(0.0, x)
            return

        self.optimizer.copy_multipliers(x, self.px)

    def _compute_affine_multipliers(self, beta_min=1.0):
        """Compute the affine scaling initial point (Section 3.6).

        Solves the KKT system with mu=0 to get the affine step, then:
          - Updates the constraint multiplier: y = y + dy
          - Updates bound duals: z = max(z + dz, beta_min)
          - Primals (x, s) are NOT changed
          - Returns the initial barrier mu = avg complementarity
        """
        x = self.vars.get_solution()
        self._update_gradient(x)

        # Solve the KKT system with mu=0 (affine direction)
        mu = 0.0
        self.optimizer.compute_residual(mu, self.vars, self.grad, self.res)
        self.optimizer.compute_diagonal(self.vars, self.diag)
        self.solver.factor(
            self._obj_scale,
            x,
            self.diag,
            post_hessian=self._hessian_scaling_fn,
        )
        self.solver.solve(self.res, self.px)

        # Extract the bound dual steps via back-substitution
        self.optimizer.compute_update(mu, self.vars, self.px, self.update)

        # Update multipliers only (y = y + dy), primals unchanged
        self.optimizer.copy_multipliers(x, self.update.get_solution())

        # Update bound duals: z = max(z + dz, beta_min)
        self.optimizer.compute_affine_start_point(
            beta_min, self.vars, self.update, self.vars
        )

        # Initial barrier = average complementarity at the new point
        barrier, _ = self.optimizer.compute_complementarity(self.vars)
        return max(barrier, beta_min)

    def _zero_multipliers(self, x):
        """Set all multipliers to zero."""
        self.optimizer.set_multipliers_value(0.0, x)
