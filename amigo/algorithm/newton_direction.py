"""Gradient evaluation, KKT factorization, and Newton solve.

Evaluates the gradient, assembles and factorizes the KKT matrix
(delegating inertia correction to InertiaCorrector when the solver
supports it), solves the condensed augmented system, asks the solver
to iteratively refine the result, and back-substitutes for the bound
duals.  The iterative-refinement kernel itself lives on the
LinearSolver base class.
"""

import numpy as np


class NewtonDirection:
    """Gradient evaluation, KKT factorization, and augmented-system solve."""

    def _update_gradient(self, x):
        """Evaluate problem functions and gradient at x."""
        alpha = self._obj_scale
        if self.distribute:
            self.mpi_problem.update(x)
            self.mpi_problem.gradient(alpha, x, self.grad)
        else:
            self.problem.update(x)
            self.problem.gradient(alpha, x, self.grad)
        self.optimizer.apply_gradient_scaling(self.grad)

    def _factorize_kkt(
        self,
        x,
        diag_base,
        inertia_corrector,
        mult_ind,
        options,
        zero_hessian_indices,
        zero_hessian_eps,
        comm_rank,
    ):
        """Assemble, regularize, and factorize the KKT matrix.

        If an InertiaCorrector is available, delegates to its factorize()
        method which implements Algorithm IC (inertia correction with
        exponential growth of delta_w).  Otherwise, factors directly with
        a simple fallback regularization on failure.

        Returns True if factorization succeeded (correct inertia or no
        inertia check), False if inertia correction exhausted its budget.
        """
        self.diag.get_array()[:] = diag_base
        self.diag.copy_host_to_device()

        if inertia_corrector is not None and hasattr(self.solver, "assemble_hessian"):
            return inertia_corrector.factorize(
                self.solver,
                x,
                self.diag,
                diag_base,
                zero_hessian_indices,
                zero_hessian_eps,
                comm_rank,
                max_corrections=options["max_inertia_corrections"],
                inertia_tolerance=options.get("inertia_tolerance", 0),
                obj_scale=self._obj_scale,
                hessian_scaling_fn=self._hessian_scaling_fn,
            )
        else:
            try:
                self.solver.factor(
                    self._obj_scale,
                    x,
                    self.diag,
                    post_hessian=self._hessian_scaling_fn,
                )
            except Exception:
                # Factorization failed: add primal regularization and retry
                diag_arr = self.diag.get_array()
                if mult_ind is not None:
                    diag_arr[~mult_ind] += 1e-4
                else:
                    diag_arr += 1e-4
                self.diag.copy_host_to_device()
                self.solver.factor(
                    self._obj_scale,
                    x,
                    self.diag,
                    post_hessian=self._hessian_scaling_fn,
                )
            return True

    def _solve_with_mu(self, mu, inertia_corrector=None, mult_ind=None):
        """Solve the augmented KKT system for the Newton direction.

        Full-space solve flow:
          1. Build condensed RHS (8-block to 4-block)
          2. Solve augmented system: K * px = res
          3. Iterative refinement on the full unreduced 8-block system
          4. Back-substitute for bound dual steps

        Requires _factorize_kkt() to have been called first.
        """
        # Step 1: Condensed RHS (8-block to 4-block)
        self.optimizer.compute_residual(mu, self.vars, self.grad, self.res)
        self.res.copy_device_to_host()
        rhs_copy = self.res.get_array().copy()

        # Step 2: Solve augmented system
        self.solver.solve(self.res, self.px)

        # Step 3: Iterative refinement on full 8-block system (solver owns it)
        if (
            inertia_corrector is not None
            and mult_ind is not None
            and hasattr(self.solver, "hess")
        ):
            self.px.copy_device_to_host()
            lbx, ubx = self._get_relaxed_bounds()
            self.solver.iterative_refinement(
                mu,
                mult_ind,
                rhs_copy,
                self.vars,
                self.grad,
                lbx,
                ubx,
                self.px,
                self.res,
                self.ir_corr,
            )

        # Step 4: Back-substitute for bound duals
        self.optimizer.compute_update(mu, self.vars, self.px, self.update)

    def _get_relaxed_bounds(self):
        """Return (lbx, ubx) with the relaxation used by the KKT assembly.

        Prefers the backend's get_lbx_relaxed/get_ubx_relaxed; falls back
        to applying the default relaxation if the backend doesn't expose
        relaxed accessors.
        """
        # TODO: move to backend: bounds should stay in C++ and the solver's iterative_refinement should fetch them itself instead
        if hasattr(self.optimizer, "get_lbx_relaxed"):
            lbx = np.array(self.optimizer.get_lbx_relaxed())
            ubx = np.array(self.optimizer.get_ubx_relaxed())
            return lbx, ubx

        lbx_orig = np.array(self.optimizer.get_lbx())
        ubx_orig = np.array(self.optimizer.get_ubx())
        brf = 1e-8
        cvt = 1e-4
        lbx = lbx_orig.copy()
        ubx = ubx_orig.copy()
        fl_orig = np.isfinite(lbx_orig)
        fu_orig = np.isfinite(ubx_orig)
        if np.any(fl_orig):
            delta_l = np.minimum(cvt, brf * np.maximum(1.0, np.abs(lbx_orig[fl_orig])))
            lbx[fl_orig] = lbx_orig[fl_orig] - delta_l
        if np.any(fu_orig):
            delta_u = np.minimum(cvt, brf * np.maximum(1.0, np.abs(ubx_orig[fu_orig])))
            ubx[fu_orig] = ubx_orig[fu_orig] + delta_u
        return lbx, ubx

    def _find_direction(
        self,
        x,
        diag_base,
        inertia_corrector,
        mult_ind,
        options,
        zero_hessian_indices,
        zero_hessian_eps,
        comm_rank,
    ):
        """Compute the Newton search direction: factorize KKT + back-solve.

        This is the main "direction finding" routine called once per
        iteration.  Combines _factorize_kkt (assemble + inertia correction)
        with _solve_with_mu (RHS + back-solve + update extraction).

        Returns True if factorization succeeded, False if inertia
        correction failed (direction is unreliable and should not be used).
        """
        ok = self._factorize_kkt(
            x,
            diag_base,
            inertia_corrector,
            mult_ind,
            options,
            zero_hessian_indices,
            zero_hessian_eps,
            comm_rank,
        )
        if not ok:
            return False
        self._solve_with_mu(self.barrier_param, inertia_corrector, mult_ind)
        return True
