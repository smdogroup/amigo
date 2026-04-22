"""Feasibility restoration phase.

Invoked when the filter line search fails to produce an acceptable
step.  Two modes are supported.  For large theta the routine
temporarily raises the barrier and takes Newton steps aimed at a 10%
constraint-violation reduction.  For small theta the filter is
assumed to be blocking optimality progress; the routine tries
KKT-descent steps and, if none are accepted, resets the filter to
unblock the solve.
"""


class FeasibilityRestoration:
    """Feasibility restoration for filter line search failure."""

    def _restoration_phase(
        self,
        inertia_corrector,
        mult_ind,
        inner_filter,
        options,
        comm_rank,
        x,
        diag_base,
        zero_hessian_indices,
        zero_hessian_eps,
    ):
        """Feasibility restoration phase.

        Returns True if restoration produced a better iterate.
        """
        max_restore = options["filter_restoration_max_iter"]
        backtrack = options["backtracking_factor"]
        max_ls = options["max_line_search_iterations"]
        tau_min = options["tau_min"]
        use_adaptive_tau = options["adaptive_tau"]
        tol = options["convergence_tolerance"]

        theta_k = self._compute_filter_theta()
        theta_start = theta_k

        # Mode 2: Small-theta restoration (filter blocking optimality)
        theta_min_restore = 1e-4 * max(1.0, getattr(self, "_filter_theta_0", theta_k))
        if theta_k < max(tol, theta_min_restore):
            res_current = self.optimizer.compute_residual(
                self.barrier_param, self.vars, self.grad, self.res
            )
            tau = (
                self._compute_adaptive_tau(self.barrier_param, tau_min)
                if use_adaptive_tau
                else options["fraction_to_boundary"]
            )
            ax, _, az, _ = self.optimizer.compute_max_step(tau, self.vars, self.update)

            alpha = 1.0
            for j in range(max_ls):
                self.optimizer.apply_step_update(
                    alpha * ax,
                    az,
                    self.vars,
                    self.update,
                    self.temp,
                )
                self._update_gradient(self.temp.get_solution())
                res_trial = self.optimizer.compute_residual(
                    self.barrier_param, self.temp, self.grad, self.res
                )
                if res_trial < res_current:
                    phi_trial = self._compute_barrier_objective(self.temp)
                    theta_trial = self._compute_filter_theta(self.temp)
                    if inner_filter.is_acceptable(phi_trial, theta_trial):
                        self.vars.copy(self.temp)
                        return True
                    if res_trial < 0.99 * res_current:
                        self.vars.copy(self.temp)
                        inner_filter.clear()
                        self._filter_theta_0 = theta_trial
                        return True
                alpha *= backtrack

            self._update_gradient(self.vars.get_solution())
            inner_filter.clear()
            self._filter_theta_0 = theta_k
            return True

        # Mode 1: Large-theta restoration (reduce constraint violation)
        phi_k = self._compute_barrier_objective(self.vars)
        inner_filter.add(phi_k, theta_k)

        target = 0.9 * theta_k
        saved_barrier = self.barrier_param

        restore_levels = [
            max(saved_barrier, 1e-2),
            0.1,
            1.0,
        ]
        restore_levels = sorted(set(mu for mu in restore_levels if mu >= saved_barrier))

        iters_per_level = max(3, max_restore // len(restore_levels))

        for restore_mu in restore_levels:
            self.barrier_param = restore_mu
            if inertia_corrector:
                inertia_corrector.update_barrier(restore_mu)

            for r in range(iters_per_level):
                x = self.vars.get_solution()
                self._update_gradient(x)
                self.optimizer.compute_diagonal(self.vars, self.diag)
                self.diag.copy_device_to_host()
                diag_r = self.diag.get_array().copy()

                self._find_direction(
                    x,
                    diag_r,
                    inertia_corrector,
                    mult_ind,
                    options,
                    zero_hessian_indices,
                    zero_hessian_eps,
                    comm_rank,
                )

                tau = (
                    self._compute_adaptive_tau(restore_mu, tau_min)
                    if use_adaptive_tau
                    else options["fraction_to_boundary"]
                )
                ax, _, az, _ = self.optimizer.compute_max_step(
                    tau, self.vars, self.update
                )

                alpha = 1.0
                step_taken = False
                for j in range(max_ls):
                    self.optimizer.apply_step_update(
                        alpha * ax,
                        alpha * az,
                        self.vars,
                        self.update,
                        self.temp,
                    )
                    self._update_gradient(self.temp.get_solution())
                    theta_trial = self._compute_filter_theta(self.temp)

                    if theta_trial < theta_k:
                        self.vars.copy(self.temp)
                        theta_k = theta_trial
                        step_taken = True
                        break
                    alpha *= backtrack

                if not step_taken:
                    break

                if theta_k < target:
                    break

            if theta_k < target:
                break

        # Restore original barrier
        self.barrier_param = saved_barrier
        if inertia_corrector:
            inertia_corrector.update_barrier(saved_barrier)
        self._update_gradient(self.vars.get_solution())
        self.optimizer.compute_diagonal(self.vars, self.diag)
        self.diag.copy_device_to_host()

        success = theta_k < theta_start * 0.99
        return success
