"""Backtracking line search using the KKT residual as the merit function.

Shortens the step from alpha=1 until Armijo sufficient decrease holds
on phi = ||F_mu(x)||.  After the first full-step rejection the routine
optionally tries one second-order correction to mitigate the Maratos
effect, reusing the existing factorization.
"""


class MeritLineSearch:
    """Backtracking Armijo line search with optional SOC."""

    def _line_search(
        self,
        alpha_x,
        alpha_z,
        options,
        comm_rank,
        tau=0.995,
        mult_ind=None,
        reject_callback=None,
    ):
        """Backtracking line search using KKT residual norm as merit function.

        Parameters
        ----------
        alpha_x, alpha_z : float
            Maximum primal/dual step from fraction-to-boundary rule.
        options : dict
            Solver options.
        comm_rank : int
            MPI rank (only rank 0 prints).
        tau : float
            Fraction-to-boundary parameter.
        mult_ind : bool array or None
            Multiplier indicator for SOC.
        reject_callback : callable or None
            Called when the line search exhausts all backtracking steps.
            If None, falls back to relaxed acceptance (accept small increase).

        Returns
        -------
        alpha : float
            Accepted step size.
        line_iters : int
            Number of backtracking iterations.
        step_accepted : bool
            True if a step was accepted.
        """
        max_iters = options["max_line_search_iterations"]
        armijo_c = options["armijo_constant"]
        use_armijo = options["use_armijo_line_search"]
        backtrack = options["backtracking_factor"]
        use_soc = options["second_order_correction"] and mult_ind is not None

        ls_baseline = self.optimizer.compute_residual(
            self.barrier_param, self.vars, self.grad, self.res
        )
        dphi_0 = -ls_baseline

        if use_soc:
            res_orig = self.res.get_array().copy()
            update_backup = self.optimizer.create_opt_vector()
            update_backup.copy(self.update)
            px_orig = self.px.get_array().copy()

        alpha = 1.0
        soc_attempted = False

        for j in range(max_iters):
            self.optimizer.apply_step_update(
                alpha * alpha_x, alpha * alpha_z, self.vars, self.update, self.temp
            )

            xt = self.temp.get_solution()
            self._update_gradient(xt)

            res_new = self.optimizer.compute_residual(
                self.barrier_param, self.temp, self.grad, self.res
            )

            if use_armijo:
                threshold = ls_baseline + armijo_c * alpha * dphi_0
                acceptable = res_new <= threshold
            else:
                acceptable = res_new < ls_baseline

            if acceptable:
                self.vars.copy(self.temp)
                return alpha, j + 1, True

            # SOC: try once after first full-step rejection
            if (
                use_soc
                and j == 0
                and not soc_attempted
                and res_new <= 10.0 * ls_baseline
            ):
                soc_attempted = True
                try:
                    res_arr = self.res.get_array()
                    c_soc = res_orig.copy()
                    c_soc[mult_ind] = res_arr[mult_ind] + alpha * res_orig[mult_ind]
                    self.res.get_array()[:] = c_soc
                    self.res.copy_host_to_device()

                    self._update_gradient(self.vars.get_solution())

                    self.solver.solve(self.res, self.px)
                    self.optimizer.compute_update(
                        self.barrier_param,
                        self.vars,
                        self.px,
                        self.update,
                    )

                    soc_ax, _, soc_az, _ = self.optimizer.compute_max_step(
                        tau, self.vars, self.update
                    )
                    self.optimizer.apply_step_update(
                        soc_ax, soc_az, self.vars, self.update, self.temp
                    )
                    self._update_gradient(self.temp.get_solution())
                    res_soc = self.optimizer.compute_residual(
                        self.barrier_param,
                        self.temp,
                        self.grad,
                        self.res,
                    )

                    if use_armijo:
                        soc_ok = res_soc <= ls_baseline + armijo_c * dphi_0
                    else:
                        soc_ok = res_soc < ls_baseline

                    if soc_ok:
                        self.vars.copy(self.temp)
                        if comm_rank == 0:
                            print(f"  SOC accepted: {ls_baseline:.2e} -> {res_soc:.2e}")
                        return 1.0, j + 1, True
                except Exception:
                    pass  # SOC failed

                # Restore original direction for backtracking
                self.update.copy(update_backup)
                self.px.get_array()[:] = px_orig
                self.px.copy_host_to_device()

            if j < max_iters - 1:
                alpha *= backtrack
                continue

            # Last iteration: relaxed acceptance / rejection
            if reject_callback is not None:
                if res_new <= ls_baseline:
                    self.vars.copy(self.temp)
                    return alpha, j + 1, True
                reject_callback()
                self._update_gradient(self.vars.get_solution())
                return alpha, j + 1, False
            elif res_new < 1.1 * ls_baseline:
                self.vars.copy(self.temp)
                return alpha, j + 1, True
            else:
                alpha = 0.01
                self.optimizer.apply_step_update(
                    alpha * alpha_x, alpha * alpha_z, self.vars, self.update, self.temp
                )
                self._update_gradient(self.temp.get_solution())
                self.vars.copy(self.temp)
                return alpha, j + 1, True

        return alpha, max_iters, False
