"""Per-iteration barrier-parameter update and direction computation.

Runs one step of the selected barrier strategy (classical heuristic
or monotone update, or the adaptive quality-function oracle), calls the KKT factorization
and direction-finding routines, and handles the rejection paths
(QF free-to-monotone switch, barrier increase on repeated rejections).
"""


class BarrierUpdate:
    """Barrier-update orchestration per iteration."""

    def _handle_zero_step_recovery(
        self, i, alpha_x_prev, alpha_z_prev, zero_step_count, comm_rank
    ):
        """Escape stuck iterates by increasing the barrier parameter.

        Only active on the non-inertia-corrector path: if three consecutive
        iterations produce a tiny step (max(alpha_x, alpha_z) < 1e-10),
        scale mu up by 10x (capped at 1.0).

        Returns the updated zero_step_count.
        """
        if i > 0 and max(alpha_x_prev, alpha_z_prev) < 1e-10:
            zero_step_count += 1
            if zero_step_count >= 3:
                old_barrier = self.barrier_param
                self.barrier_param = min(self.barrier_param * 10.0, 1.0)
                if comm_rank == 0 and self.barrier_param != old_barrier:
                    print(
                        f"  Zero step recovery: barrier "
                        f"{old_barrier:.2e} -> {self.barrier_param:.2e}"
                    )
                zero_step_count = 0
        else:
            zero_step_count = 0
        return zero_step_count

    def _initialize_qf_bounds(
        self, options, tol, compl_inf_tol, qf_mu_min, qf_mu_max, qf_state
    ):
        """Compute one-shot QF constants on the first iteration.

        On iteration 0: derive qf_mu_min from tolerances, qf_mu_max from
        initial complementarity, and snapshot initial infeasibility as
        the reference for the lower safeguard.

        Returns the updated (qf_mu_min, qf_mu_max) pair.
        """
        if qf_state["mu_min_default"]:
            qf_mu_min = min(options["mu_min"], 0.5 * min(tol, compl_inf_tol))
        if qf_mu_max < 0:
            avg_comp_init, _ = self.optimizer.compute_complementarity(self.vars)
            qf_mu_max = options["mu_max_fact"] * max(avg_comp_init, 1.0)
        if self._qf_init_dual_inf < 0:
            d0_sg, p0_sg, _ = self.optimizer.compute_kkt_error_mu(
                0.0, self.vars, self.grad
            )
            self._qf_init_dual_inf = max(1.0, d0_sg)
            self._qf_init_primal_inf = max(1.0, p0_sg)
        return qf_mu_min, qf_mu_max

    def _step_quality_function(
        self,
        options,
        mult_ind,
        inertia_corrector,
        diag_base,
        x,
        zero_hessian_indices,
        zero_hessian_eps,
        comm_rank,
        qf_free_mode,
        qf_monotone_mu,
        qf_mu_min,
        qf_mu_max,
        tol,
        compl_inf_tol,
    ):
        """Quality-function barrier update + factorize + QF mu oracle.

        Returns (qf_free_mode, qf_monotone_mu, factorize_ok).
        """
        # Mode switching (before direction)
        if not qf_free_mode:
            sufficient = self._adaptive_mu_check_sufficient_progress(options)
            if sufficient:
                if comm_rank == 0 and options.get("verbose_barrier"):
                    print("  QF: switching back to free mode")
                qf_free_mode = True
                self._adaptive_mu_remember_point(options)
            else:
                btf = options["barrier_tol_factor"]
                barrier_err = self._compute_scaled_barrier_error(
                    self.barrier_param, mult_ind
                )
                if barrier_err <= btf * self.barrier_param:
                    old_mu = self.barrier_param
                    kmu = options["mu_linear_decrease_factor"]
                    tmu = options["mu_superlinear_decrease_power"]
                    new_mu = min(kmu * old_mu, old_mu**tmu)
                    mu_floor = min(tol, compl_inf_tol) / (btf + 1.0)
                    new_mu = max(new_mu, mu_floor, qf_mu_min)
                    new_mu = min(new_mu, qf_mu_max)
                    if comm_rank == 0 and options.get("verbose_barrier"):
                        print(f"  QF monotone: {old_mu:.3e} -> {new_mu:.3e}")
                    self.barrier_param = new_mu
                    qf_monotone_mu = new_mu

        if qf_free_mode:
            sufficient = self._adaptive_mu_check_sufficient_progress(options)
            glob = options["adaptive_mu_globalization"]
            if not sufficient and glob != "never-monotone":
                qf_free_mode = False
                avg_c, _ = self.optimizer.compute_complementarity(self.vars)
                new_mu = options["adaptive_mu_monotone_init_factor"] * avg_c
                safe = self._adaptive_mu_lower_safeguard(options)
                new_mu = max(new_mu, safe, qf_mu_min)
                new_mu = min(new_mu, qf_mu_max)
                qf_monotone_mu = new_mu
                self.barrier_param = new_mu
                if comm_rank == 0:
                    print(
                        f"  QF -> monotone: mu_bar={new_mu:.3e} "
                        f"(avg_comp={avg_c:.3e})"
                    )
            elif sufficient:
                self._adaptive_mu_remember_point(options)

        # Factorize KKT system
        factorize_ok = self._factorize_kkt(
            x,
            diag_base,
            inertia_corrector,
            mult_ind,
            options,
            zero_hessian_indices,
            zero_hessian_eps,
            comm_rank,
        )

        if qf_free_mode and factorize_ok:
            result = self._compute_quality_function_mu(
                qf_mu_min, qf_mu_max, options, comm_rank
            )
            if result is not None:
                _, mu_qf = result
                mu_qf = max(mu_qf, qf_mu_min)
                safe = self._adaptive_mu_lower_safeguard(options)
                mu_qf = max(mu_qf, safe)
                mu_qf = min(mu_qf, qf_mu_max)
                self.barrier_param = mu_qf
        elif not qf_free_mode and factorize_ok:
            self._solve_with_mu(self.barrier_param)

        return qf_free_mode, qf_monotone_mu, factorize_ok

    def _step_classical(
        self,
        i,
        options,
        mult_ind,
        inertia_corrector,
        diag_base,
        x,
        zero_hessian_indices,
        zero_hessian_eps,
        comm_rank,
        res_norm,
        filter_monotone_mode,
        filter_monotone_mu,
        tol,
        compl_inf_tol,
    ):
        """Classical barrier update (heuristic or monotone) + direction.

        Returns (filter_monotone_mu, factorize_ok).
        """
        if filter_monotone_mode:
            if res_norm <= options["barrier_progress_tol"] * filter_monotone_mu:
                new_mu = max(
                    filter_monotone_mu * options["monotone_barrier_fraction"],
                    tol,
                )
                if comm_rank == 0:
                    print(
                        f"  Filter monotone: mu "
                        f"{filter_monotone_mu:.2e}->{new_mu:.2e}"
                    )
                filter_monotone_mu = new_mu
            self.barrier_param = filter_monotone_mu

        elif i > 0 and self.barrier_param > min(tol, compl_inf_tol) / (
            options["barrier_tol_factor"] + 1.0
        ):
            kappa_eps = options["barrier_tol_factor"]
            kappa_mu = options["mu_linear_decrease_factor"]
            theta_mu = options["mu_superlinear_decrease_power"]

            heuristic = options["barrier_strategy"] == "heuristic"
            if heuristic:
                comp_h, xi_h = self.optimizer.compute_complementarity(self.vars)
                should_reduce = True
            elif options["progress_based_barrier"]:
                should_reduce = self._should_reduce_barrier(
                    res_norm, self.barrier_param, kappa_eps
                )
            else:
                should_reduce = res_norm <= 0.1 * self.barrier_param

            if should_reduce:
                if heuristic:
                    mu_floor = min(tol, compl_inf_tol) / (kappa_eps + 1.0)
                    self.barrier_param, _ = self._compute_barrier_heuristic(
                        xi_h,
                        comp_h,
                        options["heuristic_barrier_gamma"],
                        options["heuristic_barrier_r"],
                        mu_floor,
                    )
                else:
                    while True:
                        mu = self.barrier_param
                        e_mu = self._compute_scaled_barrier_error(mu, mult_ind)
                        if e_mu > kappa_eps * mu:
                            break
                        old_mu = mu
                        new_mu = min(kappa_mu * mu, mu**theta_mu)
                        mu_fl = min(tol, compl_inf_tol) / (kappa_eps + 1.0)
                        new_mu = max(new_mu, mu_fl)
                        if new_mu >= old_mu:
                            break
                        self.barrier_param = new_mu

        if inertia_corrector:
            inertia_corrector.update_barrier(self.barrier_param)
        factorize_ok = self._find_direction(
            x,
            diag_base,
            inertia_corrector,
            mult_ind,
            options,
            zero_hessian_indices,
            zero_hessian_eps,
            comm_rank,
        )
        return filter_monotone_mu, factorize_ok

    def _maybe_increase_barrier(
        self,
        consecutive_rejections,
        max_rejections,
        barrier_inc,
        initial_barrier,
        comm_rank,
    ):
        """Increase barrier if we've hit max consecutive rejections."""
        if consecutive_rejections >= max_rejections:
            new_barrier = min(self.barrier_param * barrier_inc, initial_barrier)
            if new_barrier > self.barrier_param:
                if comm_rank == 0:
                    print(
                        f"  Barrier increased: {self.barrier_param:.2e} -> "
                        f"{new_barrier:.2e}"
                    )
                self.barrier_param = new_barrier
            elif comm_rank == 0:
                print(
                    f"  Barrier at max ({self.barrier_param:.2e}), "
                    f"cannot increase further"
                )
            return 0
        return consecutive_rejections

    def _switch_qf_to_monotone(self, options, qf_mu_min, qf_mu_max, comm_rank):
        """Step rejection in free mode triggers monotone fallback.

        Returns (qf_free_mode, qf_monotone_mu).
        """
        if options["adaptive_mu_globalization"] == "never-monotone":
            return True, None
        comp, _ = self.optimizer.compute_complementarity(self.vars)
        init_factor = options["adaptive_mu_monotone_init_factor"]
        qf_monotone_mu_cand = init_factor * comp
        safe_mu = self._adaptive_mu_lower_safeguard(options)
        qf_monotone_mu_cand = max(qf_monotone_mu_cand, safe_mu, qf_mu_min)
        qf_monotone_mu_cand = min(qf_monotone_mu_cand, qf_mu_max)
        self.barrier_param = qf_monotone_mu_cand
        if comm_rank == 0:
            print(
                f"  QF -> monotone (step rejected): "
                f"mu_bar={qf_monotone_mu_cand:.3e}"
            )
        return False, qf_monotone_mu_cand
