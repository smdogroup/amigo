"""Quality-function barrier oracle.

Selects a new barrier parameter mu either with a Mehrotra
predictor-corrector (sigma = (mu_aff / mu)**3 using an affine probe)
or by a golden-section search of sigma in [sigma_min, sigma_max] that
minimizes the quality function q_L(sigma).  Writes the chosen mu into
self.barrier_param and sets self.px / self.update at that mu for the
caller.
"""

import numpy as np


class BarrierQualityFunction:
    """Quality-function oracle for selecting a new barrier parameter."""

    @staticmethod
    def _golden_section(f, a, b, sigma_tol, qf_tol, max_iters):
        """Golden-section search for minimum of *f* on [a, b].

        Stops when the sigma interval is small enough, the quality-function
        values have converged, or *max_iters* is reached.

        Returns *(sigma_opt, qf_opt)*.
        """
        gfac = (3.0 - np.sqrt(5.0)) / 2.0  # ~ 0.382
        sigma_lo, sigma_up = a, b
        sigma_mid1 = sigma_lo + gfac * (sigma_up - sigma_lo)
        sigma_mid2 = sigma_lo + (1.0 - gfac) * (sigma_up - sigma_lo)
        q_lo = f(sigma_lo)
        q_up = f(sigma_up)
        qmid1 = f(sigma_mid1)
        qmid2 = f(sigma_mid2)

        for _ in range(max_iters):
            width = sigma_up - sigma_lo
            if width < sigma_tol * sigma_up:
                break
            q_all = (q_lo, q_up, qmid1, qmid2)
            qmin, qmax = min(q_all), max(q_all)
            if qmax > 0 and 1.0 - qmin / qmax < qf_tol:
                break
            if qmid1 > qmid2:
                sigma_lo = sigma_mid1
                q_lo = qmid1
                sigma_mid1 = sigma_mid2
                qmid1 = qmid2
                sigma_mid2 = sigma_lo + (1.0 - gfac) * (sigma_up - sigma_lo)
                qmid2 = f(sigma_mid2)
            else:
                sigma_up = sigma_mid2
                q_up = qmid2
                sigma_mid2 = sigma_mid1
                qmid2 = qmid1
                sigma_mid1 = sigma_lo + gfac * (sigma_up - sigma_lo)
                qmid1 = f(sigma_mid1)

        # Return the best point among all four
        best_sigma, best_q = sigma_lo, q_lo
        for s, q in ((sigma_mid1, qmid1), (sigma_mid2, qmid2), (sigma_up, q_up)):
            if q < best_q:
                best_sigma, best_q = s, q
        return best_sigma, best_q

    def _evaluate_quality_function(
        self,
        sigma,
        px0,
        dpx,
        mu_nat,
        tau,
        dual_inf,
        primal_inf,
        comp_inf,
        qf_sd,
        qf_sp,
        qf_sc,
        centrality,
        balancing_term,
    ):
        """Evaluate the quality function q_L(sigma).

        The combined step is px(sigma) = px_aff + sigma * (px_cen - px_aff),
        i.e. the exact KKT solution at mu = sigma * avg_comp.

        Returns a scalar quality value.
        """
        mu_s = sigma * mu_nat
        self.px.get_array()[:] = px0 + sigma * dpx
        self.px.copy_host_to_device()
        self.optimizer.compute_update(mu_s, self.vars, self.px, self.update)
        alpha_x, _, alpha_z, _ = self.optimizer.compute_max_step(
            tau, self.vars, self.update
        )
        self.optimizer.apply_step_update(
            alpha_x, alpha_z, self.vars, self.update, self.temp
        )
        trial_comp_sq = self.optimizer.compute_complementarity_sq(self.temp)

        qf = (
            (1.0 - alpha_z) ** 2 * dual_inf * qf_sd
            + (1.0 - alpha_x) ** 2 * primal_inf * qf_sp
            + trial_comp_sq * qf_sc
        )

        # Centrality penalty
        if centrality != "none" and trial_comp_sq > 0:
            trial_comp, trial_xi = self.optimizer.compute_complementarity(self.temp)
            trial_xi = max(trial_xi, 1e-30)
            if centrality == "log":
                qf -= trial_comp_sq * qf_sc * np.log(trial_xi)
            elif centrality == "reciprocal":
                qf += trial_comp_sq * qf_sc / trial_xi
            elif centrality == "cubed-reciprocal":
                qf += trial_comp_sq * qf_sc / trial_xi**3

        # Balancing term
        if balancing_term == "cubic":
            d_term = (1.0 - alpha_z) ** 2 * dual_inf * qf_sd
            p_term = (1.0 - alpha_x) ** 2 * primal_inf * qf_sp
            c_term = trial_comp_sq * qf_sc
            qf += max(0.0, max(d_term, p_term) - c_term) ** 3

        return qf

    def _compute_quality_function_mu(self, mu_min, mu_max, options, comm_rank):
        """Compute new mu via Mehrotra PC or golden-section quality function.

        Returns *(sigma, new_mu)* or *None* on failure.  The search
        direction self.px and self.update are set to the
        combined step at the returned mu.
        """
        avg_comp, xi = self.optimizer.compute_complementarity(self.vars)
        if avg_comp < 1e-30:
            return 1.0, self.barrier_param
        mu_nat = avg_comp  # current average complementarity

        # Solve the two linear systems (one factorization)
        # px0: affine-scaling direction (mu = 0)
        self.optimizer.compute_residual(0.0, self.vars, self.grad, self.res)
        dual_inf, primal_inf, _ = self.optimizer.compute_kkt_error(self.vars, self.grad)

        self.solver.solve(self.res, self.px)
        px0 = self.px.get_array().copy()

        # px1: centering direction (mu = avg_comp)
        self.optimizer.compute_residual(mu_nat, self.vars, self.grad, self.res)

        self.solver.solve(self.res, self.px)
        px1 = self.px.get_array().copy()

        dpx = px1 - px0  # centering component

        # Branch A: Mehrotra predictor-corrector
        if options["quality_function_predictor_corrector"]:
            self.px.get_array()[:] = px0
            self.px.copy_host_to_device()
            self.optimizer.compute_update(0.0, self.vars, self.px, self.update)
            alpha_aff_x, _, alpha_aff_z, _ = self.optimizer.compute_max_step(
                1.0, self.vars, self.update
            )
            self.optimizer.apply_step_update(
                alpha_aff_x, alpha_aff_z, self.vars, self.update, self.temp
            )
            mu_aff, _ = self.optimizer.compute_complementarity(self.temp)

            sigma_max = options["quality_function_sigma_max"]
            sigma = min((mu_aff / mu_nat) ** 3, sigma_max)
            new_mu = sigma * mu_nat

            new_mu = max(new_mu, mu_min)
            new_mu = min(new_mu, mu_max)

            if comm_rank == 0 and options.get("verbose_barrier"):
                print(
                    f"  PC: sigma={sigma:.4f}, mu={new_mu:.3e} "
                    f"(comp={avg_comp:.3e}, mu_aff={mu_aff:.3e}, "
                    f"a_aff=[{alpha_aff_x:.3f},{alpha_aff_z:.3f}])"
                )

            sigma_eff = new_mu / mu_nat if mu_nat > 0 else sigma
            self.px.get_array()[:] = px0 + sigma_eff * dpx
            self.px.copy_host_to_device()
            self.optimizer.compute_update(new_mu, self.vars, self.px, self.update)

            self._delta_aff = px0.copy()
            return sigma, new_mu

        # Branch B: Golden-section quality function search
        d_inf_qf, p_inf_qf, c_inf_qf = self.optimizer.compute_kkt_error_mu(
            0.0, self.vars, self.grad
        )
        s_d_qf, s_c_qf = self._compute_optimality_scaling(self._qf_mult_ind)
        nlp_error_qf = max(d_inf_qf / s_d_qf, p_inf_qf, c_inf_qf / s_c_qf)
        tau_qf = max(options["tau_min"], 1.0 - nlp_error_qf)
        sigma_lo_opt = max(
            options["quality_function_sigma_min"],
            mu_min / mu_nat,
        )
        sigma_up_opt = min(options["quality_function_sigma_max"], mu_max / mu_nat)
        n_gs = options["quality_function_golden_iters"]
        sigma_tol = options["quality_function_section_sigma_tol"]
        qf_tol = options["quality_function_section_qf_tol"]
        centrality = options["quality_function_centrality"]
        balancing = options["quality_function_balancing_term"]
        qf_sd, qf_sp, qf_sc = self._qf_sd, self._qf_sp, self._qf_sc

        def _eval(sigma):
            return self._evaluate_quality_function(
                sigma,
                px0,
                dpx,
                mu_nat,
                tau_qf,
                dual_inf,
                primal_inf,
                0.0,
                qf_sd,
                qf_sp,
                qf_sc,
                centrality,
                balancing,
            )

        tol_probe = max(1e-4, sigma_tol)
        sigma_1m = 1.0 - tol_probe
        qf_1 = _eval(1.0)
        qf_1m = _eval(sigma_1m)

        if comm_rank == 0 and options.get("verbose_barrier"):
            print(
                f"  QF slope: qf(1-)={qf_1m:.4e}, qf(1)={qf_1:.4e}, "
                f"search={'>' if qf_1m > qf_1 else '<'}1, "
                f"tau={tau_qf:.6f}, nlp_err={nlp_error_qf:.2e}"
            )

        if qf_1m > qf_1:
            if 1.0 >= sigma_up_opt:
                sigma_star = sigma_up_opt
            else:
                sigma_star, _ = self._golden_section(
                    _eval, 1.0, sigma_up_opt, sigma_tol, qf_tol, n_gs
                )
        else:
            gs_up = min(max(sigma_lo_opt, sigma_1m), mu_max / mu_nat)
            if sigma_lo_opt >= gs_up:
                sigma_star = sigma_lo_opt
            else:
                sigma_star, _ = self._golden_section(
                    _eval, sigma_lo_opt, gs_up, sigma_tol, qf_tol, n_gs
                )

        new_mu = sigma_star * mu_nat
        new_mu = max(new_mu, mu_min)
        new_mu = min(new_mu, mu_max)

        if comm_rank == 0 and options.get("verbose_barrier"):
            print(
                f"  QF: sigma={sigma_star:.4f}, mu={new_mu:.3e} "
                f"(comp={avg_comp:.3e})"
            )

        sigma_eff = new_mu / mu_nat if mu_nat > 0 else sigma_star
        self.px.get_array()[:] = px0 + sigma_eff * dpx
        self.px.copy_host_to_device()
        self.optimizer.compute_update(new_mu, self.vars, self.px, self.update)
        return sigma_star, new_mu
