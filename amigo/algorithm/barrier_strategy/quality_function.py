"""Quality-function (adaptive) barrier strategy.

Picks the next mu by either a Mehrotra predictor-corrector or a golden-
section search on q_L(sigma).  Wraps this oracle in the adaptive-mu
globalization from Nocedal & Wachter (2006): if the QF-picked mu stops
making progress (by KKT error or obj-constr filter), fall back to a
monotone decrease of mu until the subproblem is solved.
"""

import numpy as np

from .base import BarrierStrategy, BarrierInfo


def _golden_section(f, a, b, sigma_tol, qf_tol, max_iters):
    """Golden-section search for minimum of f on [a, b]."""
    gfac = (3.0 - np.sqrt(5.0)) / 2.0  # ~ 0.382
    lo, up = a, b
    m1 = lo + gfac * (up - lo)
    m2 = lo + (1.0 - gfac) * (up - lo)
    q_lo = f(lo)
    q_up = f(up)
    q_m1 = f(m1)
    q_m2 = f(m2)

    for _ in range(max_iters):
        width = up - lo
        if width < sigma_tol * up:
            break
        q_all = (q_lo, q_up, q_m1, q_m2)
        qmin, qmax = min(q_all), max(q_all)
        if qmax > 0 and 1.0 - qmin / qmax < qf_tol:
            break
        if q_m1 > q_m2:
            lo, q_lo = m1, q_m1
            m1, q_m1 = m2, q_m2
            m2 = lo + (1.0 - gfac) * (up - lo)
            q_m2 = f(m2)
        else:
            up, q_up = m2, q_m2
            m2, q_m2 = m1, q_m1
            m1 = lo + gfac * (up - lo)
            q_m1 = f(m1)

    best_sigma, best_q = lo, q_lo
    for s, q in ((m1, q_m1), (m2, q_m2), (up, q_up)):
        if q < best_q:
            best_sigma, best_q = s, q
    return best_sigma, best_q


class QualityFunctionBarrierStrategy(BarrierStrategy):
    """Mehrotra PC / golden-section QF oracle with adaptive-mu globalization."""

    def __init__(self, options, problem, optimizer):
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

        # Scaling factors for the QF (set in initialize())
        self.sd = 1.0
        self.sp = 1.0
        self.sc = 1.0

        # Globalization state
        self.free_mode = True
        self.refs = []  # kkt-error reference values
        self.glob_filter = []  # (f, theta) filter for obj-constr-filter
        self.init_dual_inf = -1.0
        self.init_primal_inf = -1.0

        # One-shot mu bounds (populated on first step)
        self.mu_min = self.options["mu_min"]
        self.mu_max = -1.0
        self._mu_min_default = True

        # Set values from the options
        self.tol = self.options["convergence_tolerance"]
        self.compl_inf_tol = self.options["compl_inf_tol"]

        # Temporary vectors needed for the correction step
        self.px = self.problem.create_vector()
        self.dpx = self.problem.create_vector()
        self.update = self.problem.create_vector()
        self.temp = self.optimizer.create_opt_vector()

    def initialize(self, evaluator, state):
        """Scaling factors, mult_ind, initial QF reference value."""

        if self.options["quality_function_norm_scaling"]:
            n_d, n_p, n_c = self.optimizer.get_kkt_element_counts()
            self.sd = 1.0 / max(n_d, 1)
            self.sp = 1.0 / max(n_p, 1)
            self.sc = 1.0 / max(n_c, 1)

        evaluator.evaluate_residual(state)

        # Seed the kkt-error reference with the initial error
        d0 = state.dual_infeas
        p0 = state.primal_infeas
        c0 = state.complementarity
        init_qf = d0 * self.sd + p0 * self.sp + c0 * self.sc
        self.refs.append(init_qf)

    def update_barrier(self, evaluator, state):
        """Run one QF barrier-update iteration."""

        self._initialize_bounds_once(evaluator, state)

        info = BarrierInfo()
        info.new_barrier = False
        info.mu_old = state.mu

        # Mode switching before direction
        if not self.free_mode:
            if self._sufficient_progress(evaluator, state):
                if state.comm_rank == 0 and self.options.get("verbose_barrier"):
                    print("  QF: switching back to free mode")
                self.free_mode = True
                self._remember_point(evaluator, state)
            else:
                info.new_barrier = self._monotone_reduce(state)

        if self.free_mode:
            info.new_barrier = True
            glob = self.options["adaptive_mu_globalization"]
            if glob == "never-monotone" or self._sufficient_progress(evaluator, state):
                self._remember_point(evaluator, state)
            else:
                info.new_barrier = self._enter_monotone_mode(evaluator, state)

        info.mu_new = state.mu

        return info

    def add_step_correction(self, solver, evaluator, state):
        """The step has been computed here, correct the step according to the algorithm selected"""

        if self.free_mode:
            # If we're in free mode, then use the quality function to predict mu
            mu_new = self._quality_function_mu(solver, evaluator, state)
            mu_new = max(mu_new, self.mu_min, self._lower_safeguard(state))
            mu_new = min(mu_new, self.mu_max)

            self.set_mu(state, mu_new)

            # Invalidate everything but the gradient, hessian and step
            state.invalidate(grad=False, hess=False, step=False)

        return

    def update_after_line_search(self, info, evaluator, state):
        """If free mode rejects, fall back to monotone (unless never-monotone)."""

        if not self.free_mode:
            return
        if self.options["adaptive_mu_globalization"] == "never-monotone":
            return

        # We had rejected steps, update accordingly
        if info.num_search_iters > 1:
            comp, _ = evaluator.evaluate_complementarity(state)
            mu_candidate = self.options["adaptive_mu_monotone_init_factor"] * comp
            mu_candidate = max(mu_candidate, self._lower_safeguard(state), self.mu_min)
            mu_candidate = min(mu_candidate, self.mu_max)

            self.set_mu(state, mu_candidate)
            self.free_mode = False

            self.monotone_mu = mu_candidate
            if state.comm_rank == 0:
                print(f"  QF -> monotone (step rejected): mu_bar={mu_candidate:.3e}")

            state.invalidate(grad=False, hess=False)

        return

    def _initialize_bounds_once(self, evaluator, state):
        """Populate mu bounds and initial-infeasibility refs on first call."""

        options = self.options
        if self._mu_min_default:
            self.mu_min = min(
                options["mu_min"], 0.5 * min(self.tol, self.compl_inf_tol)
            )
            self._mu_min_default = False
        if self.mu_max < 0:
            avg_comp_init, _ = evaluator.evaluate_complementarity(state)
            self.mu_max = options["mu_max_fact"] * max(avg_comp_init, 1.0)
        if self.init_dual_inf < 0:
            evaluator.evaluate_residual(state)
            self.init_dual_inf = max(1.0, state.dual_infeas)
            self.init_primal_inf = max(1.0, state.primal_infeas)

    def _monotone_reduce(self, state):
        """Monotone mu reduction when subproblem is solved."""

        btf = self.options["barrier_tol_factor"]
        barrier_err = state.kkt_error
        if barrier_err > btf * state.mu:
            return False

        kmu = self.options["mu_linear_decrease_factor"]
        tmu = self.options["mu_superlinear_decrease_power"]
        mu_new = min(kmu * state.mu, state.mu**tmu)
        floor = min(self.tol, self.compl_inf_tol) / (btf + 1.0)
        mu_new = max(mu_new, floor, self.mu_min)
        mu_new = min(mu_new, self.mu_max)

        if state.comm_rank == 0 and self.options.get("verbose_barrier"):
            print(f"  QF monotone: {state.mu:.3e} -> {mu_new:.3e}")
        self.monotone_mu = mu_new

        # Invalidate the residual and step since mu has changed
        self.set_mu(state, mu_new)
        state.invalidate(grad=False, hess=False)

        return True

    def _enter_monotone_mode(self, evaluator, state):
        """Free mode lost progress: start monotone phase."""

        self.free_mode = False
        avg_c, _ = evaluator.evaluate_complementarity(state)
        mu_new = self.options["adaptive_mu_monotone_init_factor"] * avg_c
        mu_new = max(mu_new, self._lower_safeguard(state), self.mu_min)
        mu_new = min(mu_new, self.mu_max)

        self.set_mu(state, mu_new)
        self.monotone_mu = mu_new
        state.invalidate(grad=False, hess=False, step=False)

        if state.comm_rank == 0:
            print(f"  QF -> monotone: mu_bar={mu_new:.3e} (avg_comp={avg_c:.3e})")

        return True

    def _sufficient_progress(self, evaluator, state):
        """Is free mode still making progress w.r.t. the chosen globalization?"""
        glob = self.options["adaptive_mu_globalization"]

        if glob == "never-monotone":
            return True

        if glob == "kkt-error":
            num_refs_max = self.options["adaptive_mu_kkterror_red_iters"]
            if len(self.refs) < num_refs_max:
                return True

            curr = self._kkt_quality(evaluator, state)
            red_fact = self.options["adaptive_mu_kkterror_red_fact"]
            return any(curr <= red_fact * ref for ref in self.refs)

        if glob == "obj-constr-filter":
            # Get the constraint gradient at the current point
            evaluator.evaluate_objective_and_infeasibility(state)
            f_curr = state.objective_value + state.log_barrier_value
            theta_curr = state.con_infeasibility

            m1 = min(
                self.options.get("filter_max_margin", 1.0),
                max(f_curr, theta_curr, 1e-30),
            )
            margin = self.options.get("filter_margin_fact", 1e-5) * m1
            for f_filt, theta_filt in self.glob_filter:
                if f_curr + margin < f_filt or theta_curr + margin < theta_filt:
                    return True

            return len(self.glob_filter) == 0

        return True

    def _remember_point(self, evaluator, state):
        """Record the current point as an accepted reference."""
        glob = self.options["adaptive_mu_globalization"]

        if glob == "kkt-error":
            curr = self._kkt_quality(evaluator, state)
            num_refs_max = self.options["adaptive_mu_kkterror_red_iters"]
            if len(self.refs) >= num_refs_max:
                self.refs.pop(0)
            self.refs.append(curr)
        elif glob == "obj-constr-filter":
            evaluator.evaluate_objective_and_infeasibility(state)
            f_curr = state.objective_value + state.log_barrier_value
            theta_curr = state.con_infeasibility

            self.glob_filter.append((f_curr, theta_curr))

    def _lower_safeguard(self, state):
        """Lower mu safeguard based on infeasibility progress."""
        factor = self.options["adaptive_mu_safeguard_factor"]
        if factor == 0.0:
            return 0.0
        d_inf, p_inf, _ = self.optimizer.compute_kkt_error(
            0.0, state.current, state.gradient
        )
        safe = max(
            factor * d_inf / max(self.init_dual_inf, 1.0),
            factor * p_inf / max(self.init_primal_inf, 1.0),
        )
        if self.options["adaptive_mu_globalization"] == "kkt-error" and self.refs:
            safe = min(safe, min(self.refs))
        return safe

    def _kkt_quality(self, evaluator, state):
        """Scalar KKT quality used for kkt-error globalization."""
        # TODO: move to backend - combine scaling/centrality/balancing into
        # one backend.kkt_quality(options) call.

        dual_sq, primal_sq, comp_sq = self.optimizer.compute_kkt_error(
            0.0, state.current, state.gradient
        )
        qf = dual_sq * self.sd + primal_sq * self.sp + comp_sq * self.sc

        centrality = self.options["quality_function_centrality"]
        if centrality != "none" and comp_sq > 0:
            _, xi = evaluator.evaluate_complementarity(state)
            xi = max(xi, 1e-30)
            c_term = comp_sq * self.sc
            if centrality == "log":
                qf -= c_term * np.log(xi)
            elif centrality == "reciprocal":
                qf += c_term / xi
            elif centrality == "cubed-reciprocal":
                qf += c_term / xi**3

        if self.options["quality_function_balancing_term"] == "cubic":
            d_term = dual_sq * self.sd
            p_term = primal_sq * self.sp
            c_term = comp_sq * self.sc
            qf += max(0.0, max(d_term, p_term) - c_term) ** 3

        return qf

    def _quality_function_mu(self, solver, evaluator, state):
        """Pick new mu via Mehrotra PC or golden-section QF search.

        Sets self.px and self.update at the chosen mu for the caller.
        Returns (sigma, new_mu) or None on degenerate complementarity.
        """

        avg_comp, _ = evaluator.evaluate_complementarity(state)
        if avg_comp < 1e-30:
            return 1.0, state.mu
        mu_nat = avg_comp

        # Affine (mu=0) and centering (mu=avg_comp) solves -> one factor
        evaluator.evaluate_residual_from_point(
            0.0, state.current, state.gradient, state.residual
        )

        # Evaluate the KKT error
        dual_inf, primal_inf, _ = self.optimizer.compute_kkt_error(
            0.0, state.current, state.gradient
        )
        # Solve for the update with mu = 0
        solver.solve(state.residual, self.px)

        # Solve for the update with mu = avg_comp
        self.optimizer.compute_residual(
            mu_nat, state.current, state.gradient, state.residual
        )

        # Solve for the update with mu = mu_nat but the same left-hand-side
        solver.solve(state.residual, self.dpx)

        # Set dpx = px(mu = average) - px(mu = 0)
        self.dpx.axpy(-1.0, self.px)

        # Invalidate
        state.invalidate(grad=False, hess=False)

        if self.options["quality_function_predictor_corrector"]:
            return self._mehrotra(state, self.px, self.dpx, mu_nat, avg_comp)

        return self._golden_search(
            evaluator, state, self.px, self.dpx, mu_nat, dual_inf, primal_inf, avg_comp
        )

    def _mehrotra(self, state, px0, dpx, mu_nat, avg_comp):
        # Compute the update based on a step to mu = 0
        self.optimizer.compute_update(0.0, state.current, px0, state.step)
        alpha_aff_x, _, alpha_aff_z, _ = self.optimizer.compute_max_step(
            1.0, state.current, state.step
        )

        # Now apply the update and compute the complementarity at the new point
        self.optimizer.apply_step_update(
            alpha_aff_x, alpha_aff_z, state.current, state.step, self.temp
        )
        mu_aff, _ = self.optimizer.compute_complementarity(self.temp)

        sigma_max = self.options["quality_function_sigma_max"]
        sigma = min((mu_aff / mu_nat) ** 3, sigma_max)
        mu_new = sigma * mu_nat
        mu_new = max(mu_new, self.mu_min)
        mu_new = min(mu_new, self.mu_max)

        if state.comm_rank == 0 and self.options.get("verbose_barrier"):
            print(
                f"  PC: sigma={sigma:.4f}, mu={mu_new:.3e} "
                f"(comp={avg_comp:.3e}, mu_aff={mu_aff:.3e}, "
                f"a_aff=[{alpha_aff_x:.3f},{alpha_aff_z:.3f}])"
            )

        # Compute the sigma value
        sigma_eff = sigma
        if mu_nat > 0:
            sigma_eff = mu_new / mu_nat

        # Set the step to the new update
        px0.axpy(sigma_eff, dpx)

        # Compute the full step
        self.optimizer.compute_update(mu_new, state.current, px0, state.step)

        # Compute the maximum step lengths in the primal and dual directions
        alpha_x, _, alpha_z, _ = self.optimizer.compute_max_step(
            state.tau, state.current, state.step
        )
        state.max_alpha_primal = alpha_x
        state.max_alpha_dual = alpha_z
        state.step_current = True

        # Return the new value for mu
        return mu_new

    def _golden_search(
        self,
        evaluator,
        state,
        px0,
        dpx,
        mu_nat,
        dual_inf,
        primal_inf,
        avg_comp,
    ):
        # Set up tau for trial-step probes
        d_inf_qf, p_inf_qf, c_inf_qf = self.optimizer.compute_kkt_error(
            0.0, state.current, state.gradient
        )

        # Try to get rid of this call??
        s_d_qf, s_c_qf = evaluator.compute_optimality_scaling(state)
        nlp_error_qf = max(d_inf_qf / s_d_qf, p_inf_qf, c_inf_qf / s_c_qf)

        # Set the tau_qf value
        tau_qf = max(self.options["tau_min"], 1.0 - nlp_error_qf)

        sigma_lo_opt = max(
            self.options["quality_function_sigma_min"], self.mu_min / mu_nat
        )
        sigma_up_opt = min(
            self.options["quality_function_sigma_max"], self.mu_max / mu_nat
        )
        n_gs = self.options["quality_function_golden_iters"]
        sigma_tol = self.options["quality_function_section_sigma_tol"]
        qf_tol = self.options["quality_function_section_qf_tol"]
        centrality = self.options["quality_function_centrality"]
        balancing = self.options["quality_function_balancing_term"]

        def _eval(sigma):
            return self._evaluate_qf(
                state,
                sigma,
                px0,
                dpx,
                mu_nat,
                tau_qf,
                dual_inf,
                primal_inf,
                0.0,
                centrality,
                balancing,
            )

        tol_probe = max(1e-4, sigma_tol)
        sigma_1m = 1.0 - tol_probe
        qf_1 = _eval(1.0)
        qf_1m = _eval(sigma_1m)

        if state.comm_rank == 0 and self.options.get("verbose_barrier"):
            print(
                f"  QF slope: qf(1-)={qf_1m:.4e}, qf(1)={qf_1:.4e}, "
                f"search={'>' if qf_1m > qf_1 else '<'}1, "
                f"tau={tau_qf:.6f}, nlp_err={nlp_error_qf:.2e}"
            )

        if qf_1m > qf_1:
            if 1.0 >= sigma_up_opt:
                sigma_star = sigma_up_opt
            else:
                sigma_star, _ = _golden_section(
                    _eval, 1.0, sigma_up_opt, sigma_tol, qf_tol, n_gs
                )
        else:
            gs_up = min(max(sigma_lo_opt, sigma_1m), self.mu_max / mu_nat)
            if sigma_lo_opt >= gs_up:
                sigma_star = sigma_lo_opt
            else:
                sigma_star, _ = _golden_section(
                    _eval, sigma_lo_opt, gs_up, sigma_tol, qf_tol, n_gs
                )

        mu_new = sigma_star * mu_nat
        mu_new = max(mu_new, self.mu_min)
        mu_new = min(mu_new, self.mu_max)

        if state.comm_rank == 0 and self.options.get("verbose_barrier"):
            print(
                f"  QF: sigma={sigma_star:.4f}, mu={mu_new:.3e} "
                f"(comp={avg_comp:.3e})"
            )

        # Compute the sigma value
        sigma_eff = sigma_star
        if mu_nat > 0:
            sigma_eff = mu_new / mu_nat

        # Set the step to the new update
        px0.axpy(sigma_eff, dpx)

        # Compute the full step
        self.optimizer.compute_update(mu_new, state.current, px0, state.step)

        # Compute the maximum step lengths in the primal and dual directions
        alpha_x, _, alpha_z, _ = self.optimizer.compute_max_step(
            state.tau, state.current, state.step
        )
        state.max_alpha_primal = alpha_x
        state.max_alpha_dual = alpha_z
        state.step_current = True

        # Return the new value for mu
        return mu_new

    def _evaluate_qf(
        self,
        state,
        sigma,
        px0,
        dpx,
        mu_nat,
        tau,
        dual_inf,
        primal_inf,
        comp_inf,
        centrality,
        balancing,
    ):
        """q_L(sigma) at the combined step px0 + sigma * dpx."""

        mu_s = sigma * mu_nat
        self.update.copy(px0)
        self.update.axpy(sigma, dpx)

        self.optimizer.compute_update(mu_s, state.current, self.update, state.step)
        alpha_x, _, alpha_z, _ = self.optimizer.compute_max_step(
            tau, state.current, state.step
        )
        self.optimizer.apply_step_update(
            alpha_x, alpha_z, state.current, state.step, self.temp
        )
        # Eq. 4.2: complementarity term uses the plain products, no mu target
        trial_comp_sq = self.optimizer.compute_sum_squared_complementarity(
            0.0, self.temp
        )

        qf = (
            (1.0 - alpha_z) ** 2 * dual_inf * self.sd
            + (1.0 - alpha_x) ** 2 * primal_inf * self.sp
            + trial_comp_sq * self.sc
        )

        if centrality != "none" and trial_comp_sq > 0:
            _, trial_xi = self.optimizer.compute_complementarity(self.temp)
            trial_xi = max(trial_xi, 1e-30)
            if centrality == "log":
                qf -= trial_comp_sq * self.sc * np.log(trial_xi)
            elif centrality == "reciprocal":
                qf += trial_comp_sq * self.sc / trial_xi
            elif centrality == "cubed-reciprocal":
                qf += trial_comp_sq * self.sc / trial_xi**3

        if balancing == "cubic":
            d_term = (1.0 - alpha_z) ** 2 * dual_inf * self.sd
            p_term = (1.0 - alpha_x) ** 2 * primal_inf * self.sp
            c_term = trial_comp_sq * self.sc
            qf += max(0.0, max(d_term, p_term) - c_term) ** 3

        return qf
