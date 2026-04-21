"""Globalization for the adaptive (quality-function) barrier strategy.

Decides when free-mode is making enough progress and when to fall back
to monotone decrease.  Tracks accepted reference points, computes the
primal-dual KKT quality measure used for the kkt-error rule, and
provides the lower-bound safeguard on mu based on infeasibility
progress.
"""

import numpy as np


class BarrierAdaptiveMu:
    """Globalization helpers for the adaptive (quality-function) mu strategy."""

    def _adaptive_mu_quality_function_pd(self, mult_ind, options):
        """Compute the KKT quality measure for globalization.

        Uses 2-norm-squared scaled by element counts (default).
        """
        # TODO: move to backend - the KKT-error scaling, centrality,
        # and balancing terms should combine in C++ into a single
        # backend.kkt_quality(options) that returns the scalar qf.
        dual_sq, primal_sq, comp_sq = self.optimizer.compute_kkt_error(
            self.vars, self.grad
        )
        qf = dual_sq * self._qf_sd + primal_sq * self._qf_sp + comp_sq * self._qf_sc

        # Centrality penalty
        centrality = options["quality_function_centrality"]
        if centrality != "none" and comp_sq > 0:
            _, xi = self.optimizer.compute_complementarity(self.vars)
            xi = max(xi, 1e-30)
            c_term = comp_sq * self._qf_sc
            if centrality == "log":
                qf -= c_term * np.log(xi)
            elif centrality == "reciprocal":
                qf += c_term / xi
            elif centrality == "cubed-reciprocal":
                qf += c_term / xi**3

        # Balancing term
        if options["quality_function_balancing_term"] == "cubic":
            d_term = dual_sq * self._qf_sd
            p_term = primal_sq * self._qf_sp
            c_term = comp_sq * self._qf_sc
            qf += max(0.0, max(d_term, p_term) - c_term) ** 3

        return qf

    def _adaptive_mu_check_sufficient_progress(self, options):
        """Check if free mode is making sufficient progress."""
        glob = options["adaptive_mu_globalization"]

        if glob == "never-monotone":
            return True

        if glob == "kkt-error":
            refs = self._qf_refs
            num_refs_max = options["adaptive_mu_kkterror_red_iters"]
            if len(refs) < num_refs_max:
                return True
            curr_error = self._adaptive_mu_quality_function_pd(None, options)
            red_fact = options["adaptive_mu_kkterror_red_fact"]
            for ref_val in refs:
                if curr_error <= red_fact * ref_val:
                    return True
            return False

        if glob == "obj-constr-filter":
            f_curr = self._compute_barrier_objective(self.vars)
            theta_curr = self._compute_filter_theta()
            margin = options.get("filter_margin_fact", 1e-5) * min(
                options.get("filter_max_margin", 1.0),
                max(f_curr, theta_curr, 1e-30),
            )
            for f_filt, theta_filt in self._qf_glob_filter:
                if f_curr + margin < f_filt or theta_curr + margin < theta_filt:
                    return True
            return len(self._qf_glob_filter) == 0

        return True

    def _adaptive_mu_remember_point(self, options):
        """Record current point as accepted for globalization."""
        glob = options["adaptive_mu_globalization"]

        if glob == "kkt-error":
            curr_error = self._adaptive_mu_quality_function_pd(None, options)
            num_refs_max = options["adaptive_mu_kkterror_red_iters"]
            if len(self._qf_refs) >= num_refs_max:
                self._qf_refs.pop(0)
            self._qf_refs.append(curr_error)

        elif glob == "obj-constr-filter":
            f_curr = self._compute_barrier_objective(self.vars)
            theta_curr = self._compute_filter_theta()
            self._qf_glob_filter.append((f_curr, theta_curr))

    def _adaptive_mu_lower_safeguard(self, options):
        """Compute lower mu safeguard based on infeasibility progress."""
        factor = options["adaptive_mu_safeguard_factor"]
        if factor == 0.0:
            return 0.0

        d_inf, p_inf, _ = self.optimizer.compute_kkt_error_mu(0.0, self.vars, self.grad)
        safe = max(
            factor * d_inf / max(self._qf_init_dual_inf, 1.0),
            factor * p_inf / max(self._qf_init_primal_inf, 1.0),
        )

        if options["adaptive_mu_globalization"] == "kkt-error" and self._qf_refs:
            safe = min(safe, min(self._qf_refs))

        return safe
