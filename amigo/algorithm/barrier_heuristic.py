"""LOQO-style data-driven barrier update.

Reduces mu proportional to the average complementarity, with the
proportionality factor determined by the centrality ratio xi.
"""


class BarrierHeuristic:
    """Heuristic (LOQO-style) barrier parameter update."""

    def _compute_barrier_heuristic(self, xi, complementarity, gamma, r, tol=1e-12):
        """Compute LOQO-style barrier parameter."""
        if xi > 1e-10:
            term = (1 - r) * (1 - xi) / xi
            heuristic_factor = min(term, 2.0) ** 3
        else:
            heuristic_factor = 2.0**3

        mu_new = gamma * heuristic_factor * complementarity

        mu_new = max(mu_new, tol)

        return mu_new, heuristic_factor

    def _should_reduce_barrier(self, res_norm, barrier_param, kappa_epsilon=10.0):
        """Reduce barrier when subproblem is solved: res <= kappa_eps * mu."""
        barrier_tol = kappa_epsilon * barrier_param
        return res_norm <= barrier_tol
