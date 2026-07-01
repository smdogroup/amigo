"""Base class for barrier-parameter strategies.

A BarrierStrategy owns the per-iteration barrier update: pick the next mu
(monotone rule, LOQO heuristic, or quality-function oracle) and apply it with
set_mu, which also refreshes the coupled fraction-to-boundary tau. Concrete
strategies operate on the InteriorPointState and query the Evaluator for the
gradient, residual, and complementarity.
"""

from abc import ABC, abstractmethod


class BarrierInfo:
    new_barrier: bool = False
    mu_new: float = 0.0
    mu_old: float = 0.0


class BarrierStrategy(ABC):
    def __init__(self, options={}):
        self.options = options

    def set_mu(self, state, mu):
        """Set the barrier parameter and the coupled fraction-to-boundary."""
        state.mu = mu
        if self.options["adaptive_tau"]:
            state.tau = max(self.options["tau_min"], 1.0 - mu)

    def initialize(self, evaluator, state):
        """Initialize the barrier strategy from the initial point"""
        pass

    @abstractmethod
    def update_barrier(self, evaluator, state) -> BarrierInfo:
        """Update the barrier parameter prior to factoring the KKT matrix"""
        pass

    def add_step_correction(self, solver, evalutor, state):
        """Add the correction to the step - relevant for Mehrotra P/C steps"""
        pass

    def update_after_line_search(self, info, evaluator, state):
        """
        Update any internal state required after the results of a line search

        Default behavior is to adjust the barrier parameter if small steps are taken
        """
        pass
