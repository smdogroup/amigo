"""Amigo interior-point optimizer package.

Module map, grouped by concern:

  ipm_driver.py                  Optimizer class, main loop
  ipm_state.py                   Per-iteration state dataclass
  problem_setup.py               __init__ and loop-config helpers
  iterate_initialization.py      Primal-dual starting point
  convergence_check.py           Convergence criteria
  default_options.py             Default option values

  barrier_update.py              Per-iteration barrier orchestration
  barrier_heuristic.py           LOQO-style data-driven update
  barrier_quality_function.py    Quality-function oracle
  barrier_adaptive_mu.py         Globalization for the QF strategy

  newton_direction.py            KKT factor, solve, direction
  inertia_correction.py          Algorithm IC (Wachter & Biegler 2006)
  (iterative refinement lives on LinearSolver in solvers/)

  merit_line_search.py           Backtracking Armijo with SOC
  filter_line_search.py          Filter LS + phi_mu, theta, watchdog
  filter_acceptance.py           Two-dimensional filter

  optimality_scaling.py          KKT error scaling (s_d, s_c)
  bound_safeguards.py            Slack flooring + adaptive tau

  multiplier_initialization.py   Least-squares / affine / zero start
  feasibility_restoration.py     Restoration phase
  iteration_logger.py            Iter-data assembly + progress table
  newton_diagnostics.py          Debug diagnostics
  post_optimization.py           Outputs and post-opt derivatives
  solvers/                       Linear solver implementations
"""

from .ipm_driver import Optimizer
from .inertia_correction import InertiaCorrector
from .filter_acceptance import Filter
from .default_options import get_default_options

from .solvers import (
    AmigoSolver,
    DirectCudaSolver,
    LNKSInexactSolver,
    MumpsSolver,
    PardisoSolver,
    DirectPetscSolver,
    DirectScipySolver,
)
