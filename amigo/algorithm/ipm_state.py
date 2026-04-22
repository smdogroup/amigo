"""Per-iteration state carried through the optimization loop.

Holds the scalar counters, step-size history, quality-function mode,
and filter-reset/rejection counts that change every iteration.  Static
configuration lives in the options dict; longer-lived algorithmic
state (reference points for globalization, filter theta_0) stays on
the Optimizer instance.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IpmState:
    """Mutable per-iteration state of the interior-point loop."""

    # Step tracking (updated after each accepted step)
    line_iters: int = 0
    alpha_x_prev: float = 0.0
    alpha_z_prev: float = 0.0
    x_index_prev: int = -1
    z_index_prev: int = -1

    # Convergence tracking
    prev_res_norm: float = float("inf")
    acceptable_counter: int = 0
    precision_floor_count: int = 0

    # Quality-function bounds and mode (set on first iteration)
    qf_mu_min: float = 0.0
    qf_mu_max: float = -1.0
    qf_free_mode: bool = True
    qf_monotone_mu: Optional[float] = None

    # Rejection tracking
    consecutive_rejections: int = 0
    zero_step_count: int = 0

    # Classical-barrier filter monotone fallback
    filter_monotone_mode: bool = False
    filter_monotone_mu: Optional[float] = None

    # Filter reset heuristic
    count_successive_filter_rejections: int = 0
    filter_reset_count: int = 0

    # Barrier parameter at the start of the most recent residual eval
    res_norm_mu: float = 0.0
