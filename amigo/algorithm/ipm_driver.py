"""Primal-dual interior-point optimizer.

The Optimizer class composes the algorithmic pieces from the sibling modules
and runs the main iteration loop.  Each iteration evaluates the KKT
residual, checks convergence, updates the barrier parameter, computes
a Newton direction, runs a line search, and handles step acceptance
or feasibility restoration.
"""

import time
import numpy as np

from ..model import ModelVector

from .default_options import get_default_options
from .filter_acceptance import Filter
from .filter_line_search import FilterLineSearch, WatchdogState
from .merit_line_search import MeritLineSearch
from .ipm_state import IpmState

from .problem_setup import ProblemSetup
from .iterate_initialization import IterateInitialization
from .convergence_check import (
    ConvergenceCheck,
    CONVERGED,
    CONVERGED_ACCEPTABLE,
    DIVERGED,
    PRECISION_FLOOR,
)
from .barrier_heuristic import BarrierHeuristic
from .barrier_quality_function import BarrierQualityFunction
from .barrier_adaptive_mu import BarrierAdaptiveMu
from .barrier_update import BarrierUpdate
from .newton_direction import NewtonDirection
from .optimality_scaling import OptimalityScaling
from .bound_safeguards import BoundSafeguards
from .multiplier_initialization import MultiplierInitialization
from .feasibility_restoration import FeasibilityRestoration
from .iteration_logger import IterationLogger
from .newton_diagnostics import NewtonDiagnostics
from .post_optimization import PostOptimization


class Optimizer(
    ProblemSetup,
    IterateInitialization,
    ConvergenceCheck,
    BarrierHeuristic,
    BarrierQualityFunction,
    BarrierAdaptiveMu,
    BarrierUpdate,
    NewtonDirection,
    OptimalityScaling,
    BoundSafeguards,
    MultiplierInitialization,
    MeritLineSearch,
    FilterLineSearch,
    FeasibilityRestoration,
    IterationLogger,
    NewtonDiagnostics,
    PostOptimization,
):
    """Primal-dual interior-point optimizer with filter line search.

    All algorithmic components are inherited from focused mixin classes.
    This class defines only __init__, optimize(), and get_options().
    """

    def __init__(
        self,
        model,
        x=None,
        lower=None,
        upper=None,
        solver=None,
        comm=None,
        distribute=False,
    ):
        """Initialize the optimizer.

        Parameters
        ----------
        model : Model
            The amigo model to optimize
        x : array-like, optional
            Initial point
        lower, upper : array-like, optional
            Variable bounds
        solver : Solver, optional
            Linear solver for the KKT system.
            May also be a string from ["scipy", "pardiso", "mumps"]
        comm : MPI communicator, optional
            For distributed optimization
        distribute : bool
            Whether to distribute the problem
        """
        self.barrier_param = 1.0
        self.model = model
        self.problem = self.model.get_problem()
        self.comm = comm
        self.distribute = distribute
        if self.distribute and self.comm is None:
            raise ValueError("If problem is distributed, communicator cannot be None")

        self._partition_problem()
        self._setup_initial_vectors(x, lower, upper)
        self._fill_slack_bounds()
        self._distribute_vectors()
        self._select_solver(solver)
        self._create_interior_point_backend()
        self._allocate_working_vectors()

    def get_options(self, options={}):
        return get_default_options(options)

    def optimize(self, options={}):
        """Run the interior-point optimization algorithm.

        Returns a dict with keys "converged", "iterations", "options".
        """
        start_time = time.perf_counter()
        comm_rank = self.comm.rank if self.comm is not None else 0

        options = self.get_options(options=options)
        opt_data = {"options": options, "converged": False, "iterations": []}

        max_iters = options["max_iterations"]
        base_tau = options["fraction_to_boundary"]
        tau_min = options["tau_min"]
        use_adaptive_tau = options["adaptive_tau"]
        tol = options["convergence_tolerance"]
        compl_inf_tol = options["compl_inf_tol"]
        record_components = options["record_components"]
        continuation_control = options["continuation_control"]
        max_rejections = options["max_consecutive_rejections"]
        barrier_inc = options["barrier_increase_factor"]
        initial_barrier = options["initial_barrier_param"]
        filter_reset_trigger = options["filter_reset_trigger"]
        max_filter_resets = options["max_filter_resets"]

        self.barrier_param = options["initial_barrier_param"]

        # Initialization
        self._initialize_iterate(options, comm_rank)

        x = self.vars.get_solution()
        xview = ModelVector(self.model, x=x) if not self.distribute else None

        # Loop state
        state = IpmState(qf_mu_min=options["mu_min"])
        state.res_norm_mu = self.barrier_param

        # Inertia corrector + zero-Hessian indices + QF state
        problem_ref = self.mpi_problem if self.distribute else self.problem
        mult_ind = np.array(problem_ref.get_multiplier_indicator(), dtype=bool)
        inertia_corrector = self._build_inertia_corrector(
            mult_ind, tol, options, comm_rank
        )
        zero_hessian_indices, zero_hessian_eps = self._zero_hessian_indices(
            options, comm_rank
        )

        quality_func = options["barrier_strategy"] == "quality_function"
        qf_state = self._setup_quality_function_state(quality_func, options, mult_ind)
        soc_mult_ind = mult_ind if options["second_order_correction"] else None

        # Filter line search state
        filter_ls = options["filter_line_search"]
        outer_filter = Filter() if filter_ls else None
        inner_filter = (
            Filter(
                gamma_phi=options["filter_gamma_phi"],
                gamma_theta=options["filter_gamma_theta"],
            )
            if filter_ls
            else None
        )
        if filter_ls:
            self._filter_theta_0 = None

        # Watchdog
        watchdog = WatchdogState(self.optimizer)
        watchdog.trigger = options["watchdog_shortened_iter_trigger"]
        watchdog.max_trials = options["watchdog_trial_iter_max"]

        # Main loop
        for i in range(max_iters):
            # Step A: KKT residual
            res_norm = self.optimizer.compute_residual(
                self.barrier_param, self.vars, self.grad, self.res
            )
            state.res_norm_mu = self.barrier_param
            if inertia_corrector:
                inertia_corrector.update_barrier(self.barrier_param)

            res_arr = np.array(self.res.get_array())
            theta_res = np.linalg.norm(res_arr[mult_ind])
            eta_res = np.linalg.norm(res_arr[~mult_ind])

            if filter_ls and self._filter_theta_0 is None:
                self._filter_theta_0 = self._compute_filter_theta()

            if continuation_control is not None:
                continuation_control(i, res_norm)

            # Step B: Log
            elapsed_time = time.perf_counter() - start_time
            iter_data = self._build_iter_data(
                i,
                elapsed_time,
                res_norm,
                state.line_iters,
                state.alpha_x_prev,
                state.alpha_z_prev,
                state.x_index_prev,
                state.z_index_prev,
                inertia_corrector,
                theta_res,
                eta_res,
                filter_ls,
                outer_filter,
                options,
                mult_ind,
            )
            if comm_rank == 0:
                self.write_log(i, iter_data)
            iter_data["x"] = {}
            if xview is not None:
                for name in record_components:
                    iter_data["x"][name] = xview[name].tolist()
            opt_data["iterations"].append(iter_data)

            # Step C: Convergence
            status, _, state.acceptable_counter, state.precision_floor_count = (
                self._check_convergence(
                    i,
                    options,
                    mult_ind,
                    res_norm,
                    state.prev_res_norm,
                    state.acceptable_counter,
                    state.precision_floor_count,
                    comm_rank,
                )
            )
            if status == CONVERGED:
                opt_data["converged"] = True
                break
            if status == CONVERGED_ACCEPTABLE:
                opt_data["converged"] = True
                opt_data["acceptable"] = True
                break
            if status == PRECISION_FLOOR:
                opt_data["converged"] = True
                opt_data["acceptable"] = True
                opt_data["precision_floor"] = True
                break
            if status == DIVERGED:
                break
            state.prev_res_norm = res_norm

            # Step D: Barrier update + direction
            step_rejected = False

            # Zero-step recovery (non-inertia path only)
            if not inertia_corrector:
                state.zero_step_count = self._handle_zero_step_recovery(
                    i,
                    state.alpha_x_prev,
                    state.alpha_z_prev,
                    state.zero_step_count,
                    comm_rank,
                )

            # Barrier diagonal Sigma = Z/S
            self.optimizer.compute_diagonal(self.vars, self.diag)
            self.diag.copy_device_to_host()
            diag_base = self.diag.get_array().copy()

            barrier_before = self.barrier_param

            if quality_func:
                state.qf_mu_min, state.qf_mu_max = self._initialize_qf_bounds(
                    options,
                    tol,
                    compl_inf_tol,
                    state.qf_mu_min,
                    state.qf_mu_max,
                    qf_state,
                )
                state.qf_free_mode, state.qf_monotone_mu, factorize_ok = (
                    self._step_quality_function(
                        options,
                        mult_ind,
                        inertia_corrector,
                        diag_base,
                        x,
                        zero_hessian_indices,
                        zero_hessian_eps,
                        comm_rank,
                        state.qf_free_mode,
                        state.qf_monotone_mu,
                        state.qf_mu_min,
                        state.qf_mu_max,
                        tol,
                        compl_inf_tol,
                    )
                )
            else:
                state.filter_monotone_mu, factorize_ok = self._step_classical(
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
                    state.filter_monotone_mode,
                    state.filter_monotone_mu,
                    tol,
                    compl_inf_tol,
                )

            # Reset line search state when mu changed
            if self.barrier_param != barrier_before:
                if inertia_corrector:
                    inertia_corrector.update_barrier(self.barrier_param)
                if filter_ls and inner_filter is not None:
                    inner_filter.clear()
                    self._filter_theta_0 = self._compute_filter_theta()
                watchdog.reset()
                state.count_successive_filter_rejections = 0
                state.filter_reset_count = 0

            # Inertia correction failed: reject, increase barrier if needed
            if not factorize_ok:
                step_rejected = True
                state.consecutive_rejections += 1
                if comm_rank == 0:
                    print(
                        f"  Inertia correction FAILED "
                        f"({state.consecutive_rejections}x)"
                    )
                self.barrier_param = barrier_before
                state.line_iters = 0
                state.alpha_x_prev = state.alpha_z_prev = 0.0
                state.x_index_prev = state.z_index_prev = -1
                state.consecutive_rejections = self._maybe_increase_barrier(
                    state.consecutive_rejections,
                    max_rejections,
                    barrier_inc,
                    initial_barrier,
                    comm_rank,
                )
                continue

            # Optional Newton diagnostics
            if options["check_update_step"]:
                self._run_check_update_diagnostics(comm_rank)

            rhs_norm = self.optimizer.compute_residual(
                self.barrier_param, self.vars, self.grad, self.res
            )
            if options["check_update_step"] and comm_rank == 0 and i > 0:
                self._print_newton_diagnostics(rhs_norm, state.res_norm_mu, mult_ind)

            # Compute maximum step sizes from fraction-to-boundary
            tau = (
                self._compute_adaptive_tau(self.barrier_param, tau_min)
                if use_adaptive_tau
                else base_tau
            )
            alpha_x, x_index, alpha_z, z_index = self.optimizer.compute_max_step(
                tau, self.vars, self.update
            )
            if options["equal_primal_dual_step"]:
                alpha_x = alpha_z = min(alpha_x, alpha_z)

            # Step E: Line search
            if filter_ls:
                alpha, state.line_iters, step_accepted, filter_rejected = (
                    self._filter_line_search_with_watchdog(
                        alpha_x,
                        alpha_z,
                        inner_filter,
                        options,
                        comm_rank,
                        tau,
                        soc_mult_ind,
                        watchdog,
                        factorize_ok,
                    )
                )

                # Filter reset heuristic
                if step_accepted:
                    if filter_rejected:
                        state.count_successive_filter_rejections += 1
                    else:
                        state.count_successive_filter_rejections = 0
                    if (
                        state.count_successive_filter_rejections >= filter_reset_trigger
                        and state.filter_reset_count < max_filter_resets
                    ):
                        inner_filter.clear()
                        state.filter_reset_count += 1
                        state.count_successive_filter_rejections = 0
                        if comm_rank == 0:
                            print(
                                f"  Filter reset "
                                f"({state.filter_reset_count}/{max_filter_resets})"
                            )

                # Step F: Restoration if LS failed
                if not step_accepted:
                    restored = self._restoration_phase(
                        inertia_corrector,
                        mult_ind,
                        inner_filter,
                        options,
                        comm_rank,
                        x,
                        diag_base,
                        zero_hessian_indices,
                        zero_hessian_eps,
                    )
                    if restored:
                        step_accepted = True
                        state.line_iters = 0
                        watchdog.shortened_iter = 0
                    else:
                        step_rejected = True
                        state.consecutive_rejections += 1
                        if comm_rank == 0:
                            print(
                                f"  Filter+Restoration REJECTED "
                                f"({state.consecutive_rejections}x)"
                            )

                if step_accepted:
                    n_adj = self._ensure_positive_slacks(self.vars, self.barrier_param)
                    if n_adj > 0 and comm_rank == 0:
                        print(f"  Slack adjustment: {n_adj} variable(s)")
                    self.optimizer.reset_bound_multipliers(
                        self.barrier_param,
                        1e10,
                        self.vars,
                    )
                    self._update_gradient(self.vars.get_solution())
            else:

                def _reject_step():
                    nonlocal step_rejected
                    step_rejected = True
                    state.consecutive_rejections += 1

                reject_cb = _reject_step if inertia_corrector else None
                alpha, state.line_iters, step_accepted = self._line_search(
                    alpha_x,
                    alpha_z,
                    options,
                    comm_rank,
                    tau=tau,
                    mult_ind=soc_mult_ind,
                    reject_callback=reject_cb,
                )

            # Step G: Post-step update
            if step_rejected:
                state.alpha_x_prev = state.alpha_z_prev = 0.0
                state.x_index_prev = state.z_index_prev = -1
                self.barrier_param = barrier_before

                if quality_func and state.qf_free_mode:
                    state.qf_free_mode, state.qf_monotone_mu = (
                        self._switch_qf_to_monotone(
                            options,
                            state.qf_mu_min,
                            state.qf_mu_max,
                            comm_rank,
                        )
                    )

                state.consecutive_rejections = self._maybe_increase_barrier(
                    state.consecutive_rejections,
                    max_rejections,
                    barrier_inc,
                    initial_barrier,
                    comm_rank,
                )
                if (
                    quality_func
                    and not state.qf_free_mode
                    and self.barrier_param > barrier_before
                ):
                    state.qf_monotone_mu = self.barrier_param
            else:
                state.alpha_x_prev = alpha * alpha_x
                state.alpha_z_prev = alpha_z if filter_ls else alpha * alpha_z
                state.x_index_prev = x_index
                state.z_index_prev = z_index
                state.consecutive_rejections = 0

                if filter_ls and not watchdog.in_watchdog:
                    if alpha == 1.0 or state.line_iters == 1:
                        watchdog.shortened_iter = 0
                    else:
                        watchdog.shortened_iter += 1

        if comm_rank == 0 and not opt_data.get("converged", False):
            print(f"\n{'='*70}")
            print(f"  Amigo did NOT converge (max iterations: {max_iters})")
            print(f"{'='*70}")
            print(f"  Residual                {res_norm:>20.10e}")
            print(f"  Barrier parameter       {self.barrier_param:>20.10e}")
            print(f"{'='*70}")

        return opt_data
