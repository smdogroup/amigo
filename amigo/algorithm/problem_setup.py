"""Setup helpers used by the Optimizer during construction and loop entry.

Covers vector allocation, solver selection, backend creation, MPI
distribution, slack-bound handling, the inertia corrector, zero-Hessian
indices, and the quality-function globalization state.
"""

import numpy as np

from ..amigo import InteriorPointOptimizer
from ..model import ModelVector

from .solvers import (
    AmigoSolver,
    DirectPetscSolver,
    DirectScipySolver,
    MumpsSolver,
    PardisoSolver,
)

from .inertia_correction import InertiaCorrector


class ProblemSetup:
    """Initialization and loop-config helpers."""

    def _partition_problem(self):
        if self.distribute:
            self.mpi_problem = self.problem.partition_from_root()
            self.mpi_x = self.mpi_problem.create_vector()
            self.mpi_lower = self.mpi_problem.create_vector()
            self.mpi_upper = self.mpi_problem.create_vector()

    def _setup_initial_vectors(self, x, lower, upper):
        if x is None:
            x = self.model.get_values_from_meta("value")
            self.x = x.get_vector()
        elif isinstance(x, ModelVector):
            self.x = x.get_vector()
        else:
            self.x = x

        if lower is None:
            lower = self.model.get_values_from_meta("lower")
            self.lower = lower.get_vector()
        elif isinstance(lower, ModelVector):
            self.lower = lower.get_vector()
        else:
            self.lower = lower
        if upper is None:
            upper = self.model.get_values_from_meta("upper")
            self.upper = upper.get_vector()
        elif isinstance(upper, ModelVector):
            self.upper = upper.get_vector()
        else:
            self.upper = upper

    def _fill_slack_bounds(self):
        if hasattr(self.model, "num_slacks") and self.model.num_slacks > 0:
            lb_arr = self.lower.get_array()
            ub_arr = self.upper.get_array()
            for k in range(self.model.num_slacks):
                idx = self.model.slack_indices[k]
                lb_arr[idx] = self.model._slack_meta[k]["lower"]
                ub_arr[idx] = self.model._slack_meta[k]["upper"]

    def _distribute_vectors(self):
        if self.distribute:
            self.problem.scatter_data_vector(
                self.problem.get_data_vector(),
                self.mpi_problem,
                self.mpi_problem.get_data_vector(),
            )
            self.problem.scatter_vector(self.x, self.mpi_problem, self.mpi_x)
            self.problem.scatter_vector(self.lower, self.mpi_problem, self.mpi_lower)
            self.problem.scatter_vector(self.upper, self.mpi_problem, self.mpi_upper)

    def _select_solver(self, solver):
        """Resolve solver spec (instance, string, or None) to a concrete solver."""
        if solver is None and self.distribute:
            self.solver = DirectPetscSolver(self.comm, self.mpi_problem)
        elif isinstance(solver, str):
            solver_pref = solver.lower()
            if solver_pref == "scipy":
                self.solver = DirectScipySolver(self.problem)
            elif solver_pref == "pardiso":
                self.solver = PardisoSolver(self.problem)
            elif solver_pref == "mumps":
                self.solver = MumpsSolver(self.problem)
            elif solver_pref == "amigo":
                self.solver = AmigoSolver(self.problem)
            else:
                raise ValueError(
                    f"Unknown solver string '{solver}'. "
                    "Expected one of: 'scipy', 'pardiso', 'mumps', 'amigo'."
                )
        elif solver is not None:
            self.solver = solver
        else:
            try:
                self.solver = MumpsSolver(self.problem)
            except (ImportError, Exception):
                try:
                    self.solver = PardisoSolver(self.problem)
                except (ImportError, Exception):
                    self.solver = DirectScipySolver(self.problem)

    def _create_interior_point_backend(self):
        """Create the C++ InteriorPointOptimizer backend and slack mapping."""
        if self.distribute:
            x_vec = self.mpi_x
            self.optimizer = InteriorPointOptimizer(
                self.mpi_problem, self.mpi_lower, self.mpi_upper
            )
            data_vec = self.mpi_problem.get_data_vector()
        else:
            x_vec = self.x
            self.optimizer = InteriorPointOptimizer(
                self.problem, self.lower, self.upper
            )
            data_vec = self.problem.get_data_vector()

        if self.model.num_slacks > 0 and not self.distribute:
            self.optimizer.set_slack_mapping(
                np.ascontiguousarray(self.model.slack_indices, dtype=np.int32),
                np.ascontiguousarray(
                    self.model.ineq_constraint_indices, dtype=np.int32
                ),
            )

        x_vec.copy_host_to_device()
        data_vec.copy_host_to_device()

        self.vars = self.optimizer.create_opt_vector(x_vec)
        self.update = self.optimizer.create_opt_vector()
        self.temp = self.optimizer.create_opt_vector()

    def _allocate_working_vectors(self):
        """Allocate scratch vectors for gradient, residual, direction, etc."""
        if self.distribute:
            self.grad = self.mpi_problem.create_vector()
            self.res = self.mpi_problem.create_vector()
            self.diag = self.mpi_problem.create_vector()
            self.px = self.mpi_problem.create_vector()
        else:
            self.grad = self.problem.create_vector()
            self.res = self.problem.create_vector()
            self.diag = self.problem.create_vector()
            self.px = self.problem.create_vector()
            self.ir_corr = self.problem.create_vector()

    def _build_inertia_corrector(self, mult_ind, tol, options, comm_rank):
        """Create an InertiaCorrector if the solver supports inertia queries."""
        inertia_corrector = None
        if getattr(self.solver, "supports_inertia", False):
            inertia_corrector = InertiaCorrector(mult_ind, self.barrier_param, options)
            if comm_rank == 0:
                n_primal = int(np.sum(~mult_ind))
                n_dual = int(np.sum(mult_ind))
                n_total = len(mult_ind)
                solver_name = type(self.solver).__name__
                print(f"\n  Amigo IPM ({solver_name})")
                print(
                    f"  Variables: {n_total} ({n_primal} primal, {n_dual} constraints)"
                )
                print(f"  Tolerance: {tol:.0e}  mu_init: {self.barrier_param:.0e}\n")
        return inertia_corrector

    def _zero_hessian_indices(self, options, comm_rank):
        """Resolve zero-Hessian variable names to integer indices."""
        zero_hessian_indices = None
        zero_hessian_eps = options["regularization_eps_x_zero_hessian"]
        zh_vars = options["zero_hessian_variables"]
        if zh_vars and not self.distribute:
            zero_hessian_indices = np.sort(self.model.get_indices(zh_vars))
            if comm_rank == 0:
                print(
                    f"  Variable-specific regularization: {len(zero_hessian_indices)} "
                    f"zero-Hessian vars, eps_x_zero={zero_hessian_eps:.2e}"
                )
        return zero_hessian_indices, zero_hessian_eps

    def _setup_quality_function_state(self, quality_func, options, mult_ind):
        """Pre-allocate QF globalization state on self. Returns a small dict."""
        qf_sd = qf_sp = qf_sc = 1.0
        if quality_func and options["quality_function_norm_scaling"]:
            n_d, n_p, n_c = self.optimizer.get_kkt_element_counts()
            qf_sd = 1.0 / max(n_d, 1)
            qf_sp = 1.0 / max(n_p, 1)
            qf_sc = 1.0 / max(n_c, 1)
        self._qf_sd = qf_sd
        self._qf_sp = qf_sp
        self._qf_sc = qf_sc
        self._qf_mult_ind = mult_ind
        self._qf_refs = []
        self._qf_glob_filter = []
        self._qf_init_dual_inf = -1.0
        self._qf_init_primal_inf = -1.0
        self._delta_aff = None

        if quality_func:
            d0, p0, c0 = self.optimizer.compute_kkt_error(self.vars, self.grad)
            init_qf = d0 * qf_sd + p0 * qf_sp + c0 * qf_sc
            self._qf_refs.append(init_qf)

        return {"mu_min_default": True}
