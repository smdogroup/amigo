"""Constraint-multiplier initialization.

The least-squares estimate solves the normal-equation system
[I, A^T; A, 0] for the y that minimizes the dual infeasibility norm.
The same solve is reused during the iteration to refresh the
multipliers once the iterate is nearly feasible.
"""


class MultiplierInitializer:
    def __init__(self, options, model, problem, optimizer):
        self.options = options
        self.model = model
        self.problem = problem
        self.optimizer = optimizer
        self.temp_con = problem.create_constraint_vector()

    def initialize_multipliers(self, evaluator, solver, state):
        """Initialize the constraint multipliers in state.current."""
        if self.options["init_least_squares_multipliers"]:
            self.compute_least_squares_multipliers(evaluator, solver, state)
        return

    def recompute_multipliers(self, evaluator, solver, state):
        """Refresh the constraint multipliers once the iterate is nearly
        feasible, which keeps the dual variables from diverging.

        The multipliers are recomputed only when recompute_multipliers is
        enabled, the least-squares estimate is in use, and the primal
        infeasibility is below recompute_multiplier_tol; otherwise they are
        left unchanged.
        """
        if (
            self.options["recompute_multipliers"]
            and self.options["init_least_squares_multipliers"]
            and state.primal_infeas < self.options["recompute_multiplier_tol"]
        ):
            self.compute_least_squares_multipliers(evaluator, solver, state)
        return

    def compute_least_squares_multipliers(self, evaluator, solver, state):
        """Set the constraint multipliers to their least-squares estimate.

        Solves the normal-equation system

            [ I  A^T ] [ w      ]   [ -(grad_f - zl + zu) ]
            [ A   0  ] [ lambda ] = [         0           ]

        whose (1,1) block is the identity, so lambda minimizes the dual
        infeasibility norm; w is discarded.  Used to initialize the
        multipliers and, when recompute_multipliers is set, to refresh them
        once the iterate is nearly feasible.
        """
        x = state.get_current_point()
        primal_indices = self.problem.get_primal_indices()
        con_indices = self.problem.get_constraint_indices()

        # Zero the multipliers so the gradient holds the objective term only
        x.fill_at(con_indices, 0.0)
        state.invalidate()

        # Objective gradient at the current obj_scale, constraint Hessian at obj_scale=0
        obj_scale_store = state.obj_scale
        evaluator.evaluate_gradient(state)
        state.obj_scale = 0.0
        evaluator.evaluate_hessian(state)
        state.obj_scale = obj_scale_store

        # Identity (1,1) block gives the least-squares normal equations
        state.diagonal.zero()
        state.diagonal.fill_at(primal_indices, 1.0)
        solver.factor(state.hessian, state.diagonal)

        # Right-hand side -(grad_f - zl + zu) with constraint rows zeroed
        self.optimizer.compute_dual_residual(
            state.current, state.gradient, state.residual
        )
        state.residual.scale(-1.0)
        state.residual.fill_at(con_indices, 0.0)

        update = state.step.get_solution()
        solver.solve(state.residual, update)

        # Adopt the estimate only when its magnitude is within the cap, else keep the dual at zero
        update.get_values_at(con_indices, self.temp_con)
        if self.problem.maxabs(self.temp_con) <= self.options["constr_mult_init_max"]:
            x.copy_at(con_indices, update)

        state.invalidate()
        return
