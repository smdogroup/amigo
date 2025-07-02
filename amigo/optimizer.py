import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from .amigo import InteriorPointOptimizer
from .model import ModelVector


class DirectScipySolver:
    def __init__(self, problem):
        self.problem = problem
        self.hess = self.problem.create_csr_matrix()
        self.nrows, self.ncols, self.nnz, self.rowp, self.cols = (
            self.hess.get_nonzero_structure()
        )

    def solve(self, x, diag, bx, px):
        """
        Solve the KKT syste - many different appraoches could be inserted here
        """

        # Compute the Hessian
        self.problem.hessian(x, self.hess)
        self.hess.add_diagonal(diag)

        data = self.hess.get_data()
        H = csr_matrix((data, self.cols, self.rowp), shape=(self.nrows, self.ncols))

        # Compute the solution using scipy
        px.get_array()[:] = spsolve(H, bx.get_array())

        return


class Optimizer:
    def __init__(self, model, x=None, lower=None, upper=None, solver=None):
        self.model = model
        self.problem = self.model.get_opt_problem()

        # Get the vector of initial values if none are specified
        if x is None:
            x = model.get_values_from_meta("value")
            self.x = x.get_opt_problem_vec()
        elif isinstance(x, ModelVector):
            self.x = x.get_opt_problem_vec()
        else:
            self.x = x

        # Get the lower and upper bounds if none are specified
        if lower is None:
            lower = model.get_values_from_meta("lower")
            self.lower = lower.get_opt_problem_vec()
        elif isinstance(lower, ModelVector):
            self.lower = lower.get_opt_problem_vec()
        else:
            self.lower = lower
        if upper is None:
            upper = model.get_values_from_meta("upper")
            self.upper = upper.get_opt_problem_vec()
        elif isinstance(upper, ModelVector):
            self.upper = upper.get_opt_problem_vec()
        else:
            self.upper = upper

        # Set the solver for the KKT system
        if solver is None:
            self.solver = DirectScipySolver(self.problem)
        else:
            self.solver = solver

        # Create the interior point optimizer object
        self.optimizer = InteriorPointOptimizer(self.problem, self.lower, self.upper)

        # Create data that will be used in conjunction with the optimizer
        self.vars = self.optimizer.create_opt_vector(self.x)
        self.res = self.optimizer.create_opt_vector()
        self.update = self.optimizer.create_opt_vector()
        self.temp = self.optimizer.create_opt_vector()

        # Create vectors that store problem-specific information
        self.grad = self.problem.create_vector()
        self.diag = self.problem.create_vector()
        self.px = self.problem.create_vector()
        self.bx = self.problem.create_vector()

    def write_state_vars(self, vars):

        lb = self.lower.get_array()
        ub = self.upper.get_array()

        x = vars.x.get_array()
        xs = vars.xs.get_array()
        zl = vars.zl.get_array()
        zu = vars.zu.get_array()

        num_vars = len(x)
        for i in range(num_vars):
            if i % 50 == 0:
                line = f"{'idx':>4s} {'x':>10s} {'xs':>10s} "
                line += f"{'zl':>10s} {'zu':>10s} {'lb':>10s} {'ub':>10s}"
                print(line)

            line = f"{i:4d} {x[i]:10.3e} {xs[i]:10.3e} "
            line += f"{zl[i]:10.3e} {zu[i]:10.3e} "
            line += f"{lb[i]:10.3e} {ub[i]:10.3e}"
            print(line)

    def write_log(self, iteration, iter_data):
        # Write out to the log information about this line
        if iteration % 10 == 0:
            line = f"{'iteration':>10s} "
            for name in iter_data:
                if name != "iteration":
                    line += f"{name:>15s} "
            print(line)

        line = f"{iteration:10d} "
        for name in iter_data:
            if name != "iteration":
                data = iter_data[name]
                if isinstance(data, (int, np.integer)):
                    line += f"{iter_data[name]:15d} "
                else:
                    line += f"{iter_data[name]:15.6e} "
        print(line)

        return

    def get_options(self, options={}):
        default = {
            "max_iterations": 100,
            "barrier_strategy": "monotone",
            "monotone_barrier_fraction": 0.1,
            "convergence_tolerance": 1e-6,
            "fraction_to_boundary": 0.95,
            "initial_barrier_param": 1.0,
            "max_line_search_iterations": 10,
            "check_update_step": False,
            "backtracting_factor": 0.5,
            "record_components": [],
        }

        default.update(options)
        return default

    def optimize(self, options={}):
        """
        Optimize the problem with the specified input options
        """

        # Get the set of options
        options = self.get_options(options=options)

        # Data that is recorded at each iteration
        opt_data = {"options": options, "converged": False, "iterations": []}

        barrier_param = options["initial_barrier_param"]
        max_iters = options["max_iterations"]
        tau = options["fraction_to_boundary"]
        tol = options["convergence_tolerance"]
        record_components = options["record_components"]

        # Create a view into x using the component indices
        xview = ModelVector(self.model, x=self.vars.x)

        self.optimizer.initialize_multipliers_and_slacks(self.vars)

        # Compute the gradient
        self.problem.gradient(self.vars.x, self.grad)

        line_iters = 0
        alpha_prev = 0.0
        for i in range(max_iters):
            iter_data = {}

            # Compute the complete KKT residual
            res_norm = self.optimizer.compute_residual(
                barrier_param, self.vars, self.grad, self.res
            )

            # Set information about the residual norm into the
            iter_data = {
                "iteration": i,
                "residual": res_norm,
                "barrier_param": barrier_param,
                "line_iters": line_iters,
                "alpha": alpha_prev,
            }

            self.write_log(i, iter_data)

            iter_data["x"] = {}
            for name in record_components:
                iter_data["x"][name] = xview[name].tolist()

            opt_data["iterations"].append(iter_data)

            barrier_converged = False
            if barrier_param <= 0.1 * tol and res_norm < tol:
                opt_data["converged"] = True
                break
            elif res_norm < 0.1 * barrier_param:
                barrier_converged = True

            # Check if the barrier problem has converged
            if barrier_converged:
                frac = options["monotone_barrier_fraction"]
                barrier_param *= frac

                # Re-compute the residuals with the updated barrier parameter
                res_norm = self.optimizer.compute_residual(
                    barrier_param, self.vars, self.grad, self.res
                )

            # Compute the reduced residual for the right-hand-side of the KKT system
            self.optimizer.compute_reduced_residual(self.vars, self.res, self.bx)

            # Add the diagonal contributions to the Hessian matrix
            self.optimizer.compute_diagonal(self.vars, self.diag)

            # Solve the KKT system
            self.solver.solve(self.vars.x, self.diag, self.bx, self.px)

            # Compute the full update based on the reduced variable update
            self.optimizer.compute_update_from_reduced(
                self.vars, self.res, self.px, self.update
            )

            # Check the update
            if options["check_update_step"]:
                self.optimizer.check_update(self.vars, self.res, self.update, self.hess)

            # Compute the max step in the multipliers
            alpha_x, alpha_z = self.optimizer.compute_max_step(
                tau, self.vars, self.update
            )

            # Compute the step length
            alpha = min(alpha_x, alpha_z)

            max_line_iters = options["max_line_search_iterations"]
            line_iters = 1
            for j in range(max_line_iters):
                # Copy the variable values
                self.temp.copy(self.vars)

                # Apply the update to get the new variable values at candidate step length alpha
                self.optimizer.apply_step_update(alpha, alpha, self.update, self.temp)

                # Compute the gradient at the new point
                self.problem.gradient(self.temp.x, self.grad)
                res_norm_new = self.optimizer.compute_residual(
                    barrier_param, self.temp, self.grad, self.res
                )

                if res_norm_new < res_norm or j == max_line_iters - 1:
                    self.vars.copy(self.temp)
                    break
                else:
                    line_iters += 1

                    # Apply a simple backtracking algorithm
                    alpha *= options["backtracting_factor"]

            alpha_prev = alpha

        return opt_data
