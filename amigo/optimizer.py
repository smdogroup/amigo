import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from .amigo import InteriorPointOptimizer


class Optimizer:
    def __init__(self, model, x=None, lower=None, upper=None):
        self.model = model
        self.prob = self.model.get_opt_problem()

        # Get the vector of initial values if none are specified
        if x is None:
            self.x = self.prob.create_vector()
            self.x.get_array()[:] = model.get_values_from_meta("value")
        else:
            self.x = x

        # Get the lower and upper bounds if none are specified
        if lower is None:
            self.lower = self.prob.create_vector()
            self.lower.get_array()[:] = model.get_values_from_meta("lower")
        else:
            self.lower = lower
        if upper is None:
            self.upper = self.prob.create_vector()
            self.upper.get_array()[:] = model.get_values_from_meta("upper")
        else:
            self.upper = upper

        # Create the interior point optimizer object
        self.optimizer = InteriorPointOptimizer(self.prob, self.lower, self.upper)

        # Create data that will be used in conjunction with the optimizer
        self.vars = self.optimizer.create_opt_vector(self.x)
        self.res = self.optimizer.create_opt_vector()
        self.update = self.optimizer.create_opt_vector()
        self.temp = self.optimizer.create_opt_vector()

        # Create vectors that store problem-specific information
        self.grad = self.prob.create_vector()
        self.px = self.prob.create_vector()
        self.bx = self.prob.create_vector()

        # Create hessian matrix object
        self.hess = self.prob.create_csr_matrix()
        self.nrows, self.ncols, self.nnz, self.rowp, self.cols = (
            self.hess.get_nonzero_structure()
        )

    def get_scipy_csr_mat(self, obj=None):
        if obj is None:
            data = self.hess.get_data()
        else:
            data = obj.get_data()
        return csr_matrix((data, self.cols, self.rowp), shape=(self.nrows, self.ncols))

    def _solve(self, hess, bx, px):
        """
        Solve the KKT syste - many different appraoches could be inserted here
        """

        H = self.get_scipy_csr_mat(obj=hess)

        # Compute the solution using scipy
        px.get_array()[:] = spsolve(H, bx.get_array())

        return

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

    def get_options(self, options={}):
        default = {
            "max_iterations": 100,
            "barrier_strategy": "monotone",
            "monotone_barrier_fraction": 0.1,
            "convergence_tolerance": 1e-6,
            "fraction_to_boundary": 0.95,
            "initial_barrier_param": 1.0,
            "max_line_search_iterations": 10,
        }

        default.update(options)
        return default

    def optimize(self, options={}):
        # Get the set of options
        options = self.get_options(options=options)

        barrier_param = options["initial_barrier_param"]
        max_iters = options["max_iterations"]
        tau = options["fraction_to_boundary"]

        self.optimizer.initialize_multipliers_and_slacks(self.vars)

        # Compute the gradient
        self.prob.gradient(self.vars.x, self.grad)

        line_iters = 0
        for i in range(max_iters):
            # Compute the complete KKT residual
            res_norm = self.optimizer.compute_residual(
                barrier_param, self.vars, self.grad, self.res
            )

            barrier_converged = False
            if (
                barrier_param <= 0.1 * options["convergence_tolerance"]
                and res_norm < options["convergence_tolerance"]
            ):
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

            # Compute the residual norm
            if i % 10 == 0:
                line = (
                    f"{'Iter':>10s} {'Residual':>15s} {'mu':>15s} {'Line iters':>15s}"
                )
                print(line)
            line = f"{i:10d} {res_norm:15.4e} {barrier_param:15.4e} {line_iters:15d}"
            print(line)

            # Compute the reduced residual for the right-hand-side of the KKT system
            self.optimizer.compute_reduced_residual(self.vars, self.res, self.bx)

            # Compute the Hessian
            self.prob.hessian(self.vars.x, self.hess)

            # Add the diagonal contributions to the Hessian matrix
            self.optimizer.add_diagonal(self.vars, self.hess)

            # Solve the KKT system
            self._solve(self.hess, self.bx, self.px)

            # Compute the full update based on the reduced variable update
            self.optimizer.compute_update_from_reduced(
                self.vars, self.res, self.px, self.update
            )

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
                self.prob.gradient(self.temp.x, self.grad)
                res_norm_new = self.optimizer.compute_residual(
                    barrier_param, self.temp, self.grad, self.res
                )

                if res_norm_new < res_norm or j == max_line_iters - 1:
                    self.vars.copy(self.temp)
                    break
                else:
                    line_iters += 1
                    alpha *= 0.5

            # Check the update
            # self.optimizer.check_update(self.vars, self.res, self.update, self.hess)

        return
