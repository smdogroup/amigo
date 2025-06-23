import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from .amigo import InteriorPointOptimizer


class Optimizer:
    def __init__(self, model, x=None, lower=None, upper=None, options={}):
        self.model = model
        self.prob = self.model.get_opt_problem()

        if x is None:
            self.x = self.prob.create_vector()
            self.x.get_array()[:] = model.get_values_from_meta("value")
        else:
            self.x = x

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

        # Create vectors that store problem-specific information
        self.grad = self.prob.create_vector()
        self.px = self.prob.create_vector()
        self.bx = self.prob.create_vector()

        # Create hessian matrix object
        self.hess = self.prob.create_csr_matrix()
        self.nrows, self.ncols, self.nnz, self.rowp, self.cols = (
            self.hess.get_nonzero_structure()
        )

    def _get_scipy_csr_mat(self):
        data = self.hess.get_data()
        return csr_matrix((data, self.cols, self.rowp), shape=(self.nrows, self.ncols))

    def _solve(self, hess, b, p):
        """
        Solve the KKT syste - many different appraoches could be inserted here
        """

        H = self._get_scipy_csr_mat()

        # Compute the solution using scipy
        self.px.get_array()[:] = spsolve(H, self.bx.get_array())

        return

    def optimize(self, max_iters=50):

        barrier_param = 1.0
        tau = 0.95

        self.optimizer.initialize_multipliers_and_slacks(self.vars)

        x = np.zeros((max_iters, self.x.get_array().size))

        for i in range(max_iters):
            print(f"Iteration {i}")
            x[i, :] = self.x.get_array()

            # Compute the gradient and the Hessian matrix
            self.prob.gradient(self.x, self.grad)
            self.prob.hessian(self.x, self.hess)

            # Compute the complete KKT residual
            self.optimizer.compute_residual(
                barrier_param, self.vars, self.grad, self.res
            )

            # Compute the reduced residual for the right-hand-side of the KKT system
            self.optimizer.compute_reduced_residual(self.vars, self.res, self.bx)

            # Add the diagonal contributions to the Hessian matrix
            self.optimizer.add_diagonal(self.vars, self.hess)

            # Solve the KKT system
            self._solve(self.hess, self.bx, self.px)

            # Compute the full update based on the reduced variable update
            self.optimizer.compute_update_from_reduced(
                self.vars, self.res, self.px, self.update
            )

            # Check the update
            self.optimizer.check_update(self.vars, self.res, self.update, self.hess)

            # Compute the max step in the multipliers
            alpha_x, alpha_z = self.optimizer.compute_max_step(
                tau, self.vars, self.update
            )

            # Compute the full update
            self.optimizer.apply_step_update(alpha_x, alpha_z, self.update, self.vars)

            barrier_param = np.max((1e-10, 0.5 * barrier_param))

        return x
