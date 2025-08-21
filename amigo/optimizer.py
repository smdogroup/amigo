import warnings
import numpy as np
from scipy.sparse.linalg import MatrixRankWarning
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.linalg import spsolve, eigsh

from .amigo import InteriorPointOptimizer
from .model import ModelVector
from .utils import tocsr

try:
    from petsc4py import PETSc
except:
    PETSc = None


class DirectScipySolver:
    def __init__(self, problem):
        self.problem = problem
        self.hess = self.problem.create_matrix()
        self.nrows, self.ncols, self.nnz, self.rowp, self.cols = (
            self.hess.get_nonzero_structure()
        )
        return

    def solve(self, x, diag, bx, px, zero_design_contrib=False):
        """
        Solve the KKT system - many different appraoches could be inserted here
        """

        # Compute the Hessian
        self.problem.hessian(x, self.hess, zero_design_contrib=zero_design_contrib)
        H0 = tocsr(self.hess)
        self.hess.add_diagonal(diag)

        data = self.hess.get_data()
        H = csr_matrix((data, self.cols, self.rowp), shape=(self.nrows, self.ncols))
        px.get_array()[:] = spsolve(H, bx.get_array())

        return

    def compute_eigenvalues(self, x, diag=None, k=20, sigma=0.0, which="LM"):
        """
        Compute the eigenvalues and eigenvectors
        """
        self.problem.hessian(x, self.hess, zero_design_contrib=False)
        if diag is not None:
            self.hess.add_diagonal(diag)

        data = self.hess.get_data()
        H = csr_matrix((data, self.cols, self.rowp), shape=(self.nrows, self.ncols))

        eigs, vecs = eigsh(H, k=k, sigma=sigma, which=which)

        return eigs, vecs


class DirectPetscSolver:
    def __init__(self, comm, mpi_problem):
        self.comm = comm
        self.mpi_problem = mpi_problem
        self.hess = self.mpi_problem.create_matrix()
        self.nrows_local = self.mpi_problem.get_num_variables()
        self.nrows, self.ncols, self.nnz, self.rowp, self.cols = (
            self.hess.get_nonzero_structure()
        )

        self.H = PETSc.Mat().create(comm=comm)

        s = (self.nrows_local, self.ncols)
        self.H.setSizes((s, s), bsize=1)
        self.H.setType(PETSc.Mat.Type.MPIAIJ)

        # Right-hand side and solution vector
        self.b = PETSc.Vec().createMPI(s, bsize=1, comm=comm)
        self.x = PETSc.Vec().createMPI(s, bsize=1, comm=comm)

        return

    def solve(self, x, diag, bx, px):
        # Compute the Hessian
        self.mpi_problem.hessian(x, self.hess)
        self.hess.add_diagonal(diag)

        # Extract the Hessian entries
        data = self.hess.get_data()

        nnz = self.rowp[self.nrows_local]
        self.H.zeroEntries()
        self.H.setValuesCSR(
            self.rowp[: self.nrows_local + 1], self.cols[:nnz], data[:nnz]
        )
        self.H.assemble()

        # Create KSP solver
        ksp = PETSc.KSP().create(comm=self.comm)
        ksp.setOperators(self.H)
        ksp.setTolerances(rtol=1e-16)
        ksp.setType("preonly")  # Do not iterate — direct solve only

        pc = ksp.getPC()
        pc.setType("cholesky")
        pc.setFactorSolverType("mumps")

        M = pc.getFactorMatrix()
        M.setMumpsIcntl(6, 5)  # Reordering strategy
        M.setMumpsIcntl(7, 2)  # Use scaling
        M.setMumpsIcntl(13, 1)  # Control
        M.setMumpsIcntl(24, 1)
        M.setMumpsIcntl(4, 1)  # Set verbosity of the output
        M.setMumpsCntl(1, 0.01)

        # ksp.setMonitor(
        #     lambda ksp, its, rnorm: print(f"Iter {its}: ||r|| = {rnorm:.3e}")
        # )
        ksp.setUp()

        # Solve the system
        self.b.getArray()[:] = bx.get_array()[: self.nrows_local]
        ksp.solve(self.b, self.x)
        px.get_array()[: self.nrows_local] = self.x.getArray()[:]

        # if self.comm.size == 1:
        #     H = tocsr(self.hess)
        #     px.get_array()[:] = spsolve(H, bx.get_array())

        return


class Optimizer:
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
        self.model = model
        self.problem = self.model.get_opt_problem()

        self.comm = comm
        self.distribute = distribute
        if self.distribute and self.comm is None:
            raise ValueError("If problem is distributed, communicator cannot be None")

        # Partition the problem
        if self.distribute:
            self.mpi_problem = self.problem.partition_from_root()
            self.mpi_x = self.mpi_problem.create_vector()
            self.mpi_lower = self.mpi_problem.create_vector()
            self.mpi_upper = self.mpi_problem.create_vector()

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

        # Distribute the initial point
        if self.distribute:
            # Scatter the data vector
            self.problem.scatter_data_vector(
                self.problem.get_data_vector(),
                self.mpi_problem,
                self.mpi_problem.get_data_vector(),
            )

            # Scatter the initial point and bound variables
            self.problem.scatter_vector(self.x, self.mpi_problem, self.mpi_x)
            self.problem.scatter_vector(self.lower, self.mpi_problem, self.mpi_lower)
            self.problem.scatter_vector(self.upper, self.mpi_problem, self.mpi_upper)

        # Set the solver for the KKT system
        if solver is None and self.distribute:
            self.solver = DirectPetscSolver(self.comm, self.mpi_problem)
        elif solver is None:
            self.solver = DirectScipySolver(self.problem)
        else:
            self.solver = solver

        # Create the interior point optimizer object
        if self.distribute:
            x_vec = self.mpi_x
            self.optimizer = InteriorPointOptimizer(
                self.mpi_problem, self.mpi_lower, self.mpi_upper
            )
        else:
            x_vec = self.x
            self.optimizer = InteriorPointOptimizer(
                self.problem, self.lower, self.upper
            )

        # Create data that will be used in conjunction with the optimizer
        self.vars = self.optimizer.create_opt_vector(x_vec)
        self.update = self.optimizer.create_opt_vector()
        self.temp = self.optimizer.create_opt_vector()

        # Create vectors that store problem-specific information
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

        return

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
            "gamma_peanlty": 1e3,
            "equal_primal_dual_step": False,
            "init_least_squares_multipliers": True,
            "init_affine_step_multipliers": False,
        }

        default.update(options)
        return default

    def _compute_least_squares_multipliers(self):
        """
        Compute the least squares multiplier estimates by solving the system of equations

        [ I  A^{T} ][ w      ] = - [ (g - zl + zu) ]
        [ A     0  ][ lambda ] =   [             0 ]

        Note that the w components are discarded.
        """

        # Get which variables are multipliers/constraints
        if self.distribute:
            is_mult = self.mpi_problem.get_multiplier_indicator()
        else:
            is_mult = self.problem.get_multiplier_indicator()
        is_mult_array = is_mult.get_array()

        # Set the diagonal entries of the matrix
        diag_array = self.diag.get_array()
        diag_array[:] = 1.0
        diag_array[is_mult_array != 0] = 0.0

        # Compute the residual
        self.optimizer.compute_residual(0.0, 0.0, self.vars, self.grad, self.res)

        # Zero the constraint contributions
        res_array = self.res.get_array()
        res_array[is_mult_array != 0] = 0.0

        # Find the solution values
        x = self.vars.get_solution()
        self.solver.solve(x, self.diag, self.res, self.px, zero_design_contrib=True)

        # Update the multiplier values
        x_array = x.get_array()
        x_array[is_mult_array != 0] = self.px.get_array()[is_mult_array != 0]

        return

    def _compute_affine_multipliers(self, gamma_penalty=1e3, beta_min=1.0):
        """
        Compute the affine multipliers
        """

        # Compute the gradient at the new point with the updated multipliers
        x = self.vars.get_solution()
        if self.distribute:
            self.mpi_problem.gradient(x, self.grad)
        else:
            self.problem.gradient(x, self.grad)

        # Compute the residual
        mu = 0.0
        self.optimizer.compute_residual(
            mu, gamma_penalty, self.vars, self.grad, self.res
        )

        # Add the diagonal contributions to the Hessian matrix
        self.optimizer.compute_diagonal(self.vars, self.diag)

        # Solve the KKT system
        self.solver.solve(x, self.diag, self.res, self.px)

        # Compute the full update based on the reduced variable update
        self.optimizer.compute_update(
            mu, gamma_penalty, self.vars, self.px, self.update
        )

        self.optimizer.compute_affine_start_point(
            beta_min, self.vars, self.update, self.temp
        )

        # Copy over the design point
        xt_array = self.temp.get_solution().get_array()
        x_array = self.vars.get_solution().get_array()
        xt_array[:] = x_array[:]

        # Now, compute the updates based
        barrier = self.optimizer.compute_complementarity(self.temp)

        self.vars.copy(self.temp)

        return barrier

    def _add_regularization_terms(self, diag, eps_x=1e-4, eps_z=1e-4):
        if self.distribute:
            is_mult = self.mpi_problem.get_multiplier_indicator()
        else:
            is_mult = self.problem.get_multiplier_indicator()
        is_mult_array = is_mult.get_array()

        # Add regularization terms
        diag_array = diag.get_array()
        diag_array[is_mult_array == 0] += eps_x
        diag_array[is_mult_array == 1] -= eps_z

        return

    def _zero_multipliers(self, x):
        # Zero the multiplier contributions
        if self.distribute:
            is_mult = self.mpi_problem.get_multiplier_indicator()
        else:
            is_mult = self.problem.get_multiplier_indicator()
        is_mult_array = is_mult.get_array()

        x_array = x.get_array()
        x_array[is_mult_array != 0] = 0.0

        return

    def optimize(self, options={}):
        """
        Optimize the problem with the specified input options
        """

        # Set the communicator rank
        comm_rank = 0
        if self.comm is not None:
            comm_rank = self.comm.rank

        # Get the set of options
        options = self.get_options(options=options)

        # Data that is recorded at each iteration
        opt_data = {"options": options, "converged": False, "iterations": []}

        self.barrier_param = options["initial_barrier_param"]
        self.gamma_penalty = options["gamma_peanlty"]
        max_iters = options["max_iterations"]
        tau = options["fraction_to_boundary"]
        tol = options["convergence_tolerance"]
        record_components = options["record_components"]

        # Get the x/multiplier solution vector from the optimization variables
        x = self.vars.get_solution()

        # Create a view into x using the component indices
        xview = None
        if not self.distribute:
            xview = ModelVector(self.model, x=x)

        # Zero the multipliers so that the gradient consists of the objective gradient and
        # constraint values
        self._zero_multipliers(x)

        # Compute the gradient
        self.problem.gradient(x, self.grad)

        # Set the initial point and slack variable
        self.optimizer.initialize_multipliers_and_slacks(
            self.barrier_param, self.grad, self.vars
        )

        # Initialize the multipliers
        if options["init_affine_step_multipliers"]:
            # Compute the
            self._compute_least_squares_multipliers()

            # Compute the affine step multipliers and
            self.barrier_param = self._compute_affine_multipliers(
                beta_min=self.barrier_param, gamma_penalty=self.gamma_penalty
            )
        elif options["init_least_squares_multipliers"]:
            self._compute_least_squares_multipliers()

        # Compute the gradient
        if self.distribute:
            self.mpi_problem.gradient(x, self.grad)
        else:
            self.problem.gradient(x, self.grad)

        line_iters = 0
        alpha_x_prev = 0.0
        alpha_z_prev = 0.0
        x_index_prev = -1
        z_index_prev = -1

        for i in range(max_iters):
            iter_data = {}

            # Compute the complete KKT residual
            res_norm = self.optimizer.compute_residual(
                self.barrier_param, self.gamma_penalty, self.vars, self.grad, self.res
            )

            # Set information about the residual norm into the
            iter_data = {
                "iteration": i,
                "residual": res_norm,
                "barrier_param": self.barrier_param,
                "line_iters": line_iters,
                "alpha_x": alpha_x_prev,
                "x_index": x_index_prev,
                "alpha_z": alpha_z_prev,
                "z_index": z_index_prev,
            }

            if comm_rank == 0:
                self.write_log(i, iter_data)

            iter_data["x"] = {}
            if xview is not None:
                for name in record_components:
                    iter_data["x"][name] = xview[name].tolist()

            opt_data["iterations"].append(iter_data)

            barrier_converged = False
            if self.barrier_param <= 0.1 * tol and res_norm < tol:
                opt_data["converged"] = True
                break
            elif res_norm < 0.1 * self.barrier_param:
                barrier_converged = True

            # Check if the barrier problem has converged
            if barrier_converged:
                frac = options["monotone_barrier_fraction"]
                self.barrier_param *= frac

                # Re-compute the reduced residual for the right-hand-side of the KKT system
                res_norm = self.optimizer.compute_residual(
                    self.barrier_param,
                    self.gamma_penalty,
                    self.vars,
                    self.grad,
                    self.res,
                )

            # Add the diagonal contributions to the Hessian matrix
            self.optimizer.compute_diagonal(self.vars, self.diag)

            # Solve the KKT system with the computed diagonal entries
            self.solver.solve(x, self.diag, self.res, self.px)

            # Compute the full update based on the reduced variable update
            self.optimizer.compute_update(
                self.barrier_param, self.gamma_penalty, self.vars, self.px, self.update
            )

            # Check the update
            if options["check_update_step"]:
                if self.distribute:
                    hess = self.mpi_problem.create_matrix()
                else:
                    hess = self.problem.create_matrix()
                self.optimizer.check_update(
                    self.barrier_param,
                    self.gamma_penalty,
                    self.grad,
                    self.vars,
                    self.update,
                    hess,
                )

            # Compute the max step in the multipliers
            alpha_x, x_index, alpha_z, z_index = self.optimizer.compute_max_step(
                tau, self.vars, self.update
            )

            # Set the line search step length for the primal and dual variables to be equal to one another
            if options["equal_primal_dual_step"]:
                alpha_x = alpha_z = min(alpha_x, alpha_z)

            # Compute the step length
            max_line_iters = options["max_line_search_iterations"]
            alpha = 1.0
            line_iters = 1
            for j in range(max_line_iters):
                # Apply the update to get the new variable values at candidate step length alpha
                self.optimizer.apply_step_update(
                    alpha * alpha_x, alpha * alpha_z, self.vars, self.update, self.temp
                )

                # Compute the gradient at the new point
                xt = self.temp.get_solution()
                if self.distribute:
                    self.mpi_problem.gradient(xt, self.grad)
                else:
                    self.problem.gradient(xt, self.grad)

                # Compute the residual at the perturbed point
                res_norm_new = self.optimizer.compute_residual(
                    self.barrier_param,
                    self.gamma_penalty,
                    self.temp,
                    self.grad,
                    self.res,
                )

                if res_norm_new < res_norm or j == max_line_iters - 1:
                    self.vars.copy(self.temp)
                    break
                else:
                    line_iters += 1

                    # Apply a simple backtracking algorithm
                    alpha *= options["backtracting_factor"]

            alpha_x_prev = alpha * alpha_x
            alpha_z_prev = alpha * alpha_z
            x_index_prev = x_index
            z_index_prev = z_index

        return opt_data
