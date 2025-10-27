import warnings
import numpy as np
from scipy.sparse.linalg import MatrixRankWarning
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu, eigsh

from .amigo import InteriorPointOptimizer
from .model import ModelVector
from .utils import tocsr

try:
    from petsc4py import PETSc
except:
    PETSc = None


def gmres(mult, precon, b, x, msub=20, rtol=1e-2, atol=1e-30):
    # Allocate the Hessenberg - this allocates a full matrix
    H = np.zeros((msub + 1, msub))

    # Allocate small arrays of size m
    res = np.zeros(msub + 1)

    # Store the normal rotations
    Qsin = np.zeros(msub)
    Qcos = np.zeros(msub)

    # Allocate the subspaces
    W = np.zeros((msub + 1, len(x)))
    Z = np.zeros((msub, len(x)))

    # Perform the initialization: copy over b to W[0] and
    W[0, :] = b[:]
    beta = np.linalg.norm(W[0, :])

    x[:] = 0.0
    if beta < atol:
        return

    W[0, :] /= beta
    res[0] = beta

    # Perform the matrix-vector products
    niters = 0
    for i in range(msub):
        # Apply the preconditioner
        precon(W[i, :], Z[i, :])

        # Compute the matrix-vector product
        mult(Z[i, :], W[i + 1, :])

        # Perform modified Gram-Schmidt orthogonalization
        for j in range(i + 1):
            H[j, i] = np.dot(W[j, :], W[i + 1, :])
            W[i + 1, :] -= H[j, i] * W[j, :]

        # Compute the norm of the orthogonalized vector and
        # normalize it
        H[i + 1, i] = np.linalg.norm(W[i + 1, :])
        W[i + 1, :] /= H[i + 1, i]

        # Apply the Givens rotations
        for j in range(i):
            h1 = H[j, i]
            h2 = H[j + 1, i]
            H[j, i] = h1 * Qcos[j] + h2 * Qsin[j]
            H[j + 1, i] = -h1 * Qsin[j] + h2 * Qcos[j]

        # Compute the contribution to the Givens rotation
        # for the current entry
        h1 = H[i, i]
        h2 = H[i + 1, i]

        # Modification for complex from Saad pg. 193
        sq = np.sqrt(h1**2 + h2**2)
        Qsin[i] = h2 / sq
        Qcos[i] = h1 / sq

        # Apply the newest Givens rotation to the last entry
        H[i, i] = h1 * Qcos[i] + h2 * Qsin[i]
        H[i + 1, i] = -h1 * Qsin[i] + h2 * Qcos[i]

        # Update the residual
        h1 = res[i]
        res[i] = h1 * Qcos[i]
        res[i + 1] = -h1 * Qsin[i]

        if np.fabs(res[i + 1]) < rtol * beta:
            niters = i
            break

    # Compute the linear combination
    for i in range(niters, -1, -1):
        for j in range(i + 1, msub):
            res[i] -= H[i, j] * res[j]
        res[i] /= H[i, i]

    # Form the linear combination
    for i in range(msub):
        x += res[i] * Z[i]

    return


class DirectScipySolver:
    def __init__(self, problem):
        self.problem = problem
        self.hess = self.problem.create_matrix()
        self.nrows, self.ncols, self.nnz, self.rowp, self.cols = (
            self.hess.get_nonzero_structure()
        )

        self.lu = None
        return

    def factor(self, x, diag, zero_design_contrib=False):
        """
        Compute and factor the Hessian matrix
        """

        # Compute the Hessian
        self.problem.hessian(x, self.hess, zero_design_contrib=zero_design_contrib)
        self.hess.add_diagonal(diag)

        # Build the CSR matrix and convert to CSC
        shape = (self.nrows, self.ncols)
        data = self.hess.get_data()
        H = csr_matrix((data, self.cols, self.rowp), shape=shape).tocsc()

        # Compute the LU factorization
        self.lu = splu(H, permc_spec="COLAMD", diag_pivot_thresh=1.0)

        return

    def solve(self, bx, px):
        """
        Solve the KKT system
        """

        px.get_array()[:] = self.lu.solve(bx.get_array())

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


class LNKSInexactSolver:
    def __init__(
        self,
        problem,
        model=None,
        state_vars=None,
        residuals=None,
        state_indices=None,
        res_indices=None,
        gmres_subspace_size=20,
        gmres_rtol=1e-2,
    ):
        self.problem = problem
        self.hess = self.problem.create_matrix()
        self.gmres_subspace_size = gmres_subspace_size
        self.gmres_rtol = gmres_rtol

        if state_indices is None:
            self.state_indices = np.sort(model.get_indices(state_vars))
        else:
            self.state_indices = np.sort(state_indices)

        if res_indices is None:
            self.res_indices = np.sort(model.get_indices(residuals))
        else:
            self.res_indices = np.sort(res_indices)

        if len(self.state_indices) != len(self.res_indices):
            raise ValueError("Residual and states must be of the same dimension")

        # Determine the design indices based on the remaining values
        all_states = np.concatenate((self.state_indices, self.res_indices))
        upper = model.num_variables
        self.design_indices = np.sort(np.setdiff1d(np.arange(upper), all_states))

        return

    def factor(self, x, diag, zero_design_contrib=False):

        # Compute the Hessian
        self.problem.hessian(x, self.hess, zero_design_contrib=zero_design_contrib)
        self.hess.add_diagonal(diag)

        # Extract the submatrices
        self.Hmat = tocsr(self.hess)
        self.Hxx = self.Hmat[self.design_indices, :][:, self.design_indices]
        self.A = self.Hmat[self.res_indices, :][:, self.state_indices]
        self.dRdx = self.Hmat[self.res_indices, :][:, self.design_indices]

        # Factor the matrices required
        self.A_lu = splu(self.A.tocsc(), permc_spec="COLAMD", diag_pivot_thresh=1.0)
        self.Hxx_lu = splu(self.Hxx.tocsc(), permc_spec="COLAMD", diag_pivot_thresh=1.0)

        return

    def mult(self, x, y):
        y[:] = self.Hmat @ x

        return

    def precon(self, b, x):
        x[self.res_indices] = self.A_lu.solve(b[self.state_indices], trans="T")
        bx = b[self.design_indices] - self.dRdx.T @ x[self.res_indices]
        x[self.design_indices] = self.Hxx_lu.solve(bx)
        bu = b[self.res_indices] - self.dRdx @ x[self.design_indices]
        x[self.state_indices] = self.A_lu.solve(bu)

        return

    def solve(self, bx, px):
        gmres(
            self.mult,
            self.precon,
            bx.get_array(),
            px.get_array(),
            msub=self.gmres_subspace_size,
            rtol=self.gmres_rtol,
        )

        return


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

    def factor(self, x, diag):
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
        self.ksp = PETSc.KSP().create(comm=self.comm)
        self.ksp.setOperators(self.H)
        self.ksp.setTolerances(rtol=1e-16)
        self.ksp.setType("preonly")  # Do not iterate — direct solve only

        pc = self.ksp.getPC()
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
        self.ksp.setUp()

    def solve(self, bx, px):

        # Solve the system
        self.b.getArray()[:] = bx.get_array()[: self.nrows_local]
        self.ksp.solve(self.b, self.x)
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
        self.problem = self.model.get_problem()

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
            self.x = x.get_vector()
        elif isinstance(x, ModelVector):
            self.x = x.get_vector()
        else:
            self.x = x

        # Get the lower and upper bounds if none are specified
        if lower is None:
            lower = model.get_values_from_meta("lower")
            self.lower = lower.get_vector()
        elif isinstance(lower, ModelVector):
            self.lower = lower.get_vector()
        else:
            self.lower = lower
        if upper is None:
            upper = model.get_values_from_meta("upper")
            self.upper = upper.get_vector()
        elif isinstance(upper, ModelVector):
            self.upper = upper.get_vector()
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
            # Heuristic barrier parameter options
            "heuristic_barrier_gamma": 0.1,
            "heuristic_barrier_r": 0.95,
            "verbose_barrier": False,
        }

        for name in options:
            if name in default:
                default[name] = options[name]
            else:
                raise ValueError(f"Unrecognized option {name}")

        return default

    def _compute_complementarity_and_uniformity(self):
        """
        Compute both complementarity and uniformity measure in a single pass.
        Returns (complementarity, uniformity) where:
        - complementarity is the average: y^T w / m
        - uniformity ξ = min_i [w_i y_i / (y^T w / m)]
        """
        complementarity, uniformity = self.optimizer.compute_complementarity(
            self.vars, True
        )
        return complementarity, uniformity

    def _compute_barrier_heuristic(self, xi, complementarity, gamma, r):
        """
        Compute the heuristic barrier parameter.
        Formula: μ = γ * min((1-r)*(1-ξ)/ξ, 2)^3 * complementarity
        """
        if xi > 1e-10:
            term = (1 - r) * (1 - xi) / xi
            heuristic_factor = min(term, 2.0) ** 3
        else:
            heuristic_factor = 2.0**3

        return gamma * heuristic_factor * complementarity, heuristic_factor

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
        self.solver.factor(x, self.diag, zero_design_contrib=True)
        self.solver.solve(self.res, self.px)

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
        self.solver.factor(x, self.diag)
        self.solver.solve(self.res, self.px)

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

        # Storage for heuristic barrier table (to display after optimization)
        heuristic_data = []

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

            # Compute and store xi and complementarity for heuristic (display table after optimization)
            if (
                options["barrier_strategy"] == "heuristic"
                and options["verbose_barrier"]
            ):
                complementarity, xi = self._compute_complementarity_and_uniformity()
                iter_data["xi"] = xi
                iter_data["complementarity"] = complementarity
                heuristic_data.append(
                    {"iteration": i, "xi": xi, "complementarity": complementarity}
                )

            iter_data["x"] = {}
            if xview is not None:
                for name in record_components:
                    iter_data["x"][name] = xview[name].tolist()

            opt_data["iterations"].append(iter_data)

            barrier_converged = False
            if self.barrier_param <= 0.999 * tol and res_norm < tol:
                opt_data["converged"] = True
                break
            elif res_norm <= 0.1 * self.barrier_param:
                barrier_converged = True

            # Check if the barrier problem has converged and update barrier parameter
            if options["barrier_strategy"] == "heuristic":
                # Heuristic barrier update - compute xi and complementarity if not already done
                if "xi" in iter_data:
                    xi = iter_data["xi"]
                    complementarity = iter_data["complementarity"]
                else:
                    xi = self._compute_uniformity_measure()
                    complementarity = self.optimizer.compute_complementarity(self.vars)

                self.barrier_param, _ = self._compute_barrier_heuristic(
                    xi,
                    complementarity,
                    options["heuristic_barrier_gamma"],
                    options["heuristic_barrier_r"],
                )

                # Re-compute the reduced residual for the right-hand-side of the KKT system
                res_norm = self.optimizer.compute_residual(
                    self.barrier_param,
                    self.gamma_penalty,
                    self.vars,
                    self.grad,
                    self.res,
                )
            elif barrier_converged:
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
            self.solver.factor(x, self.diag)
            self.solver.solve(self.res, self.px)

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

            # # Check if this is a line search

            # # Compute the derivative of the line search function
            # self._compute_line_search_deriv()

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

        # Print heuristic barrier table after optimization completes
        if (
            comm_rank == 0
            and options["barrier_strategy"] == "heuristic"
            and options["verbose_barrier"]
        ):
            if len(heuristic_data) > 0:
                print()
                print(f"{'iteration':>10s} {'xi':>15s} {'complementarity':>15s}")
                for idx, data in enumerate(heuristic_data):
                    if idx % 10 == 0 and idx > 0:
                        print(
                            f"{'iteration':>10s} {'xi':>15s} {'complementarity':>15s}"
                        )
                    print(
                        f"{data['iteration']:10d} {data['xi']:15.6e} {data['complementarity']:15.6e}"
                    )

        return opt_data

    def compute_output(self):
        output = self.model.create_output_vector()

        x = self.vars.get_solution()
        self.problem.compute_output(x, output.get_vector())

        return output

    def compute_post_opt_derivatives(self, of=[], wrt=[], method="adjoint"):
        """
        Compute the post-optimality derivatives of the outputs
        """

        of_indices, of_map = self.model.get_indices_and_map(of)
        wrt_indices, wrt_map = self.model.get_indices_and_map(wrt)

        # Allocate space for the derivative
        dfdx = np.zeros((len(of_indices), len(wrt_indices)))

        # Get the x/multiplier solution vector from the optimization variables
        x = self.vars.get_solution()

        out_wrt_input = self.problem.create_output_jacobian_wrt_input()
        self.problem.output_jacobian_wrt_input(x, out_wrt_input)
        out_wrt_input = tocsr(out_wrt_input)

        out_wrt_data = self.problem.create_output_jacobian_wrt_data()
        self.problem.output_jacobian_wrt_data(x, out_wrt_data)
        out_wrt_data = tocsr(out_wrt_data)

        grad_wrt_data = self.problem.create_gradient_jacobian_wrt_data()
        self.problem.gradient_jacobian_wrt_data(x, grad_wrt_data)
        grad_wrt_data = tocsr(grad_wrt_data)

        # Add the diagonal contributions to the Hessian matrix
        self.optimizer.compute_diagonal(self.vars, self.diag)

        # Factor the KKT system
        self.solver.factor(x, self.diag)

        if method == "adjoint":
            for i in range(len(of_indices)):
                idx = of_indices[i]

                self.res.get_array()[:] = -out_wrt_input[idx, :].toarray()
                self.solver.solve(self.res, self.px)

                adjx = grad_wrt_data.T @ self.px.get_array()

                dfdx[i, :] = out_wrt_data[idx, wrt_indices] + adjx[wrt_indices]
        elif method == "direct":
            # Convert to CSC since we will access the columns of this matrix
            grad_wrt_data = grad_wrt_data.tocsc()

            for i in range(len(wrt_indices)):
                idx = wrt_indices[i]

                self.res.get_array()[:] = -grad_wrt_data[:, idx].toarray().flatten()
                self.solver.solve(self.res, self.px)

                dirx = out_wrt_input @ self.px.get_array()

                dfdx[:, i] = out_wrt_data[of_indices, idx] + dirx[of_indices]

        return dfdx, of_map, wrt_map
