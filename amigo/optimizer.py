import sys
import time
import numpy as np
from collections import deque
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from .amigo import InteriorPointOptimizer, MemoryLocation
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


def _find_diag_indices(rowp, cols, nrows):
    """Find the CSR data-array index of each diagonal entry (row == col)."""
    diag_idx = np.empty(nrows, dtype=np.intp)
    for i in range(nrows):
        start, end = rowp[i], rowp[i + 1]
        row_cols = cols[start:end]
        pos = np.searchsorted(row_cols, i)
        if pos < len(row_cols) and row_cols[pos] == i:
            diag_idx[i] = start + pos
        else:
            diag_idx[i] = start  # fallback (should not happen)
    return diag_idx


class _HessianDiagMixin:
    """Shared Hessian-diagonal helpers for CSR-based solvers.

    Requires subclass to have: self.problem, self.hess, self._diag_indices.
    """

    def assemble_hessian(self, alpha, x):
        """Assemble Lagrangian Hessian and return its diagonal.

        Leaves the assembled matrix in self.hess for a subsequent
        add_diagonal_and_factor() call.  Cost: one Hessian evaluation +
        one device-to-host copy.  No factorization.
        """
        self.problem.hessian(alpha, x, self.hess)
        self.hess.copy_data_device_to_host()
        return self.hess.get_data()[self._diag_indices].copy()

    def get_hessian_diagonal(self, alpha, x):
        """Evaluate Hessian and return its diagonal. O(n), no factorization."""
        self.problem.hessian(alpha, x, self.hess)
        self.hess.copy_data_device_to_host()
        return self.hess.get_data()[self._diag_indices]


class DirectCudaSolver:
    def __init__(self, problem, pivot_eps=1e-12):
        self.problem = problem

        try:
            from .amigo import CSRMatFactorCuda
        except:
            raise NotImplementedError("Amigo compiled without CUDA support")

        loc = MemoryLocation.DEVICE_ONLY
        self.hess = self.problem.create_matrix(loc)
        self.solver = CSRMatFactorCuda(self.hess, pivot_eps)

    def factor(self, alpha, x, diag):
        self.problem.hessian(alpha, x, self.hess)
        self.problem.add_diagonal(diag, self.hess)
        self.solver.factor()

    def solve(self, bx, px):
        self.solver.solve(bx, px)


class DirectScipySolver(_HessianDiagMixin):
    def __init__(self, problem):
        self.problem = problem
        loc = MemoryLocation.HOST_AND_DEVICE
        self.hess = self.problem.create_matrix(loc)
        self.nrows, self.ncols, self.nnz, self.rowp, self.cols = (
            self.hess.get_nonzero_structure()
        )
        self._diag_indices = _find_diag_indices(self.rowp, self.cols, self.nrows)
        self.lu = None

    def add_diagonal_and_factor(self, diag):
        """Add diagonal to the already-assembled Hessian and factorize.

        Must be called after assemble_hessian().  Modifies self.hess
        in-place, so a subsequent retry must use factor() (which
        re-assembles from scratch).
        """
        self.problem.add_diagonal(diag, self.hess)
        self.hess.copy_data_device_to_host()
        shape = (self.nrows, self.ncols)
        data = self.hess.get_data()
        H = csr_matrix((data, self.cols, self.rowp), shape=shape).tocsc()
        self.lu = splu(H, permc_spec="COLAMD", diag_pivot_thresh=1.0)

    def factor(self, alpha, x, diag):
        """Assemble Hessian, add diagonal, and factorize (one shot).

        Used for inertia-correction retries where we need a fresh
        assembly (since add_diagonal_and_factor mutates self.hess).
        """
        self.problem.hessian(alpha, x, self.hess)
        self.problem.add_diagonal(diag, self.hess)
        self.hess.copy_data_device_to_host()
        shape = (self.nrows, self.ncols)
        data = self.hess.get_data()
        H = csr_matrix((data, self.cols, self.rowp), shape=shape).tocsc()
        self.lu = splu(H, permc_spec="COLAMD", diag_pivot_thresh=1.0)

    def solve(self, bx, px):
        """Solve the KKT system."""
        bx.copy_device_to_host()
        px.get_array()[:] = self.lu.solve(bx.get_array())
        px.copy_host_to_device()

    def solve_array(self, rhs):
        """Solve K*x = rhs using existing factorization. Returns numpy array."""
        return self.lu.solve(rhs)

    def get_inertia(self):
        """Approximate inertia from LU diagonal (heuristic).

        With diag_pivot_thresh=1.0 (strong diagonal pivoting), the signs of
        U's diagonal approximate the eigenvalue signs of the original matrix.
        Returns (n_positive, n_negative).
        """
        if self.lu is None:
            raise RuntimeError("Must call factor() before get_inertia()")
        u_diag = self.lu.U.diagonal()
        return int(np.sum(u_diag > 0)), int(np.sum(u_diag < 0))


class PardisoSolver(_HessianDiagMixin):
    """Sparse LDL^T solver via Intel MKL PARDISO with inertia detection.

    Uses symmetric indefinite factorization (mtype=-2) which provides
    exact inertia (positive/negative eigenvalue counts) after factorization.
    This enables inertia correction for nonconvex optimization.

    The upper-triangle sparsity structure is cached at construction time
    so that factor() only copies data (O(nnz)) without rebuilding the
    CSR structure. Passing the same matrix object to pypardiso lets it
    skip symbolic analysis (phase 11) after the first factorization.

    Falls back to DirectScipySolver if pypardiso is not installed.
    """

    def __init__(self, problem):
        from pypardiso import PyPardisoSolver

        self.problem = problem
        loc = MemoryLocation.HOST_AND_DEVICE
        self.hess = self.problem.create_matrix(loc)
        self.nrows, self.ncols, self.nnz, self.rowp, self.cols = (
            self.hess.get_nonzero_structure()
        )
        # mtype=-2: real symmetric indefinite
        self.pardiso = PyPardisoSolver(mtype=-2)

        # Pre-compute upper-triangle mask (sparsity structure is fixed).
        # For each entry in the full CSR, mark True if col >= row.
        upper_mask = np.empty(self.nnz, dtype=bool)
        for i in range(self.nrows):
            start, end = self.rowp[i], self.rowp[i + 1]
            upper_mask[start:end] = self.cols[start:end] >= i
        self._upper_mask = upper_mask

        # Build upper-triangle CSR structure once (indices/indptr are fixed).
        upper_cols = self.cols[upper_mask]
        upper_indptr = np.zeros(self.nrows + 1, dtype=self.rowp.dtype)
        for i in range(self.nrows):
            start, end = self.rowp[i], self.rowp[i + 1]
            upper_indptr[i + 1] = upper_indptr[i] + int(
                np.sum(self.cols[start:end] >= i)
            )
        upper_data = np.zeros(len(upper_cols))

        # Persistent CSR matrix — same object passed to pardiso every time
        # so symbolic analysis (phase 11) runs once, then only numerical
        # factorization (phase 22) on subsequent calls.
        self._matrix = csr_matrix(
            (upper_data, upper_cols.copy(), upper_indptr.copy()),
            shape=(self.nrows, self.ncols),
        )

        # Pre-compute diagonal entry indices for fast diagonal extraction
        self._diag_indices = _find_diag_indices(self.rowp, self.cols, self.nrows)

    def add_diagonal_and_factor(self, diag):
        """Add diagonal to the already-assembled Hessian and LDL^T factorize.

        Must be called after assemble_hessian().  Modifies self.hess
        in-place, so a subsequent retry must use factor() (which
        re-assembles from scratch).
        """
        self.problem.add_diagonal(diag, self.hess)
        self.hess.copy_data_device_to_host()
        data = self.hess.get_data()
        self._matrix.data[:] = data[self._upper_mask]
        self.pardiso.factorize(self._matrix)

    def factor(self, alpha, x, diag):
        """Assemble Hessian, add diagonal, and LDL^T factorize (one shot).

        Used for inertia-correction retries where we need a fresh
        assembly (since add_diagonal_and_factor mutates self.hess).
        """
        self.problem.hessian(alpha, x, self.hess)
        self.problem.add_diagonal(diag, self.hess)
        self.hess.copy_data_device_to_host()
        data = self.hess.get_data()
        self._matrix.data[:] = data[self._upper_mask]
        self.pardiso.factorize(self._matrix)

    def get_inertia(self):
        """Return (n_positive, n_negative) eigenvalue counts from LDL^T.

        Uses PARDISO iparm(22) and iparm(23) (1-based Fortran indexing),
        which are iparm[21] and iparm[22] in 0-based Python indexing.
        """
        return int(self.pardiso.iparm[21]), int(self.pardiso.iparm[22])

    def solve(self, bx, px):
        """Solve the KKT system."""
        bx.copy_device_to_host()
        px.get_array()[:] = self.pardiso.solve(self._matrix, bx.get_array())
        px.copy_host_to_device()

    def solve_array(self, rhs):
        """Solve K*x = rhs using existing factorization. Returns numpy array."""
        return self.pardiso.solve(self._matrix, rhs)


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
        loc = MemoryLocation.HOST_AND_DEVICE
        self.hess = self.problem.create_matrix(loc)
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

    def factor(self, alpha, x, diag):

        # Compute the Hessian
        self.problem.hessian(alpha, x, self.hess)
        self.problem.add_diagonal(diag, self.hess)
        self.hess.copy_data_device_to_host()

        # Extract the submatrices
        self.Hmat = tocsr(self.hess)
        self.Hxx = self.Hmat[self.design_indices, :][:, self.design_indices]
        self.A = self.Hmat[self.res_indices, :][:, self.state_indices]
        self.dRdx = self.Hmat[self.res_indices, :][:, self.design_indices]

        # Factor the matrices required
        self.A_lu = splu(self.A.tocsc(), permc_spec="COLAMD", diag_pivot_thresh=1.0)
        self.Hxx_lu = splu(self.Hxx.tocsc(), permc_spec="COLAMD", diag_pivot_thresh=1.0)

    def mult(self, x, y):
        y[:] = self.Hmat @ x

    def precon(self, b, x):
        x[self.res_indices] = self.A_lu.solve(b[self.state_indices], trans="T")
        bx = b[self.design_indices] - self.dRdx.T @ x[self.res_indices]
        x[self.design_indices] = self.Hxx_lu.solve(bx)
        bu = b[self.res_indices] - self.dRdx @ x[self.design_indices]
        x[self.state_indices] = self.A_lu.solve(bu)

    def solve(self, bx, px):
        bx.copy_device_to_host()
        gmres(
            self.mult,
            self.precon,
            bx.get_array(),
            px.get_array(),
            msub=self.gmres_subspace_size,
            rtol=self.gmres_rtol,
        )
        px.copy_host_to_device()


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

    def factor(self, alpha, x, diag):
        # Compute the Hessian
        self.mpi_problem.hessian(alpha, x, self.hess)
        self.mpi_problem.add_diagonal(diag, self.hess)

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

        self.ksp.setUp()

    def solve(self, bx, px):

        # Solve the system
        self.b.getArray()[:] = bx.get_array()[: self.nrows_local]
        self.ksp.solve(self.b, self.x)
        px.get_array()[: self.nrows_local] = self.x.getArray()[:]


class CurvatureProbeConvexifier:
    """Three-layer inertia regularization for nonconvex interior-point methods.

    Layer 1 — Diagonal Hessian regularization (per-variable, exact):
        Extract W_ii (diagonal of the Lagrangian Hessian) from the assembled
        matrix.  For each primal variable where W_ii + Sigma_ii < 0, set
        E_ii = |W_ii + Sigma_ii| + eps to flip the diagonal to positive.
        Handles all diagonal negative curvature with zero lag and no blind spots.

    Layer 2 — Targeted inertia correction (nonconvex primal vars):
        After Layer 1, factorize and check inertia via LDL^T.  If wrong,
        add scalar delta only to primal variables that appear in nonconvex
        constraints (identified from KKT sparsity at init).  Handles
        off-diagonal coupling (e.g. friction circle) with minimal distortion.

    Layer 3 — Global inertia correction (safety net):
        If targeted correction fails, add delta to ALL primal variables.
        Guarantees correct inertia under any circumstance.

    eps_z (virtual control) is applied selectively to nonconvex constraint
    multipliers and tracks the barrier parameter mu.
    """

    def __init__(
        self, options, barrier_param, model, problem, solver, distribute, tol=1e-6
    ):
        # eps_z / VC parameters
        self.cz = options["convex_eps_z_coeff"]
        self.eps_z_floor = min(1e-10, tol * 1e-2)
        self.eps_z = max(self.eps_z_floor, self.cz * barrier_param)
        self.tol = tol
        self._barrier = barrier_param
        self.theta = 0.0
        self.eta = 0.0
        self.vc_floor = 0.0
        self.max_rejections = options["max_consecutive_rejections"]
        self.barrier_inc = options["barrier_increase_factor"]
        self.initial_barrier = options["initial_barrier_param"]
        self.step_rejected = False
        self.consecutive_rejections = 0

        self.numerical_eps = 1e-12

        # Inertia correction state
        self.last_inertia_delta = 0.0

        # Multiplier indicator
        self.mult_ind = np.array(problem.get_multiplier_indicator(), dtype=bool)
        self.n_neg = 0
        self.max_reg = 0.0

        # Resolve nonconvex constraint MULTIPLIER indices (for selective eps_z)
        nc_constraints = options["nonconvex_constraints"]
        if nc_constraints and not distribute:
            self._nonconvex_indices = np.sort(model.get_indices(nc_constraints))
            print(
                f"  Selective eps_z: {len(self._nonconvex_indices)} "
                f"nonconvex constraint entries"
            )
        else:
            self._nonconvex_indices = None

        # Resolve nonconvex PRIMAL variable indices (for targeted inertia)
        # From the KKT sparsity: primal columns coupled to nonconvex rows
        self._nonconvex_primal_indices = None
        if self._nonconvex_indices is not None and hasattr(solver, "rowp"):
            nc_primal = set()
            for k in self._nonconvex_indices:
                row_cols = solver.cols[solver.rowp[k] : solver.rowp[k + 1]]
                nc_primal.update(row_cols[~self.mult_ind[row_cols]])
            self._nonconvex_primal_indices = np.array(sorted(nc_primal))
            print(
                f"  Targeted inertia: {len(self._nonconvex_primal_indices)} "
                f"nonconvex primal vars"
            )

    def _compute_schur_diagonal(self, solver, diag_base):
        """Diagonal of J^T |D^{-1}| J (Schur complement contribution)."""
        data = solver.hess.get_data()
        rowp = solver.rowp
        cols = solver.cols
        n = len(self.mult_ind)

        # Constraint diagonal weights: |1/D_kk|
        D_abs = np.abs(diag_base[self.mult_ind])
        safe = D_abs > 1e-30
        w = np.zeros_like(D_abs)
        w[safe] = 1.0 / D_abs[safe]

        # Per-row weight: nonzero only for constraint rows
        row_weight = np.zeros(n)
        row_weight[self.mult_ind] = w

        # Expand to per-NNZ-entry weight via CSR row lengths
        row_nnz = np.diff(rowp)
        entry_weight = np.repeat(row_weight, row_nnz)

        # J_ki^2 * w_k for each NNZ entry in constraint rows
        weighted_sq = data**2 * entry_weight

        # Scatter-add to column indices (only primal columns get nonzero)
        schur_diag = np.zeros(n)
        np.add.at(schur_diag, cols, weighted_sq)

        return schur_diag[~self.mult_ind]

    def build_regularization(
        self,
        diag,
        W_diag,
        diag_base,
        zero_hessian_indices=None,
        zero_hessian_eps=None,
        solver=None,
    ):
        """Layer 1: per-variable regularization from Hessian diagonal + Schur complement."""
        diag.copy_device_to_host()
        diag_arr = diag.get_array()
        n = len(diag_arr)

        primal = ~self.mult_ind

        full_diag = W_diag + diag_base
        effective_diag = full_diag.copy()
        if solver is not None and hasattr(solver, "hess"):
            schur_contrib = self._compute_schur_diagonal(solver, diag_base)
            effective_diag[primal] += schur_contrib

        reg = np.full(n, self.numerical_eps)
        neg_curv = primal & (effective_diag < 0)
        reg[neg_curv] = np.abs(effective_diag[neg_curv]) + self.numerical_eps

        if zero_hessian_indices is not None and zero_hessian_eps is not None:
            np.maximum(
                reg[zero_hessian_indices],
                zero_hessian_eps,
                out=reg[zero_hessian_indices],
            )

        self.n_neg = int(np.sum(neg_curv))
        primal_reg = reg[primal]
        self.max_reg = float(np.max(primal_reg)) if len(primal_reg) > 0 else 0.0

        diag_arr[primal] += reg[primal]

        # Dual eps_z (selective to nonconvex constraints)
        eps_z_num = self.eps_z_floor
        diag_arr[self.mult_ind] -= eps_z_num
        if self._nonconvex_indices is not None and self.eps_z > eps_z_num:
            diag_arr[self._nonconvex_indices] -= self.eps_z - eps_z_num
        elif self._nonconvex_indices is None and self.eps_z > eps_z_num:
            diag_arr[self.mult_ind] -= self.eps_z - eps_z_num

        diag.copy_host_to_device()

    # --- Shared interface ---

    def decompose_residual(self, res, vars):
        """Compute theta (feasibility), eta (optimality), and vc_floor."""
        res_arr = np.array(res.get_array())
        self.theta = np.linalg.norm(res_arr[self.mult_ind])
        self.eta = np.linalg.norm(res_arr[~self.mult_ind])

        x_sol = np.array(vars.get_solution())
        if self._nonconvex_indices is not None:
            lam_nc_norm = np.linalg.norm(x_sol[self._nonconvex_indices])
        else:
            lam_nc_norm = np.linalg.norm(x_sol[self.mult_ind])
        self.vc_floor = self.eps_z * lam_nc_norm

    def update_eps_z(self, barrier_param):
        self.eps_z = max(self.eps_z_floor, self.cz * barrier_param)

    def should_force_barrier_reduction(self):
        """Force barrier reduction when eps_z limits convergence."""
        total_res = max(self.theta, self.eta)
        if total_res < 1e-30:
            return False
        # theta must be limited by eps_z AND be a meaningful part of the residual
        return self.theta <= 3.0 * self.vc_floor and self.theta > 0.1 * total_res

    def begin_iteration(self, barrier_param):
        self._barrier = barrier_param
        self.update_eps_z(barrier_param)
        if self.step_rejected:
            self.step_rejected = False

    def reject_step(self):
        """Mark step as rejected."""
        self.step_rejected = True
        self.consecutive_rejections += 1

    def handle_rejection_escape(self, barrier_param):
        if self.consecutive_rejections >= self.max_rejections:
            new_barrier = min(barrier_param * self.barrier_inc, self.initial_barrier)
            if new_barrier > barrier_param:
                self.eps_z = self.cz * new_barrier
                self.consecutive_rejections = 0
                return new_barrier, True
            else:
                self.consecutive_rejections = 0
                return barrier_param, False
        return barrier_param, None

    def check_stagnation(
        self, stagnation_count, threshold, barrier_param, barrier_fraction, tol
    ):
        if stagnation_count >= threshold and barrier_param > tol:
            new_barrier = max(barrier_param * barrier_fraction, tol)
            return new_barrier, True
        return barrier_param, False

    def iter_data(self):
        return {
            "eps_x": self.max_reg if self.max_reg > 0 else self.numerical_eps,
            "eps_z": self.eps_z,
            "theta": self.theta,
            "eta": self.eta,
            "vc_floor": self.vc_floor,
            "inertia_delta": self.last_inertia_delta,
            "n_neg": self.n_neg,
        }


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
        """
        Initialize the optimizer.

        Parameters
        ----------
        model : Model
            The amigo model to optimize
        x : array-like, optional
            Initial point
        lower, upper : array-like, optional
            Variable bounds
        solver : Solver, optional
            Linear solver for KKT system
        comm : MPI communicator, optional
            For distributed optimization
        distribute : bool
            Whether to distribute the problem
        """
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
        # Prefer PardisoSolver (LDL^T with inertia detection) when available
        if solver is None and self.distribute:
            self.solver = DirectPetscSolver(self.comm, self.mpi_problem)
        elif solver is None:
            try:
                self.solver = PardisoSolver(self.problem)
            except (ImportError, Exception):
                self.solver = DirectScipySolver(self.problem)
        else:
            self.solver = solver

        # Create the interior point optimizer object
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

        # Copy the essential information from host to device
        x_vec.copy_host_to_device()
        data_vec.copy_host_to_device()

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
        sys.stdout.flush()

    def get_options(self, options={}):
        default = {
            "max_iterations": 100,
            "barrier_strategy": "heuristic",  # "heuristic", "monotone", "quality_function"
            "monotone_barrier_fraction": 0.1,
            "convergence_tolerance": 1e-8,
            "fraction_to_boundary": 0.95,
            "initial_barrier_param": 1.0,
            "max_line_search_iterations": 10,
            "check_update_step": False,
            "backtracking_factor": 0.5,
            "record_components": [],
            "gamma_penalty": 1e3,
            "equal_primal_dual_step": False,
            "init_least_squares_multipliers": True,
            "init_affine_step_multipliers": False,
            # Heuristic barrier parameter options
            "heuristic_barrier_gamma": 0.1,
            "heuristic_barrier_r": 0.95,
            "verbose_barrier": False,
            "continuation_control": None,
            # Regularization options
            "regularization_eps_x": 1e-8,  # Primal regularization
            "regularization_eps_z": 1e-8,  # Dual regularization
            "adaptive_regularization": True,  # Increase regularization on failure
            "max_regularization": 1e-2,  # Maximum regularization value
            "regularization_increase_factor": 10.0,  # Factor to increase regularization
            # Line search options
            "armijo_constant": 1e-4,  # Armijo sufficient decrease constant
            "use_armijo_line_search": True,  # Use Armijo condition
            # Acceptable convergence (like IPOPT)
            "acceptable_tol": None,  # Acceptable residual threshold (None = 100x convergence_tolerance)
            "acceptable_iter": 10,  # Iterations without progress to trigger acceptable
            # Advanced features
            "adaptive_tau": True,  # Adaptive fraction-to-boundary
            "tau_min": 0.99,  # Minimum tau value
            "progress_based_barrier": True,  # Only reduce barrier when making progress
            "barrier_progress_tol": 10.0,  # kappa_epsilon factor for barrier subproblem tolerance
            # Variable-specific regularization for zero-Hessian (linearly-appearing) variables
            "zero_hessian_variables": [],  # Variable names (e.g., ["dyn.qdot"])
            "regularization_eps_x_zero_hessian": 1.0,  # Strong eps_x for those variables
            # Three-layer inertia regularization (see CurvatureProbeConvexifier)
            # Layer 1: per-variable reg from Hessian diagonal
            # Layer 2: targeted inertia correction on nonconvex primal vars
            # Layer 3: global inertia correction (fallback)
            "curvature_probe_convexification": False,
            "convex_eps_z_coeff": 1.0,  # C_z: eps_z = C_z * mu
            "nonconvex_constraints": [],  # Constraint names for selective eps_z
            "max_consecutive_rejections": 5,  # Before barrier increase
            "barrier_increase_factor": 5.0,  # Barrier *= this when stuck
            "max_inertia_corrections": 8,  # Max refactorizations for inertia
            # Quality function barrier (Nocedal-Wachter-Waltz 2009, Section 4)
            "quality_function_sigma_max": 1000.0,  # Upper bound for sigma search
            "quality_function_golden_iters": 12,  # Golden section iterations
            "quality_function_kappa_free": 0.9999,  # Nonmonotone acceptance ratio (paper: kappa=0.9999)
            "quality_function_l_max": 5,  # Lookback window for free mode
            "quality_function_norm_scaling": True,  # Section 6: divide each term by element count
        }

        for name in options:
            if name in default:
                default[name] = options[name]
            else:
                raise ValueError(f"Unrecognized option {name}")

        return default

    def _compute_barrier_heuristic(self, xi, complementarity, gamma, r, tol=1e-12):
        """Compute LOQO-style barrier parameter (NWW 2009, eq 3.6)."""
        if xi > 1e-10:
            term = (1 - r) * (1 - xi) / xi
            heuristic_factor = min(term, 2.0) ** 3
        else:
            heuristic_factor = 2.0**3

        mu_new = gamma * heuristic_factor * complementarity

        # Floor at tolerance (only safeguard in the paper)
        mu_new = max(mu_new, tol)

        return mu_new, heuristic_factor

    def _compute_adaptive_tau(self, barrier_param, tau_min=0.99):
        """Adaptive fraction-to-boundary: tau = max(tau_min, 1 - mu)."""
        return max(tau_min, 1.0 - barrier_param)

    def _should_reduce_barrier(self, res_norm, barrier_param, kappa_epsilon=10.0):
        """Reduce barrier when subproblem is solved: res <= kappa_eps * mu."""
        barrier_tol = kappa_epsilon * barrier_param
        return res_norm <= barrier_tol

    def _golden_section_search(self, f, a, b, n_iters=12):
        """Golden section search for minimum of f on [a, b].

        Returns (x_opt, f_opt).
        """
        gr = (np.sqrt(5.0) + 1.0) / 2.0
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        fc = f(c)
        fd = f(d)

        for _ in range(n_iters):
            if b - a < b * 1e-2:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - (b - a) / gr
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + (b - a) / gr
                fd = f(d)

        return (c, fc) if fc < fd else (d, fd)

    def _compute_quality_function_barrier(
        self, tau_min, use_adaptive_tau, options, comm_rank
    ):
        """Choose barrier parameter via quality function (NWW 2009, Section 4).

        Returns (sigma_star, mu_new).
        """
        avg_comp, xi = self.optimizer.compute_complementarity(self.vars)
        if avg_comp < 1e-30:
            return 1.0, self.barrier_param

        mu_nat = avg_comp  # x^T z / n

        # Affine-scaling RHS (mu=0) and dual/primal infeasibility norms
        dual_infeas_sq, primal_infeas_sq = (
            self.optimizer.compute_residual_and_infeasibility(
                0.0, self.gamma_penalty, self.vars, self.grad, self.res
            )
        )

        # Delta(0): affine scaling direction (mu=0)
        rhs0 = self.res.get_array().copy()
        if hasattr(self.solver, "solve_array"):
            px0 = self.solver.solve_array(rhs0).copy()
        else:
            self.solver.solve(self.res, self.px)
            px0 = self.px.get_array().copy()

        # Delta(1): centering direction (mu=avg_comp)
        self.optimizer.compute_residual(
            mu_nat, self.gamma_penalty, self.vars, self.grad, self.res
        )
        if hasattr(self.solver, "solve_array"):
            px1 = self.solver.solve_array(self.res.get_array().copy()).copy()
        else:
            self.solver.solve(self.res, self.px)
            px1 = self.px.get_array().copy()

        dpx = px1 - px0

        # sigma_min floor (paper Sec 4)
        tol_qf = options["convergence_tolerance"]
        mu_floor = max(tol_qf * 0.01, 1e-14)
        sigma_min_floor = (mu_floor / mu_nat) if mu_nat > mu_floor else 1e-6

        # Fixed tau for q_L probing (paper eq 2.8)
        tau_qf = options["fraction_to_boundary"]

        # Section 6 norm scaling (computed once in optimize(), stored on self)
        qf_sd = self._qf_sd
        qf_sp = self._qf_sp
        qf_sc = self._qf_sc

        def eval_qf(sigma):
            """Evaluate q_L(sigma) — paper eq (4.2)."""
            mu_s = max(sigma * mu_nat, mu_floor)
            self.px.get_array()[:] = px0 + sigma * dpx
            self.px.copy_host_to_device()
            self.optimizer.compute_update(
                mu_s, self.gamma_penalty, self.vars, self.px, self.update
            )
            alpha_x, _, alpha_z, _ = self.optimizer.compute_max_step(
                tau_qf, self.vars, self.update
            )
            self.optimizer.apply_step_update(
                alpha_x, alpha_z, self.vars, self.update, self.temp
            )
            trial_comp_sq = self.optimizer.compute_complementarity_sq(self.temp)
            return (
                (1.0 - alpha_z) ** 2 * dual_infeas_sq * qf_sd
                + (1.0 - alpha_x) ** 2 * primal_infeas_sq * qf_sp
                + trial_comp_sq * qf_sc
            )

        # Interval selection (paper Section 4): test q_L(0.99) vs q_L(1)
        # to choose [sigma_min, 1] or [1, sigma_max], then golden section.
        sigma_min = max(1e-6, sigma_min_floor)
        sigma_max = options["quality_function_sigma_max"]
        n_gs_iters = options["quality_function_golden_iters"]

        qL_099 = eval_qf(0.99)
        qL_1 = eval_qf(1.0)

        if qL_099 <= qL_1:
            sigma_star, _ = self._golden_section_search(
                eval_qf, sigma_min, 1.0, n_iters=n_gs_iters
            )
        else:
            sigma_star, _ = self._golden_section_search(
                eval_qf, 1.0, sigma_max, n_iters=n_gs_iters
            )

        mu_new = sigma_star * mu_nat

        if comm_rank == 0:
            print(
                f"  QF: sigma={sigma_star:.4f}, "
                f"mu={mu_new:.4e} (comp={avg_comp:.4e})"
            )

        return sigma_star, mu_new

    def _compute_least_squares_multipliers(self):
        """
        Compute the least squares multiplier estimates by solving the system of equations

        [ I  A^{T} ][ w      ] = - [ (g - zl + zu) ]
        [ A     0  ][ lambda ] =   [             0 ]

        Note that the w components are discarded.

        On input to the function, the design variables stored in self.vars.get_solution() are
        at the initial point and the multipliers are initialized to zero.
        """

        # Compute the residual based on the gradient. At this point the multipliers are zero
        x = self.vars.get_solution()
        self.optimizer.compute_residual(0.0, 0.0, self.vars, self.grad, self.res)

        # Zero out the constraint contributions from the right-hand-side
        self.optimizer.set_multipliers_value(0.0, self.res)

        # Set the diagonal entries - zero everywhere except in entries with the design variables
        # where the diagonal = 1.0
        self.diag.zero()
        self.optimizer.set_design_vars_value(1.0, self.diag)

        # Set up the system above (Note that alpha = 0.0, zeroing the objective contributions
        # to the Hessian. Since the multipliers are also zero, they contribute nothing to the
        # design-variable Hessian. Only the constraint Jacobians remain.)
        self.solver.factor(0.0, x, self.diag)
        self.solver.solve(self.res, self.px)

        # Copy the multipiler values from self.px to x
        self.optimizer.copy_multipliers(x, self.px)

    def _compute_affine_multipliers(self, gamma_penalty=1e3, beta_min=1.0):
        """
        Compute the affine multipliers
        """

        # Compute the gradient at the new point with the updated multipliers
        x = self.vars.get_solution()
        if self.distribute:
            self.mpi_problem.gradient(1.0, x, self.grad)
        else:
            self.problem.gradient(1.0, x, self.grad)

        # Compute the residual
        mu = 0.0
        self.optimizer.compute_residual(
            mu, gamma_penalty, self.vars, self.grad, self.res
        )

        # Add the diagonal contributions to the Hessian matrix
        self.optimizer.compute_diagonal(self.vars, self.diag)

        # Solve the KKT system
        self.solver.factor(1.0, x, self.diag)
        self.solver.solve(self.res, self.px)

        # Compute the full update based on the reduced variable update
        self.optimizer.compute_update(
            mu, gamma_penalty, self.vars, self.px, self.update
        )

        self.optimizer.compute_affine_start_point(
            beta_min, self.vars, self.update, self.temp
        )

        # Copy the multipliers to the solution
        xt = self.temp.get_solution()
        self.optimizer.copy_multipliers(x, xt)

        # Now, compute the updates based
        barrier, _ = self.optimizer.compute_complementarity(self.temp)

        self.vars.copy(self.temp)

        return barrier

    def _add_regularization_terms(
        self,
        diag,
        eps_x=1e-4,
        eps_z=1e-4,
        zero_hessian_indices=None,
        eps_x_zero=None,
        nonconvex_indices=None,
    ):
        """
        Add regularization to the KKT diagonal: +eps_x for primal, -eps_z for dual.

        When nonconvex_indices is provided, eps_z is applied selectively:
        tiny eps on all multipliers, full eps_z only on nonconvex constraints.
        """
        diag.copy_device_to_host()
        d = diag.get_array()

        problem = self.mpi_problem if self.distribute else self.problem
        mult = np.array(problem.get_multiplier_indicator(), dtype=bool)

        if nonconvex_indices is not None:
            eps_z_num = 1e-10
            d[~mult] += eps_x
            d[mult] -= eps_z_num
            if eps_z > eps_z_num:
                d[nonconvex_indices] -= eps_z - eps_z_num
        else:
            d[~mult] += eps_x
            d[mult] -= eps_z

        if zero_hessian_indices is not None and eps_x_zero is not None:
            d[zero_hessian_indices] += eps_x_zero - eps_x

        diag.copy_host_to_device()

    def _zero_multipliers(self, x):
        self.optimizer.set_multipliers_value(0.0, x)

    def _update_gradient(self, x):
        """Evaluate problem functions and gradient at x."""
        if self.distribute:
            self.mpi_problem.update(x)
            self.mpi_problem.gradient(1.0, x, self.grad)
        else:
            self.problem.update(x)
            self.problem.gradient(1.0, x, self.grad)

    def _factorize_kkt(
        self,
        x,
        diag_base,
        probe_convex,
        options,
        zero_hessian_indices,
        zero_hessian_eps,
        comm_rank,
    ):
        """Regularize and factorize KKT matrix (three-layer inertia correction)."""
        self.diag.get_array()[:] = diag_base
        self.diag.copy_host_to_device()

        if probe_convex and hasattr(self.solver, "assemble_hessian"):
            # --- Layer 1: Diagonal Hessian regularization ---
            W_diag = self.solver.assemble_hessian(1.0, x)
            probe_convex.build_regularization(
                self.diag,
                W_diag,
                diag_base,
                zero_hessian_indices,
                zero_hessian_eps,
                solver=self.solver,
            )

            # First factorization uses the already-assembled Hessian
            reg_diag = self.diag.get_array().copy()
            primal_mask = ~probe_convex.mult_ind
            n_primal = int(np.sum(primal_mask))
            n_dual = int(np.sum(probe_convex.mult_ind))
            has_inertia = hasattr(self.solver, "get_inertia")
            max_corrections = options["max_inertia_corrections"]
            delta = probe_convex.last_inertia_delta / 3.0

            # Targeted inertia correction variable set
            nc_primal = probe_convex._nonconvex_primal_indices
            use_targeted = nc_primal is not None and len(nc_primal) > 0

            inertia_ok = False
            try:
                self.solver.add_diagonal_and_factor(self.diag)
                if has_inertia:
                    n_pos, n_neg = self.solver.get_inertia()
                    inertia_ok = n_pos == n_primal and n_neg == n_dual
                else:
                    inertia_ok = True
            except Exception:
                pass  # factorization failed; proceed to correction

            if not inertia_ok and has_inertia:
                # --- Layers 2/3: Inertia correction ---
                # Layer 2: targeted delta on nonconvex primal vars (first half)
                # Layer 3: global delta on all primal vars (second half)
                delta = max(delta, 1e-8)
                for attempt in range(max_corrections):
                    self.diag.get_array()[:] = reg_diag
                    if use_targeted and attempt < max_corrections // 2:
                        self.diag.get_array()[nc_primal] += delta
                    else:
                        if attempt == max_corrections // 2 and comm_rank == 0:
                            print(
                                f"  Inertia: targeted failed, " f"switching to global"
                            )
                        self.diag.get_array()[primal_mask] += delta
                    self.diag.copy_host_to_device()

                    try:
                        self.solver.factor(1.0, x, self.diag)
                    except Exception as e:
                        delta = max(1e-8, delta * 10)
                        if comm_rank == 0:
                            print(f"  Inertia: factor failed, " f"delta -> {delta:.2e}")
                        if attempt == max_corrections - 1:
                            raise e
                        continue

                    n_pos, n_neg = self.solver.get_inertia()
                    if n_pos == n_primal and n_neg == n_dual:
                        inertia_ok = True
                        break
                    delta = max(1e-8, delta * 10)
                    if comm_rank == 0:
                        layer = (
                            "targeted"
                            if use_targeted and attempt < max_corrections // 2
                            else "global"
                        )
                        print(
                            f"  Inertia ({layer}): expected "
                            f"({n_primal}+, {n_dual}-), got "
                            f"({n_pos}+, {n_neg}-), "
                            f"delta -> {delta:.2e}"
                        )

            probe_convex.last_inertia_delta = delta
        elif probe_convex:
            # Solver without assemble_hessian: use Layer 1 with separate call
            if hasattr(self.solver, "get_hessian_diagonal"):
                W_diag = self.solver.get_hessian_diagonal(1.0, x)
            else:
                W_diag = np.zeros(len(diag_base))
            probe_convex.build_regularization(
                self.diag, W_diag, diag_base, zero_hessian_indices, zero_hessian_eps
            )
            try:
                self.solver.factor(1.0, x, self.diag)
            except Exception:
                # Fallback: add uniform delta
                self.diag.get_array()[~probe_convex.mult_ind] += 1e-4
                self.diag.copy_host_to_device()
                self.solver.factor(1.0, x, self.diag)
        else:
            # Plain scalar regularization with adaptive retry
            eps_x = options["regularization_eps_x"]
            eps_z = options["regularization_eps_z"]
            self._add_regularization_terms(
                self.diag, eps_x, eps_z, zero_hessian_indices, zero_hessian_eps
            )

            max_reg = options["max_regularization"]
            reg_factor = options["regularization_increase_factor"]
            for attempt in range(15):
                try:
                    self.solver.factor(1.0, x, self.diag)
                    break
                except Exception as e:
                    if options["adaptive_regularization"] and eps_x < max_reg:
                        eps_x *= reg_factor
                        eps_z *= reg_factor
                        if comm_rank == 0:
                            print(f"  Factorization failed, eps_x -> {eps_x:.2e}")
                        self.diag.get_array()[:] = diag_base
                        self.diag.copy_host_to_device()
                        self._add_regularization_terms(
                            self.diag,
                            eps_x,
                            eps_z,
                            zero_hessian_indices,
                            zero_hessian_eps,
                        )
                    else:
                        raise e

    def _solve_with_mu(self, mu):
        """Back-solve for Newton direction with given barrier parameter.

        Requires _factorize_kkt() to have been called first.
        Modifies self.res, self.px, and self.update in-place.
        """
        self.optimizer.compute_residual(
            mu, self.gamma_penalty, self.vars, self.grad, self.res
        )
        self.solver.solve(self.res, self.px)
        self.optimizer.compute_update(
            mu, self.gamma_penalty, self.vars, self.px, self.update
        )

    def _find_direction(
        self,
        x,
        diag_base,
        probe_convex,
        options,
        zero_hessian_indices,
        zero_hessian_eps,
        comm_rank,
    ):
        """Compute Newton direction: factorize + solve.

        Convenience wrapper that calls _factorize_kkt then _solve_with_mu.
        """
        self._factorize_kkt(
            x,
            diag_base,
            probe_convex,
            options,
            zero_hessian_indices,
            zero_hessian_eps,
            comm_rank,
        )
        self._solve_with_mu(self.barrier_param)

    def _line_search(self, alpha_x, alpha_z, convex, options, comm_rank):
        """Backtracking line search on ||KKT|| merit function.

        Returns (alpha, line_iters, step_accepted).
        """
        max_iters = options["max_line_search_iterations"]
        armijo_c = options["armijo_constant"]
        use_armijo = options["use_armijo_line_search"]
        backtrack = options["backtracking_factor"]

        ls_baseline = self.optimizer.compute_residual(
            self.barrier_param, self.gamma_penalty, self.vars, self.grad, self.res
        )
        dphi_0 = -ls_baseline

        alpha = 1.0
        for j in range(max_iters):
            self.optimizer.apply_step_update(
                alpha * alpha_x, alpha * alpha_z, self.vars, self.update, self.temp
            )

            xt = self.temp.get_solution()
            self._update_gradient(xt)

            res_new = self.optimizer.compute_residual(
                self.barrier_param, self.gamma_penalty, self.temp, self.grad, self.res
            )

            # Acceptance criterion
            if use_armijo:
                threshold = ls_baseline + armijo_c * alpha * dphi_0
                acceptable = res_new <= threshold
            else:
                acceptable = res_new < ls_baseline

            if acceptable:
                self.vars.copy(self.temp)
                return alpha, j + 1, True

            if j < max_iters - 1:
                alpha *= backtrack
                continue

            # Last iteration: relaxed acceptance / rejection
            if convex:
                if res_new <= ls_baseline:
                    self.vars.copy(self.temp)
                    return alpha, j + 1, True
                convex.reject_step()
                self._update_gradient(self.vars.get_solution())
                if comm_rank == 0:
                    print(f"  Step REJECTED ({convex.consecutive_rejections}x)")
                return alpha, j + 1, False
            elif res_new < 1.1 * ls_baseline:
                self.vars.copy(self.temp)
                if comm_rank == 0 and res_new >= ls_baseline:
                    print(f"  Warning: Accepted step with slight increase")
                return alpha, j + 1, True
            else:
                alpha = 0.01
                self.optimizer.apply_step_update(
                    alpha * alpha_x, alpha * alpha_z, self.vars, self.update, self.temp
                )
                self._update_gradient(self.temp.get_solution())
                self.vars.copy(self.temp)
                if comm_rank == 0:
                    print(f"  Warning: Line search failed, taking minimal step")
                return alpha, j + 1, True

        return alpha, max_iters, False

    def optimize(self, options={}):
        """
        Optimize the problem with the specified input options
        """

        # Keep track of the optimization time
        start_time = time.perf_counter()

        # Set the communicator rank
        comm_rank = 0
        if self.comm is not None:
            comm_rank = self.comm.rank

        # Get the set of options
        options = self.get_options(options=options)

        # Data that is recorded at each iteration
        opt_data = {"options": options, "converged": False, "iterations": []}

        self.barrier_param = options["initial_barrier_param"]
        self.gamma_penalty = options["gamma_penalty"]
        max_iters = options["max_iterations"]
        base_tau = options["fraction_to_boundary"]
        tau_min = options["tau_min"]
        use_adaptive_tau = options["adaptive_tau"]
        tol = options["convergence_tolerance"]
        record_components = options["record_components"]
        continuation_control = options["continuation_control"]

        # Get the x/multiplier solution vector from the optimization variables
        x = self.vars.get_solution()

        # Create a view into x using the component indices
        xview = None
        if not self.distribute:
            xview = ModelVector(self.model, x=x)

        self._zero_multipliers(x)
        self._update_gradient(x)

        # Initialize multipliers and slacks
        self.optimizer.initialize_multipliers_and_slacks(
            self.barrier_param, self.grad, self.vars
        )

        if options["init_affine_step_multipliers"]:
            self._compute_least_squares_multipliers()
            self.barrier_param = self._compute_affine_multipliers(
                beta_min=self.barrier_param, gamma_penalty=self.gamma_penalty
            )
        elif options["init_least_squares_multipliers"]:
            self._compute_least_squares_multipliers()

        self._update_gradient(x)

        line_iters = 0
        alpha_x_prev = 0.0
        alpha_z_prev = 0.0
        x_index_prev = -1
        z_index_prev = -1

        # Stagnation tracking for acceptable convergence
        acceptable_tol = options["acceptable_tol"]
        if acceptable_tol is None:
            acceptable_tol = tol * 100
        acceptable_iter = options["acceptable_iter"]
        prev_res_norm = float("inf")
        best_res_norm = float(
            "inf"
        )  # Track best residual to detect stagnation across cycles
        stagnation_count = 0
        precision_floor_count = 0  # Count of bit-identical residuals

        # Curvature probe convexification with inertia correction
        problem_ref = self.mpi_problem if self.distribute else self.problem
        probe_convex = None
        convex = None
        if options["curvature_probe_convexification"]:
            probe_convex = CurvatureProbeConvexifier(
                options,
                self.barrier_param,
                self.model,
                problem_ref,
                self.solver,
                self.distribute,
                tol=tol,
            )
            convex = probe_convex
            if comm_rank == 0:
                n_primal = int(np.sum(~probe_convex.mult_ind))
                solver_name = type(self.solver).__name__
                print(
                    f"  Three-layer inertia regularization "
                    f"({solver_name}): {n_primal} primal vars"
                )

        zero_step_count = 0  # Zero-step recovery (plain path only)

        # Resolve zero-Hessian variable indices
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

        # Quality function state (Algorithm A, NWW 2009)
        quality_func = options["barrier_strategy"] == "quality_function"
        qf_free_mode = True
        qf_kappa = options["quality_function_kappa_free"]
        qf_l_max = options["quality_function_l_max"]
        qf_kkt_history = deque(maxlen=qf_l_max + 1)
        qf_mu_floor = max(tol * 0.01, 1e-14)
        qf_monotone_mu = None
        qf_M_k_at_entry = None
        # Section 6 norm scaling: divide each term by element count
        qf_sd = qf_sp = qf_sc = 1.0  # default: no scaling
        if quality_func and options["quality_function_norm_scaling"]:
            n_d, n_p, n_c = self.optimizer.get_kkt_element_counts()
            qf_sd = 1.0 / max(n_d, 1)
            qf_sp = 1.0 / max(n_p, 1)
            qf_sc = 1.0 / max(n_c, 1)
            if comm_rank == 0:
                print(f"  QF norm scaling: n_d={n_d}, n_p={n_p}, n_c={n_c}")
        self._qf_sd = qf_sd
        self._qf_sp = qf_sp
        self._qf_sc = qf_sc

        # Seed phi_0 so M_0 = Phi_0 (Algorithm A)
        if quality_func:
            d0, p0, c0 = self.optimizer.compute_kkt_error(self.vars, self.grad)
            qf_kkt_history.append(d0 * qf_sd + p0 * qf_sp + c0 * qf_sc)

        for i in range(max_iters):
            # Compute the complete KKT residual
            res_norm = self.optimizer.compute_residual(
                self.barrier_param, self.gamma_penalty, self.vars, self.grad, self.res
            )

            if convex:
                convex.update_eps_z(self.barrier_param)
                convex.decompose_residual(self.res, self.vars)

            # Compute the elapsed time
            elapsed_time = time.perf_counter() - start_time

            # Apply the continuation strategy if any
            if continuation_control is not None:
                continuation_control(i, res_norm)

            iter_data = {
                "iteration": i,
                "time": elapsed_time,
                "residual": res_norm,
                "barrier_param": self.barrier_param,
                "line_iters": line_iters,
                "alpha_x": alpha_x_prev,
                "x_index": x_index_prev,
                "alpha_z": alpha_z_prev,
                "z_index": z_index_prev,
            }
            if convex:
                iter_data.update(convex.iter_data())

            # Compute and store xi and complementarity for heuristic (display table after optimization)
            if (
                options["barrier_strategy"] == "heuristic"
                and options["verbose_barrier"]
            ):
                complementarity, xi = self.optimizer.compute_complementarity(self.vars)
                iter_data["xi"] = xi
                iter_data["complementarity"] = complementarity

            if comm_rank == 0:
                self.write_log(i, iter_data)

            iter_data["x"] = {}
            if xview is not None:
                for name in record_components:
                    iter_data["x"][name] = xview[name].tolist()

            opt_data["iterations"].append(iter_data)

            # Check convergence
            if res_norm < tol:
                opt_data["converged"] = True
                break

            # Precision floor detection: bit-identical residuals mean we've
            # hit numerical limits. Exit immediately instead of cycling.
            rel_change = abs(res_norm - prev_res_norm) / max(res_norm, 1e-30)
            if rel_change < 1e-14 and i > 0:
                precision_floor_count += 1
            else:
                precision_floor_count = 0

            if precision_floor_count >= 3 and res_norm < acceptable_tol:
                if comm_rank == 0:
                    print(
                        f"  Precision floor: residual {res_norm:.6e} unchanged "
                        f"for {precision_floor_count} iterations"
                    )
                opt_data["converged"] = True
                opt_data["acceptable"] = True
                opt_data["precision_floor"] = True
                break

            # Check for stagnation and acceptable convergence
            # Track best residual (robust to barrier cycling that resets prev_res_norm)
            if res_norm < 0.99 * best_res_norm:
                best_res_norm = res_norm
                stagnation_count = 0
            elif res_norm < 0.99 * prev_res_norm:
                # Improving vs previous iteration but not vs best — don't reset
                stagnation_count += 1
            else:
                stagnation_count += 1

            # Acceptable convergence: stuck for N iterations at acceptable residual
            if stagnation_count >= acceptable_iter and res_norm < acceptable_tol:
                if comm_rank == 0:
                    print(
                        f"  Acceptable convergence: residual {res_norm:.6e} < {acceptable_tol:.0e} for {stagnation_count} iterations"
                    )
                opt_data["converged"] = True
                opt_data["acceptable"] = True
                break

            if convex and res_norm >= tol:
                new_barrier, should_reset = convex.check_stagnation(
                    stagnation_count,
                    2 * acceptable_iter,
                    self.barrier_param,
                    options["monotone_barrier_fraction"],
                    tol,
                )
                if should_reset:
                    if comm_rank == 0:
                        print(
                            f"  Stagnation: forcing barrier {self.barrier_param:.2e} -> "
                            f"{new_barrier:.2e}"
                        )
                    self.barrier_param = new_barrier
                    stagnation_count = 0
                    # Sync QF monotone mu so the QF block doesn't override it
                    if quality_func and not qf_free_mode:
                        qf_monotone_mu = new_barrier
                        if comm_rank == 0:
                            print(f"  QF monotone mu synced: {qf_monotone_mu:.2e}")

            prev_res_norm = res_norm

            if convex:
                convex.begin_iteration(self.barrier_param)
            else:
                # Legacy zero-step recovery
                if i > 0 and max(alpha_x_prev, alpha_z_prev) < 1e-10:
                    zero_step_count += 1
                    if zero_step_count >= 3:
                        old_barrier = self.barrier_param
                        self.barrier_param = min(self.barrier_param * 10.0, 1.0)
                        if comm_rank == 0 and self.barrier_param != old_barrier:
                            print(
                                f"  Zero step recovery: barrier {old_barrier:.2e} -> {self.barrier_param:.2e}"
                            )
                        zero_step_count = 0
                else:
                    zero_step_count = 0

            # Compute and cache the base diagonal (barrier Sigma)
            self.optimizer.compute_diagonal(self.vars, self.diag)
            self.diag.copy_device_to_host()
            diag_base = self.diag.get_array().copy()

            barrier_before = self.barrier_param

            if quality_func:
                # --- Quality function barrier (Algorithm A, NWW 2009) ---
                self._factorize_kkt(
                    x,
                    diag_base,
                    probe_convex,
                    options,
                    zero_hessian_indices,
                    zero_hessian_eps,
                    comm_rank,
                )

                if qf_free_mode:
                    sigma_opt, mu_qf = self._compute_quality_function_barrier(
                        tau_min, use_adaptive_tau, options, comm_rank
                    )
                    self.barrier_param = max(mu_qf, qf_mu_floor)
                else:
                    # Monotone mode: fixed mu, decrease when subproblem solved
                    if res_norm <= options["barrier_progress_tol"] * qf_monotone_mu:
                        new_mono_mu = max(
                            qf_monotone_mu * options["monotone_barrier_fraction"],
                            qf_mu_floor,
                        )
                        if comm_rank == 0:
                            print(
                                f"  QF monotone mu: "
                                f"{qf_monotone_mu:.4e} -> {new_mono_mu:.4e}"
                            )
                        qf_monotone_mu = new_mono_mu
                    self.barrier_param = qf_monotone_mu
                    if comm_rank == 0:
                        print(f"  QF monotone step: " f"mu={self.barrier_param:.4e}")

                if convex:
                    convex.update_eps_z(self.barrier_param)

                self._solve_with_mu(self.barrier_param)
            else:
                # --- Existing heuristic/monotone barrier strategy ---
                heuristic = options["barrier_strategy"] == "heuristic"
                if heuristic:
                    complementarity, xi = self.optimizer.compute_complementarity(
                        self.vars
                    )

                if options["progress_based_barrier"]:
                    should_reduce = self._should_reduce_barrier(
                        res_norm, self.barrier_param, options["barrier_progress_tol"]
                    )
                elif heuristic:
                    should_reduce = True
                else:
                    should_reduce = res_norm <= 0.1 * self.barrier_param

                # VC floor override
                if (
                    convex
                    and not should_reduce
                    and self.barrier_param > tol
                    and convex.should_force_barrier_reduction()
                ):
                    should_reduce = True
                    if comm_rank == 0:
                        print(
                            f"  VC floor: theta={convex.theta:.2e} <= "
                            f"3*vc={3*convex.vc_floor:.2e}, "
                            f"forcing barrier reduction"
                        )

                if should_reduce:
                    if heuristic:
                        self.barrier_param, _ = self._compute_barrier_heuristic(
                            xi,
                            complementarity,
                            options["heuristic_barrier_gamma"],
                            options["heuristic_barrier_r"],
                            tol,
                        )
                    else:
                        self.barrier_param = max(
                            self.barrier_param * options["monotone_barrier_fraction"],
                            tol,
                        )

                # Direction finding (regularize, factor, solve)
                self._find_direction(
                    x,
                    diag_base,
                    probe_convex,
                    options,
                    zero_hessian_indices,
                    zero_hessian_eps,
                    comm_rank,
                )

            # Check the update (debug only)
            if options["check_update_step"]:
                hess = (
                    self.mpi_problem if self.distribute else self.problem
                ).create_matrix()
                self.optimizer.check_update(
                    self.barrier_param,
                    self.gamma_penalty,
                    self.grad,
                    self.vars,
                    self.update,
                    hess,
                )

            # Compute step sizes
            tau = (
                self._compute_adaptive_tau(self.barrier_param, tau_min)
                if use_adaptive_tau
                else base_tau
            )
            alpha_x, x_index, alpha_z, z_index = self.optimizer.compute_max_step(
                tau, self.vars, self.update
            )

            # Diagnostic: Newton direction norms
            if comm_rank == 0 and i < 10:
                problem_ref = self.mpi_problem if self.distribute else self.problem
                mult_ind = (
                    convex.mult_ind
                    if convex
                    else np.array(problem_ref.get_multiplier_indicator(), dtype=bool)
                )
                dx_full = np.array(self.update.get_solution())
                print(
                    f"  Newton: ||dx||={np.linalg.norm(dx_full[~mult_ind]):.2e}, "
                    f"||dlam||={np.linalg.norm(dx_full[mult_ind]):.2e}, "
                    f"a_x={alpha_x:.4f}, a_z={alpha_z:.4f}"
                )

            if options["equal_primal_dual_step"]:
                alpha_x = alpha_z = min(alpha_x, alpha_z)

            # Line search
            alpha, line_iters, step_accepted = self._line_search(
                alpha_x, alpha_z, convex, options, comm_rank
            )

            if convex and convex.step_rejected:
                # Step rejected: stay at current point, undo barrier reduction
                alpha_x_prev = 0.0
                alpha_z_prev = 0.0
                x_index_prev = -1
                z_index_prev = -1
                self.barrier_param = barrier_before

                # Algorithm A: rejected step in free mode -> monotone mode
                if quality_func and qf_free_mode:
                    comp, _ = self.optimizer.compute_complementarity(self.vars)
                    qf_monotone_mu_cand = max(0.8 * comp, qf_mu_floor)
                    M_k = max(qf_kkt_history)
                    if qf_monotone_mu_cand > qf_mu_floor:
                        qf_free_mode = False
                        qf_M_k_at_entry = M_k
                        qf_monotone_mu = qf_monotone_mu_cand
                        if comm_rank == 0:
                            print(
                                f"  QF -> monotone (step rejected): "
                                f"mu_bar={qf_monotone_mu:.4e}"
                            )
                    elif comm_rank == 0:
                        print(f"  QF: comp at floor ({comp:.2e}), " f"skip monotone")

                new_barrier, increased = convex.handle_rejection_escape(
                    self.barrier_param
                )
                if increased is True:
                    if comm_rank == 0:
                        print(
                            f"  Barrier increased: {self.barrier_param:.2e} -> "
                            f"{new_barrier:.2e}"
                        )
                    self.barrier_param = new_barrier
                    if quality_func and not qf_free_mode:
                        qf_monotone_mu = new_barrier
                        if comm_rank == 0:
                            print(f"  QF monotone mu synced: {new_barrier:.2e}")
                elif increased is False:
                    if comm_rank == 0:
                        print(
                            f"  Barrier at max ({self.barrier_param:.2e}), "
                            f"cannot increase further"
                        )
            else:
                alpha_x_prev = alpha * alpha_x
                alpha_z_prev = alpha * alpha_z
                x_index_prev = x_index
                z_index_prev = z_index

                if convex:
                    convex.consecutive_rejections = 0

                # Algorithm A: evaluate Phi, update mode for next iteration
                if quality_func:
                    dual_sq, primal_sq, comp_sq = self.optimizer.compute_kkt_error(
                        self.vars, self.grad
                    )
                    phi_new = dual_sq * qf_sd + primal_sq * qf_sp + comp_sq * qf_sc

                    M_k = max(qf_kkt_history)

                    if qf_free_mode:
                        if phi_new > qf_kappa * M_k:
                            # Failed nonmonotone check -> monotone mode
                            comp, _ = self.optimizer.compute_complementarity(self.vars)
                            qf_monotone_mu_cand = max(0.8 * comp, qf_mu_floor)
                            if qf_monotone_mu_cand > qf_mu_floor:
                                qf_free_mode = False
                                qf_M_k_at_entry = M_k
                                qf_monotone_mu = qf_monotone_mu_cand
                                if comm_rank == 0:
                                    print(
                                        f"  QF -> monotone: "
                                        f"Phi={phi_new:.4e} > "
                                        f"kappa*M={qf_kappa * M_k:.4e}"
                                    )
                                    print(
                                        f"  QF monotone: "
                                        f"mu_bar={qf_monotone_mu:.4e}"
                                    )
                            elif comm_rank == 0:
                                print(
                                    f"  QF: comp at floor ({comp:.2e}), "
                                    f"skip monotone"
                                )
                        else:
                            qf_kkt_history.append(phi_new)
                    else:
                        if phi_new <= qf_kappa * qf_M_k_at_entry:
                            qf_kkt_history.append(phi_new)
                            qf_free_mode = True
                            qf_monotone_mu = None
                            qf_M_k_at_entry = None
                            if comm_rank == 0:
                                print(f"  QF resume free: " f"Phi={phi_new:.4e}")

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
        self.solver.factor(1.0, x, self.diag)

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
