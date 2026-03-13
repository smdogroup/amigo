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

    def factor_from_host(self, diag):
        """Factor from BISC-modified host data, adding diagonal in-place.

        BISC edits self.hess host data directly. This method adds diag to
        the host diagonal and factorizes without a device->host copy (which
        would overwrite BISC's modifications).
        """
        diag_arr = diag.get_array()
        data = self.hess.get_data()
        data[self._diag_indices] += diag_arr
        shape = (self.nrows, self.ncols)
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

    def factor_from_host(self, diag):
        """Factor from BISC-modified host data, adding diagonal in-place.

        BISC edits self.hess host data directly. This method adds diag to
        the host diagonal and factorizes without a device->host copy (which
        would overwrite BISC's modifications).
        """
        diag_arr = diag.get_array()
        data = self.hess.get_data()
        data[self._diag_indices] += diag_arr
        self._matrix.data[:] = data[self._upper_mask]
        self.pardiso.factorize(self._matrix)

    def factor(self, alpha, x, diag, _debug_inertia=False):
        """Assemble Hessian, add diagonal, and LDL^T factorize (one shot).

        Used for inertia-correction retries where we need a fresh
        assembly (since add_diagonal_and_factor mutates self.hess).
        """
        self.problem.hessian(alpha, x, self.hess)
        self.problem.add_diagonal(diag, self.hess)
        self.hess.copy_data_device_to_host()
        data = self.hess.get_data()
        if _debug_inertia:
            full_diag = data[self._diag_indices]
            diag_arr = diag.get_array()
            print(
                f"    [DEBUG] diag vector: min={diag_arr.min():.2e}, "
                f"max={diag_arr.max():.2e}, "
                f"n_positive={np.sum(diag_arr > 0)}, "
                f"n_large={np.sum(diag_arr > 1e3)}"
            )
            print(
                f"    [DEBUG] CSR diagonal: min={full_diag.min():.2e}, "
                f"max={full_diag.max():.2e}, "
                f"n_positive={np.sum(full_diag > 0)}, "
                f"n_large={np.sum(full_diag > 1e3)}"
            )
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
    """IPOPT-style inertia correction (Algorithm IC) for interior-point methods.

    Applies IPOPT Algorithm IC (Wachter & Biegler 2006, Section 3.1):
    delta_w on primal diagonal, delta_c on constraint diagonal, with
    exponential growth retry until correct inertia is achieved.

    eps_z (virtual control) is applied selectively to nonconvex constraint
    multipliers and tracks the barrier parameter mu.
    """

    def __init__(
        self, options, barrier_param, model, problem, solver, distribute, tol=1e-6
    ):
        # eps_z / VC parameters
        self.cz = options["convex_eps_z_coeff"]
        # No unconditional constraint regularization: IPOPT never applies
        # delta_c unless inertia correction detects zero eigenvalues.
        # PARDISO Bunch-Kaufman pivoting handles D=0 entries fine.
        # Nonzero eps_z_floor creates bias eps_z*||lam|| that kills
        # quadratic convergence.
        self.eps_z_floor = 0.0
        self.eps_z = 0.0  # Set properly after _nonconvex_indices resolved
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

        # Inertia correction state (IPOPT Algorithm IC, Sec 3.1)
        self.last_inertia_delta = 0.0  # delta_w^last
        self._ic_iteration = 0  # iteration counter for early-singularity detection
        self._ic_always_dw = False  # if first 3 iters needed delta_w > 0
        self._ic_always_dc = False  # if first 3 iters needed delta_c > 0

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

        # Now set eps_z: only nonzero when nonconvex constraints exist
        if self._nonconvex_indices is not None:
            self.eps_z = max(self.eps_z_floor, self.cz * barrier_param)

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

        # Dual eps_z (selective to nonconvex constraints only)
        # No blanket eps_z: it creates bias eps_z*||lam|| that
        # prevents quadratic convergence. Only explicitly marked
        # nonconvex constraints get eps_z.
        if self._nonconvex_indices is not None and self.eps_z > 0:
            diag_arr[self._nonconvex_indices] -= self.eps_z

        diag.copy_host_to_device()

    def decompose_residual(self, res, vars):
        """Compute theta (feasibility), eta (optimality), and vc_floor."""
        res_arr = np.array(res.get_array())
        self.theta = np.linalg.norm(res_arr[self.mult_ind])
        self.eta = np.linalg.norm(res_arr[~self.mult_ind])

        x_sol = np.array(vars.get_solution())
        if self._nonconvex_indices is not None:
            lam_nc_norm = np.linalg.norm(x_sol[self._nonconvex_indices])
            self.vc_floor = self.eps_z * lam_nc_norm
        else:
            self.vc_floor = 0.0

    def update_eps_z(self, barrier_param):
        if self._nonconvex_indices is not None:
            self.eps_z = max(self.eps_z_floor, self.cz * barrier_param)
        else:
            self.eps_z = 0.0

    def should_force_barrier_reduction(self):
        """Force barrier reduction when eps_z limits convergence."""
        total_res = max(self.theta, self.eta)
        if total_res < 1e-30:
            return False
        # Only trigger when we're close enough to convergence that eps_z
        # is actually the bottleneck. With large residuals (>1), the barrier
        # should be reduced by the normal progress-based mechanism, not VC floor.
        if total_res > 1.0:
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


class Filter:
    """Outer bi-objective filter for Algorithm B (NWW 2009).

    Tracks (psi, theta) pairs where:
      psi   = sqrt(dual_sq + comp_sq)  (optimality + complementarity)
      theta = sqrt(primal_sq)          (raw constraint violation)

    A trial point is acceptable if (psi + delta, theta + delta) is not
    dominated by any filter entry.  The margin delta adapts to the
    current KKT error, preventing trivially small improvements.

    The filter never resets — it monitors progress on the original NLP
    across barrier parameter changes (Algorithm B, Section 5.2).
    """

    def __init__(self, kappa_1=1e-5, kappa_2=1.0):
        self.entries = []  # List of (psi, theta) tuples
        self.kappa_1 = kappa_1
        self.kappa_2 = kappa_2

    def margin(self, phi_kkt):
        """Compute filter margin delta_k = kappa_1 * min(kappa_2, Phi_k)."""
        return self.kappa_1 * min(self.kappa_2, phi_kkt)

    def is_acceptable(self, psi, theta, phi_kkt):
        """True if (psi + delta, theta + delta) is not dominated."""
        delta = self.margin(phi_kkt)
        psi_m = psi + delta
        theta_m = theta + delta
        for psi_f, theta_f in self.entries:
            if psi_m >= psi_f and theta_m >= theta_f:
                return False
        return True

    def add(self, psi, theta):
        """Add entry and remove dominated pairs."""
        self.entries = [
            (p, t) for p, t in self.entries if not (p >= psi and t >= theta)
        ]
        self.entries.append((psi, theta))

    def __len__(self):
        return len(self.entries)


class Optimizer:
    @property
    def barrier_param(self):
        return self._barrier_param

    @barrier_param.setter
    def barrier_param(self, value):
        self._barrier_param = value

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
        self._barrier_param = 1.0
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

        # Ensure constraint bounds always come from model meta data.
        # When users pass explicit lower/upper vectors (e.g. for primal
        # bounds), constraint rows may have incorrect defaults (0/0),
        # causing the C++ backend to misclassify inequalities as equalities.
        self._merge_constraint_bounds(model)

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

    def _merge_constraint_bounds(self, model):
        """Overlay constraint bounds from model meta data onto self.lower/upper.

        The C++ InteriorPointOptimizer classifies multiplier rows as
        equality (lb == ub) or inequality (lb != ub) based on bounds.
        When users supply explicit bound vectors, constraint rows may
        default to 0/0, misclassifying inequalities as equalities.
        """
        lb_arr = self.lower.get_array()
        ub_arr = self.upper.get_array()

        for comp_name, comp in model.comp.items():
            for con_name in comp.get_constraint_names():
                full_name = comp_name + "." + con_name
                meta = model.get_meta(full_name)
                idx = model.get_indices(full_name)
                lb_arr[idx] = meta["lower"]
                ub_arr[idx] = meta["upper"]

    def write_log(self, iteration, iter_data):
        # Write out to the log information about this line
        if iteration % 10 == 0:
            line = f"{'iteration':>10s} "
            for name in iter_data:
                if name != "iteration" and not isinstance(
                    iter_data[name], (list, dict)
                ):
                    line += f"{name:>15s} "
            print(line)

        line = f"{iteration:10d} "
        for name in iter_data:
            if name != "iteration":
                data = iter_data[name]
                if isinstance(data, (list, dict)):
                    continue
                elif isinstance(data, (int, np.integer)):
                    line += f"{data:15d} "
                else:
                    line += f"{data:15.6e} "
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
            "riccati_solve_compare": False,
            "backtracking_factor": 0.5,
            "record_components": [],
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
            "second_order_correction": True,
            # Filter line search: bi-objective (theta, psi) acceptance.
            # Accepts steps improving feasibility OR merit, with feasibility
            # restoration when both fail.
            "filter_line_search": False,
            "filter_gamma_theta": 1e-5,  # Sufficient feasibility decrease
            "filter_gamma_psi": 1e-5,  # Sufficient merit decrease
            "filter_delta": 1e-4,  # Armijo constant for switching condition
            "filter_s_theta": 1.1,  # Switching condition exponent
            "filter_restoration_max_iter": 20,  # Max restoration iterations
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
            # IPOPT Algorithm IC inertia correction (see CurvatureProbeConvexifier)
            "curvature_probe_convexification": False,
            "convex_eps_z_coeff": 1.0,  # C_z: eps_z = C_z * mu
            "nonconvex_constraints": [],  # Constraint names for selective eps_z
            "max_consecutive_rejections": 5,  # Before barrier increase
            "barrier_increase_factor": 5.0,  # Barrier *= this when stuck
            "max_inertia_corrections": 40,  # Max refactorizations for inertia (IPOPT: ~45)
            # Block PSD convexification (per-block eigendecomposition)
            "block_psd_convexification": False,
            "block_psd_hub_threshold": 50,  # Hub degree threshold
            # Eigenvalue modification mode: "reflect" or "clip".
            # "reflect": lambda~ = max(|lambda|, eps). Preserves curvature magnitude.
            # "clip":    lambda~ = max(lambda, eps). Minimal Frobenius perturbation.
            "block_psd_eigenvalue_mode": "reflect",
            # Sweep direction: "forward" or "backward".
            # Forward: R_k = S_p[k,k] - C_{k-1}^T R~_{k-1}^{-1} C_{k-1}.
            # Backward: R_k = S_p[k,k] - C_k R~_{k+1}^{-1} C_k^T.
            # Both guarantee S_p > 0 via Sylvester's inertia law.
            "block_psd_sweep_direction": "backward",
            # Max condition number for modified blocks (condition-based eps).
            "block_psd_kappa_max": 1e8,
            # Max block size for chain levels (BFS fallback).
            # Chains with any level exceeding this become small blocks.
            "block_psd_max_block_size": 64,
            # Delta-skip tier (Tier 2): small inertia defects get uniform delta
            # instead of full BISC. Practical efficiency improvement.
            "block_psd_delta_skip": True,
            "block_psd_delta_skip_max_defect": 10,
            "block_psd_delta_skip_initial": 1e-8,
            "block_psd_delta_skip_max": 1e-4,
            "block_psd_force_bisc": False,  # Skip inertia gate (testing)
            "quality_function_sigma_max": 4.0,
            "quality_function_golden_iters": 12,
            "quality_function_kappa_free": 0.9999,
            "quality_function_l_max": 5,
            "quality_function_norm_scaling": True,
            # BISC-adaptive sigma_max: reduces search range when BISC makes large
            # modifications (distorted Hessian -> linearized model less trustworthy).
            # sigma_max_eff = 1 + (sigma_max-1) / (1 + lambda * mod_ratio)
            # -> sigma_max as mod_ratio->0 (near solution), -> 1 when heavily modified.
            "quality_function_bisc_lambda": 2.0,
            # Mehrotra predictor-corrector (replaces golden section in free mode).
            # Predictor: solve at mu=0 (affine) -> alpha_aff -> comp_aff.
            # sigma_pc = min(1, (comp_aff / comp)^3), mu_pc = sigma_pc * comp.
            # Corrector: px0 + sigma_pc * dpx (exact by linearity of KKT RHS in mu).
            # No extra factorization; replaces O(n_gs) quality function evaluations.
            "quality_function_predictor_corrector": True,
            # Maximum mu decay per iteration in PC (IPOPT-style kappa_mu).
            # mu_new >= kappa * mu_old. Prevents mu crash when alpha_aff_x
            # jumps between iterations (e.g. 0.6->0.9 makes sigma_step
            # drop from 0.16->0.01, crashing mu 40x without this guard).
            "pc_kappa_mu_decay": 0.2,
        }

        for name in options:
            if name in default:
                default[name] = options[name]
            else:
                raise ValueError(f"Unrecognized option {name}")

        return default

    def _compute_barrier_heuristic(self, xi, complementarity, gamma, r, tol=1e-12):
        """Compute LOQO-style barrier parameter."""
        if xi > 1e-10:
            term = (1 - r) * (1 - xi) / xi
            heuristic_factor = min(term, 2.0) ** 3
        else:
            heuristic_factor = 2.0**3

        mu_new = gamma * heuristic_factor * complementarity

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
        self,
        tau_min,
        use_adaptive_tau,
        options,
        comm_rank,
        probe_convex=None,
    ):
        """Choose barrier parameter via Mehrotra predictor-corrector or golden section.

        Two directions are computed from the already-factorized KKT matrix:
          px0 : affine-scaling direction (mu=0)
          px1 : centering direction      (mu=avg_comp)
          dpx = px1 - px0

        The corrector direction at any mu_pc is exactly px0 + (mu_pc/mu_nat)*dpx
        because the KKT RHS is affinely linear in mu: r(mu) = r(0) + (mu/mu_nat)*r(mu_nat).
        No additional factorization is required.

        Mehrotra predictor-corrector (default):
          1. Affine max-step alpha_aff from px0 direction.
          2. Trial affine complementarity comp_aff.
          3. sigma_pc = min(1, (comp_aff / comp)^3)  [Nocedal & Wright, Alg 14.3]
          4. Corrector = px0 + sigma_pc * dpx.

        Golden-section fallback (quality_function_predictor_corrector=False):
          Searches sigma in [sigma_min, sigma_max] minimizing q_L(sigma).
        """
        avg_comp, xi = self.optimizer.compute_complementarity(self.vars)
        if avg_comp < 1e-30:
            return 1.0, self.barrier_param

        mu_nat = avg_comp  # x^T z / n

        # Affine-scaling RHS (mu=0) and dual/primal infeasibility norms
        dual_infeas_sq, primal_infeas_sq = (
            self.optimizer.compute_residual_and_infeasibility(
                0.0, self.vars, self.grad, self.res
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
        self.optimizer.compute_residual(mu_nat, self.vars, self.grad, self.res)
        if hasattr(self.solver, "solve_array"):
            px1 = self.solver.solve_array(self.res.get_array().copy()).copy()
        else:
            self.solver.solve(self.res, self.px)
            px1 = self.px.get_array().copy()

        dpx = px1 - px0

        tol_qf = options["convergence_tolerance"]
        mu_floor = max(tol_qf * 0.01, 1e-14)

        tau_qf = options["fraction_to_boundary"]

        # Diagnostic string (mod_ratio printed in both branches)
        ratio_str = ""
        if probe_convex is not None and hasattr(probe_convex, "modification_ratio"):
            ratio_str = f", mod_ratio={probe_convex.modification_ratio:.3f}"

        # -----------------------------------------------------------------------
        # Branch A: Mehrotra predictor-corrector
        # -----------------------------------------------------------------------
        if options["quality_function_predictor_corrector"]:
            # Affine update: compute full primal-dual step from px0
            self.px.get_array()[:] = px0
            self.px.copy_host_to_device()
            self.optimizer.compute_update(0.0, self.vars, self.px, self.update)
            alpha_aff_x, _, alpha_aff_z, _ = self.optimizer.compute_max_step(
                tau_qf, self.vars, self.update
            )
            # Trial affine point -> complementarity
            self.optimizer.apply_step_update(
                alpha_aff_x, alpha_aff_z, self.vars, self.update, self.temp
            )
            avg_comp_aff, _ = self.optimizer.compute_complementarity(self.temp)

            # Mehrotra sigma with NLP safeguards.
            # Classical component: how much complementarity the affine step reduces.
            sigma_aff = min(1.0, (avg_comp_aff / mu_nat) ** 3)
            # Primal-constraint component: when alpha_aff_x is small the affine
            # direction is primal-infeasible, so more centering is needed.
            # (1-alpha_aff_x)^2 -> 0 when full affine step is feasible (LP case),
            # -> large when primal boundary is tight (NLP case).
            sigma_step = (1.0 - alpha_aff_x) ** 2
            # Decay safeguard (IPOPT kappa_mu): prevents mu from crashing
            # when alpha_aff_x jumps between iterations.  mu_new >= kappa*mu_old.
            kappa_decay = options["pc_kappa_mu_decay"]
            sigma_decay = (
                kappa_decay * self.barrier_param / mu_nat
                if mu_nat > 0 and self.barrier_param > mu_floor
                else 0.0
            )
            # Floor component: keeps mu_pc >= mu_floor.
            sigma_floor = mu_floor / mu_nat if mu_nat > mu_floor else 1.0
            sigma_pc = min(1.0, max(sigma_aff, sigma_step, sigma_decay, sigma_floor))
            mu_pc = max(sigma_pc * mu_nat, mu_floor)

            if comm_rank == 0:
                print(
                    f"  PC: sigma={sigma_pc:.4f} "
                    f"(aff={sigma_aff:.4f}, step={sigma_step:.4f}, "
                    f"decay={sigma_decay:.4f}), "
                    f"mu={mu_pc:.4e} "
                    f"(comp={avg_comp:.4e}, comp_aff={avg_comp_aff:.4e}, "
                    f"a_aff={alpha_aff_x:.3f}){ratio_str}"
                )

            # Corrector direction: px0 + sigma_pc*dpx is the exact KKT solution at
            # mu=mu_pc (no extra factorization; linearity of r(mu) in mu guarantees
            # this is identical to re-solving K p = r(mu_pc) from scratch).
            self.px.get_array()[:] = px0 + sigma_pc * dpx
            self.px.copy_host_to_device()
            self.optimizer.compute_update(mu_pc, self.vars, self.px, self.update)
            return sigma_pc, mu_pc

        # -----------------------------------------------------------------------
        # Branch B: golden-section search over q_L(sigma)
        # -----------------------------------------------------------------------
        sigma_min_floor = (mu_floor / mu_nat) if mu_nat > mu_floor else 1e-6
        sigma_min = max(1e-6, sigma_min_floor)
        sigma_max = options["quality_function_sigma_max"]
        n_gs_iters = options["quality_function_golden_iters"]

        qf_sd = self._qf_sd
        qf_sp = self._qf_sp
        qf_sc = self._qf_sc

        # BISC-adaptive sigma_max: shrink search range when BISC makes large
        # modifications (distorted Hessian -> linearized model less trustworthy).
        if probe_convex is not None and hasattr(probe_convex, "modification_ratio"):
            ratio = probe_convex.modification_ratio
            if ratio > 0:
                lambda_b = options["quality_function_bisc_lambda"]
                sigma_max = 1.0 + (sigma_max - 1.0) / (1.0 + lambda_b * ratio)

        def eval_qf(sigma):
            """Evaluate the linear quality function q_L(sigma)."""
            mu_s = max(sigma * mu_nat, mu_floor)
            self.px.get_array()[:] = px0 + sigma * dpx
            self.px.copy_host_to_device()
            self.optimizer.compute_update(mu_s, self.vars, self.px, self.update)
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
            sigma_max_str = (
                f", sigma_max={sigma_max:.2f}"
                if probe_convex is not None
                and hasattr(probe_convex, "modification_ratio")
                else ""
            )
            print(
                f"  QF: sigma={sigma_star:.4f}, "
                f"mu={mu_new:.4e} (comp={avg_comp:.4e}){ratio_str}{sigma_max_str}"
            )

        self.px.get_array()[:] = px0 + sigma_star * dpx
        self.px.copy_host_to_device()
        self.optimizer.compute_update(mu_new, self.vars, self.px, self.update)

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
        self.optimizer.compute_residual(0.0, self.vars, self.grad, self.res)

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

    def _compute_affine_multipliers(self, beta_min=1.0):
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
        self.optimizer.compute_residual(mu, self.vars, self.grad, self.res)

        # Add the diagonal contributions to the Hessian matrix
        self.optimizer.compute_diagonal(self.vars, self.diag)

        # Solve the KKT system
        self.solver.factor(1.0, x, self.diag)
        self.solver.solve(self.res, self.px)

        # Compute the full update based on the reduced variable update
        self.optimizer.compute_update(mu, self.vars, self.px, self.update)

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
        """Regularize and factorize KKT matrix (IPOPT Algorithm IC)."""
        self.diag.get_array()[:] = diag_base
        self.diag.copy_host_to_device()

        if probe_convex and hasattr(self.solver, "assemble_hessian"):
            self.solver.assemble_hessian(1.0, x)

            # ==========================================================
            # IPOPT Algorithm IC (Wachter & Biegler 2006, Section 3.1)
            # The KKT system is modified with delta_w on primal diagonal
            # and -delta_c on constraint diagonal:
            #   [W + Sigma + delta_w*I    J^T     ] [dx]
            #   [       J             -delta_c*I   ] [dl]
            # delta_c is only applied when zero eigenvalues are detected
            # (indicating rank-deficient constraint Jacobian).
            # ==========================================================
            primal_mask = ~probe_convex.mult_ind
            n_primal = int(np.sum(primal_mask))
            n_dual = int(np.sum(probe_convex.mult_ind))
            n_total = n_primal + n_dual
            has_inertia = hasattr(self.solver, "get_inertia")
            mu = probe_convex._barrier

            # IPOPT constants (Section 3.1, after Algorithm IC)
            dw_min = 1e-20  # delta_bar_w^min
            dw_0 = 1e-4  # delta_bar_w^0
            dw_max = 1e40  # delta_bar_w^max
            dc_bar = 1e-8  # delta_bar_c (approx sqrt(eps_mach))
            kw_minus = 1.0 / 3  # kappa_w^-
            kw_plus = 8.0  # kappa_w^+
            kw_plus_bar = 100.0  # kappa_bar_w^+ (IPOPT default)
            kc = 0.25  # kappa_c

            # Baseline: small eps on primal for numerical stability,
            # -eps_z on inequality constraint rows for condensed formulation.
            # Equality constraints have D = 0 from compute_diagonal; adding
            # eps_z_floor creates tiny eigenvalues (O(1e-7)) that inflate
            # ||KKT^{-1}|| to O(1e7), destroying quadratic Newton convergence.
            # PARDISO handles D = 0 fine via Bunch-Kaufman 2x2 pivoting.
            # The IC-2 delta_c handles rank-deficient Jacobians separately.
            diag_arr = self.diag.get_array()
            diag_arr[primal_mask] += probe_convex.numerical_eps
            # No unconditional eps_z on constraint rows: it creates
            # non-vanishing bias eps_z*||lam|| that kills quadratic
            # convergence. IPOPT only applies delta_c during inertia
            # correction (IC-2). Selective eps_z for nonconvex constraints
            # only (when explicitly configured).
            if probe_convex._nonconvex_indices is not None and probe_convex.eps_z > 0:
                ineq_dual_mask = probe_convex.mult_ind & (diag_arr < -1e-30)
                nc_ineq = np.isin(
                    probe_convex._nonconvex_indices,
                    np.where(ineq_dual_mask)[0],
                )
                nc_ineq_idx = probe_convex._nonconvex_indices[nc_ineq]
                if len(nc_ineq_idx) > 0:
                    diag_arr[nc_ineq_idx] -= probe_convex.eps_z
            if zero_hessian_indices is not None and zero_hessian_eps is not None:
                np.maximum(
                    diag_arr[zero_hessian_indices],
                    zero_hessian_eps,
                    out=diag_arr[zero_hessian_indices],
                )
            self.diag.copy_host_to_device()

            if not has_inertia:
                # No inertia available: just factor with baseline
                try:
                    self.solver.add_diagonal_and_factor(self.diag)
                except Exception:
                    diag_arr[primal_mask] += dw_0
                    self.diag.copy_host_to_device()
                    self.solver.factor(1.0, x, self.diag)
            else:
                # reg_diag = baseline diagonal (before any delta_w/delta_c)
                reg_diag = self.diag.get_array().copy()
                last_delta = probe_convex.last_inertia_delta
                ic_iter = probe_convex._ic_iteration
                probe_convex._ic_iteration += 1
                max_corrections = options["max_inertia_corrections"]

                # --- IC-1: Try with minimal modification ---
                ic1_dw = 0.0
                ic1_dc = 0.0
                if probe_convex._ic_always_dw and last_delta > 0:
                    ic1_dw = max(dw_min, kw_minus * last_delta)
                if probe_convex._ic_always_dc:
                    ic1_dc = dc_bar * mu**kc

                if ic1_dw > 0 or ic1_dc > 0:
                    self.diag.get_array()[:] = reg_diag
                    if ic1_dw > 0:
                        self.diag.get_array()[primal_mask] += ic1_dw
                    if ic1_dc > 0:
                        self.diag.get_array()[probe_convex.mult_ind] -= ic1_dc
                    self.diag.copy_host_to_device()

                inertia_ok = False
                has_zero_eigs = False
                n_pos = n_neg = 0
                try:
                    if ic1_dw > 0 or ic1_dc > 0:
                        self.solver.factor(1.0, x, self.diag)
                    else:
                        self.solver.add_diagonal_and_factor(self.diag)
                    n_pos, n_neg = self.solver.get_inertia()
                    inertia_ok = n_pos == n_primal and n_neg == n_dual
                    has_zero_eigs = n_pos + n_neg < n_total
                except Exception:
                    has_zero_eigs = True  # Factor failure = singular

                if inertia_ok:
                    # IC-1 success: pure Newton (no modification)
                    if ic1_dw > 0:
                        probe_convex.last_inertia_delta = ic1_dw
                    else:
                        # Reset: unmodified KKT has correct inertia
                        probe_convex.last_inertia_delta = 0.0
                else:
                    # --- IC-2: delta_c for zero eigenvalues ---
                    delta_c = dc_bar * mu**kc if has_zero_eigs else 0.0

                    # --- IC-3: initial delta_w ---
                    if last_delta == 0.0:
                        delta_w = dw_0
                    else:
                        delta_w = max(dw_min, kw_minus * last_delta)

                    # --- IC-4 through IC-6 retry loop ---
                    for attempt in range(max_corrections):
                        self.diag.get_array()[:] = reg_diag
                        self.diag.get_array()[primal_mask] += delta_w
                        if delta_c > 0:
                            self.diag.get_array()[probe_convex.mult_ind] -= delta_c
                        self.diag.copy_host_to_device()

                        try:
                            self.solver.factor(1.0, x, self.diag)
                            n_pos, n_neg = self.solver.get_inertia()
                        except Exception:
                            n_pos = n_neg = 0  # Treat as wrong

                        if n_pos == n_primal and n_neg == n_dual:
                            # IC-4: success
                            probe_convex.last_inertia_delta = delta_w
                            if ic_iter < 3:
                                probe_convex._ic_always_dw = True
                            if delta_c > 0 and ic_iter < 3:
                                probe_convex._ic_always_dc = True
                            inertia_ok = True
                            if comm_rank == 0:
                                print(
                                    f"  Inertia correction: "
                                    f"delta_w={delta_w:.2e}, "
                                    f"delta_c={delta_c:.2e}, "
                                    f"attempts={attempt + 1}"
                                )
                            break

                        # IC-5: grow delta_w
                        if last_delta == 0.0:
                            delta_w *= kw_plus_bar  # x100
                        else:
                            delta_w *= kw_plus  # x8

                        # IC-6: abort if too large
                        if delta_w > dw_max:
                            if comm_rank == 0:
                                print(
                                    f"  Inertia: delta_w="
                                    f"{delta_w:.2e} > max, "
                                    f"aborting correction"
                                )
                            break

                        if comm_rank == 0:
                            print(
                                f"  Inertia: expected "
                                f"({n_primal}+, {n_dual}-), "
                                f"got ({n_pos}+, {n_neg}-), "
                                f"delta_w -> {delta_w:.2e}"
                            )

                    if not inertia_ok:
                        # Save for warm-start even on failure
                        probe_convex.last_inertia_delta = delta_w
                        if ic_iter < 3:
                            probe_convex._ic_always_dw = True
        else:
            # No convexifier or no assemble_hessian: just factor
            try:
                self.solver.factor(1.0, x, self.diag)
            except Exception:
                diag_arr = self.diag.get_array()
                if probe_convex:
                    diag_arr[~probe_convex.mult_ind] += 1e-4
                else:
                    diag_arr += 1e-4
                self.diag.copy_host_to_device()
                self.solver.factor(1.0, x, self.diag)

    def _solve_with_mu(self, mu):
        """Back-solve for Newton direction with given barrier parameter.

        Requires _factorize_kkt() to have been called first.
        Modifies self.res, self.px, and self.update in-place.
        """
        self.optimizer.compute_residual(mu, self.vars, self.grad, self.res)
        self.solver.solve(self.res, self.px)
        self.optimizer.compute_update(mu, self.vars, self.px, self.update)

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

    def _line_search(
        self,
        alpha_x,
        alpha_z,
        convex,
        options,
        comm_rank,
        tau=0.995,
        mult_ind=None,
    ):
        """Backtracking line search on ||KKT|| merit function.

        Returns (alpha, line_iters, step_accepted).
        """
        max_iters = options["max_line_search_iterations"]
        armijo_c = options["armijo_constant"]
        use_armijo = options["use_armijo_line_search"]
        backtrack = options["backtracking_factor"]
        use_soc = options["second_order_correction"] and mult_ind is not None

        ls_baseline = self.optimizer.compute_residual(
            self.barrier_param, self.vars, self.grad, self.res
        )
        dphi_0 = -ls_baseline

        if use_soc:
            res_orig = self.res.get_array().copy()
            update_backup = self.optimizer.create_opt_vector()
            update_backup.copy(self.update)
            px_orig = self.px.get_array().copy()

        alpha = 1.0
        soc_attempted = False

        for j in range(max_iters):
            self.optimizer.apply_step_update(
                alpha * alpha_x, alpha * alpha_z, self.vars, self.update, self.temp
            )

            xt = self.temp.get_solution()
            self._update_gradient(xt)

            res_new = self.optimizer.compute_residual(
                self.barrier_param, self.temp, self.grad, self.res
            )

            if use_armijo:
                threshold = ls_baseline + armijo_c * alpha * dphi_0
                acceptable = res_new <= threshold
            else:
                acceptable = res_new < ls_baseline

            if acceptable:
                self.vars.copy(self.temp)
                return alpha, j + 1, True

            # SOC: try once after first full-step rejection, but only if
            # the trial residual isn't too far above baseline (otherwise the
            # correction overshoots and produces a worse point).
            if (
                use_soc
                and j == 0
                and not soc_attempted
                and res_new <= 10.0 * ls_baseline
            ):
                soc_attempted = True
                try:
                    # self.res already holds the residual at self.temp (the trial
                    # point), computed just above for res_new. Its constraint rows
                    # [mult_ind] capture c(x+dx) — the nonlinear correction that SOC
                    # is designed to exploit. Replace only the primal stationarity
                    # rows [~mult_ind] with those from the current point (res_orig),
                    # which are unchanged by the step.
                    res_arr = self.res.get_array()
                    res_arr[~mult_ind] = res_orig[~mult_ind]
                    self.res.copy_host_to_device()

                    # Restore gradient to current point before back-solve so that
                    # compute_update uses the correct stationarity residual.
                    self._update_gradient(self.vars.get_solution())

                    # Back-solve (reuses existing factorization)
                    self.solver.solve(self.res, self.px)
                    self.optimizer.compute_update(
                        self.barrier_param,
                        self.vars,
                        self.px,
                        self.update,
                    )

                    # SOC trial
                    soc_ax, _, soc_az, _ = self.optimizer.compute_max_step(
                        tau, self.vars, self.update
                    )
                    self.optimizer.apply_step_update(
                        soc_ax, soc_az, self.vars, self.update, self.temp
                    )
                    self._update_gradient(self.temp.get_solution())
                    res_soc = self.optimizer.compute_residual(
                        self.barrier_param,
                        self.temp,
                        self.grad,
                        self.res,
                    )

                    if use_armijo:
                        soc_ok = res_soc <= ls_baseline + armijo_c * dphi_0
                    else:
                        soc_ok = res_soc < ls_baseline

                    if soc_ok:
                        self.vars.copy(self.temp)
                        if comm_rank == 0:
                            print(f"  SOC accepted: {ls_baseline:.2e} -> {res_soc:.2e}")
                        return 1.0, j + 1, True

                    if comm_rank == 0:
                        print(f"  SOC rejected: {res_soc:.2e} vs {ls_baseline:.2e}")
                except Exception:
                    if comm_rank == 0:
                        print("  SOC failed, continuing backtracking")

                # Restore original direction for backtracking
                self.update.copy(update_backup)
                self.px.get_array()[:] = px_orig
                self.px.copy_host_to_device()

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

    def _compute_barrier_objective(self, vars):
        """Compute IPOPT barrier objective phi = f(x) - mu * sum(ln(s)).

        f(x) is extracted from the Lagrangian by subtracting multiplier terms.
        The gradient and problem state must be current at the vars point
        (i.e. _update_gradient must have been called).
        """
        problem = self.mpi_problem if self.distribute else self.problem
        x_vec = vars.get_solution()

        # L(x,lam) = f(x) + lam^T c(x)
        lagr = problem.lagrangian(1.0, x_vec)

        # Subtract lam^T c(x): at constraint indices, g holds c(x) values
        # and x holds the multiplier values
        g = np.array(self.grad.get_array())
        x_arr = np.array(x_vec.get_array())
        mult_sum = np.dot(x_arr[self._filter_mult_ind], g[self._filter_mult_ind])
        f_obj = lagr - mult_sum

        # Barrier log terms: -mu * sum(ln(barrier_vars))
        barrier_log = self.optimizer.compute_barrier_log_sum(self.barrier_param, vars)
        return f_obj + barrier_log

    def _compute_filter_theta(self, vars=None):
        """Compute theta = ||c(x)|| (barrier-independent constraint violation)."""
        if vars is None:
            vars = self.vars
        _, primal_sq, _ = self.optimizer.compute_kkt_error(vars, self.grad)
        return np.sqrt(primal_sq)

    def _filter_line_search(
        self,
        alpha_x,
        alpha_z,
        convex,
        inner_filter,
        options,
        comm_rank,
        tau=0.995,
        mult_ind=None,
        phi_current=None,
    ):
        """Inner filter line search (IPOPT Algorithm A, Waechter & Biegler 2006).

        Uses the IPOPT bi-objective filter with measures:
          theta = ||c(x)|| (constraint violation, barrier-independent)
          phi   = f(x) - mu * sum(ln(s)) (barrier objective)

        Case 1 (switching): theta_k < theta_min AND dphi < 0 AND switching
            inequality holds -> accept with Armijo on phi. Filter NOT augmented.
        Case 2 (filter): sufficient decrease in theta OR phi, acceptable
            to the filter. Filter IS augmented with current (phi_k, theta_k).

        IPOPT key detail: only the PRIMAL step is backtracked; the dual
        step uses the full alpha_z from fraction-to-boundary.

        Returns (alpha, line_iters, step_accepted).
        """
        max_iters = options["max_line_search_iterations"]
        backtrack = options["backtracking_factor"]
        gamma_theta = options["filter_gamma_theta"]
        gamma_phi = options["filter_gamma_psi"]  # reuse option for phi
        delta_switch = options["filter_delta"]  # switching condition delta
        s_theta = options["filter_s_theta"]
        s_phi = 2.3  # IPOPT default exponent for switching inequality
        eta_phi = 1e-4  # Armijo constant for phi
        use_soc = options["second_order_correction"] and mult_ind is not None
        tol = options["convergence_tolerance"]

        # Current point measures
        theta_k = self._compute_filter_theta()
        phi_k = phi_current  # computed in main loop at current point

        # theta_min for switching condition (IPOPT Eq. 21: 1e-4 * max(1, theta_0))
        theta_0 = getattr(self, "_filter_theta_0", theta_k)
        theta_min = 1e-4 * max(1.0, theta_0)

        # FIX 4: theta_max uses theta_0 (set once per barrier subproblem),
        # NOT theta_k. IPOPT Eq. 21: theta^max = 1e4 * max(1, theta(x_0)).
        theta_max = 1e4 * max(1.0, theta_0)

        # FIX 1: Analytical dphi from C++ (IPOPT Eq. 19-20).
        # dphi = nabla phi_mu(x_k)^T d_k, computed from the KKT residual,
        # solution, and slack data. This replaces the finite-difference
        # estimation that was unreliable at large step sizes.
        dphi = self.optimizer.compute_barrier_dphi(
            self.barrier_param,
            self.vars,
            self.update,
            self.res,
            self.px,
            self.diag,
        )

        # Switching condition (IPOPT Eq. 19): checked once at the start.
        # switching_active means we use Armijo on phi instead of filter.
        # Numerical safeguard: don't activate switching when |dphi| is so
        # small that the Armijo condition can't be verified in floating point.
        # Need |dphi| > eps_machine * |phi| for meaningful Armijo check.
        switching_eligible = theta_k < theta_min
        switching_active = False
        dphi_threshold = 1e-13 * max(1.0, abs(phi_k))
        if switching_eligible and dphi < -dphi_threshold:
            lhs = (-dphi) ** s_phi
            rhs = delta_switch * max(theta_k, 1e-30) ** s_theta
            switching_active = lhs > rhs

        # Near-feasible regime: theta so small that filter conditions become
        # numerically degenerate (require detecting O(gamma_phi * theta_k)
        # changes in phi, which may be below floating-point precision).
        # In this regime, fall back to KKT residual reduction acceptance.
        # Use theta_min (same as switching eligibility) as the threshold:
        # when theta < theta_min, we're in the "optimality" regime.
        near_feasible = theta_k < theta_min

        if comm_rank == 0:
            mode = "switch" if switching_active else "filter"
            if near_feasible:
                mode += "+nf"
            print(
                f"  dphi={dphi:.2e}, theta_k={theta_k:.2e}, "
                f"theta_min={theta_min:.2e}, mode={mode}"
            )

        # FIX 3: alpha_min (IPOPT Eq. 23) — minimum step size before
        # restoration. IPOPT multiplies by gamma_alpha (alpha_min_frac=0.05).
        gamma_alpha = 0.05
        if switching_active:
            # Case I: alpha_min from switching / Armijo
            alpha_min_val = gamma_alpha * min(
                gamma_theta,
                gamma_phi * theta_k / (-dphi),
                (
                    delta_switch * theta_k**s_theta / (-dphi) ** s_phi
                    if (-dphi) ** s_phi > 1e-30
                    else 1.0
                ),
            )
        else:
            # Case II: alpha_min from filter sufficient decrease
            if dphi < 0:
                alpha_min_val = gamma_alpha * min(
                    gamma_theta,
                    gamma_phi * theta_k / (-dphi),
                )
            else:
                # dphi >= 0: only theta decrease can satisfy filter
                alpha_min_val = gamma_alpha * gamma_theta

        # Compute current KKT residual for near-feasible fallback
        if near_feasible:
            res_kkt_current = self.optimizer.compute_residual(
                self.barrier_param,
                self.vars,
                self.grad,
                self.res,
            )

        if use_soc:
            # SOC needs the condensed Newton residual at the current point
            if not near_feasible:
                self.optimizer.compute_residual(
                    self.barrier_param,
                    self.vars,
                    self.grad,
                    self.res,
                )
            res_orig = self.res.get_array().copy()
            update_backup = self.optimizer.create_opt_vector()
            update_backup.copy(self.update)
            px_orig = self.px.get_array().copy()

        alpha = 1.0
        soc_attempted = False

        def _check_acceptance(theta_t, phi_t, alpha_trial):
            """Check if trial (theta_t, phi_t) is acceptable (IPOPT filter).

            Returns (accepted: bool, is_switching: bool).
            """
            # Reject if theta exceeds maximum
            if theta_t > theta_max:
                return False, False
            # Case 1: switching with Armijo (IPOPT Eq. 20)
            if switching_active and dphi < 0:
                armijo_ok = phi_t <= phi_k + eta_phi * alpha_trial * dphi
                if armijo_ok:
                    return True, True
            # Case 2: sufficient decrease + filter acceptance (IPOPT Eq. 18)
            sufficient = (
                theta_t <= (1.0 - gamma_theta) * theta_k
                or phi_t <= phi_k - gamma_phi * theta_k
            )
            if sufficient and inner_filter.is_acceptable(phi_t, theta_t, abs(phi_k)):
                return True, False
            return False, False

        for j in range(max_iters):
            # FIX 3: Check alpha_min before trial evaluation
            if alpha * alpha_x < alpha_min_val and j > 0:
                if comm_rank == 0:
                    print(
                        f"  Filter LS: alpha={alpha:.2e} < "
                        f"alpha_min={alpha_min_val:.2e}, "
                        f"triggering restoration"
                    )
                self._update_gradient(self.vars.get_solution())
                return alpha, j, False

            # IPOPT: only backtrack PRIMAL step; dual uses full alpha_z
            self.optimizer.apply_step_update(
                alpha * alpha_x,
                alpha_z,
                self.vars,
                self.update,
                self.temp,
            )
            self._update_gradient(self.temp.get_solution())

            # Trial measures: theta (barrier-independent) and phi (barrier obj)
            _, primal_sq, _ = self.optimizer.compute_kkt_error(self.temp, self.grad)
            theta_trial = np.sqrt(primal_sq)
            phi_trial = self._compute_barrier_objective(self.temp)

            accepted, is_switching = _check_acceptance(
                theta_trial, phi_trial, alpha * alpha_x
            )

            # Near-feasible fallback: when theta is so small that filter
            # conditions are numerically degenerate, accept based on KKT
            # residual reduction. This avoids the cycling where the filter
            # blocks all steps and restoration just resets the filter.
            if not accepted and near_feasible and j == 0:
                res_trial = self.optimizer.compute_residual(
                    self.barrier_param,
                    self.temp,
                    self.grad,
                    self.res,
                )
                if res_trial < res_kkt_current:
                    accepted = True
                    is_switching = False  # don't augment filter
                    if comm_rank == 0:
                        print(
                            f"  Near-feasible accept: res "
                            f"{res_kkt_current:.2e}->{res_trial:.2e}, "
                            f"theta {theta_k:.2e}->{theta_trial:.2e} "
                            f"(a={alpha:.2e})"
                        )

            if accepted:
                # IPOPT: only augment filter for Case 2 (not switching)
                if not is_switching:
                    inner_filter.add(phi_k, theta_k)
                self.vars.copy(self.temp)
                if (
                    comm_rank == 0
                    and not near_feasible
                    and (
                        theta_trial < 0.9 * theta_k
                        or phi_trial < phi_k - 0.1 * abs(phi_k)
                    )
                ):
                    mode = "switch" if is_switching else "filter"
                    print(
                        f"  Filter[{mode}]: theta {theta_k:.2e}->{theta_trial:.2e}, "
                        f"phi {phi_k:.2e}->{phi_trial:.2e} (a={alpha:.2e})"
                    )
                return alpha, j + 1, True

            # FIX 2: SOC condition (IPOPT A-5.5).
            # Only attempt SOC when the full step INCREASED infeasibility
            # (theta_trial >= theta_k). If theta decreased, the step is
            # already making progress on feasibility — SOC not needed.
            if use_soc and j == 0 and not soc_attempted and theta_trial >= theta_k:
                soc_attempted = True
                try:
                    # Compute condensed residual at trial for SOC RHS
                    self.optimizer.compute_residual(
                        self.barrier_param,
                        self.temp,
                        self.grad,
                        self.res,
                    )
                    res_arr = self.res.get_array()
                    res_arr[~mult_ind] = res_orig[~mult_ind]
                    self.res.copy_host_to_device()
                    self._update_gradient(self.vars.get_solution())

                    self.solver.solve(self.res, self.px)
                    self.optimizer.compute_update(
                        self.barrier_param,
                        self.vars,
                        self.px,
                        self.update,
                    )

                    soc_ax, _, soc_az, _ = self.optimizer.compute_max_step(
                        tau, self.vars, self.update
                    )
                    self.optimizer.apply_step_update(
                        soc_ax, soc_az, self.vars, self.update, self.temp
                    )
                    self._update_gradient(self.temp.get_solution())
                    _, p_sq, _ = self.optimizer.compute_kkt_error(self.temp, self.grad)
                    theta_soc = np.sqrt(p_sq)
                    phi_soc = self._compute_barrier_objective(self.temp)

                    soc_ok, soc_switching = _check_acceptance(
                        theta_soc, phi_soc, soc_ax
                    )
                    if soc_ok:
                        if not soc_switching:
                            inner_filter.add(phi_k, theta_k)
                        self.vars.copy(self.temp)
                        if comm_rank == 0:
                            print(
                                f"  SOC accepted: theta {theta_k:.2e}"
                                f"->{theta_soc:.2e}, "
                                f"phi {phi_k:.2e}->{phi_soc:.2e}"
                            )
                        return 1.0, j + 1, True

                    if comm_rank == 0:
                        print(
                            f"  SOC rejected: theta={theta_soc:.2e},"
                            f" phi={phi_soc:.2e}"
                        )
                except Exception:
                    if comm_rank == 0:
                        print("  SOC failed, continuing backtracking")

                self.update.copy(update_backup)
                self.px.get_array()[:] = px_orig
                self.px.copy_host_to_device()

            if j < max_iters - 1:
                alpha *= backtrack
                continue

            # All backtracking failed -> signal for restoration
            if comm_rank == 0:
                print(
                    f"  Filter LS exhausted: theta={theta_k:.2e}, "
                    f"phi={phi_k:.2e}, last alpha={alpha:.2e}"
                )
            self._update_gradient(self.vars.get_solution())
            return alpha, j + 1, False

        return alpha, max_iters, False

    def _restoration_phase(
        self,
        convex,
        inner_filter,
        options,
        comm_rank,
        x,
        diag_base,
        probe_convex,
        zero_hessian_indices,
        zero_hessian_eps,
    ):
        """IPOPT-style feasibility restoration (Section 3.3).

        Two modes depending on current theta:
        1. theta large: increase barrier and take Newton steps to reduce theta
        2. theta tiny (< tol): the filter is blocking optimality progress.
           Try to find a point acceptable to the filter by taking a Newton
           step and accepting if the overall KKT residual decreases (IPOPT
           tiny-step / acceptable-point mechanism).

        Returns True if restoration succeeded.
        """
        max_restore = options["filter_restoration_max_iter"]
        backtrack = options["backtracking_factor"]
        max_ls = options["max_line_search_iterations"]
        tau_min = options["tau_min"]
        use_adaptive_tau = options["adaptive_tau"]
        tol = options["convergence_tolerance"]

        # Current infeasibility (barrier-independent)
        theta_k = self._compute_filter_theta()
        theta_start = theta_k

        # --- Mode 2: theta is small, filter is blocking optimality ---
        # Use same theta_min as switching condition: 1e-4 * max(1, theta_0).
        # When theta is below this, Mode 1 (reduce theta by 10%) can't help
        # because theta is already small relative to the problem scale.
        theta_min_restore = 1e-4 * max(1.0, getattr(self, "_filter_theta_0", theta_k))
        if theta_k < max(tol, theta_min_restore):
            if comm_rank == 0:
                print(
                    f"  Restoration (small-theta): theta={theta_k:.2e}, "
                    f"bypassing feasibility restore"
                )
            # Compute current full KKT residual
            res_current = self.optimizer.compute_residual(
                self.barrier_param, self.vars, self.grad, self.res
            )
            tau = (
                self._compute_adaptive_tau(self.barrier_param, tau_min)
                if use_adaptive_tau
                else options["fraction_to_boundary"]
            )
            ax, _, az, _ = self.optimizer.compute_max_step(tau, self.vars, self.update)

            # Try full Newton step first, then backtrack
            alpha = 1.0
            for j in range(max_ls):
                self.optimizer.apply_step_update(
                    alpha * ax,
                    az,
                    self.vars,
                    self.update,
                    self.temp,
                )
                self._update_gradient(self.temp.get_solution())
                res_trial = self.optimizer.compute_residual(
                    self.barrier_param, self.temp, self.grad, self.res
                )
                # Accept if KKT residual decreases (even slightly)
                if res_trial < res_current:
                    phi_trial = self._compute_barrier_objective(self.temp)
                    theta_trial = self._compute_filter_theta(self.temp)
                    # Check filter acceptability (don't add to filter)
                    if inner_filter.is_acceptable(
                        phi_trial, theta_trial, abs(phi_trial)
                    ):
                        self.vars.copy(self.temp)
                        if comm_rank == 0:
                            print(
                                f"  Restoration accepted: res {res_current:.2e}"
                                f"->{res_trial:.2e} (a={alpha:.2e})"
                            )
                        return True
                    # Not filter-acceptable, but KKT improving: accept anyway
                    # if residual decrease is significant
                    if res_trial < 0.99 * res_current:
                        self.vars.copy(self.temp)
                        # Clear filter to unblock (new barrier subproblem)
                        inner_filter.entries.clear()
                        self._filter_theta_0 = theta_trial
                        if comm_rank == 0:
                            print(
                                f"  Restoration accepted (filter reset): "
                                f"res {res_current:.2e}->{res_trial:.2e} "
                                f"(a={alpha:.2e})"
                            )
                        return True
                alpha *= backtrack

            # No KKT-descent step found. Since theta is already small,
            # the filter is simply clogging optimality progress.
            # Reset the filter to unblock: this is safe because theta
            # is small (constraints are nearly satisfied).
            self._update_gradient(self.vars.get_solution())
            inner_filter.entries.clear()
            self._filter_theta_0 = theta_k
            if comm_rank == 0:
                print(
                    f"  Restoration (small-theta): filter reset "
                    f"(theta={theta_k:.2e}, res={res_current:.2e})"
                )
            return True

        # --- Mode 1: standard feasibility restoration (theta is large) ---
        # Add current point to filter with barrier objective
        phi_k = self._compute_barrier_objective(self.vars)
        inner_filter.add(phi_k, theta_k)

        target = 0.9 * theta_k
        if comm_rank == 0:
            print(f"  Restoration: theta={theta_k:.2e}, target={target:.2e}")

        saved_barrier = self.barrier_param

        # Try progressively higher barriers to relax bounds and allow movement.
        restore_levels = [
            max(saved_barrier, 1e-2),
            0.1,
            1.0,
        ]
        restore_levels = sorted(set(mu for mu in restore_levels if mu >= saved_barrier))

        iters_per_level = max(3, max_restore // len(restore_levels))

        for restore_mu in restore_levels:
            self.barrier_param = restore_mu
            if convex:
                convex.update_eps_z(restore_mu)

            if comm_rank == 0:
                print(
                    f"  Restoration barrier: {saved_barrier:.2e} -> "
                    f"{restore_mu:.2e}"
                )

            for r in range(iters_per_level):
                x = self.vars.get_solution()
                self._update_gradient(x)
                self.optimizer.compute_diagonal(self.vars, self.diag)
                self.diag.copy_device_to_host()
                diag_r = self.diag.get_array().copy()

                self._find_direction(
                    x,
                    diag_r,
                    probe_convex,
                    options,
                    zero_hessian_indices,
                    zero_hessian_eps,
                    comm_rank,
                )

                tau = (
                    self._compute_adaptive_tau(restore_mu, tau_min)
                    if use_adaptive_tau
                    else options["fraction_to_boundary"]
                )
                ax, _, az, _ = self.optimizer.compute_max_step(
                    tau, self.vars, self.update
                )

                # Backtracking: accept any step reducing theta.
                alpha = 1.0
                step_taken = False
                for j in range(max_ls):
                    self.optimizer.apply_step_update(
                        alpha * ax,
                        alpha * az,
                        self.vars,
                        self.update,
                        self.temp,
                    )
                    self._update_gradient(self.temp.get_solution())
                    _, p_sq, _ = self.optimizer.compute_kkt_error(self.temp, self.grad)
                    theta_trial = np.sqrt(p_sq)

                    if theta_trial < theta_k:
                        self.vars.copy(self.temp)
                        theta_k = theta_trial
                        step_taken = True
                        break
                    alpha *= backtrack

                if comm_rank == 0:
                    print(
                        f"  Restoration iter: theta={theta_k:.2e} "
                        f"(a={alpha:.2e}, mu_R={restore_mu:.2e})"
                    )

                if not step_taken:
                    break

                if theta_k < target:
                    break

            if theta_k < target:
                if comm_rank == 0:
                    print(f"  Restoration complete: theta={theta_k:.2e}")
                break

        # Restore original barrier
        self.barrier_param = saved_barrier
        if convex:
            convex.update_eps_z(saved_barrier)
        self._update_gradient(self.vars.get_solution())
        self.optimizer.compute_diagonal(self.vars, self.diag)
        self.diag.copy_device_to_host()

        success = theta_k < theta_start * 0.99
        if comm_rank == 0 and not success:
            print(
                f"  Restoration failed: theta={theta_k:.2e} "
                f"(started {theta_start:.2e})"
            )
        return success

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
                beta_min=self.barrier_param
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

        # Convexification setup
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
                    f"  IPOPT Algorithm IC " f"({solver_name}): {n_primal} primal vars"
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

        quality_func = options["barrier_strategy"] == "quality_function"
        qf_free_mode = True
        qf_kappa = options["quality_function_kappa_free"]
        qf_l_max = options["quality_function_l_max"]
        qf_kkt_history = deque(maxlen=qf_l_max + 1)
        qf_mu_floor = max(tol * 0.01, 1e-14)
        qf_monotone_mu = None
        qf_M_k_at_entry = None
        qf_sd = qf_sp = qf_sc = 1.0
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

        if quality_func:
            d0, p0, c0 = self.optimizer.compute_kkt_error(self.vars, self.grad)
            qf_kkt_history.append(d0 * qf_sd + p0 * qf_sp + c0 * qf_sc)

        soc_mult_ind = None
        if options["second_order_correction"]:
            if convex:
                soc_mult_ind = convex.mult_ind
            else:
                soc_mult_ind = np.array(
                    problem_ref.get_multiplier_indicator(), dtype=bool
                )

        # Filter line search setup (IPOPT-style + Algorithm B outer)
        filter_ls = options["filter_line_search"]
        outer_filter = Filter() if filter_ls else None
        inner_filter = Filter(kappa_1=1e-5, kappa_2=1.0) if filter_ls else None
        filter_monotone_mode = False  # True when filter rejected step
        filter_monotone_mu = None

        # Multiplier indicator for barrier objective computation
        if filter_ls:
            if convex:
                self._filter_mult_ind = convex.mult_ind
            else:
                self._filter_mult_ind = np.array(
                    problem_ref.get_multiplier_indicator(), dtype=bool
                )
            # Track theta_0 per barrier subproblem for switching condition
            self._filter_theta_0 = None  # set after first gradient eval

        res_norm_mu = self.barrier_param  # mu used for res_norm
        rhs_norm = 0.0  # baseline at solve mu (set after direction finding)

        for i in range(max_iters):
            # Compute the complete KKT residual
            res_norm = self.optimizer.compute_residual(
                self.barrier_param, self.vars, self.grad, self.res
            )
            res_norm_mu = self.barrier_param  # track which mu was used

            if convex:
                convex.update_eps_z(self.barrier_param)
                convex.decompose_residual(self.res, self.vars)

            # Initialize theta_0 for this barrier subproblem (switching cond.)
            if filter_ls and self._filter_theta_0 is None:
                self._filter_theta_0 = self._compute_filter_theta()

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
            if filter_ls:
                iter_data["filter_size"] = len(outer_filter)

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

            # Convergence diagnostics
            if comm_rank == 0 and i > 0:
                ratio = res_norm / prev_res_norm if prev_res_norm > 0 else 0
                # Same-mu ratio: res_norm at current mu vs rhs_norm at the mu
                # used for the previous Newton direction (both at same mu)
                smu_ratio = res_norm / max(rhs_norm, 1e-30) if rhs_norm > 0 else 0
                d_sq, p_sq, c_sq = self.optimizer.compute_kkt_error(
                    self.vars, self.grad
                )
                print(
                    f"  ratio={ratio:.4f}, same_mu={smu_ratio:.4f}, "
                    f"dual={np.sqrt(d_sq):.2e}, "
                    f"primal={np.sqrt(p_sq):.2e}, "
                    f"comp={np.sqrt(c_sq):.2e}"
                )

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

            # IPOPT Algorithm A, Step A-3: barrier subproblem convergence.
            # When the current iterate solves the barrier subproblem to
            # sufficient accuracy, reduce mu and reset the inner filter.
            #
            # IPOPT uses E_mu = max(dual_err, primal_err, comp_err).
            # All three components must be small before reducing mu.
            # Missing the dual term causes premature mu reduction, making
            # the KKT system ill-conditioned and destroying quadratic convergence.
            if filter_ls and i > 0 and self.barrier_param > tol:
                kappa_eps_ipopt = 10.0
                kappa_mu_ipopt = 0.2
                theta_mu_ipopt = 1.5
                comp_val, _ = self.optimizer.compute_complementarity(self.vars)
                d_sq_a3, p_sq_a3, _ = self.optimizer.compute_kkt_error(
                    self.vars, self.grad
                )
                e_mu = max(np.sqrt(d_sq_a3), np.sqrt(p_sq_a3), comp_val)
                if e_mu <= kappa_eps_ipopt * self.barrier_param:
                    old_mu = self.barrier_param
                    new_mu = max(
                        tol / 10.0,
                        min(kappa_mu_ipopt * old_mu, old_mu**theta_mu_ipopt),
                    )
                    if new_mu < old_mu:
                        self.barrier_param = new_mu
                        if convex:
                            convex.update_eps_z(self.barrier_param)
                        inner_filter.entries.clear()
                        # Reset theta_0 for the new barrier subproblem
                        self._filter_theta_0 = self._compute_filter_theta()
                        if comm_rank == 0:
                            print(
                                f"  IPOPT A-3: mu {old_mu:.2e} -> "
                                f"{new_mu:.2e} "
                                f"(E_mu={e_mu:.2e}, filter reset)"
                            )

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

            # IPOPT filter LS path: barrier is handled by Step A-3 above.
            # Just compute the search direction with the current mu.
            if filter_ls:
                self._find_direction(
                    x,
                    diag_base,
                    probe_convex,
                    options,
                    zero_hessian_indices,
                    zero_hessian_eps,
                    comm_rank,
                )
            elif filter_monotone_mode:
                if res_norm <= options["barrier_progress_tol"] * filter_monotone_mu:
                    new_mu = max(
                        filter_monotone_mu * options["monotone_barrier_fraction"],
                        tol,
                    )
                    if comm_rank == 0:
                        print(
                            f"  Filter monotone: mu "
                            f"{filter_monotone_mu:.2e}->{new_mu:.2e}"
                        )
                    filter_monotone_mu = new_mu
                self.barrier_param = filter_monotone_mu
                if convex:
                    convex.update_eps_z(self.barrier_param)

                self._find_direction(
                    x,
                    diag_base,
                    probe_convex,
                    options,
                    zero_hessian_indices,
                    zero_hessian_eps,
                    comm_rank,
                )
            elif quality_func:
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
                        tau_min,
                        use_adaptive_tau,
                        options,
                        comm_rank,
                        probe_convex=probe_convex,
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

                if not qf_free_mode:
                    self._solve_with_mu(self.barrier_param)

                # Riccati solve comparison (QF path)
                if (
                    probe_convex
                    and hasattr(probe_convex, "_riccati_chain_factors")
                    and probe_convex._bisc_tier == 3
                    and probe_convex._riccati_chain_factors
                    and options.get("riccati_solve_compare", False)
                ):
                    # Compute RHS at the actual barrier param (self.px was
                    # set by QF to solve at this mu, not at mu_nat)
                    self.optimizer.compute_residual(
                        self.barrier_param, self.vars, self.grad, self.res
                    )
                    self.res.copy_device_to_host()
                    rhs_arr = self.res.get_array().copy()
                    # Re-solve with PARDISO for a clean comparison
                    px_pardiso = self.solver.solve_array(rhs_arr).copy()
                    p_riccati = probe_convex.riccati_solve(
                        rhs_arr, diag_base, self.solver
                    )

                    for ci, chain_ci in enumerate(probe_convex.bfs_chains):
                        chain_vars = np.concatenate(chain_ci)
                        p_par = px_pardiso[chain_vars]
                        p_ric = p_riccati[chain_vars]
                        norm_par = np.linalg.norm(p_par)
                        rel_err = np.linalg.norm(p_par - p_ric) / max(norm_par, 1e-30)
                        if comm_rank == 0:
                            print(
                                f"  Riccati vs PARDISO chain {ci}: "
                                f"||dp||={norm_par:.2e}, "
                                f"rel_err={rel_err:.2e}"
                            )
            else:
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

            # Riccati solve comparison (prototype diagnostic)
            if (
                probe_convex
                and hasattr(probe_convex, "_riccati_chain_factors")
                and probe_convex._bisc_tier == 3
                and probe_convex._riccati_chain_factors
                and options.get("riccati_solve_compare", False)
            ):
                self.res.copy_device_to_host()
                self.px.copy_device_to_host()
                rhs_arr = self.res.get_array().copy()
                px_pardiso = self.px.get_array().copy()
                p_riccati = probe_convex.riccati_solve(rhs_arr, diag_base, self.solver)
                for ci, chain_ci in enumerate(probe_convex.bfs_chains):
                    chain_vars = np.concatenate(chain_ci)
                    p_par = px_pardiso[chain_vars]
                    p_ric = p_riccati[chain_vars]
                    norm_par = np.linalg.norm(p_par)
                    rel_err = np.linalg.norm(p_par - p_ric) / max(norm_par, 1e-30)
                    if comm_rank == 0:
                        print(
                            f"  Riccati vs PARDISO chain {ci}: "
                            f"||dp||={norm_par:.2e}, "
                            f"rel_err={rel_err:.2e}"
                        )

            # Check the update (debug only)
            if options["check_update_step"]:
                hess = (
                    self.mpi_problem if self.distribute else self.problem
                ).create_matrix()
                self.optimizer.check_update(
                    self.barrier_param,
                    self.grad,
                    self.vars,
                    self.update,
                    hess,
                )

            # Solve accuracy diagnostic: check ||K*px - rhs||
            if options["check_update_step"] and comm_rank == 0:
                if hasattr(self.solver, "hess"):
                    from scipy.sparse import csr_matrix as _sp_csr

                    self.res.copy_device_to_host()
                    self.px.copy_device_to_host()
                    _rhs = self.res.get_array().copy()
                    _px = self.px.get_array().copy()
                    _data = self.solver.hess.get_data()
                    _nr, _nc, _nnz, _rp, _cl = self.solver.hess.get_nonzero_structure()
                    _K = _sp_csr((_data, _cl, _rp), shape=(_nr, _nc))
                    _Kpx = _K @ _px
                    _serr = np.linalg.norm(_Kpx - _rhs)
                    _rnrm = np.linalg.norm(_rhs)
                    print(
                        f"  Solve accuracy: ||K*px-rhs||={_serr:.2e}, "
                        f"||rhs||={_rnrm:.2e}, "
                        f"rel={_serr / max(_rnrm, 1e-30):.2e}"
                    )
                    # RHS decomposed by variable type
                    _mi = getattr(self, "_filter_mult_ind", None)
                    if _mi is None and probe_convex is not None:
                        _mi = getattr(probe_convex, "mult_ind", None)
                    if _mi is not None:
                        _rhs_primal = np.linalg.norm(_rhs[~_mi])
                        _rhs_constr = np.linalg.norm(_rhs[_mi])
                        _dx_norm = np.linalg.norm(_px[~_mi])
                        _dlam_norm = np.linalg.norm(_px[_mi])
                        print(
                            f"  RHS: primal={_rhs_primal:.3e}, "
                            f"constr={_rhs_constr:.3e}"
                        )
                        print(
                            f"  Step: ||dx||={_dx_norm:.3e}, "
                            f"||dlam||={_dlam_norm:.3e}"
                        )

            # Baseline residual at the mu used for the Newton direction.
            # May differ from res_norm (top of loop) if Step A-3 changed mu.
            rhs_norm = self.optimizer.compute_residual(
                self.barrier_param, self.vars, self.grad, self.res
            )

            # Newton quality diagnostic: check residual at full step
            if comm_rank == 0 and i > 0:
                mu_changed = abs(self.barrier_param - res_norm_mu) > 1e-30
                # Apply full step to temp
                self.optimizer.apply_step_update(
                    1.0, 1.0, self.vars, self.update, self.temp
                )
                # Update gradient at trial point
                self._update_gradient(self.temp.get_solution())
                # Compute condensed residual at trial point (same mu as direction)
                trial_res = self.optimizer.compute_residual(
                    self.barrier_param,
                    self.temp,
                    self.grad,
                    self.res,
                )
                # Also check full KKT error components at trial
                d_sq_t, p_sq_t, c_sq_t = self.optimizer.compute_kkt_error(
                    self.temp, self.grad
                )
                # Restore gradient at current point
                self._update_gradient(self.vars.get_solution())
                nq_ratio = trial_res / max(rhs_norm, 1e-30)
                print(
                    f"  Newton quality: ||F(x+dx)||={trial_res:.2e}, "
                    f"||F_mu(x)||={rhs_norm:.2e}, "
                    f"ratio={nq_ratio:.4f}, "
                    f"quadratic={trial_res/max(rhs_norm**2,1e-30):.2e}"
                    f"{', mu_changed' if mu_changed else ''}"
                )
                print(
                    f"  Trial KKT: dual={np.sqrt(d_sq_t):.2e}, "
                    f"primal={np.sqrt(p_sq_t):.2e}, "
                    f"comp={np.sqrt(c_sq_t):.2e}"
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
            if comm_rank == 0:
                problem_ref = self.mpi_problem if self.distribute else self.problem
                mult_ind = (
                    convex.mult_ind
                    if convex
                    else np.array(problem_ref.get_multiplier_indicator(), dtype=bool)
                )
                dx_full = np.array(self.update.get_solution())
                dlam_vec = dx_full[mult_ind]
                dx_vec = dx_full[~mult_ind]
                print(
                    f"  Newton: ||dx||={np.linalg.norm(dx_vec):.2e}, "
                    f"||dlam||={np.linalg.norm(dlam_vec):.2e}, "
                    f"a_x={alpha_x:.4f}, a_z={alpha_z:.4f}"
                )
                # Detailed dlam diagnostic (top contributors)
                dlam_abs = np.abs(dlam_vec)
                top_k = min(5, len(dlam_abs))
                top_idx = np.argsort(dlam_abs)[-top_k:][::-1]
                # Get multiplier indices in the full KKT vector
                mult_positions = np.where(mult_ind)[0]
                x_sol = np.array(self.vars.get_solution())
                print(f"  dlam top-{top_k}: ", end="")
                for rank, li in enumerate(top_idx):
                    gi = mult_positions[li]  # global index
                    print(
                        f"[{gi}]={dlam_vec[li]:+.4e}(lam={x_sol[gi]:+.4e})",
                        end="  " if rank < top_k - 1 else "\n",
                    )
                # Bound multiplier complementarity: top z*gap deviations
                try:
                    vars_pt = self.vars
                    ipo = self.optimizer
                    zl_arr = np.array(vars_pt.get_zl())
                    zu_arr = np.array(vars_pt.get_zu())
                    x_arr = np.array(vars_pt.get_solution())
                    n_var = ipo.get_num_design_variables()
                    lbx = np.array(ipo.get_lbx())
                    ubx = np.array(ipo.get_ubx())
                    gap_l = x_arr[:n_var] - lbx
                    gap_u = ubx - x_arr[:n_var]
                    comp_l = zl_arr * gap_l
                    comp_u = zu_arr * gap_u
                    mu = self.barrier_param
                    dev_l = np.abs(comp_l - mu)
                    dev_u = np.abs(comp_u - mu)
                    all_dev = np.concatenate([dev_l, dev_u])
                    all_comp = np.concatenate([comp_l, comp_u])
                    top5 = np.argsort(all_dev)[-min(5, len(all_dev)) :][::-1]
                    print(f"  bound comp top-5 |z*gap-mu|: ", end="")
                    for rank, ci in enumerate(top5):
                        side = "L" if ci < n_var else "U"
                        vi = ci % n_var
                        print(
                            f"x[{vi}]{side}={all_comp[ci]:.2e}(dev={all_dev[ci]:.2e})",
                            end="  " if rank < len(top5) - 1 else "\n",
                        )
                except Exception as e:
                    print(f"  (bound comp diagnostic failed: {e})")

            if options["equal_primal_dual_step"]:
                alpha_x = alpha_z = min(alpha_x, alpha_z)

            # Line search: use inner filter LS when enabled, else standard
            if filter_ls and convex:
                # Compute barrier objective at current point (gradient is current)
                phi_current = self._compute_barrier_objective(self.vars)
                alpha, line_iters, step_accepted = self._filter_line_search(
                    alpha_x,
                    alpha_z,
                    convex,
                    inner_filter,
                    options,
                    comm_rank,
                    tau=tau,
                    mult_ind=soc_mult_ind,
                    phi_current=phi_current,
                )

                # If filter LS failed -> feasibility restoration
                if not step_accepted:
                    restored = self._restoration_phase(
                        convex,
                        inner_filter,
                        options,
                        comm_rank,
                        x,
                        diag_base,
                        probe_convex,
                        zero_hessian_indices,
                        zero_hessian_eps,
                    )
                    if restored:
                        # Restoration moved the point; re-enter main loop
                        step_accepted = True
                        line_iters = 0
                    else:
                        # Both filter LS and restoration failed: reject step
                        convex.reject_step()
                        if comm_rank == 0:
                            print(
                                f"  Filter+Restoration REJECTED "
                                f"({convex.consecutive_rejections}x)"
                            )

                # Update gradient at the new point for the next iteration.
                if step_accepted:
                    # FIX 5: IPOPT Eq. 16 — reset bound multipliers after
                    # step acceptance to prevent divergence. Clamps each z_i
                    # to [mu/(kappa*gap), kappa*mu/gap] with kappa=1e10.
                    self.optimizer.reset_bound_multipliers(
                        self.barrier_param,
                        1e10,
                        self.vars,
                    )
                    self._update_gradient(self.vars.get_solution())
            else:
                alpha, line_iters, step_accepted = self._line_search(
                    alpha_x,
                    alpha_z,
                    convex,
                    options,
                    comm_rank,
                    tau=tau,
                    mult_ind=soc_mult_ind,
                )

            if convex and convex.step_rejected:
                # Step rejected: stay at current point, undo barrier reduction
                alpha_x_prev = 0.0
                alpha_z_prev = 0.0
                x_index_prev = -1
                z_index_prev = -1
                self.barrier_param = barrier_before

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
                # Filter LS doesn't backtrack dual; merit LS does
                alpha_z_prev = alpha_z if filter_ls else alpha * alpha_z
                x_index_prev = x_index
                z_index_prev = z_index

                if convex:
                    convex.consecutive_rejections = 0

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
