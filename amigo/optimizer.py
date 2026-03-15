import os
import sys
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu, eigsh

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
        return

    def solve(self, bx, px):
        self.solver.solve(bx, px)
        return


class DirectScipySolver:
    def __init__(self, problem):
        self.problem = problem
        loc = MemoryLocation.HOST_AND_DEVICE
        self.hess = self.problem.create_matrix(loc)
        self.nrows, self.ncols, self.nnz, self.rowp, self.cols = (
            self.hess.get_nonzero_structure()
        )

        self.lu = None
        return

    def factor(self, alpha, x, diag):
        """
        Compute and factor the Hessian matrix
        """

        # Compute the Hessian and add the diagonal values
        self.problem.hessian(alpha, x, self.hess)
        self.problem.add_diagonal(diag, self.hess)
        self.hess.copy_data_device_to_host()

        # Build the CSR matrix and convert to CSC
        shape = (self.nrows, self.ncols)
        data = self.hess.get_data()
        H = csr_matrix((data, self.cols, self.rowp), shape=shape).tocsc()

        # Compute the LU factorization
        self.lu = splu(H, permc_spec="COLAMD", diag_pivot_thresh=1.0)

    def solve(self, bx, px):
        """
        Solve the KKT system
        """

        bx.copy_device_to_host()
        px.get_array()[:] = self.lu.solve(bx.get_array())
        px.copy_host_to_device()

        return

    def compute_eigenvalues(self, alpha, x, diag=None, k=20, sigma=0.0, which="LM"):
        """
        if self.lu is None:
            raise RuntimeError("Must call factor() before get_inertia()")
        u_diag = self.lu.U.diagonal()
        return int(np.sum(u_diag > 0)), int(np.sum(u_diag < 0))


class MumpsSolver(_HessianDiagMixin):
    """Sparse LDL^T solver via MUMPS with inertia detection.

    Uses the MUMPS C interface (dmumps_c) via ctypes to perform symmetric
    indefinite factorization (sym=2) with Bunch-Kaufman pivoting. Provides
    exact inertia counts from MUMPS info arrays after factorization.

    Requires libdmumps.dll (and its dependencies: libblas.dll, liblapack.dll,
    libmumps_common.dll, libpord.dll, libmpiseq.dll, etc.) to be on the DLL
    search path. These are provided by the mumps-build third-party repo.
    """

    def __init__(self, problem):
        import ctypes
        import sys

        self._ct = ctypes
        try:
            if sys.platform == "win32":
                self._libmumps = ctypes.CDLL("libdmumps.dll")
            else:
                lib_dir = os.environ.get("MUMPS_LIB_DIR", "")
                lib_path = os.path.join(lib_dir, "libdmumps.so") if lib_dir else "libdmumps.so"
                self._libmumps = ctypes.CDLL(lib_path)
        except OSError:
            raise ImportError(
                "MUMPS library not found. On Windows, ensure libdmumps.dll "
                "is on the DLL search path. On Linux, install libmumps-dev "
                "or set MUMPS_LIB_DIR."
            )
        self._dmumps_c = self._libmumps.dmumps_c
        self._dmumps_c.restype = None

        self.problem = problem
        loc = MemoryLocation.HOST_AND_DEVICE
        self.hess = self.problem.create_matrix(loc)
        self.nrows, self.ncols, self.nnz, self.rowp, self.cols = (
            self.hess.get_nonzero_structure()
        )
        self._diag_indices = _find_diag_indices(self.rowp, self.cols, self.nrows)

        # Build COO triplet arrays from CSR (MUMPS uses 1-based COO)
        # Only store lower triangle for sym=2 (symmetric indefinite)
        irn_list = []
        jcn_list = []
        data_map = []
        for i in range(self.nrows):
            for k in range(self.rowp[i], self.rowp[i + 1]):
                j = self.cols[k]
                if j <= i:  # lower triangle
                    irn_list.append(i + 1)
                    jcn_list.append(j + 1)
                    data_map.append(k)
        self._irn = np.array(irn_list, dtype=np.int32)
        self._jcn = np.array(jcn_list, dtype=np.int32)
        self._data_map = np.array(data_map, dtype=np.intc)
        self._nnz_lower = len(irn_list)
        self._a = np.empty(self._nnz_lower, dtype=np.float64)

        # Build the MUMPS struct via ctypes
        self._build_struct()

        # Initialize MUMPS (job=-1)
        self._mumps.job = -1
        self._mumps.par = 1
        self._mumps.sym = 2  # symmetric indefinite (LDL^T)
        self._mumps.comm_fortran = -987654  # MPISEQ sequential
        self._call_mumps()

        # Set ICNTL parameters
        self._mumps.icntl[0] = -1  # suppress error output
        self._mumps.icntl[1] = -1  # suppress diagnostic output
        self._mumps.icntl[2] = -1  # suppress global info output
        self._mumps.icntl[3] = 0  # no output
        self._mumps.icntl[6] = 5  # ordering: METIS if available
        self._mumps.icntl[7] = 0  # no scaling (preserve KKT structure)
        self._mumps.icntl[12] = 1  # ScaLAPACK (no effect in sequential)
        self._mumps.icntl[13] = 1000  # percent increase in workspace (IPOPT default)
        self._mumps.icntl[23] = 0  # no null pivot detection
        self._mumps.cntl[0] = 1e-6  # pivot threshold (IPOPT default: minimal pivoting)

        # Set matrix structure
        self._mumps.n = self.nrows
        self._mumps.nz = int(self._nnz_lower) if self._nnz_lower < 2**31 else 0
        self._mumps.nnz = self._nnz_lower
        self._mumps.irn = self._irn.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self._mumps.jcn = self._jcn.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self._mumps.a = self._a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Symbolic analysis (job=1) - done once
        self._mumps.job = 1
        self._call_mumps()
        if self._mumps.infog[0] < 0:
            raise RuntimeError(
                f"MUMPS analysis failed: infog(1)={self._mumps.infog[0]}"
            )

        self._rhs = np.empty(self.nrows, dtype=np.float64)

    def _build_struct(self):
        """Build the ctypes Structure matching DMUMPS_STRUC_C."""
        ct = self._ct

        class DMUMPS_STRUC_C(ct.Structure):
            _fields_ = [
                ("sym", ct.c_int),
                ("par", ct.c_int),
                ("job", ct.c_int),
                ("comm_fortran", ct.c_int),
                ("icntl", ct.c_int * 60),
                ("keep", ct.c_int * 500),
                ("cntl", ct.c_double * 15),
                ("dkeep", ct.c_double * 230),
                ("keep8", ct.c_int64 * 150),
                ("n", ct.c_int),
                ("nblk", ct.c_int),
                ("nz_alloc", ct.c_int),
                ("nz", ct.c_int),
                ("nnz", ct.c_int64),
                ("irn", ct.POINTER(ct.c_int)),
                ("jcn", ct.POINTER(ct.c_int)),
                ("a", ct.POINTER(ct.c_double)),
                ("nz_loc", ct.c_int),
                ("nnz_loc", ct.c_int64),
                ("irn_loc", ct.POINTER(ct.c_int)),
                ("jcn_loc", ct.POINTER(ct.c_int)),
                ("a_loc", ct.POINTER(ct.c_double)),
                ("nelt", ct.c_int),
                ("eltptr", ct.POINTER(ct.c_int)),
                ("eltvar", ct.POINTER(ct.c_int)),
                ("a_elt", ct.POINTER(ct.c_double)),
                ("blkptr", ct.POINTER(ct.c_int)),
                ("blkvar", ct.POINTER(ct.c_int)),
                ("perm_in", ct.POINTER(ct.c_int)),
                ("sym_perm", ct.POINTER(ct.c_int)),
                ("uns_perm", ct.POINTER(ct.c_int)),
                ("colsca", ct.POINTER(ct.c_double)),
                ("rowsca", ct.POINTER(ct.c_double)),
                ("colsca_from_mumps", ct.c_int),
                ("rowsca_from_mumps", ct.c_int),
                ("colsca_loc", ct.POINTER(ct.c_double)),
                ("rowsca_loc", ct.POINTER(ct.c_double)),
                ("rowind", ct.POINTER(ct.c_int)),
                ("colind", ct.POINTER(ct.c_int)),
                ("pivots", ct.POINTER(ct.c_double)),
                ("rhs", ct.POINTER(ct.c_double)),
                ("redrhs", ct.POINTER(ct.c_double)),
                ("rhs_sparse", ct.POINTER(ct.c_double)),
                ("sol_loc", ct.POINTER(ct.c_double)),
                ("rhs_loc", ct.POINTER(ct.c_double)),
                ("rhsintr", ct.POINTER(ct.c_double)),
                ("irhs_sparse", ct.POINTER(ct.c_int)),
                ("irhs_ptr", ct.POINTER(ct.c_int)),
                ("isol_loc", ct.POINTER(ct.c_int)),
                ("irhs_loc", ct.POINTER(ct.c_int)),
                ("glob2loc_rhs", ct.POINTER(ct.c_int)),
                ("glob2loc_sol", ct.POINTER(ct.c_int)),
                ("nrhs", ct.c_int),
                ("lrhs", ct.c_int),
                ("lredrhs", ct.c_int),
                ("nz_rhs", ct.c_int),
                ("lsol_loc", ct.c_int),
                ("nloc_rhs", ct.c_int),
                ("lrhs_loc", ct.c_int),
                ("nsol_loc", ct.c_int),
                ("schur_mloc", ct.c_int),
                ("schur_nloc", ct.c_int),
                ("schur_lld", ct.c_int),
                ("mblock", ct.c_int),
                ("nblock", ct.c_int),
                ("nprow", ct.c_int),
                ("npcol", ct.c_int),
                ("ld_rhsintr", ct.c_int),
                ("info", ct.c_int * 80),
                ("infog", ct.c_int * 80),
                ("rinfo", ct.c_double * 40),
                ("rinfog", ct.c_double * 40),
                ("deficiency", ct.c_int),
                ("pivnul_list", ct.POINTER(ct.c_int)),
                ("mapping", ct.POINTER(ct.c_int)),
                ("singular_values", ct.POINTER(ct.c_double)),
                ("size_schur", ct.c_int),
                ("listvar_schur", ct.POINTER(ct.c_int)),
                ("schur", ct.POINTER(ct.c_double)),
                ("wk_user", ct.POINTER(ct.c_double)),
                ("version_number", ct.c_char * 32),
                ("ooc_tmpdir", ct.c_char * 1024),
                ("ooc_prefix", ct.c_char * 256),
                ("write_problem", ct.c_char * 1024),
                ("lwk_user", ct.c_int),
                ("save_dir", ct.c_char * 1024),
                ("save_prefix", ct.c_char * 256),
                ("metis_options", ct.c_int * 40),
                ("instance_number", ct.c_int),
            ]

        self._DMUMPS_STRUC_C = DMUMPS_STRUC_C
        self._mumps = DMUMPS_STRUC_C()
        self._dmumps_c.argtypes = [ct.POINTER(DMUMPS_STRUC_C)]

    def _call_mumps(self):
        self._dmumps_c(self._ct.byref(self._mumps))

    def _factorize_current(self):
        self._mumps.job = 2
        self._call_mumps()
        if self._mumps.infog[0] < 0:
            raise RuntimeError(
                f"MUMPS factorize failed: infog(1)={self._mumps.infog[0]}, "
                f"infog(2)={self._mumps.infog[1]}"
            )

    def _update_values(self):
        data = self.hess.get_data()
        self._a[:] = data[self._data_map]

    def add_diagonal_and_factor(self, diag):
        self.problem.add_diagonal(diag, self.hess)
        self.hess.copy_data_device_to_host()
        self._update_values()
        self._factorize_current()

    def factor(self, alpha, x, diag):
        self.problem.hessian(alpha, x, self.hess)
        self.problem.add_diagonal(diag, self.hess)
        self.hess.copy_data_device_to_host()
        self._update_values()
        self._factorize_current()

    def get_inertia(self):
        """Return (n_positive, n_negative) from MUMPS infog(12).

        infog(12) = number of negative pivots in LDL^T factorization.
        n_positive is inferred as n - n_negative (no zero pivot detection).
        """
        n_neg = int(self._mumps.infog[11])
        n_pos = self.nrows - n_neg
        return n_pos, n_neg

    def solve(self, bx, px):
        bx.copy_device_to_host()
        self._rhs[:] = bx.get_array()
        self._mumps.rhs = self._rhs.ctypes.data_as(self._ct.POINTER(self._ct.c_double))
        self._mumps.nrhs = 1
        self._mumps.lrhs = self.nrows
        self._mumps.job = 3
        self._call_mumps()
        if self._mumps.infog[0] < 0:
            raise RuntimeError(f"MUMPS solve failed: infog(1)={self._mumps.infog[0]}")
        px.get_array()[:] = self._rhs
        px.copy_host_to_device()

    def solve_array(self, rhs):
        self._rhs[:] = rhs
        self._mumps.rhs = self._rhs.ctypes.data_as(self._ct.POINTER(self._ct.c_double))
        self._mumps.nrhs = 1
        self._mumps.lrhs = self.nrows
        self._mumps.job = 3
        self._call_mumps()
        return self._rhs.copy()

    def __del__(self):
        try:
            self._mumps.job = -2
            self._call_mumps()
        except Exception:
            pass


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

    def factor(self, alpha, x, diag, _debug_inertia=False):
        """Assemble Hessian, add diagonal, and LDL^T factorize (one shot).

        Used for inertia-correction retries where we need a fresh
        assembly (since add_diagonal_and_factor mutates self.hess).
        """
        self.problem.hessian(alpha, x, self.hess)
        if diag is not None:
            self.problem.add_diagonal(diag, self.hess)

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

        return

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

class InertiaCorrector:
    """IPOPT Algorithm IC: inertia correction for the KKT system.

    Reference: Wachter & Biegler, "On the Implementation of an Interior-Point
    Filter Line-Search Algorithm for Large-Scale Nonlinear Programming",
    Mathematical Programming 106(1), 2006, Section 3.1.

    The KKT matrix for an interior-point method has the structure:

        [W + Sigma    J^T ] [dx ]   [r_d]
        [   J          0  ] [dl ] = [r_p]

    where W is the Hessian of the Lagrangian, Sigma contains the barrier
    terms, and J is the constraint Jacobian.  For a correct Newton step,
    this matrix must have inertia (n, m, 0) -- i.e., n positive eigenvalues
    (one per primal variable) and m negative eigenvalues (one per constraint).

    When the inertia is wrong (nonconvex Hessian or rank-deficient Jacobian),
    this class adds regularization:
      - delta_w * I  on the primal block  (fixes wrong positive eigencount)
      - -delta_c * I on the constraint block (fixes zero eigenvalues)

    The regularization magnitudes follow the exponential growth schedule
    from IPOPT Algorithm IC (steps IC-1 through IC-6).

    Parameters
    ----------
    mult_ind : bool array
        True for constraint (dual) rows, False for primal rows in the
        KKT system.  Size = total KKT dimension.
    barrier_param : float
        Current barrier parameter mu.
    options : dict
        Optimizer options.  Relevant keys:
        - "convex_eps_z_coeff": coefficient C_z for eps_z = C_z * mu
        - "nonconvex_constraints": list of constraint names that are
          known to be nonconvex (receive selective eps_z regularization).
    model : Model, optional
        Required if nonconvex_constraints is specified.
    solver : Solver, optional
        Not used directly; kept for API consistency.
    distribute : bool
        If True, skip nonconvex constraint index resolution (MPI mode).
    """

    def __init__(
        self,
        mult_ind,
        barrier_param,
        options,
        model=None,
        solver=None,
        distribute=False,
    ):
        self.mult_ind = mult_ind
        self._barrier = barrier_param
        self.numerical_eps = 1e-12

        # Warm-start state for delta_w across iterations (IPOPT Sec 3.1).
        # If previous iterations needed regularization, start with a
        # fraction of the last successful delta_w instead of zero.
        self.last_inertia_delta = 0.0
        self._ic_iteration = 0  # iteration counter for early warm-start
        self._ic_always_dw = False  # True after first few corrections needed dw
        self._ic_always_dc = False  # True after first few corrections needed dc

        # Selective eps_z: extra negative regularization on constraint
        # rows corresponding to nonconvex constraints.  This acts as a
        # "virtual control" that relaxes the condensed Hessian, preventing
        # the optimizer from over-regularizing the entire system.
        self.cz = options.get("convex_eps_z_coeff", 10.0)
        self.eps_z = 0.0
        self._nonconvex_indices = None

        nc_constraints = options.get("nonconvex_constraints", [])
        if nc_constraints and model is not None and not distribute:
            self._nonconvex_indices = np.sort(model.get_indices(nc_constraints))
            self.eps_z = self.cz * barrier_param
            print(
                f"  Selective eps_z: {len(self._nonconvex_indices)} "
                f"nonconvex constraint entries"
            )

    def update_barrier(self, mu):
        """Synchronize internal barrier parameter and recompute eps_z."""
        self._barrier = mu
        if self._nonconvex_indices is not None:
            self.eps_z = self.cz * mu
        else:
            self.eps_z = 0.0

    def factorize(
        self,
        solver,
        optimizer_self,
        x,
        diag,
        diag_base,
        zero_hessian_indices,
        zero_hessian_eps,
        comm_rank,
        max_corrections=40,
        inertia_tolerance=0,
    ):
        """Assemble the KKT matrix, check inertia, and correct if needed.

        Implements IPOPT Algorithm IC (steps IC-1 through IC-6):
          IC-1: Try factorization with minimal or no modification.
          IC-2: If zero eigenvalues detected, add delta_c on constraint block.
          IC-3: Choose initial delta_w (warm-start from last iteration or dw_0).
          IC-4: Factorize with (delta_w, delta_c). If inertia correct, done.
          IC-5: Grow delta_w by kw_plus (x8) or kw_plus_bar (x100).
          IC-6: Abort if delta_w exceeds dw_max.

        Parameters
        ----------
        solver : solver object
            Must support assemble_hessian(), add_diagonal_and_factor(),
            factor(), and optionally get_inertia().
        optimizer_self : Optimizer
            Parent optimizer (unused, kept for API compatibility).
        x : ModelVector
            Current primal-dual solution vector.
        diag : ModelVector
            Diagonal vector to modify (barrier Sigma terms).
        diag_base : ndarray
            Unmodified copy of the diagonal (for resetting between retries).
        zero_hessian_indices : ndarray or None
            Indices of variables with zero Hessian contribution (linear
            variables), which need stronger regularization.
        zero_hessian_eps : float or None
            Regularization value for zero-Hessian variables.
        comm_rank : int
            MPI rank (only rank 0 prints diagnostics).
        max_corrections : int
            Maximum number of factorization retries before giving up.
        """
        # Step 0: Assemble the Hessian matrix W + Sigma (no diagonal mods yet)
        solver.assemble_hessian(1.0, x)

        primal_mask = ~self.mult_ind
        n_primal = int(np.sum(primal_mask))
        n_dual = int(np.sum(self.mult_ind))
        n_total = n_primal + n_dual
        has_inertia = hasattr(solver, "get_inertia")
        itol = inertia_tolerance

        def _inertia_ok(np_, nn_):
            return (
                abs(np_ - n_primal) <= itol
                and abs(nn_ - n_dual) <= itol
                and np_ + nn_ >= n_total - itol
            )

        mu = self._barrier

        # IPOPT Algorithm IC constants (Table 3 in Wachter & Biegler 2006)
        dw_min = 1e-20  # delta_bar_w^min: smallest meaningful delta_w
        dw_0 = 1e-4  # delta_bar_w^0:   initial delta_w when no history
        dw_max = 1e40  # delta_bar_w^max: abort threshold
        dc_bar = 1e-8  # delta_bar_c:     base delta_c (~sqrt(eps_mach))
        kw_minus = 1.0 / 3  # kappa_w^-:  shrink factor for warm-start delta_w
        kw_plus = 8.0  # kappa_w^+:  growth factor (with warm-start)
        kw_plus_bar = 100.0  # kappa_w^+:  growth factor (no warm-start)
        kc = 0.25  # kappa_c:    exponent for delta_c = dc_bar * mu^kc

        # Build the baseline diagonal modification:
        #   - Small numerical eps on primal rows (prevents exact singularity)
        #   - Selective eps_z on nonconvex constraint rows (virtual control)
        #   - Stronger eps on zero-Hessian (purely linear) variables
        diag_arr = diag.get_array()
        diag_arr[primal_mask] += self.numerical_eps

        if self._nonconvex_indices is not None and self.eps_z > 0:
            # Only apply eps_z to nonconvex constraints that are active
            # inequalities (identified by negative diagonal entries)
            ineq_dual_mask = self.mult_ind & (diag_arr < -1e-30)
            nc_ineq = np.isin(
                self._nonconvex_indices,
                np.where(ineq_dual_mask)[0],
            )
            nc_ineq_idx = self._nonconvex_indices[nc_ineq]
            if len(nc_ineq_idx) > 0:
                diag_arr[nc_ineq_idx] -= self.eps_z

        if zero_hessian_indices is not None and zero_hessian_eps is not None:
            # Variables that appear linearly in the objective have zero
            # Hessian contribution; ensure at least eps_x_zero regularization
            np.maximum(
                diag_arr[zero_hessian_indices],
                zero_hessian_eps,
                out=diag_arr[zero_hessian_indices],
            )
        diag.copy_host_to_device()

        if not has_inertia:
            # Solver cannot report inertia (e.g. CUDA): factor with baseline
            # diagonal and fall back to dw_0 regularization on failure
            try:
                solver.add_diagonal_and_factor(diag)
            except Exception:
                diag_arr[primal_mask] += dw_0
                diag.copy_host_to_device()
                solver.factor(1.0, x, diag)
        else:
            # Save baseline diagonal for resetting between correction attempts
            reg_diag = diag.get_array().copy()
            last_delta = self.last_inertia_delta
            ic_iter = self._ic_iteration
            self._ic_iteration += 1

            # IC-1: Try with minimal modification
            # If previous iterations consistently needed corrections,
            # start with a reduced version of the last successful delta_w
            # (warm-start) instead of trying unmodified first.
            ic1_dw = 0.0
            ic1_dc = 0.0
            if self._ic_always_dw and last_delta > 0:
                ic1_dw = max(dw_min, kw_minus * last_delta)
            if self._ic_always_dc:
                ic1_dc = dc_bar * mu**kc

            if ic1_dw > 0 or ic1_dc > 0:
                diag.get_array()[:] = reg_diag
                if ic1_dw > 0:
                    diag.get_array()[primal_mask] += ic1_dw
                if ic1_dc > 0:
                    diag.get_array()[self.mult_ind] -= ic1_dc
                diag.copy_host_to_device()

            # Factorize and check inertia
            inertia_ok = False
            has_zero_eigs = False
            n_pos = n_neg = 0
            try:
                if ic1_dw > 0 or ic1_dc > 0:
                    solver.factor(1.0, x, diag)
                else:
                    solver.add_diagonal_and_factor(diag)
                n_pos, n_neg = solver.get_inertia()
                inertia_ok = _inertia_ok(n_pos, n_neg)
                has_zero_eigs = n_pos + n_neg < n_total - itol
            except Exception:
                has_zero_eigs = True  # Factorization failure => singular

            if inertia_ok:
                # IC-1 success: unmodified (or minimally modified) KKT works
                self.last_inertia_delta = ic1_dw if ic1_dw > 0 else 0.0
            else:
                # IC-2: Handle zero eigenvalues
                # Zero eigenvalues indicate rank-deficient constraint Jacobian.
                # Add -delta_c on the constraint diagonal to regularize.
                delta_c = dc_bar * mu**kc if has_zero_eigs else 0.0

                # IC-3: Choose initial delta_w
                if last_delta == 0.0:
                    delta_w = dw_0  # No history: start at delta_bar_w^0
                else:
                    delta_w = max(dw_min, kw_minus * last_delta)  # Warm-start

                # IC-4 through IC-6: Retry loop
                for attempt in range(max_corrections):
                    # Reset diagonal and apply current (delta_w, delta_c)
                    diag.get_array()[:] = reg_diag
                    diag.get_array()[primal_mask] += delta_w
                    if delta_c > 0:
                        diag.get_array()[self.mult_ind] -= delta_c
                    diag.copy_host_to_device()

                    try:
                        solver.factor(1.0, x, diag)
                        n_pos, n_neg = solver.get_inertia()
                    except Exception as e:
                        n_pos = n_neg = 0
                        if comm_rank == 0:
                            print(f"  Factorize error: {e}")

                    # IC-4: Check if inertia is now correct
                    if _inertia_ok(n_pos, n_neg):
                        self.last_inertia_delta = delta_w
                        # Learn: if corrections were needed early, always
                        # warm-start in future iterations
                        if ic_iter < 3:
                            self._ic_always_dw = True
                        if delta_c > 0 and ic_iter < 3:
                            self._ic_always_dc = True
                        inertia_ok = True
                        if comm_rank == 0:
                            print(
                                f"  Inertia correction: "
                                f"delta_w={delta_w:.2e}, "
                                f"delta_c={delta_c:.2e}, "
                                f"attempts={attempt + 1}"
                            )
                        break

                    # IC-5: Grow delta_w exponentially
                    if last_delta == 0.0:
                        delta_w *= kw_plus_bar  # x100 (aggressive, no history)
                    else:
                        delta_w *= kw_plus  # x8 (moderate, with history)

                    # IC-6: Abort if delta_w exceeds maximum
                    if delta_w > dw_max:
                        if comm_rank == 0:
                            print(
                                f"  Inertia: delta_w={delta_w:.2e} > max, "
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
                    # Save delta_w for warm-start even on failure
                    self.last_inertia_delta = delta_w
                    if ic_iter < 3:
                        self._ic_always_dw = True


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
        self.barrier_param = 1.0
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
        # AMIGO_SOLVER env var: "scipy", "mumps", "pardiso" (default: auto)
        if solver is None and self.distribute:
            self.solver = DirectPetscSolver(self.comm, self.mpi_problem)
        elif solver is None:
            solver_pref = os.environ.get("AMIGO_SOLVER", "").lower()
            if solver_pref == "scipy":
                self.solver = DirectScipySolver(self.problem)
            elif solver_pref == "pardiso":
                self.solver = PardisoSolver(self.problem)
            else:
                try:
                    self.solver = MumpsSolver(self.problem)
                except (ImportError, Exception):
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

        return

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
                if isinstance(data, (int, np.integer)):
                    line += f"{iter_data[name]:15d} "
                else:
                    line += f"{iter_data[name]:15.6e} "
        print(line)
        sys.stdout.flush()

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
            # IPOPT Algorithm IC inertia correction (auto-detected from solver)
            "convex_eps_z_coeff": 1.0,  # C_z: eps_z = C_z * mu
            "nonconvex_constraints": [],  # Constraint names for selective eps_z
            "max_consecutive_rejections": 5,  # Before barrier increase
            "barrier_increase_factor": 5.0,  # Barrier *= this when stuck
            "max_inertia_corrections": 40,  # Max refactorizations for inertia (IPOPT: ~45)
            "inertia_tolerance": 0,  # Allow n_pos/n_neg to differ by this many from expected
            "quality_function_sigma_max": 4.0,
            "quality_function_golden_iters": 12,
            "quality_function_kappa_free": 0.9999,
            "quality_function_l_max": 5,
            "quality_function_norm_scaling": True,
            "quality_function_predictor_corrector": True,
            "pc_kappa_mu_decay": 0.2,
        }

        for name in options:
            if name in default:
                default[name] = options[name]
            else:
                raise ValueError(f"Unrecognized option {name}")

        return default

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

        # Branch A: Mehrotra predictor-corrector
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
                    f"a_aff={alpha_aff_x:.3f})"
                )

            # Corrector direction: px0 + sigma_pc*dpx is the exact KKT solution at
            # mu=mu_pc (no extra factorization; linearity of r(mu) in mu guarantees
            # this is identical to re-solving K p = r(mu_pc) from scratch).
            self.px.get_array()[:] = px0 + sigma_pc * dpx
            self.px.copy_host_to_device()
            self.optimizer.compute_update(mu_pc, self.vars, self.px, self.update)
            return sigma_pc, mu_pc

        # Branch B: golden-section search over q_L(sigma)
        sigma_min_floor = (mu_floor / mu_nat) if mu_nat > mu_floor else 1e-6
        sigma_min = max(1e-6, sigma_min_floor)
        sigma_max = options["quality_function_sigma_max"]
        n_gs_iters = options["quality_function_golden_iters"]

        qf_sd = self._qf_sd
        qf_sp = self._qf_sp
        qf_sc = self._qf_sc

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
            print(
                f"  QF: sigma={sigma_star:.4f}, "
                f"mu={mu_new:.4e} (comp={avg_comp:.4e})"
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

        return

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

    def _add_regularization_terms(self, diag, eps_x=1e-4, eps_z=1e-4):
        self.optimizer.set_design_vars_value(eps_x, diag)
        self.optimizer.set_multipliers_value(-eps_z, diag)

        return

    def _zero_multipliers(self, x):
        # Zero the multiplier contributions
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
        inertia_corrector,
        mult_ind,
        options,
        zero_hessian_indices,
        zero_hessian_eps,
        comm_rank,
    ):
        """Assemble, regularize, and factorize the KKT matrix.

        If an InertiaCorrector is available, delegates to its factorize()
        method which implements IPOPT Algorithm IC (inertia correction with
        exponential growth of delta_w).  Otherwise, factors directly with
        a simple fallback regularization on failure.

        After this call, the solver is ready for back-solves via
        solver.solve().
        """
        self.diag.get_array()[:] = diag_base
        self.diag.copy_host_to_device()

        if inertia_corrector is not None and hasattr(self.solver, "assemble_hessian"):
            inertia_corrector.factorize(
                self.solver,
                self,
                x,
                self.diag,
                diag_base,
                zero_hessian_indices,
                zero_hessian_eps,
                comm_rank,
                max_corrections=options["max_inertia_corrections"],
                inertia_tolerance=options.get("inertia_tolerance", 0),
            )
        else:
            try:
                self.solver.factor(1.0, x, self.diag)
            except Exception:
                # Factorization failed: add primal regularization and retry
                diag_arr = self.diag.get_array()
                if mult_ind is not None:
                    diag_arr[~mult_ind] += 1e-4
                else:
                    diag_arr += 1e-4
                self.diag.copy_host_to_device()
                self.solver.factor(1.0, x, self.diag)

    def _solve_with_mu(self, mu):
        """Compute the Newton direction by back-solving the factorized KKT system.

        Given the factorized KKT matrix K from _factorize_kkt(), this computes:
          1. RHS residual r(mu) from the KKT conditions at barrier param mu
          2. Reduced-space step px = K^{-1} r
          3. Full primal-dual update from px

        Requires _factorize_kkt() to have been called first.
        """
        self.optimizer.compute_residual(mu, self.vars, self.grad, self.res)
        self.solver.solve(self.res, self.px)
        self.optimizer.compute_update(mu, self.vars, self.px, self.update)

    def _find_direction(
        self,
        x,
        diag_base,
        inertia_corrector,
        mult_ind,
        options,
        zero_hessian_indices,
        zero_hessian_eps,
        comm_rank,
    ):
        """Compute the Newton search direction: factorize KKT + back-solve.

        This is the main "direction finding" routine called once per
        iteration.  Combines _factorize_kkt (assemble + inertia correction)
        with _solve_with_mu (RHS + back-solve + update extraction).
        """
        self._factorize_kkt(
            x,
            diag_base,
            inertia_corrector,
            mult_ind,
            options,
            zero_hessian_indices,
            zero_hessian_eps,
            comm_rank,
        )
        self._solve_with_mu(self.barrier_param)

    def _print_newton_diagnostics(self, rhs_norm, res_norm_mu, mult_ind):
        """Print detailed Newton step diagnostics (check_update_step only).

        Evaluates the residual at the full Newton step to assess:
        - Newton quality ratio: ||F(x+dx)|| / ||F(x)|| (should be << 1)
        - Quadratic convergence indicator: ||F(x+dx)|| / ||F(x)||^2
        - Per-component KKT errors at the trial point
        """
        mu_changed = abs(self.barrier_param - res_norm_mu) > 1e-30

        # Evaluate residual at full Newton step (temporary)
        self.optimizer.apply_step_update(1.0, 1.0, self.vars, self.update, self.temp)
        self._update_gradient(self.temp.get_solution())
        trial_res = self.optimizer.compute_residual(
            self.barrier_param,
            self.temp,
            self.grad,
            self.res,
        )
        d_sq_t, p_sq_t, c_sq_t = self.optimizer.compute_kkt_error(self.temp, self.grad)
        # Restore gradient at current point
        self._update_gradient(self.vars.get_solution())

        nq_ratio = trial_res / max(rhs_norm, 1e-30)
        print(
            f"  Newton quality: ||F(x+dx)||={trial_res:.2e}, "
            f"||F_mu(x)||={rhs_norm:.2e}, "
            f"ratio={nq_ratio:.4f}, "
            f"quadratic={trial_res / max(rhs_norm**2, 1e-30):.2e}"
            f"{', mu_changed' if mu_changed else ''}"
        )
        print(
            f"  Trial KKT: dual={np.sqrt(d_sq_t):.2e}, "
            f"primal={np.sqrt(p_sq_t):.2e}, "
            f"comp={np.sqrt(c_sq_t):.2e}"
        )

        # Newton direction norms and step sizes
        dx_full = np.array(self.update.get_solution())
        dlam_vec = dx_full[mult_ind]
        dx_vec = dx_full[~mult_ind]
        print(
            f"  Newton: ||dx||={np.linalg.norm(dx_vec):.2e}, "
            f"||dlam||={np.linalg.norm(dlam_vec):.2e}"
        )

        # Top-5 multiplier step contributors
        dlam_abs = np.abs(dlam_vec)
        top_k = min(5, len(dlam_abs))
        top_idx = np.argsort(dlam_abs)[-top_k:][::-1]
        mult_positions = np.where(mult_ind)[0]
        x_sol = np.array(self.vars.get_solution())
        print(f"  dlam top-{top_k}: ", end="")
        for rank, li in enumerate(top_idx):
            gi = mult_positions[li]
            print(
                f"[{gi}]={dlam_vec[li]:+.4e}(lam={x_sol[gi]:+.4e})",
                end="  " if rank < top_k - 1 else "\n",
            )

        # Top-5 bound complementarity deviations |z*gap - mu|
        try:
            zl_arr = np.array(self.vars.get_zl())
            zu_arr = np.array(self.vars.get_zu())
            x_arr = np.array(self.vars.get_solution())
            n_var = self.optimizer.get_num_design_variables()
            lbx = np.array(self.optimizer.get_lbx())
            ubx = np.array(self.optimizer.get_ubx())
            dvi = np.where(~mult_ind)[0]
            x_design = x_arr[dvi]
            gap_l = x_design - lbx
            gap_u = ubx - x_design
            finite_l = np.isfinite(lbx)
            finite_u = np.isfinite(ubx)
            comp_l = np.where(finite_l, zl_arr * gap_l, 0.0)
            comp_u = np.where(finite_u, zu_arr * gap_u, 0.0)
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

    def _line_search(
        self,
        alpha_x,
        alpha_z,
        options,
        comm_rank,
        tau=0.995,
        mult_ind=None,
        reject_callback=None,
    ):
        """Backtracking line search using KKT residual norm as merit function.

        Backtracks alpha from 1.0 until the Armijo sufficient decrease
        condition is satisfied: ||F(x + alpha*dx)|| <= ||F(x)|| + c * alpha * dphi.

        Optionally tries a second-order correction (SOC) after the first
        full-step rejection.  SOC replaces the constraint part of the RHS
        with the trial point's nonlinear residual and re-solves (no re-factor).

        Parameters
        ----------
        alpha_x, alpha_z : float
            Maximum primal/dual step from fraction-to-boundary rule.
        options : dict
            Solver options.
        comm_rank : int
            MPI rank (only rank 0 prints).
        tau : float
            Fraction-to-boundary parameter.
        mult_ind : bool array or None
            Multiplier indicator for SOC.
        reject_callback : callable or None
            Called when the line search exhausts all backtracking steps.
            If None, falls back to relaxed acceptance (accept small increase).

        Returns
        -------
        alpha : float
            Accepted step size.
        line_iters : int
            Number of backtracking iterations.
        step_accepted : bool
            True if a step was accepted.
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
            if reject_callback is not None:
                if res_new <= ls_baseline:
                    self.vars.copy(self.temp)
                    return alpha, j + 1, True
                reject_callback()
                self._update_gradient(self.vars.get_solution())
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
        """Compute the barrier objective: phi_mu(x) = f(x) - mu * sum(ln(s)).

        This is used as one of the two filter measures.  f(x) is extracted
        from the Lagrangian L = f + lam^T c by subtracting the multiplier
        term.  Requires _update_gradient() to have been called at vars.
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
        inner_filter,
        options,
        comm_rank,
        tau=0.995,
        mult_ind=None,
        phi_current=None,
    ):
        """IPOPT Algorithm A filter line search (Waechter & Biegler 2006).

        The filter tracks two measures at each iterate:
          theta = ||c(x)||   -- constraint violation (barrier-independent)
          phi   = f(x) - mu * sum(ln(s))  -- barrier objective

        A trial step is accepted under one of two conditions:

        Case 1 (switching): When theta is small (near-feasible) and the
            search direction is a descent direction for phi, accept with
            Armijo sufficient decrease on phi.  The filter is NOT augmented
            (we trust that the step is making optimality progress).

        Case 2 (filter): Accept if the trial point shows sufficient
            decrease in EITHER theta OR phi, AND is acceptable to the
            filter (not dominated by any previous entry).  The filter IS
            augmented with the current (phi_k, theta_k).

        Key IPOPT detail: only the PRIMAL step size alpha is backtracked.
        The dual step uses the full alpha_z from fraction-to-boundary,
        since dual variables don't affect the filter measures.

        Parameters
        ----------
        alpha_x, alpha_z : float
            Maximum primal/dual step from fraction-to-boundary rule.
        inner_filter : Filter
            The inner filter for the current barrier subproblem.
        options : dict
            Optimizer options.
        comm_rank : int
            MPI rank (only rank 0 prints diagnostics).
        tau : float
            Fraction-to-boundary parameter.
        mult_ind : bool array or None
            Multiplier indicator for SOC.
        phi_current : float
            Barrier objective at the current iterate.

        Returns
        -------
        alpha : float
            Accepted step size.
        line_iters : int
            Number of backtracking iterations.
        step_accepted : bool
            False if step was rejected (triggers restoration).
        """
        max_iters = options["max_line_search_iterations"]
        backtrack = options["backtracking_factor"]
        gamma_theta = options["filter_gamma_theta"]
        gamma_phi = options["filter_gamma_psi"]
        delta_switch = options["filter_delta"]
        s_theta = options["filter_s_theta"]
        s_phi = 2.3  # IPOPT default exponent for switching inequality
        eta_phi = 1e-4  # Armijo constant for barrier objective
        use_soc = options["second_order_correction"] and mult_ind is not None
        tol = options["convergence_tolerance"]

        # Evaluate current-point measures
        theta_k = self._compute_filter_theta()
        phi_k = phi_current

        # theta_min and theta_max (IPOPT Eq. 21), based on theta_0 which
        # is set once at the start of each barrier subproblem
        theta_0 = getattr(self, "_filter_theta_0", theta_k)
        theta_min = 1e-4 * max(1.0, theta_0)
        theta_max = 1e4 * max(1.0, theta_0)

        # Directional derivative of barrier objective along search direction
        # (IPOPT Eq. 19).  Computed analytically from the C++ backend.
        dphi = self.optimizer.compute_barrier_dphi(
            self.barrier_param,
            self.vars,
            self.update,
            self.res,
            self.px,
            self.diag,
        )

        # Switching condition (IPOPT Eq. 19)
        # When theta is small (near-feasible) and dphi is sufficiently
        # negative, use Armijo on phi instead of the filter.
        switching_eligible = theta_k < theta_min
        switching_active = False
        dphi_threshold = 1e-13 * max(1.0, abs(phi_k))
        if switching_eligible and dphi < -dphi_threshold:
            switching_active = (-dphi) ** s_phi > delta_switch * max(
                theta_k, 1e-30
            ) ** s_theta

        # Near-feasible regime: theta so small that filter conditions become
        # numerically degenerate.  Fall back to KKT residual acceptance.
        near_feasible = theta_k < theta_min

        if comm_rank == 0:
            mode = "switch" if switching_active else "filter"
            if near_feasible:
                mode += "+nf"
            print(
                f"  dphi={dphi:.2e}, theta_k={theta_k:.2e}, "
                f"theta_min={theta_min:.2e}, mode={mode}"
            )

        # Minimum step size alpha_min (IPOPT Eq. 23)
        # If alpha falls below alpha_min, trigger feasibility restoration.
        gamma_alpha = 0.05
        if switching_active:
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
            if dphi < 0:
                alpha_min_val = gamma_alpha * min(
                    gamma_theta,
                    gamma_phi * theta_k / (-dphi),
                )
            else:
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
            # Save current-point data for second-order correction (SOC).
            # SOC replaces the constraint part of the RHS with the trial
            # point's constraint residual, then re-solves (no re-factor).
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
            """Check if trial point is acceptable (IPOPT Eqs. 18-20).

            Returns (accepted, is_switching):
              - is_switching=True  -> Case 1 (Armijo on phi, no filter update)
              - is_switching=False -> Case 2 (filter acceptance, augment filter)
            """
            if theta_t > theta_max:
                return False, False
            # Case 1: switching condition -- Armijo on phi (IPOPT Eq. 20)
            if switching_active and dphi < 0:
                if phi_t <= phi_k + eta_phi * alpha_trial * dphi:
                    return True, True
            # Case 2: sufficient decrease in theta OR phi (IPOPT Eq. 18)
            sufficient = (
                theta_t <= (1.0 - gamma_theta) * theta_k
                or phi_t <= phi_k - gamma_phi * theta_k
            )
            if sufficient and inner_filter.is_acceptable(phi_t, theta_t, abs(phi_k)):
                return True, False
            return False, False

        # Backtracking loop
        for j in range(max_iters):
            # Check alpha_min: if step too small, trigger restoration
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

            # Second-order correction (IPOPT Algorithm A, step A-5.5).
            # SOC corrects for the linearization error in constraints.
            # Only attempt when the full step INCREASED infeasibility
            # (theta_trial >= theta_k). If theta decreased, the step is
            # already improving feasibility -- SOC is not needed.
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
        inertia_corrector,
        mult_ind,
        inner_filter,
        options,
        comm_rank,
        x,
        diag_base,
        zero_hessian_indices,
        zero_hessian_eps,
    ):
        """Feasibility restoration phase (IPOPT Section 3.3).

        Called when the filter line search fails to find an acceptable step.
        Two modes depending on the current constraint violation theta:

        Mode 1 (theta large): The iterate is infeasible.  Temporarily
            increase the barrier parameter and take Newton steps aimed at
            reducing theta by at least 10%.  Higher barrier relaxes the
            bound constraints, giving the optimizer more room to improve
            feasibility.  The current point is added to the filter before
            starting.

        Mode 2 (theta small): The constraints are nearly satisfied but
            the filter is blocking optimality progress.  Try to find a
            step that reduces the KKT residual.  If no such step exists,
            reset the filter (safe because constraints are satisfied).

        Returns True if restoration produced a better iterate.
        """
        max_restore = options["filter_restoration_max_iter"]
        backtrack = options["backtracking_factor"]
        max_ls = options["max_line_search_iterations"]
        tau_min = options["tau_min"]
        use_adaptive_tau = options["adaptive_tau"]
        tol = options["convergence_tolerance"]

        theta_k = self._compute_filter_theta()
        theta_start = theta_k

        # Mode 2: Small-theta restoration (filter blocking optimality)
        # When theta is below theta_min, reducing feasibility further won't
        # help -- the filter is simply preventing optimality steps.
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

        # Mode 1: Large-theta restoration (reduce constraint violation)
        phi_k = self._compute_barrier_objective(self.vars)
        inner_filter.add(phi_k, theta_k)

        target = 0.9 * theta_k
        if comm_rank == 0:
            print(f"  Restoration: theta={theta_k:.2e}, target={target:.2e}")

        saved_barrier = self.barrier_param

        # Try progressively higher barriers to relax bound constraints,
        # giving the optimizer more room to reduce infeasibility.
        restore_levels = [
            max(saved_barrier, 1e-2),
            0.1,
            1.0,
        ]
        restore_levels = sorted(set(mu for mu in restore_levels if mu >= saved_barrier))

        iters_per_level = max(3, max_restore // len(restore_levels))

        for restore_mu in restore_levels:
            self.barrier_param = restore_mu
            if inertia_corrector:
                inertia_corrector.update_barrier(restore_mu)

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
                    inertia_corrector,
                    mult_ind,
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
        if inertia_corrector:
            inertia_corrector.update_barrier(saved_barrier)
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
        """Run the interior-point optimization algorithm.

        Implements a primal-dual interior-point method with barrier parameter
        reduction.  The main loop structure is:

          1. Evaluate KKT residual and check convergence
          2. Update barrier parameter mu (strategy-dependent)
          3. Compute Newton search direction (factorize KKT + back-solve)
          4. Line search (filter-based or merit-function-based)
          5. Accept or reject step; handle restoration if needed

        Parameters
        ----------
        options : dict
            Solver options (see get_options() for defaults and descriptions).

        Returns
        -------
        opt_data : dict
            Optimization results with keys:
            - "converged": bool
            - "iterations": list of per-iteration data dicts
            - "options": the resolved options dict
        """
        start_time = time.perf_counter()

        comm_rank = 0
        if self.comm is not None:
            comm_rank = self.comm.rank

        options = self.get_options(options=options)
        opt_data = {"options": options, "converged": False, "iterations": []}

        # 1. Unpack frequently-used options
        self.barrier_param = options["initial_barrier_param"]
        self.gamma_penalty = options["gamma_peanlty"]
        max_iters = options["max_iterations"]
        tau = options["fraction_to_boundary"]
        tol = options["convergence_tolerance"]
        record_components = options["record_components"]
        continuation_control = options["continuation_control"]

        # 2. Initialize primal-dual variables
        x = self.vars.get_solution()

        xview = None
        if not self.distribute:
            xview = ModelVector(self.model, x=x)

        # Zero the multipliers so that the gradient consists of the objective gradient and
        # constraint values
        self._zero_multipliers(x)

        # Initialize slack variables and multipliers from the barrier parameter
        self.optimizer.initialize_multipliers_and_slacks(
            self.barrier_param, self.grad, self.vars
        )

        # Optionally compute better initial multiplier estimates
        if options["init_affine_step_multipliers"]:
            self._compute_least_squares_multipliers()

            # Compute the affine step multipliers and the new barrier parameter
            self.barrier_param = self._compute_affine_multipliers(
                beta_min=self.barrier_param, gamma_penalty=self.gamma_penalty
            )
        elif options["init_least_squares_multipliers"]:
            self._compute_least_squares_multipliers()

        # Compute the gradient
        if self.distribute:
            self.mpi_problem.update(x)
            self.mpi_problem.gradient(1.0, x, self.grad)
        else:
            self.problem.update(x)
            self.problem.gradient(1.0, x, self.grad)

        # Step size tracking (for log output)
        line_iters = 0
        alpha_x_prev = 0.0
        alpha_z_prev = 0.0
        x_index_prev = -1
        z_index_prev = -1

        # 3. Convergence and stagnation tracking
        acceptable_tol = options["acceptable_tol"]
        if acceptable_tol is None:
            acceptable_tol = tol * 100
        acceptable_iter = options["acceptable_iter"]
        prev_res_norm = float("inf")
        best_res_norm = float("inf")
        stagnation_count = 0
        precision_floor_count = 0

        # 4. Inertia correction setup (auto-detect from solver capability)
        problem_ref = self.mpi_problem if self.distribute else self.problem
        mult_ind = np.array(problem_ref.get_multiplier_indicator(), dtype=bool)
        inertia_corrector = None
        if hasattr(self.solver, "get_inertia"):
            inertia_corrector = InertiaCorrector(
                mult_ind,
                self.barrier_param,
                options,
                self.model,
                self.solver,
                self.distribute,
            )
            if comm_rank == 0:
                n_primal = int(np.sum(~mult_ind))
                solver_name = type(self.solver).__name__
                print(
                    f"  IPOPT Algorithm IC " f"({solver_name}): {n_primal} primal vars"
                )

        # Step rejection tracking: when the line search fails repeatedly,
        # increase the barrier parameter to escape ill-conditioned regions.
        step_rejected = False
        consecutive_rejections = 0
        max_rejections = options["max_consecutive_rejections"]
        barrier_inc = options["barrier_increase_factor"]
        initial_barrier = options["initial_barrier_param"]
        zero_step_count = 0  # Zero-step recovery (no inertia corrector path)

        # Zero-Hessian variable indices: variables that appear linearly
        # in the objective and need stronger regularization
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

        # 5. Quality function barrier strategy setup (if selected)
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

        soc_mult_ind = mult_ind if options["second_order_correction"] else None

        # 6. Filter line search setup (IPOPT Algorithm A + Algorithm B outer)
        filter_ls = options["filter_line_search"]
        outer_filter = Filter() if filter_ls else None
        inner_filter = Filter(kappa_1=1e-5, kappa_2=1.0) if filter_ls else None
        filter_monotone_mode = False  # True when filter rejected step
        filter_monotone_mu = None

        # Multiplier indicator for barrier objective computation
        if filter_ls:
            self._filter_mult_ind = mult_ind
            # Track theta_0 per barrier subproblem for switching condition
            self._filter_theta_0 = None  # set after first gradient eval

        res_norm_mu = self.barrier_param
        rhs_norm = 0.0

        # MAIN OPTIMIZATION LOOP
        for i in range(max_iters):
            # Step A: Evaluate KKT residual at current iterate
            res_norm = self.optimizer.compute_residual(
                self.barrier_param, self.gamma_penalty, self.vars, self.grad, self.res
            )
            res_norm_mu = self.barrier_param

            if inertia_corrector:
                inertia_corrector.update_barrier(self.barrier_param)

            # Decompose residual: theta = feasibility, eta = optimality
            res_arr = np.array(self.res.get_array())
            theta = np.linalg.norm(res_arr[mult_ind])
            eta = np.linalg.norm(res_arr[~mult_ind])

            if filter_ls and self._filter_theta_0 is None:
                self._filter_theta_0 = self._compute_filter_theta()

            # Step B: Log iteration data
            elapsed_time = time.perf_counter() - start_time

            if continuation_control is not None:
                continuation_control(i, res_norm)

            # Set information about the residual norm into the
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
            if inertia_corrector:
                iter_data.update(
                    {
                        "eps_z": inertia_corrector.eps_z,
                        "theta": theta,
                        "eta": eta,
                        "inertia_delta": inertia_corrector.last_inertia_delta,
                    }
                )
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
                heuristic_data.append(
                    {"iteration": i, "xi": xi, "complementarity": complementarity}
                )

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

            # Step C: Convergence and termination checks
            if res_norm < tol:
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

            # Stagnation check: force barrier reduction if stuck too long
            if inertia_corrector and res_norm >= tol:
                if stagnation_count >= 2 * acceptable_iter and self.barrier_param > tol:
                    new_barrier = max(
                        self.barrier_param * options["monotone_barrier_fraction"], tol
                    )
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

            # Step D: Barrier parameter update
            # IPOPT Algorithm A, Step A-3: reduce mu when the barrier
            # subproblem is solved to sufficient accuracy.
            # E_mu = max(dual_err, primal_err, comp_err) <= kappa_eps * mu.
            # All three KKT components must be small -- missing the dual
            # term causes premature mu reduction and ill-conditioning.
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
                        old_mu / 10.0,  # cap: never reduce by more than 10x
                    )
                    if new_mu < old_mu:
                        self.barrier_param = new_mu
                        if inertia_corrector:
                            inertia_corrector.update_barrier(self.barrier_param)
                        inner_filter.entries.clear()
                        # Reset theta_0 for the new barrier subproblem
                        self._filter_theta_0 = self._compute_filter_theta()
                        if comm_rank == 0:
                            print(
                                f"  IPOPT A-3: mu {old_mu:.2e} -> "
                                f"{new_mu:.2e} "
                                f"(E_mu={e_mu:.2e}, filter reset)"
                            )

            # Step E: Compute search direction
            step_rejected = False
            if inertia_corrector:
                inertia_corrector.update_barrier(self.barrier_param)
            else:
                # Legacy zero-step recovery (no inertia corrector)
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

            # Solve the KKT system with the computed diagonal entries
            self.solver.factor(1.0, x, self.diag)
            self.solver.solve(self.res, self.px)

            # IPOPT filter LS path: barrier is handled by Step A-3 above.
            # Just compute the search direction with the current mu.
            if filter_ls:
                self._find_direction(
                    x,
                    diag_base,
                    inertia_corrector,
                    mult_ind,
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
                if inertia_corrector:
                    inertia_corrector.update_barrier(self.barrier_param)

                self._find_direction(
                    x,
                    diag_base,
                    inertia_corrector,
                    mult_ind,
                    options,
                    zero_hessian_indices,
                    zero_hessian_eps,
                    comm_rank,
                )
            elif quality_func:
                self._factorize_kkt(
                    x,
                    diag_base,
                    inertia_corrector,
                    mult_ind,
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

                if inertia_corrector:
                    inertia_corrector.update_barrier(self.barrier_param)

                if not qf_free_mode:
                    self._solve_with_mu(self.barrier_param)
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
                        kappa_mono = options["monotone_barrier_fraction"]
                        self.barrier_param = max(
                            kappa_mono * self.barrier_param,
                            tol,
                        )

                # Direction finding (regularize, factor, solve)
                self._find_direction(
                    x,
                    diag_base,
                    inertia_corrector,
                    mult_ind,
                    options,
                    zero_hessian_indices,
                    zero_hessian_eps,
                    comm_rank,
                )

            # Check the update (debug only)
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
                    _mi = mult_ind
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
            # May differ from res_norm (top of loop) if Step D changed mu.
            rhs_norm = self.optimizer.compute_residual(
                self.barrier_param, self.vars, self.grad, self.res
            )

            # Detailed Newton diagnostics (expensive, only with check_update_step)
            if options["check_update_step"] and comm_rank == 0 and i > 0:
                self._print_newton_diagnostics(rhs_norm, res_norm_mu, mult_ind)

            # Compute maximum step sizes from fraction-to-boundary rule
            tau = (
                self._compute_adaptive_tau(self.barrier_param, tau_min)
                if use_adaptive_tau
                else base_tau
            )
            alpha_x, x_index, alpha_z, z_index = self.optimizer.compute_max_step(
                tau, self.vars, self.update
            )

            if comm_rank == 0:
                print(f"  a_x={alpha_x:.4f}, a_z={alpha_z:.4f}")

            if options["equal_primal_dual_step"]:
                alpha_x = alpha_z = min(alpha_x, alpha_z)

            # Step F: Line search and step acceptance
            if filter_ls:
                # Compute barrier objective at current point (gradient is current)
                phi_current = self._compute_barrier_objective(self.vars)
                alpha, line_iters, step_accepted = self._filter_line_search(
                    alpha_x,
                    alpha_z,
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
                        line_iters = 0
                    else:
                        step_rejected = True
                        consecutive_rejections += 1
                        if comm_rank == 0:
                            print(
                                f"  Filter+Restoration REJECTED "
                                f"({consecutive_rejections}x)"
                            )

                # Update gradient at the new point for the next iteration.
                if step_accepted:
                    # FIX 5: IPOPT Eq. 16 -- reset bound multipliers after
                    # step acceptance to prevent divergence. Clamps each z_i
                    # to [mu/(kappa*gap), kappa*mu/gap] with kappa=1e10.
                    self.optimizer.reset_bound_multipliers(
                        self.barrier_param,
                        1e10,
                        self.vars,
                    )
                    self._update_gradient(self.vars.get_solution())
            else:
                # Build reject callback for line search
                def _reject_step():
                    nonlocal step_rejected, consecutive_rejections
                    step_rejected = True
                    consecutive_rejections += 1
                    if comm_rank == 0:
                        print(f"  Step REJECTED ({consecutive_rejections}x)")

                reject_cb = _reject_step if inertia_corrector else None
                alpha, line_iters, step_accepted = self._line_search(
                    alpha_x,
                    alpha_z,
                    options,
                    comm_rank,
                    tau=tau,
                    mult_ind=soc_mult_ind,
                    reject_callback=reject_cb,
                )

            # Step G: Post-step update (rejection handling or acceptance)
            if step_rejected:
                alpha_x_prev = 0.0
                alpha_z_prev = 0.0
                x_index_prev = -1
                z_index_prev = -1
                self.barrier_param = barrier_before

                    # Apply a simple backtracking algorithm
                    alpha *= options["backtracting_factor"]

                # Handle rejection escape: increase barrier after too many rejections
                if consecutive_rejections >= max_rejections:
                    new_barrier = min(self.barrier_param * barrier_inc, initial_barrier)
                    if new_barrier > self.barrier_param:
                        if inertia_corrector:
                            inertia_corrector.eps_z = inertia_corrector.cz * new_barrier
                        consecutive_rejections = 0
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
                    else:
                        consecutive_rejections = 0
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

                consecutive_rejections = 0

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
