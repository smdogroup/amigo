import os
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

    def factor(self, alpha, x, diag, post_hessian=None):
        """Assemble Hessian, add diagonal, and factorize (one shot).

        Used for inertia-correction retries where we need a fresh
        assembly (since add_diagonal_and_factor mutates self.hess).
        """
        self.problem.hessian(alpha, x, self.hess)
        if post_hessian is not None:
            self.hess.copy_data_device_to_host()
            post_hessian(self.hess)
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


class MumpsSolver(_HessianDiagMixin):
    """Sparse symmetric indefinite solver via MUMPS (LDL^T with inertia).

    Uses the MUMPS C interface (dmumps_c) via ctypes. Provides exact
    inertia counts from MUMPS info arrays after factorization.

    Requires coin-or/ThirdParty-Mumps (with METIS ordering and scaling).
    Windows: build via MSYS2 with mingw-w64-x86_64-metis.
    Linux: apt install libmumps-dev or conda install mumps-seq.
    Mac: brew install brewsci/num/mumps.
    """

    @staticmethod
    def _load_mumps_library():
        """Locate and load the MUMPS shared library.

        Search order: MUMPS_LIB_DIR env var, coin-or ThirdParty-Mumps
        install, conda environment, system PATH.
        """
        import ctypes

        lib_dir = os.environ.get("MUMPS_LIB_DIR", "")

        # Platform-specific library names and search paths
        if sys.platform == "win32":
            # Register dependency directories for Windows DLL resolution
            for d in [
                r"C:\msys64\mingw64\bin",
                os.path.expanduser("~/mumps-coinor/bin"),
            ]:
                if os.path.isdir(d):
                    os.add_dll_directory(d)

            names = ["libcoinmumps-3.dll", "libdmumps.dll", "dmumps.dll"]
            search_dirs = [
                lib_dir,
                os.path.expanduser("~/mumps-coinor/bin"),
            ]
            conda = os.environ.get("CONDA_PREFIX", "")
            if conda:
                search_dirs.append(os.path.join(conda, "Library", "bin"))
        elif sys.platform == "darwin":
            names = ["libdmumps.dylib"]
            search_dirs = [lib_dir] if lib_dir else []
        else:
            names = ["libdmumps.so"]
            search_dirs = [lib_dir] if lib_dir else []

        # Try each directory + name combination, then bare names for PATH
        for d in search_dirs:
            if not d:
                continue
            for name in names:
                path = os.path.join(d, name)
                try:
                    return ctypes.CDLL(path)
                except OSError:
                    pass
        for name in names:
            try:
                return ctypes.CDLL(name)
            except OSError:
                pass

        raise ImportError(
            "MUMPS library not found. "
            "Windows: build coin-or/ThirdParty-Mumps via MSYS2. "
            "Linux: apt install libmumps-dev or conda install mumps-seq. "
            "Mac: brew install brewsci/num/mumps. "
            "Or set MUMPS_LIB_DIR to the directory containing the library."
        )

    def __init__(self, problem):
        import ctypes

        self._ct = ctypes
        self._libmumps = self._load_mumps_library()
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
        row_idx = np.repeat(np.arange(self.nrows, dtype=np.int32), np.diff(self.rowp))
        col_idx = np.array(self.cols, dtype=np.int32)
        lower_mask = col_idx <= row_idx
        self._irn = row_idx[lower_mask] + 1
        self._jcn = col_idx[lower_mask] + 1
        self._data_map = np.nonzero(lower_mask)[0].astype(np.intc)
        self._nnz_lower = int(lower_mask.sum())
        self._a = np.empty(self._nnz_lower, dtype=np.float64)

        # Build the MUMPS struct via ctypes
        self._build_struct()

        # Initialize MUMPS (job=-1)
        self._mumps.job = -1
        self._mumps.par = 1
        self._mumps.sym = 2  # symmetric indefinite (LDL^T)
        self._mumps.comm_fortran = -987654  # MPISEQ sequential
        self._call_mumps()

        # MUMPS solver parameters
        self._mumps.icntl[0] = -1  # ICNTL(1):  suppress error output
        self._mumps.icntl[1] = -1  # ICNTL(2):  suppress diagnostic output
        self._mumps.icntl[2] = -1  # ICNTL(3):  suppress global info
        self._mumps.icntl[3] = 0  # ICNTL(4):  no output
        self._mumps.icntl[5] = 7  # ICNTL(6):  permuting and scaling
        self._mumps.icntl[6] = 7  # ICNTL(7):  pivot ordering (automatic)
        self._mumps.icntl[7] = 77  # ICNTL(8):  scaling (automatic)
        self._mumps.icntl[9] = 0  # ICNTL(10): no iterative refinement
        self._mumps.icntl[12] = 1  # ICNTL(13): proper inertia detection
        self._mumps.icntl[13] = 1000  # ICNTL(14): workspace increase %
        # ICNTL(24) = 0: do NOT enable null pivot detection during normal
        # factorization (matches IPOPT).  When enabled, near-zero negative
        # pivots can be misclassified as "null", corrupting the inertia count.
        self._mumps.icntl[23] = 0  # ICNTL(24): null pivot detection OFF
        self._mumps.cntl[0] = 1e-6  # CNTL(1):  pivot tolerance

        # Set matrix structure and values pointer
        self._mumps.n = self.nrows
        self._mumps.nz = int(self._nnz_lower) if self._nnz_lower < 2**31 else 0
        self._mumps.nnz = self._nnz_lower
        self._mumps.irn = self._irn.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self._mumps.jcn = self._jcn.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self._mumps.a = self._a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Symbolic analysis deferred to first factorization (with real values)
        self._have_symbolic = False

        self._rhs = np.empty(self.nrows, dtype=np.float64)
        self._mumps.rhs = self._rhs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self._mumps.nrhs = 1
        self._mumps.lrhs = self.nrows

    def _build_struct(self):
        """Build the ctypes Structure matching DMUMPS_STRUC_C."""
        ct = self._ct
        import sys

        if sys.platform == "win32":
            # Layout matching the Windows MUMPS build (newer/custom version)
            _mumps_fields = [
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
        else:
            # Layout matching MUMPS 5.x (Linux/Mac)
            _mumps_fields = [
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
                ("rhs", ct.POINTER(ct.c_double)),
                ("redrhs", ct.POINTER(ct.c_double)),
                ("rhs_sparse", ct.POINTER(ct.c_double)),
                ("sol_loc", ct.POINTER(ct.c_double)),
                ("rhs_loc", ct.POINTER(ct.c_double)),
                ("irhs_sparse", ct.POINTER(ct.c_int)),
                ("irhs_ptr", ct.POINTER(ct.c_int)),
                ("isol_loc", ct.POINTER(ct.c_int)),
                ("irhs_loc", ct.POINTER(ct.c_int)),
                ("nrhs", ct.c_int),
                ("lrhs", ct.c_int),
                ("lredrhs", ct.c_int),
                ("nz_rhs", ct.c_int),
                ("lsol_loc", ct.c_int),
                ("nloc_rhs", ct.c_int),
                ("lrhs_loc", ct.c_int),
                ("schur_mloc", ct.c_int),
                ("schur_nloc", ct.c_int),
                ("schur_lld", ct.c_int),
                ("mblock", ct.c_int),
                ("nblock", ct.c_int),
                ("nprow", ct.c_int),
                ("npcol", ct.c_int),
                ("info", ct.c_int * 80),
                ("infog", ct.c_int * 80),
                ("rinfo", ct.c_double * 40),
                ("rinfog", ct.c_double * 40),
                ("deficiency", ct.c_int),
                ("pivnul_list", ct.POINTER(ct.c_int)),
                ("mapping", ct.POINTER(ct.c_int)),
                ("size_schur", ct.c_int),
                ("listvar_schur", ct.POINTER(ct.c_int)),
                ("schur", ct.POINTER(ct.c_double)),
                ("instance_number", ct.c_int),
                ("wk_user", ct.POINTER(ct.c_double)),
                ("version_number", ct.c_char * 32),
                ("ooc_tmpdir", ct.c_char * 256),
                ("ooc_prefix", ct.c_char * 64),
                ("write_problem", ct.c_char * 256),
                ("lwk_user", ct.c_int),
                ("save_dir", ct.c_char * 256),
                ("save_prefix", ct.c_char * 256),
                ("metis_options", ct.c_int * 40),
            ]

        class DMUMPS_STRUC_C(ct.Structure):
            _fields_ = _mumps_fields

        self._DMUMPS_STRUC_C = DMUMPS_STRUC_C
        self._mumps = DMUMPS_STRUC_C()
        self._dmumps_c.argtypes = [ct.POINTER(DMUMPS_STRUC_C)]

    def _call_mumps(self):
        self._dmumps_c(self._ct.byref(self._mumps))

    def _factorize_current(self):
        if not self._have_symbolic:
            self._mumps.job = 1  # symbolic analysis with actual values
            self._call_mumps()
            if self._mumps.infog[0] < 0:
                raise RuntimeError(
                    f"MUMPS analysis failed: infog(1)={self._mumps.infog[0]}"
                )
            self._have_symbolic = True
        self._mumps.job = 2  # numerical factorization
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

    def factor(self, alpha, x, diag, post_hessian=None):
        self.problem.hessian(alpha, x, self.hess)
        if post_hessian is not None:
            self.hess.copy_data_device_to_host()
            post_hessian(self.hess)
        self.problem.add_diagonal(diag, self.hess)
        self.hess.copy_data_device_to_host()
        self._update_values()
        self._factorize_current()

    def get_inertia(self):
        """Return (n_positive, n_negative) from MUMPS infog(12).

        infog(12) = number of negative pivots in LDL^T factorization.
        With ICNTL(24)=0 (null pivot detection off, matching IPOPT),
        all pivots are classified as positive or negative.
        """
        n_neg = int(self._mumps.infog[11])
        n_pos = self.nrows - n_neg
        return n_pos, n_neg

    def solve(self, bx, px):
        bx.copy_device_to_host()
        self._rhs[:] = bx.get_array()
        self._mumps.job = 3
        self._call_mumps()
        if self._mumps.infog[0] < 0:
            raise RuntimeError(f"MUMPS solve failed: infog(1)={self._mumps.infog[0]}")
        px.get_array()[:] = self._rhs
        px.copy_host_to_device()

    def solve_array(self, rhs):
        self._rhs[:] = rhs
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

    def factor(self, alpha, x, diag, _debug_inertia=False, post_hessian=None):
        """Assemble Hessian, add diagonal, and LDL^T factorize (one shot).

        Used for inertia-correction retries where we need a fresh
        assembly (since add_diagonal_and_factor mutates self.hess).
        """
        self.problem.hessian(alpha, x, self.hess)
        if post_hessian is not None:
            self.hess.copy_data_device_to_host()
            post_hessian(self.hess)
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


class InertiaCorrector:
    """Inertia correction for the KKT system (Algorithm IC, Wachter & Biegler 2006).

    Manages primal (delta_w) and constraint (delta_c) regularization to
    ensure correct inertia (n positive, m negative eigenvalues):
      - ConsiderNewSystem: save last perturbation, reset current to zero.
        If structurally degenerate, pre-apply delta_c / delta_x.
      - PerturbForSingularity: add delta_c first, then delta_x.
      - PerturbForWrongInertia: grow delta_x; on overflow, add delta_c
        and restart delta_x search.
      - finalize_test: structural degeneracy detection after consecutive
        iterations needing the same perturbation type.
      - IncreaseQuality: when too few negative eigenvalues, try improving
        pivot tolerance before treating as singular.
    """

    # Degeneracy status
    _NOT_YET = 0
    _NOT_DEGEN = 1
    _DEGENERATE = 2

    # Test status for finalize_test
    _NO_TEST = 0
    _TEST_DC0_DX0 = 1
    _TEST_DC1_DX0 = 2
    _TEST_DC0_DX1 = 3
    _TEST_DC1_DX1 = 4

    def __init__(self, mult_ind, barrier_param, options):
        self.mult_ind = mult_ind
        self._barrier = barrier_param
        self.numerical_eps = 1e-12

        # Perturbation state
        self._delta_x_last = 0.0
        self._delta_c_last = 0.0
        self._delta_x_curr = 0.0
        self._delta_c_curr = 0.0

        # Exposed for iterative refinement
        self.last_delta_w = 0.0
        self.last_delta_c = 0.0

        # Algorithm IC constants (Table 3, Wachter & Biegler 2006)
        self._dw_init = 1e-4  # first_hessian_perturbation
        self._dw_min = 1e-20  # min_hessian_perturbation
        self._dw_max = 1e20  # max_hessian_perturbation
        self._kw_inc = 8.0  # perturb_inc_fact
        self._kw_first_inc = 100.0  # perturb_inc_fact_first
        self._kw_dec = 1.0 / 3  # perturb_dec_fact
        self._dc_val = 1e-8  # jacobian_regularization_value
        self._dc_exp = 0.25  # jacobian_regularization_exponent

        # Structural degeneracy detection
        self._hess_degen = self._NOT_YET
        self._jac_degen = self._NOT_YET
        self._degen_iters = 0
        self._degen_iters_max = 3
        self._test_status = self._NO_TEST

        # Adaptive pivot tolerance
        self._pivtol = 1e-6
        self._pivtolmax = 0.1

    def update_barrier(self, mu):
        """Synchronize internal barrier parameter."""
        self._barrier = mu

    def _delta_cd(self):
        """Constraint regularization: delta_c = delta_cd_val * mu^delta_cd_exp."""
        return self._dc_val * self._barrier**self._dc_exp

    def _get_deltas_for_wrong_inertia(self):
        """Grow delta_x geometrically. Returns False if delta_x exceeds max."""
        prev = self._delta_x_curr
        if self._delta_x_curr == 0.0:
            if self._delta_x_last == 0.0:
                self._delta_x_curr = self._dw_init
            else:
                self._delta_x_curr = max(
                    self._dw_min, self._delta_x_last * self._kw_dec
                )
        else:
            if (
                self._delta_x_last == 0.0
                or 1e5 * self._delta_x_last < self._delta_x_curr
            ):
                self._delta_x_curr *= self._kw_first_inc
            else:
                self._delta_x_curr *= self._kw_inc

        if self._delta_x_curr > self._dw_max:
            # Revert: the overflowed value was never used in a factorization
            self._delta_x_curr = prev
            self._delta_x_last = 0.0
            return False
        return True

    def _perturb_for_wrong_inertia(self):
        """Perturb for wrong inertia (too many negative eigenvalues).

        Calls finalize_test, then grows delta_x.
        On overflow with delta_c==0: add delta_c and restart delta_x.
        """
        self._finalize_test()
        if self._get_deltas_for_wrong_inertia():
            return True
        # delta_x overflow: if delta_c==0, add it and retry from scratch
        if self._delta_c_curr == 0.0:
            self._delta_c_curr = self._delta_cd()
            self._delta_x_curr = 0.0
            if self._hess_degen == self._DEGENERATE:
                self._hess_degen = self._NOT_YET
            self._test_status = self._NO_TEST
            return self._get_deltas_for_wrong_inertia()
        return False

    def _perturb_for_singularity(self):
        """Perturb for singular system (too few negative eigenvalues).

        Handles the degeneracy test state machine for singular systems.
        """
        if self._hess_degen == self._NOT_YET or self._jac_degen == self._NOT_YET:
            # Degeneracy test state machine
            ts = self._test_status

            if ts == self._TEST_DC0_DX0:
                # Haven't tried anything yet for this matrix
                if self._jac_degen == self._NOT_YET:
                    # Try adding delta_c only (test if jac is degenerate)
                    self._delta_c_curr = self._delta_cd()
                    self._test_status = self._TEST_DC1_DX0
                else:
                    # jac known, hess NOT_YET: try delta_x only
                    if not self._get_deltas_for_wrong_inertia():
                        return False
                    self._test_status = self._TEST_DC0_DX1

            elif ts == self._TEST_DC1_DX0:
                # Already tried delta_c>0, delta_x=0 — still singular.
                # Now try delta_x>0, delta_c=0
                self._delta_c_curr = 0.0
                if not self._get_deltas_for_wrong_inertia():
                    return False
                self._test_status = self._TEST_DC0_DX1

            elif ts == self._TEST_DC0_DX1:
                # Tried delta_x>0, delta_c=0 — still singular.
                # Now try both.
                self._delta_c_curr = self._delta_cd()
                if not self._get_deltas_for_wrong_inertia():
                    return False
                self._test_status = self._TEST_DC1_DX1

            elif ts == self._TEST_DC1_DX1:
                # Both active — just grow delta_x.
                if not self._get_deltas_for_wrong_inertia():
                    return False

            # else: NO_TEST should not occur here

        else:
            # Both hess/jac degeneracy resolved
            if self._delta_c_curr > 0.0:
                # Already perturbing constraints: treat like wrong inertia
                if not self._get_deltas_for_wrong_inertia():
                    return False
            else:
                # First singular encounter: add constraint regularization
                self._delta_c_curr = self._delta_cd()

        return True

    def _finalize_test(self):
        """Conclude degeneracy test after successful factorization.

        After degen_iters_max consecutive iterations needing the same
        perturbation type, declare structural degeneracy.
        """
        ts = self._test_status
        if ts == self._NO_TEST:
            return

        if ts == self._TEST_DC0_DX0:
            if self._hess_degen == self._NOT_YET:
                self._hess_degen = self._NOT_DEGEN
            if self._jac_degen == self._NOT_YET:
                self._jac_degen = self._NOT_DEGEN
        elif ts == self._TEST_DC1_DX0:
            if self._hess_degen == self._NOT_YET:
                self._hess_degen = self._NOT_DEGEN
            if self._jac_degen == self._NOT_YET:
                self._degen_iters += 1
                if self._degen_iters >= self._degen_iters_max:
                    self._jac_degen = self._DEGENERATE
        elif ts == self._TEST_DC0_DX1:
            if self._jac_degen == self._NOT_YET:
                self._jac_degen = self._NOT_DEGEN
            if self._hess_degen == self._NOT_YET:
                self._degen_iters += 1
                if self._degen_iters >= self._degen_iters_max:
                    self._hess_degen = self._DEGENERATE
        elif ts == self._TEST_DC1_DX1:
            self._degen_iters += 1
            if self._degen_iters >= self._degen_iters_max:
                self._hess_degen = self._DEGENERATE
                self._jac_degen = self._DEGENERATE

        self._test_status = self._NO_TEST

    def _consider_new_system(self):
        """Prepare for a new KKT system.

        Save last perturbation, reset current to zero. Pre-apply delta_c
        if Jacobian is structurally degenerate, and pre-populate delta_x
        if Hessian is structurally degenerate.
        """
        self._finalize_test()

        # Note: pivot tolerance is NOT reset here — it persists across
        # iterations, matching IPOPT's ConsiderNewSystem behavior.  Once
        # IncreaseQuality raises pivtol, the solver keeps the more
        # accurate setting for subsequent systems.

        # Save last perturbation
        if self._delta_x_curr > 0.0:
            self._delta_x_last = self._delta_x_curr
        if self._delta_c_curr > 0.0:
            self._delta_c_last = self._delta_c_curr

        # Set up degeneracy test for this iteration
        if self._hess_degen == self._NOT_YET or self._jac_degen == self._NOT_YET:
            self._test_status = self._TEST_DC0_DX0
        else:
            self._test_status = self._NO_TEST

        # Pre-apply delta_c if Jacobian structurally degenerate
        if self._jac_degen == self._DEGENERATE:
            self._delta_c_curr = self._delta_cd()
        else:
            self._delta_c_curr = 0.0

        # Pre-apply delta_x if Hessian structurally degenerate
        if self._hess_degen == self._DEGENERATE:
            self._delta_x_curr = 0.0
            self._get_deltas_for_wrong_inertia()
        else:
            self._delta_x_curr = 0.0

    def factorize(
        self,
        solver,
        x,
        diag,
        diag_base,
        zero_hessian_indices,
        zero_hessian_eps,
        comm_rank,
        max_corrections=40,
        inertia_tolerance=0,
        obj_scale=1.0,
        hessian_scaling_fn=None,
    ):
        """Assemble, regularize, and factorize the KKT matrix."""
        solver.assemble_hessian(obj_scale, x)
        if hessian_scaling_fn is not None:
            hessian_scaling_fn(solver.hess)

        primal_mask = ~self.mult_ind
        n_primal = int(np.sum(primal_mask))
        n_dual = int(np.sum(self.mult_ind))
        n_total = n_primal + n_dual
        itol = inertia_tolerance

        def _ok(np_, nn_):
            return (
                abs(np_ - n_primal) <= itol
                and abs(nn_ - n_dual) <= itol
                and np_ + nn_ >= n_total - itol
            )

        # Build baseline diagonal: Sigma + small numerical eps on primals
        diag_arr = diag.get_array()
        diag_arr[primal_mask] += self.numerical_eps
        if zero_hessian_indices is not None and zero_hessian_eps is not None:
            np.maximum(
                diag_arr[zero_hessian_indices],
                zero_hessian_eps,
                out=diag_arr[zero_hessian_indices],
            )
        diag.copy_host_to_device()

        # No inertia check available: simple fallback
        if not hasattr(solver, "get_inertia"):
            try:
                solver.add_diagonal_and_factor(diag)
            except Exception:
                diag_arr[primal_mask] += self._dw_init
                diag.copy_host_to_device()
                solver.factor(obj_scale, x, diag, post_hessian=hessian_scaling_fn)
            return

        reg_diag = diag.get_array().copy()

        # Prepare new system: save last perturbation, reset current
        self._consider_new_system()

        # Sync pivot tolerance to solver
        if hasattr(solver, "_mumps"):
            solver._mumps.cntl[0] = self._pivtol
        augsys_improved = False

        def _apply_and_factor(first=False):
            """Apply perturbation, factorize. Returns (n_pos, n_neg, singular)."""
            diag.get_array()[:] = reg_diag
            if self._delta_x_curr > 0:
                diag.get_array()[primal_mask] += self._delta_x_curr
            if self._delta_c_curr > 0:
                diag.get_array()[self.mult_ind] -= self._delta_c_curr
            diag.copy_host_to_device()
            try:
                if first:
                    solver.add_diagonal_and_factor(diag)
                else:
                    solver.factor(obj_scale, x, diag, post_hessian=hessian_scaling_fn)
                return *solver.get_inertia(), False
            except Exception:
                return 0, 0, True

        # Main retry loop
        for attempt in range(max_corrections + 1):
            n_pos, n_neg, singular = _apply_and_factor(first=(attempt == 0))

            if not singular and _ok(n_pos, n_neg):
                # Success
                self.last_delta_w = self._delta_x_curr
                self.last_delta_c = self._delta_c_curr
                if self._delta_x_curr > 0 and comm_rank == 0:
                    print(
                        f"  Inertia correction: "
                        f"delta_w={self._delta_x_curr:.2e}, "
                        f"delta_c={self._delta_c_curr:.2e}, "
                        f"attempts={attempt + 1}"
                    )
                return True

            if comm_rank == 0 and not singular:
                print(
                    f"  Inertia: expected ({n_primal}+, {n_dual}-), "
                    f"got ({n_pos}+, {n_neg}-), "
                    f"dw={self._delta_x_curr:.1e}, pivtol={self._pivtol:.1e}"
                )

            # Dispatch based on failure type
            if singular and n_dual > 0:
                if not self._perturb_for_singularity():
                    break
            elif not singular and n_neg < n_dual:
                # Too few negatives: IncreaseQuality first, then singular
                assume_singular = True
                if not augsys_improved:
                    augsys_improved = self._increase_quality(solver)
                    if augsys_improved:
                        assume_singular = False
                if assume_singular:
                    if not self._perturb_for_singularity():
                        break
            else:
                # SYMSOLVER_WRONG_INERTIA (too many negatives) or
                # SYMSOLVER_SINGULAR with no constraints
                if not self._perturb_for_wrong_inertia():
                    if comm_rank == 0:
                        print(
                            f"  Inertia: delta_w={self._delta_x_curr:.2e} "
                            f"> max, aborting correction"
                        )
                    break

        # Inertia correction failed — store last actually-applied values
        self.last_delta_w = self._delta_x_curr
        self.last_delta_c = self._delta_c_curr
        return False

    def _increase_quality(self, solver):
        """Increase pivot tolerance: pivtol = min(pivtolmax, sqrt(pivtol))."""
        if self._pivtol >= self._pivtolmax:
            return False
        self._pivtol = min(self._pivtolmax, self._pivtol**0.5)
        if hasattr(solver, "_mumps"):
            solver._mumps.cntl[0] = self._pivtol
        return True


class Filter:
    """Filter for the line search.

    Stores (phi, theta) pairs with margins baked in at add-time.
    Acceptance is a strict dominance check: a trial point is acceptable
    if it is NOT dominated by any filter entry.

    The margins match Eq. 18:
      phi_entry   = phi_ref - gamma_phi * theta_ref
      theta_entry = (1 - gamma_theta) * theta_ref
    """

    def __init__(self, gamma_phi=1e-8, gamma_theta=1e-5):
        self.entries = []
        self.gamma_phi = gamma_phi
        self.gamma_theta = gamma_theta

    def is_acceptable(self, phi_trial, theta_trial):
        """True if trial is not dominated by any filter entry.

        A trial is acceptable to an entry if at least one coordinate
        is <= the entry. Rejection requires STRICT > in ALL coordinates.
        """
        for phi_f, theta_f in self.entries:
            if phi_trial > phi_f and theta_trial > theta_f:
                return False
        return True

    def add(self, phi, theta):
        """Add entry with margins (Eq. 18), remove dominated."""
        phi_entry = phi - self.gamma_phi * theta
        theta_entry = (1.0 - self.gamma_theta) * theta
        self.entries = [
            (p, t) for p, t in self.entries if not (p >= phi_entry and t >= theta_entry)
        ]
        self.entries.append((phi_entry, theta_entry))

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

        # Fill slack bounds from inequality constraint metadata.
        # User-provided vectors default to 0 at slack positions.
        if hasattr(model, "num_slacks") and model.num_slacks > 0:
            lb_arr = self.lower.get_array()
            ub_arr = self.upper.get_array()
            for k in range(model.num_slacks):
                idx = model.slack_indices[k]
                lb_arr[idx] = model._slack_meta[k]["lower"]
                ub_arr[idx] = model._slack_meta[k]["upper"]

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

        # Register slack-to-constraint mapping (if model has inequalities)
        if self.model.num_slacks > 0 and not self.distribute:
            self.optimizer.set_slack_mapping(
                np.ascontiguousarray(self.model.slack_indices, dtype=np.int32),
                np.ascontiguousarray(
                    self.model.ineq_constraint_indices, dtype=np.int32
                ),
            )

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
        if iteration % 20 == 0:
            print(
                f"{'iter':>4s}  {'nlp_error':>9s}  {'objective':>12s}  "
                f"{'inf_pr':>9s}  {'inf_du':>9s}  {'compl':>9s}  "
                f"{'mu':>9s}  {'||d||':>9s}  {'delta_w':>8s}  "
                f"{'alpha_x':>8s}  {'alpha_z':>8s}  "
                f"{'ls':>2s}  {'filt':>4s}"
            )

        mu = iter_data.get("barrier_param", 1.0)
        delta_w = iter_data.get("inertia_delta", 0.0)
        nlp_err = iter_data.get("nlp_error", 0.0)
        obj = iter_data.get("objective", 0.0)
        inf_pr = iter_data.get("inf_pr", 0.0)
        inf_du = iter_data.get("inf_du", 0.0)
        compl = iter_data.get("compl", 0.0)
        step_norm = iter_data.get("step_norm", 0.0)
        ax = iter_data.get("alpha_x", 0.0)
        az = iter_data.get("alpha_z", 0.0)
        ls = iter_data.get("line_iters", 0)
        fsize = iter_data.get("filter_size", 0)

        dw_str = f"{delta_w:8.1e}" if delta_w > 0 else f"{'---':>8s}"

        print(
            f"{iteration:4d}  {nlp_err:9.2e}  {obj:12.5e}  "
            f"{inf_pr:9.2e}  {inf_du:9.2e}  {compl:9.2e}  "
            f"{mu:9.2e}  {step_norm:9.2e}  {dw_str}  "
            f"{ax:8.2e}  {az:8.2e}  "
            f"{ls:2d}  {fsize:4d}"
        )
        sys.stdout.flush()

    def get_options(self, options={}):
        default = {
            "max_iterations": 100,
            "barrier_strategy": "heuristic",  # "heuristic", "monotone", "quality_function"
            "monotone_barrier_fraction": 0.1,
            "convergence_tolerance": 1e-8,
            "dual_inf_tol": 1.0,
            "constr_viol_tol": 1e-4,
            "compl_inf_tol": 1e-4,
            "diverging_iterates_tol": 1e20,
            "fraction_to_boundary": 0.95,
            "initial_barrier_param": 0.1,
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
            "filter_line_search": True,
            "filter_gamma_theta": 1e-5,
            "filter_gamma_phi": 1e-8,
            "filter_delta": 1.0,
            "filter_s_theta": 1.1,
            "filter_s_phi": 2.3,
            "filter_eta_phi": 1e-8,
            "filter_max_soc": 4,
            "filter_kappa_soc": 0.99,
            "filter_restoration_max_iter": 20,  # Max restoration iterations
            # Acceptable convergence
            "acceptable_tol": 1e-6,
            "acceptable_iter": 15,
            "acceptable_dual_inf_tol": 1e10,
            "acceptable_constr_viol_tol": 1e-2,
            "acceptable_compl_inf_tol": 1e-2,
            # Advanced features
            "adaptive_tau": True,  # Adaptive fraction-to-boundary
            "tau_min": 0.99,  # Minimum tau value
            "progress_based_barrier": True,  # Only reduce barrier when making progress
            "barrier_progress_tol": 10.0,  # kappa_epsilon factor for barrier subproblem tolerance
            # Variable-specific regularization for zero-Hessian (linearly-appearing) variables
            "zero_hessian_variables": [],  # Variable names (e.g., ["dyn.qdot"])
            "regularization_eps_x_zero_hessian": 1.0,  # Strong eps_x for those variables
            # Algorithm IC inertia correction (auto-detected from solver)
            "max_consecutive_rejections": 5,  # Before barrier increase
            "barrier_increase_factor": 5.0,  # Barrier *= this when stuck
            "max_inertia_corrections": 40,
            "inertia_tolerance": 0,  # Allow n_pos/n_neg to differ by this many from expected
            "quality_function_sigma_max": 4.0,
            "quality_function_golden_iters": 12,
            "quality_function_kappa_free": 0.9999,
            "quality_function_l_max": 5,
            "quality_function_norm_scaling": True,
            "quality_function_predictor_corrector": True,
            "pc_kappa_mu_decay": 0.2,
            "nlp_scaling_max_gradient": 100.0,
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
        self.optimizer.compute_residual(0.0, self.vars, self.grad, self.res)
        dual_infeas_sq, primal_infeas_sq, _ = self.optimizer.compute_kkt_error(
            self.vars, self.grad
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
        mu_floor = tol_qf * 0.1  # don't drive mu far below convergence tolerance

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

            # Mehrotra centering parameter (Nocedal-Wachter-Waltz, eq 3.5).
            # Pure data-driven: sigma small when affine step reduces comp well,
            # large when it doesn't.  Free/fixed mode toggle provides robustness.
            sigma_pc = min(1.0, max(0.0, avg_comp_aff / mu_nat) ** 3)
            mu_pc = max(sigma_pc * mu_nat, mu_floor)

            if comm_rank == 0:
                print(
                    f"  PC: sigma={sigma_pc:.4f}, "
                    f"mu={mu_pc:.4e} "
                    f"(comp={avg_comp:.4e}, "
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

    def _compute_least_squares_multipliers(self, lambda_max=1e3):
        """Least-squares constraint multiplier initialization (Section 3.6).

        Solves for lambda that minimizes the dual infeasibility norm:

            [ I  A^T ] [ w      ]   [ -(grad_f - zl + zu) ]
            [ A   0  ] [ lambda ] = [         0           ]

        The (1,1) block is I (no Hessian), making this a least-squares
        normal equation. w is discarded; lambda is the multiplier estimate.

        Safeguard: if ||lambda||_inf > lambda_max, discard and set to 0.
        """
        x = self.vars.get_solution()

        # Build RHS: -(grad_f - zl + zu) for primals, 0 for constraints
        self.optimizer.compute_dual_residual_vector(self.vars, self.grad, self.res)
        self.res.get_array()[:] *= -1.0
        self.res.copy_host_to_device()

        # Factor [I, A^T; A, 0]: W_factor=0 (no Hessian), diag=I on primals
        self.diag.zero()
        self.optimizer.set_design_vars_value(1.0, self.diag)
        self.diag.copy_host_to_device()
        self.solver.factor(0.0, x, self.diag, post_hessian=self._hessian_scaling_fn)
        self.solver.solve(self.res, self.px)

        # Safeguard: discard if multipliers are too large
        self.px.copy_device_to_host()
        px_arr = self.px.get_array()
        problem_ref = self.mpi_problem if self.distribute else self.problem
        mult_ind = np.array(problem_ref.get_multiplier_indicator(), dtype=bool)
        lam_vals = px_arr[mult_ind]
        if len(lam_vals) > 0 and np.max(np.abs(lam_vals)) > lambda_max:
            self.optimizer.set_multipliers_value(0.0, x)
            return

        self.optimizer.copy_multipliers(x, self.px)

    def _compute_affine_multipliers(self, beta_min=1.0):
        """Compute the affine scaling initial point (Section 3.6).

        Solves the KKT system with mu=0 to get the affine step, then:
          - Updates the constraint multiplier: y = y + dy
          - Updates bound duals: z = max(z + dz, beta_min)
          - Primals (x, s) are NOT changed
          - Returns the initial barrier mu = avg complementarity
        """
        x = self.vars.get_solution()
        self._update_gradient(x)

        # Solve the KKT system with mu=0 (affine direction)
        mu = 0.0
        self.optimizer.compute_residual(mu, self.vars, self.grad, self.res)
        self.optimizer.compute_diagonal(self.vars, self.diag)
        self.solver.factor(
            self._obj_scale,
            x,
            self.diag,
            post_hessian=self._hessian_scaling_fn,
        )
        self.solver.solve(self.res, self.px)

        # Extract the bound dual steps via back-substitution
        self.optimizer.compute_update(mu, self.vars, self.px, self.update)

        # Update multipliers only (y = y + dy), primals unchanged
        self.optimizer.copy_multipliers(x, self.update.get_solution())

        # Update bound duals: z = max(z + dz, beta_min)
        self.optimizer.compute_affine_start_point(
            beta_min, self.vars, self.update, self.vars
        )

        # Initial barrier = average complementarity at the new point
        barrier, _ = self.optimizer.compute_complementarity(self.vars)
        return max(barrier, beta_min)

    def _add_regularization_terms(
        self,
        diag,
        eps_x=1e-4,
        eps_z=1e-4,
        zero_hessian_indices=None,
        eps_x_zero=None,
    ):
        """Add regularization to the KKT diagonal: +eps_x for primal, -eps_z for dual."""
        diag.copy_device_to_host()
        d = diag.get_array()

        problem = self.mpi_problem if self.distribute else self.problem
        mult = np.array(problem.get_multiplier_indicator(), dtype=bool)

        d[~mult] += eps_x
        d[mult] -= eps_z

        if zero_hessian_indices is not None and eps_x_zero is not None:
            d[zero_hessian_indices] += eps_x_zero - eps_x

        diag.copy_host_to_device()

    def _zero_multipliers(self, x):
        self.optimizer.set_multipliers_value(0.0, x)

    def _update_gradient(self, x):
        """Evaluate problem functions and gradient at x."""
        alpha = self._obj_scale
        if self.distribute:
            self.mpi_problem.update(x)
            self.mpi_problem.gradient(alpha, x, self.grad)
        else:
            self.problem.update(x)
            self.problem.gradient(alpha, x, self.grad)
        self.optimizer.apply_gradient_scaling(self.grad)

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
        method which implements Algorithm IC (inertia correction with
        exponential growth of delta_w).  Otherwise, factors directly with
        a simple fallback regularization on failure.

        Returns True if factorization succeeded (correct inertia or no
        inertia check), False if inertia correction exhausted its budget.
        """
        self.diag.get_array()[:] = diag_base
        self.diag.copy_host_to_device()

        if inertia_corrector is not None and hasattr(self.solver, "assemble_hessian"):
            return inertia_corrector.factorize(
                self.solver,
                x,
                self.diag,
                diag_base,
                zero_hessian_indices,
                zero_hessian_eps,
                comm_rank,
                max_corrections=options["max_inertia_corrections"],
                inertia_tolerance=options.get("inertia_tolerance", 0),
                obj_scale=self._obj_scale,
                hessian_scaling_fn=self._hessian_scaling_fn,
            )
        else:
            try:
                self.solver.factor(
                    self._obj_scale,
                    x,
                    self.diag,
                    post_hessian=self._hessian_scaling_fn,
                )
            except Exception:
                # Factorization failed: add primal regularization and retry
                diag_arr = self.diag.get_array()
                if mult_ind is not None:
                    diag_arr[~mult_ind] += 1e-4
                else:
                    diag_arr += 1e-4
                self.diag.copy_host_to_device()
                self.solver.factor(
                    self._obj_scale,
                    x,
                    self.diag,
                    post_hessian=self._hessian_scaling_fn,
                )
            return True

    def _iterative_refinement(self, mu, mult_ind, condensed_rhs, max_steps=10):
        """Iterative refinement for the full-space KKT solve.

        Residuals are computed on the full 8-block unreduced Newton system
        (x, s, y_c, y_d, z_L, z_U, v_L, v_U) and the residual ratio uses
        the uncondensed max-norm across all blocks.  Corrections are obtained
        by condensing the residual into the augmented system, solving with
        the existing factorization, and back-substituting for bound duals.
        """
        px = self.px.get_array()
        n = len(px)

        # Current-point data (frozen for the entire IR)
        x = np.array(self.vars.get_solution().get_array())
        g = np.array(self.grad.get_array())
        zl = np.array(self.vars.get_zl())
        zu = np.array(self.vars.get_zu())
        # Use relaxed bounds, consistent with KKT matrix assembly.
        # get_lbx()/get_ubx() return the original bounds, so replicate
        # the same relaxation that the backend applied at startup.
        if hasattr(self.optimizer, "get_lbx_relaxed"):
            lbx = np.array(self.optimizer.get_lbx_relaxed())
            ubx = np.array(self.optimizer.get_ubx_relaxed())
        else:
            lbx_orig = np.array(self.optimizer.get_lbx())
            ubx_orig = np.array(self.optimizer.get_ubx())
            brf = 1e-8  # bound_relax_factor
            cvt = 1e-4  # constr_viol_tol
            lbx = lbx_orig.copy()
            ubx = ubx_orig.copy()
            fl_orig = np.isfinite(lbx_orig)
            fu_orig = np.isfinite(ubx_orig)
            if np.any(fl_orig):
                delta_l = np.minimum(
                    cvt, brf * np.maximum(1.0, np.abs(lbx_orig[fl_orig]))
                )
                lbx[fl_orig] = lbx_orig[fl_orig] - delta_l
            if np.any(fu_orig):
                delta_u = np.minimum(
                    cvt, brf * np.maximum(1.0, np.abs(ubx_orig[fu_orig]))
                )
                ubx[fu_orig] = ubx_orig[fu_orig] + delta_u

        # Index sets
        pi = np.where(~mult_ind)[0]  # primal indices (design + slacks)
        ci = np.where(mult_ind)[0]  # constraint indices (eq + ineq)

        # Gaps and finite-bound masks (primal-sized)
        gap_l = x[pi] - lbx
        gap_u = ubx - x[pi]
        fl = np.isfinite(lbx)
        fu = np.isfinite(ubx)
        gl = np.where(fl, gap_l, 1.0)  # safe for division
        gu = np.where(fu, gap_u, 1.0)

        # Sigma = zl/gap_l + zu/gap_u (barrier diagonal, primal-sized)
        sigma = np.where(fl, zl / gl, 0.0) + np.where(fu, zu / gu, 0.0)

        # Initial bound dual steps from back-substitution
        dx = px[pi]
        dzl = np.where(fl, (mu - gl * zl - zl * dx) / gl, 0.0)
        dzu = np.where(fu, (mu - gu * zu + zu * dx) / gu, 0.0)

        # Augmented matrix K for matvec (sparse, includes Sigma on diagonal)
        self.solver.hess.copy_data_device_to_host()
        hdata = np.array(self.solver.hess.get_data())
        K = csr_matrix((hdata, self.solver.cols, self.solver.rowp), shape=(n, n))

        # Full 8-block RHS (constant across IR steps)
        # Block 1 (x stationarity): rhs_x = -(grad[pi] - zl + zu)
        # Block 2 (constraint feas): rhs_c = -(grad[ci] - lbh) = condensed_rhs[ci]
        # Block 3 (compl lower):     rhs_zl = mu - gap_l * zl
        # Block 4 (compl upper):     rhs_zu = mu - gap_u * zu
        rhs_x = -(g[pi] - zl + zu)
        rhs_c = condensed_rhs[ci]
        rhs_zl = np.where(fl, mu - gl * zl, 0.0)
        rhs_zu = np.where(fu, mu - gu * zu, 0.0)

        # RHS norm for ratio (max over all 8 blocks)
        nrm_rhs = max(
            np.max(np.abs(rhs_x)) if len(rhs_x) > 0 else 0.0,
            np.max(np.abs(rhs_c)) if len(rhs_c) > 0 else 0.0,
            np.max(np.abs(rhs_zl)) if len(rhs_zl) > 0 else 0.0,
            np.max(np.abs(rhs_zu)) if len(rhs_zu) > 0 else 0.0,
        )

        # Refinement parameters
        residual_ratio_max = 1e-10
        residual_improvement_factor = 0.999999999
        min_refinement_steps = 1
        max_cond = 1e6
        residual_ratio_old = 1e30

        n_ir_steps = 0
        for step in range(max_steps):
            # Compute full 8-block residual
            Kpx = K.dot(px)

            # Block 1 (x): e_x = rhs_x - (K*px)[pi] + Sigma*dx + dzl - dzu
            #   K contains W+Sigma+delta_x, so subtract Sigma*dx to get W+delta_x part
            e_x = rhs_x - Kpx[pi] + sigma * dx + dzl - dzu

            # Block 2 (constraints): e_c = rhs_c - (K*px)[ci]
            e_c = rhs_c - Kpx[ci]

            # Block 3 (compl lower): e_zl = rhs_zl - zl*dx - gap_l*dzl
            e_zl = np.where(fl, rhs_zl - zl * dx - gl * dzl, 0.0)

            # Block 4 (compl upper): e_zu = rhs_zu + zu*dx - gap_u*dzu
            e_zu = np.where(fu, rhs_zu + zu * dx - gu * dzu, 0.0)

            # Residual ratio
            # Max-norm over ALL blocks (uncondensed)
            nrm_resid = max(
                np.max(np.abs(e_x)) if len(e_x) > 0 else 0.0,
                np.max(np.abs(e_c)) if len(e_c) > 0 else 0.0,
                np.max(np.abs(e_zl)) if len(e_zl) > 0 else 0.0,
                np.max(np.abs(e_zu)) if len(e_zu) > 0 else 0.0,
            )

            # Solution norm over ALL components (px + dzl + dzu)
            nrm_res = max(
                np.max(np.abs(px)) if len(px) > 0 else 0.0,
                np.max(np.abs(dzl)) if len(dzl) > 0 else 0.0,
                np.max(np.abs(dzu)) if len(dzu) > 0 else 0.0,
            )

            residual_ratio = nrm_resid / (min(nrm_res, max_cond * nrm_rhs) + nrm_rhs)

            if residual_ratio < residual_ratio_max and step >= min_refinement_steps:
                break

            # Stall detection (improvement factor check)
            if (
                step >= min_refinement_steps
                and residual_ratio > residual_improvement_factor * residual_ratio_old
            ):
                break
            residual_ratio_old = residual_ratio

            # Condense residual for correction solve
            # Fold complementarity residual into primal rows:
            #   e_cond[pi] = e_x + e_zl/gap_l - e_zu/gap_u
            #   e_cond[ci] = e_c
            e_cond = np.zeros(n)
            e_cond[pi] = (
                e_x + np.where(fl, e_zl / gl, 0.0) - np.where(fu, e_zu / gu, 0.0)
            )
            e_cond[ci] = e_c

            # Solve correction with existing factorization
            corr = self.solver.solve_array(e_cond)

            # Back-substitute for bound dual corrections
            dc = corr[pi]
            dzl += np.where(fl, (e_zl - zl * dc) / gl, 0.0)
            dzu += np.where(fu, (e_zu + zu * dc) / gu, 0.0)

            # Accumulate into solution
            px[:] += corr
            dx = px[pi]
            n_ir_steps += 1

        pass

    def _solve_with_mu(self, mu, inertia_corrector=None, mult_ind=None):
        """Solve the augmented KKT system for the Newton direction.

        Full-space solve flow:
          1. Build condensed RHS (8-block to 4-block)
          2. Solve augmented system: K * px = res
          3. Iterative refinement on the full unreduced 8-block system
          4. Back-substitute for bound dual steps

        Requires _factorize_kkt() to have been called first.
        """
        # Step 1: Condensed RHS (8-block to 4-block)
        self.optimizer.compute_residual(mu, self.vars, self.grad, self.res)
        self.res.copy_device_to_host()
        rhs_copy = self.res.get_array().copy()

        # Step 2: Solve augmented system
        self.solver.solve(self.res, self.px)

        # Step 3: Iterative refinement on full 8-block system
        if (
            inertia_corrector is not None
            and mult_ind is not None
            and hasattr(self.solver, "hess")
        ):
            self.px.copy_device_to_host()
            self._iterative_refinement(mu, mult_ind, rhs_copy)

        # Step 4: Back-substitute for bound duals
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

        Returns True if factorization succeeded, False if inertia
        correction failed (direction is unreliable and should not be used).
        """
        ok = self._factorize_kkt(
            x,
            diag_base,
            inertia_corrector,
            mult_ind,
            options,
            zero_hessian_indices,
            zero_hessian_eps,
            comm_rank,
        )
        if not ok:
            return False
        self._solve_with_mu(self.barrier_param, inertia_corrector, mult_ind)
        return True

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

                    pass  # SOC rejected
                except Exception:
                    pass  # SOC failed

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
                return alpha, j + 1, True
            else:
                alpha = 0.01
                self.optimizer.apply_step_update(
                    alpha * alpha_x, alpha * alpha_z, self.vars, self.update, self.temp
                )
                self._update_gradient(self.temp.get_solution())
                self.vars.copy(self.temp)
                return alpha, j + 1, True

        return alpha, max_iters, False

    def _compute_barrier_objective(self, vars):
        """Barrier objective phi_mu = f(x) - mu * sum(ln(gaps)).

        f(x) is obtained by evaluating L(x, lam=0) = alpha * f(x).
        """
        problem = self.mpi_problem if self.distribute else self.problem
        x_vec = vars.get_solution()
        x_arr = x_vec.get_array()

        # Zero multipliers, evaluate L(x,0) = f(x), restore
        mult_ind = np.array(
            (
                self.mpi_problem if self.distribute else self.problem
            ).get_multiplier_indicator(),
            dtype=bool,
        )
        lam_backup = x_arr[mult_ind].copy()
        x_arr[mult_ind] = 0.0
        x_vec.copy_host_to_device()
        f_obj = problem.lagrangian(self._obj_scale, x_vec)
        x_arr[mult_ind] = lam_backup
        x_vec.copy_host_to_device()

        barrier_log = self.optimizer.compute_barrier_log_sum(self.barrier_param, vars)
        return f_obj + barrier_log

    def _compute_filter_theta(self, vars=None):
        """Compute theta = ||c(x)||_1 (1-norm)."""
        if vars is None:
            vars = self.vars
        return self.optimizer.compute_constraint_violation_1norm(vars, self.grad)

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
        """Filter line search (Algorithm A, Wächter & Biegler 2006).

        Implements the filter line search from Wächter & Biegler 2006:
        theta uses 1-norm, Compare_le tolerance in Armijo/sufficient-decrease,
        post-acceptance filter augmentation via f-type and Armijo re-check,
        and SOC acceptance uses the original alpha_primal_test.

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
            Accepted step size (fraction of alpha_x).
        line_iters : int
            Number of backtracking iterations.
        step_accepted : bool
            False if step was rejected (triggers restoration).
        """
        # Constants
        max_iters = options["max_line_search_iterations"]
        alpha_red = options["backtracking_factor"]
        gamma_theta = options["filter_gamma_theta"]
        gamma_phi = options["filter_gamma_phi"]
        delta = options["filter_delta"]
        s_theta = options["filter_s_theta"]
        s_phi = options["filter_s_phi"]
        eta_phi = options["filter_eta_phi"]
        alpha_min_frac = 0.05
        max_soc = options["filter_max_soc"]
        kappa_soc = options["filter_kappa_soc"]
        obj_max_inc = 5.0
        use_soc = options["second_order_correction"] and mult_ind is not None
        EPS10 = 10.0 * np.finfo(float).eps

        # Reference values
        ref_theta = self._compute_filter_theta()
        ref_barr = phi_current
        ref_dphi = self.optimizer.compute_barrier_dphi(
            self.barrier_param,
            self.vars,
            self.update,
            self.res,
            self.px,
            self.diag,
        )

        # theta_min, theta_max (Eq. 21)
        theta_0 = getattr(self, "_filter_theta_0", ref_theta)
        theta_min = 1e-4 * max(1.0, theta_0)
        theta_max = 1e4 * max(1.0, theta_0)

        # Alpha_min (Eq. 23)
        alpha_min = gamma_theta
        if ref_dphi < 0.0:
            alpha_min = min(gamma_theta, gamma_phi * ref_theta / (-ref_dphi))
            if ref_theta <= theta_min:
                alpha_min = min(
                    alpha_min,
                    delta * ref_theta**s_theta / (-ref_dphi) ** s_phi,
                )
        alpha_min *= alpha_min_frac

        # f-type switching condition (Eq. 19)
        def _is_ftype(alpha_test):
            return (
                ref_dphi < 0.0
                and alpha_test * (-ref_dphi) ** s_phi > delta * ref_theta**s_theta
            )

        # Armijo condition (Eq. 20, with Compare_le)
        def _armijo_holds(trial_barr, alpha_test):
            return (
                trial_barr - ref_barr
            ) - eta_phi * alpha_test * ref_dphi <= EPS10 * abs(ref_barr)

        # Acceptable to current iterate (Eq. 18, with Compare_le)
        def _acceptable_to_iterate(trial_barr, trial_theta):
            if trial_barr > ref_barr:
                basval = 1.0
                if abs(ref_barr) > 10.0:
                    basval = np.log10(abs(ref_barr))
                if np.log10(max(trial_barr - ref_barr, 1e-300)) > obj_max_inc + basval:
                    return False
            return (
                trial_theta - (1.0 - gamma_theta) * ref_theta <= EPS10 * abs(ref_theta)
            ) or (
                (trial_barr - ref_barr) - (-gamma_phi * ref_theta)
                <= EPS10 * abs(ref_barr)
            )

        # Check acceptability of trial point
        def _check_acceptance(trial_barr, trial_theta, alpha_test):
            if trial_theta > theta_max:
                return False
            if alpha_test > 0.0 and _is_ftype(alpha_test) and ref_theta <= theta_min:
                return _armijo_holds(trial_barr, alpha_test)
            if not _acceptable_to_iterate(trial_barr, trial_theta):
                return False
            return inner_filter.is_acceptable(trial_barr, trial_theta)

        # Post-acceptance filter augmentation.
        # Augment filter unless f-type and Armijo both hold.
        def _update_filter(trial_barr, alpha_test):
            if not (_is_ftype(alpha_test) and _armijo_holds(trial_barr, alpha_test)):
                inner_filter.add(ref_barr, ref_theta)

        # SOC state backup
        if use_soc:
            self.optimizer.compute_residual(
                self.barrier_param, self.vars, self.grad, self.res
            )
            res_orig = self.res.get_array().copy()
            update_backup = self.optimizer.create_opt_vector()
            update_backup.copy(self.update)
            px_orig = self.px.get_array().copy()

        # Backtracking loop
        alpha_primal = alpha_x
        n_steps = 0

        while alpha_primal > alpha_min or n_steps == 0:
            if n_steps >= max_iters:
                break

            # Trial point
            self.optimizer.apply_step_update(
                alpha_primal, alpha_z, self.vars, self.update, self.temp
            )
            self._update_gradient(self.temp.get_solution())

            trial_theta = self._compute_filter_theta(self.temp)
            trial_barr = self._compute_barrier_objective(self.temp)

            alpha_primal_test = alpha_primal

            if _check_acceptance(trial_barr, trial_theta, alpha_primal_test):
                _update_filter(trial_barr, alpha_primal_test)
                self.vars.copy(self.temp)
                return alpha_primal / alpha_x, n_steps + 1, True

            # SOC (Algorithm A, step A-5.5): first trial, infeasibility increased
            if use_soc and n_steps == 0 and trial_theta >= ref_theta:
                self.optimizer.compute_residual(
                    self.barrier_param, self.temp, self.grad, self.res
                )
                c_soc = self.res.get_array().copy()
                c_soc[~mult_ind] = res_orig[~mult_ind]
                alpha_soc = alpha_primal
                theta_soc_old = trial_theta
                soc_accepted = False

                for soc_count in range(max_soc):
                    if soc_count > 0 and trial_theta > kappa_soc * theta_soc_old:
                        break
                    theta_soc_old = trial_theta

                    self.res.get_array()[:] = c_soc
                    self.res.copy_host_to_device()
                    self._update_gradient(self.vars.get_solution())

                    try:
                        self.solver.solve(self.res, self.px)
                    except Exception:
                        break
                    self.optimizer.compute_update(
                        self.barrier_param, self.vars, self.px, self.update
                    )

                    soc_ax, _, soc_az, _ = self.optimizer.compute_max_step(
                        tau, self.vars, self.update
                    )
                    self.optimizer.apply_step_update(
                        soc_ax, soc_az, self.vars, self.update, self.temp
                    )
                    self._update_gradient(self.temp.get_solution())

                    trial_theta = self._compute_filter_theta(self.temp)
                    trial_barr = self._compute_barrier_objective(self.temp)

                    # Acceptance uses original alpha_primal_test
                    if _check_acceptance(trial_barr, trial_theta, alpha_primal_test):
                        _update_filter(trial_barr, alpha_primal_test)
                        self.vars.copy(self.temp)
                        soc_accepted = True
                        break

                    # Accumulate: c_soc = alpha_soc * c_soc + c(trial)
                    self.optimizer.compute_residual(
                        self.barrier_param, self.temp, self.grad, self.res
                    )
                    new_c = self.res.get_array().copy()
                    c_soc[mult_ind] = alpha_soc * c_soc[mult_ind] + new_c[mult_ind]
                    alpha_soc = soc_ax

                if soc_accepted:
                    return 1.0, n_steps + 1, True

                # Restore original direction for continued backtracking
                self.update.copy(update_backup)
                self.px.get_array()[:] = px_orig
                self.px.copy_host_to_device()

            alpha_primal *= alpha_red
            n_steps += 1

        # All backtracking exhausted — trigger restoration
        self._update_gradient(self.vars.get_solution())
        return alpha_primal / alpha_x, n_steps, False

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
        """Feasibility restoration phase (Section 3.3).

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
                    if inner_filter.is_acceptable(phi_trial, theta_trial):
                        self.vars.copy(self.temp)
                        return True
                    # Not filter-acceptable, but KKT improving: accept anyway
                    # if residual decrease is significant
                    if res_trial < 0.99 * res_current:
                        self.vars.copy(self.temp)
                        # Clear filter to unblock (new barrier subproblem)
                        inner_filter.entries.clear()
                        self._filter_theta_0 = theta_trial
                        return True
                alpha *= backtrack

            # No KKT-descent step found. Since theta is already small,
            # the filter is simply clogging optimality progress.
            # Reset the filter to unblock: this is safe because theta
            # is small (constraints are nearly satisfied).
            self._update_gradient(self.vars.get_solution())
            inner_filter.entries.clear()
            self._filter_theta_0 = theta_k
            return True

        # Mode 1: Large-theta restoration (reduce constraint violation)
        phi_k = self._compute_barrier_objective(self.vars)
        inner_filter.add(phi_k, theta_k)

        target = 0.9 * theta_k
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
                    theta_trial = self._compute_filter_theta(self.temp)

                    if theta_trial < theta_k:
                        self.vars.copy(self.temp)
                        theta_k = theta_trial
                        step_taken = True
                        break
                    alpha *= backtrack

                if not step_taken:
                    break

                if theta_k < target:
                    break

            if theta_k < target:
                break

        # Restore original barrier
        self.barrier_param = saved_barrier
        if inertia_corrector:
            inertia_corrector.update_barrier(saved_barrier)
        self._update_gradient(self.vars.get_solution())
        self.optimizer.compute_diagonal(self.vars, self.diag)
        self.diag.copy_device_to_host()

        success = theta_k < theta_start * 0.99
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
        self._obj_scale = 1.0
        self._hessian_scaling_fn = None
        # self._debug_iter = True
        self.barrier_param = options["initial_barrier_param"]
        max_iters = options["max_iterations"]
        base_tau = options["fraction_to_boundary"]
        tau_min = options["tau_min"]
        use_adaptive_tau = options["adaptive_tau"]
        tol = options["convergence_tolerance"]
        record_components = options["record_components"]
        continuation_control = options["continuation_control"]

        # 2. Initialize primal-dual variables
        x = self.vars.get_solution()

        xview = None
        if not self.distribute:
            xview = ModelVector(self.model, x=x)

        # Iterate initialization sequence:
        #   1. Relax bounds
        #   2. Push x into bounds, set z = 1.0
        #   3. Initialize slacks to s = c(x), re-push into bounds
        #   4. Recompute gradient at (pushed x, initialized s)
        #   5. Least-squares constraint multiplier init
        #   6. Recompute gradient at final (x, lam) for the main loop

        # Step 1: Relax bounds (default: bound_relax_factor = 1e-8)
        self.optimizer.relax_bounds(1e-8, options["constr_viol_tol"])

        # Step 2: Project design variables into bounds, initialize z = 1.0
        self._zero_multipliers(x)
        self.optimizer.initialize_multipliers_and_slacks(
            self.barrier_param, self.grad, self.vars
        )

        # Step 3: Initialize slacks to s = d(x), then push into bounds.
        # Set s = d(x), then push into bounds.
        # Gradient is already evaluated at the pushed x from Step 2.
        # initialize_slacks recovers d(x) = grad[ci] + s_old from the
        # SlackCouplingGroup contribution (grad[ci] = c(x) - s_old).
        if self.optimizer.has_slacks():
            self._update_gradient(x)
            self.optimizer.initialize_slacks(self.grad, self.vars)
            x.copy_host_to_device()
            self.optimizer.initialize_multipliers_and_slacks(
                self.barrier_param, self.grad, self.vars
            )

        # Step 4: Recompute gradient at the pushed x with lam=0
        self._update_gradient(x)

        # Step 4b: Gradient-based NLP scaling from initial point.
        nlp_max_grad = options["nlp_scaling_max_gradient"]
        if nlp_max_grad > 0 and not self.distribute:
            self.optimizer.compute_nlp_scaling(x, self.grad, max_gradient=nlp_max_grad)
            self._obj_scale = self.optimizer.get_obj_scale()
            if self.optimizer.has_scaling():
                self._hessian_scaling_fn = self.optimizer.apply_hessian_scaling
                self._update_gradient(x)
                if comm_rank == 0:
                    print(f"  NLP scaling: obj_scale={self._obj_scale:.4e}")

        # Step 5: Least-squares constraint multiplier initialization
        if options["init_affine_step_multipliers"]:
            self._compute_least_squares_multipliers()
            self.barrier_param = self._compute_affine_multipliers(
                beta_min=self.barrier_param
            )
        elif options["init_least_squares_multipliers"]:
            self._compute_least_squares_multipliers()

        # Step 6: Recompute gradient at final (x, lam) for the main loop
        self._update_gradient(x)

        pass

        # Step size tracking (for log output)
        line_iters = 0
        alpha_x_prev = 0.0
        alpha_z_prev = 0.0
        x_index_prev = -1
        z_index_prev = -1

        # 3. Convergence tracking
        dual_inf_tol = options["dual_inf_tol"]
        constr_viol_tol = options["constr_viol_tol"]
        compl_inf_tol = options["compl_inf_tol"]
        diverging_iterates_tol = options["diverging_iterates_tol"]
        acceptable_tol = options["acceptable_tol"]
        acceptable_iter = options["acceptable_iter"]
        acceptable_dual_inf_tol = options["acceptable_dual_inf_tol"]
        acceptable_constr_viol_tol = options["acceptable_constr_viol_tol"]
        acceptable_compl_inf_tol = options["acceptable_compl_inf_tol"]
        acceptable_counter = 0
        prev_res_norm = float("inf")
        precision_floor_count = 0

        # 4. Inertia correction setup (auto-detect from solver capability)
        problem_ref = self.mpi_problem if self.distribute else self.problem
        mult_ind = np.array(problem_ref.get_multiplier_indicator(), dtype=bool)
        inertia_corrector = None
        if hasattr(self.solver, "get_inertia"):
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
        qf_mu_floor = tol * 0.1  # mu should not go far below convergence tolerance
        qf_monotone_mu = None
        qf_M_k_at_entry = None
        qf_sd = qf_sp = qf_sc = 1.0
        if quality_func and options["quality_function_norm_scaling"]:
            n_d, n_p, n_c = self.optimizer.get_kkt_element_counts()
            qf_sd = 1.0 / max(n_d, 1)
            qf_sp = 1.0 / max(n_p, 1)
            qf_sc = 1.0 / max(n_c, 1)
        self._qf_sd = qf_sd
        self._qf_sp = qf_sp
        self._qf_sc = qf_sc

        if quality_func:
            d0, p0, c0 = self.optimizer.compute_kkt_error(self.vars, self.grad)
            qf_kkt_history.append(d0 * qf_sd + p0 * qf_sp + c0 * qf_sc)

        soc_mult_ind = mult_ind if options["second_order_correction"] else None

        # 6. Filter line search setup (Algorithm A + Algorithm B outer)
        filter_ls = options["filter_line_search"]
        outer_filter = Filter() if filter_ls else None
        inner_filter = (
            Filter(
                gamma_phi=options["filter_gamma_phi"],
                gamma_theta=options["filter_gamma_theta"],
            )
            if filter_ls
            else None
        )
        filter_monotone_mode = False  # True when filter rejected step
        filter_monotone_mu = None

        if filter_ls:
            # Track theta_0 per barrier subproblem for switching condition
            self._filter_theta_0 = None  # set after first gradient eval

        res_norm_mu = self.barrier_param
        rhs_norm = 0.0

        # MAIN OPTIMIZATION LOOP
        for i in range(max_iters):
            # Step A: Evaluate KKT residual at current iterate
            res_norm = self.optimizer.compute_residual(
                self.barrier_param, self.vars, self.grad, self.res
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
                        "theta": theta,
                        "eta": eta,
                        "inertia_delta": inertia_corrector.last_delta_w,
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

            # Compute NLP-level quantities for display
            # These are at mu=0 (true NLP error, not barrier error)
            d_inf_log, p_inf_log, c_inf_log = self.optimizer.compute_kkt_error_mu(
                0.0, self.vars, self.grad
            )
            # Scaling factors for overall error (same as convergence check)
            _zl_log = np.array(self.vars.get_zl())
            _zu_log = np.array(self.vars.get_zu())
            _lbx_log = np.array(self.optimizer.get_lbx())
            _ubx_log = np.array(self.optimizer.get_ubx())
            _z_sum_log = float(np.sum(np.abs(_zl_log)) + np.sum(np.abs(_zu_log)))
            _n_bounds_log = int(
                np.sum(np.isfinite(_lbx_log)) + np.sum(np.isfinite(_ubx_log))
            )
            _xlam_log = self.vars.get_solution().get_array()
            _lam_sum_log = float(
                sum(abs(_xlam_log[j]) for j in range(len(_xlam_log)) if mult_ind[j])
            )
            _n_lam_log = int(np.sum(mult_ind))
            _s_max_log = 100.0
            _s_c_log = (
                max(_s_max_log, _z_sum_log / max(_n_bounds_log, 1)) / _s_max_log
                if _n_bounds_log > 0
                else 1.0
            )
            _n_all_log = _n_lam_log + _n_bounds_log
            _s_d_log = (
                max(_s_max_log, (_lam_sum_log + _z_sum_log) / max(_n_all_log, 1))
                / _s_max_log
                if _n_all_log > 0
                else 1.0
            )
            _overall_log = max(d_inf_log / _s_d_log, p_inf_log, c_inf_log / _s_c_log)

            iter_data["inf_pr"] = p_inf_log
            iter_data["inf_du"] = d_inf_log
            iter_data["compl"] = c_inf_log
            iter_data["nlp_error"] = _overall_log
            iter_data["objective"] = self._compute_barrier_objective(self.vars)
            iter_data["step_norm"] = float(np.max(np.abs(self.px.get_array())))

            if comm_rank == 0:
                self.write_log(i, iter_data)

            iter_data["x"] = {}
            if xview is not None:
                for name in record_components:
                    iter_data["x"][name] = xview[name].tolist()

            opt_data["iterations"].append(iter_data)

            # Step C: Convergence checks

            # Compute NLP error components at mu_target=0
            d_inf_nlp, p_inf_nlp, c_inf_nlp = self.optimizer.compute_kkt_error_mu(
                0.0, self.vars, self.grad
            )

            # Scaling factors (same as barrier update)
            s_max_conv = 100.0
            zl_conv = np.array(self.vars.get_zl())
            zu_conv = np.array(self.vars.get_zu())
            lbx_conv = np.array(self.optimizer.get_lbx())
            ubx_conv = np.array(self.optimizer.get_ubx())
            z_sum_conv = float(np.sum(np.abs(zl_conv)) + np.sum(np.abs(zu_conv)))
            n_bounds_conv = int(
                np.sum(np.isfinite(lbx_conv)) + np.sum(np.isfinite(ubx_conv))
            )
            xlam_conv = self.vars.get_solution().get_array()
            lam_sum_conv = float(
                sum(abs(xlam_conv[j]) for j in range(len(xlam_conv)) if mult_ind[j])
            )
            n_lam_conv = int(np.sum(mult_ind))

            if n_bounds_conv == 0:
                s_c_conv = 1.0
            else:
                s_c_conv = max(s_max_conv, z_sum_conv / n_bounds_conv) / s_max_conv
            n_all_conv = n_lam_conv + n_bounds_conv
            if n_all_conv == 0:
                s_d_conv = 1.0
            else:
                s_d_conv = (
                    max(s_max_conv, (lam_sum_conv + z_sum_conv) / n_all_conv)
                    / s_max_conv
                )

            overall_error = max(d_inf_nlp / s_d_conv, p_inf_nlp, c_inf_nlp / s_c_conv)

            # Primary convergence: ALL 4 conditions must hold
            if (
                overall_error <= tol
                and d_inf_nlp <= dual_inf_tol
                and p_inf_nlp <= constr_viol_tol
                and c_inf_nlp <= compl_inf_tol
            ):
                if comm_rank == 0:
                    obj_final = self._compute_barrier_objective(self.vars)
                    print(f"\n{'='*70}")
                    print(f"  Amigo converged in {i} iterations")
                    print(f"{'='*70}")
                    print(f"  Objective value         {obj_final:>20.10e}")
                    print(f"  NLP error               {overall_error:>20.10e}")
                    print(f"  Primal infeasibility    {p_inf_nlp:>20.10e}")
                    print(f"  Dual infeasibility      {d_inf_nlp:>20.10e}")
                    print(f"  Complementarity         {c_inf_nlp:>20.10e}")
                    print(f"  Barrier parameter       {self.barrier_param:>20.10e}")
                    print(f"  Total iterations        {i:>20d}")
                    print(f"{'='*70}")
                opt_data["converged"] = True
                break

            # Divergence check
            x_max = np.max(np.abs(xlam_conv))
            if x_max > diverging_iterates_tol:
                if comm_rank == 0:
                    print(f"  Diverging iterates: max |x| = {x_max:.2e}")
                break

            # Acceptable convergence
            # Check if current iterate satisfies relaxed tolerances
            is_acceptable = (
                overall_error <= acceptable_tol
                and d_inf_nlp <= acceptable_dual_inf_tol
                and p_inf_nlp <= acceptable_constr_viol_tol
                and c_inf_nlp <= acceptable_compl_inf_tol
            )
            if acceptable_iter > 0 and is_acceptable:
                acceptable_counter += 1
                if acceptable_counter >= acceptable_iter:
                    if comm_rank == 0:
                        obj_acc = self._compute_barrier_objective(self.vars)
                        print(f"\n{'='*70}")
                        print(
                            f"  Amigo converged to acceptable point in {i} iterations"
                        )
                        print(f"{'='*70}")
                        print(f"  Objective value         {obj_acc:>20.10e}")
                        print(f"  NLP error               {overall_error:>20.10e}")
                        print(f"  Primal infeasibility    {p_inf_nlp:>20.10e}")
                        print(f"  Dual infeasibility      {d_inf_nlp:>20.10e}")
                        print(f"  Complementarity         {c_inf_nlp:>20.10e}")
                        print(f"  Total iterations        {i:>20d}")
                        print(f"{'='*70}")
                    opt_data["converged"] = True
                    opt_data["acceptable"] = True
                    break
            else:
                acceptable_counter = 0

            # Precision floor: bit-identical residuals → numerical limit
            rel_change = abs(res_norm - prev_res_norm) / max(res_norm, 1e-30)
            if rel_change < 1e-14 and i > 0:
                precision_floor_count += 1
            else:
                precision_floor_count = 0
            if precision_floor_count >= 3 and is_acceptable:
                if comm_rank == 0:
                    print(
                        f"  Precision floor: residual unchanged "
                        f"for {precision_floor_count} iterations"
                    )
                opt_data["converged"] = True
                opt_data["acceptable"] = True
                opt_data["precision_floor"] = True
                break

            prev_res_norm = res_norm

            # Step D+E: Barrier update + search direction
            step_rejected = False
            factorize_ok = True
            if not filter_ls:
                if inertia_corrector:
                    inertia_corrector.update_barrier(self.barrier_param)
                else:
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

            pass

            # Barrier diagonal Sigma (eq. 13).
            # compute_diagonal writes (not accumulates) each entry.
            self.optimizer.compute_diagonal(self.vars, self.diag)
            self.diag.copy_device_to_host()
            diag_base = self.diag.get_array().copy()

            barrier_before = self.barrier_param

            # Filter LS path: monotone A-3 barrier update.
            # E_mu = max(||dual||, ||primal||, max_i|s_i*z_i - mu|)
            # Using max deviation (not avg comp) prevents premature reduction
            # when individual pairs are far from the central path.
            if filter_ls:
                if i > 0 and self.barrier_param > tol:
                    # Monotone barrier update constants
                    kappa_eps = 10.0  # barrier_tol_factor
                    kappa_mu = 0.2  # mu_linear_decrease_factor
                    theta_mu = 1.5  # mu_superlinear_decrease_power
                    s_max = 100.0  # scaling threshold
                    compl_inf_tol = 1e-4  # complementarity tolerance

                    # Barrier update while-loop
                    mu_changed = False
                    while True:
                        mu = self.barrier_param

                        # E_mu: barrier error
                        d_inf, p_inf, c_inf = self.optimizer.compute_kkt_error_mu(
                            mu, self.vars, self.grad
                        )

                        # Scaling (ComputeOptimalityErrorScaling, line 3663-3700)
                        # Count only FINITE bounds for dimensions.
                        zl_arr = np.array(self.vars.get_zl())
                        zu_arr = np.array(self.vars.get_zu())
                        lbx = np.array(self.optimizer.get_lbx())
                        ubx = np.array(self.optimizer.get_ubx())
                        z_sum = float(np.sum(np.abs(zl_arr)) + np.sum(np.abs(zu_arr)))
                        n_bounds = int(
                            np.sum(np.isfinite(lbx)) + np.sum(np.isfinite(ubx))
                        )

                        # s_c: average bound dual magnitude
                        if n_bounds == 0:
                            s_c = 1.0
                        else:
                            s_c = max(s_max, z_sum / n_bounds) / s_max

                        # s_d: average of ALL multipliers
                        xlam = self.vars.get_solution().get_array()
                        lam_sum = float(
                            sum(abs(xlam[j]) for j in range(len(xlam)) if mult_ind[j])
                        )
                        n_lam = int(np.sum(mult_ind))
                        n_all = n_lam + n_bounds
                        if n_all == 0:
                            s_d = 1.0
                        else:
                            s_d = max(s_max, (lam_sum + z_sum) / n_all) / s_max

                        # E_mu = max(dual/s_d, primal, compl/s_c)
                        # Note: primal is UNSCALED, compl IS scaled
                        e_mu = max(d_inf / s_d, p_inf, c_inf / s_c)

                        kappa_eps_mu = kappa_eps * mu

                        if e_mu > kappa_eps_mu:
                            break  # Subproblem not yet solved

                        # Compute new mu and tau
                        old_mu = mu
                        new_mu = min(kappa_mu * mu, mu**theta_mu)
                        mu_floor = min(tol, compl_inf_tol) / (kappa_eps + 1.0)
                        new_mu = max(new_mu, mu_floor)

                        if new_mu >= old_mu:
                            break  # Can't reduce further

                        self.barrier_param = new_mu
                        mu_changed = True

                        # Continue while-loop to check if we can reduce again

                    # Reset filter when mu changed
                    if mu_changed:
                        if inertia_corrector:
                            inertia_corrector.update_barrier(self.barrier_param)
                        inner_filter.entries.clear()
                        self._filter_theta_0 = self._compute_filter_theta()
                factorize_ok = self._find_direction(
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

                factorize_ok = self._find_direction(
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
                factorize_ok = self._factorize_kkt(
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
                if inertia_corrector:
                    inertia_corrector.update_barrier(self.barrier_param)

                if not qf_free_mode and factorize_ok:
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
                factorize_ok = self._find_direction(
                    x,
                    diag_base,
                    inertia_corrector,
                    mult_ind,
                    options,
                    zero_hessian_indices,
                    zero_hessian_eps,
                    comm_rank,
                )

            # Inertia correction failed: skip line search, force rejection.
            # This matches IPOPT's behavior where SolveOnce returns false
            # and the step is not taken.
            if not factorize_ok:
                step_rejected = True
                consecutive_rejections += 1
                if comm_rank == 0:
                    print(
                        f"  Inertia correction FAILED " f"({consecutive_rejections}x)"
                    )
                # Restore barrier and skip to Step G
                self.barrier_param = barrier_before
                alpha = 0.0
                line_iters = 0
                alpha_x_prev = 0.0
                alpha_z_prev = 0.0
                x_index_prev = -1
                z_index_prev = -1
                # Handle rejection escape
                if consecutive_rejections >= max_rejections:
                    new_barrier = min(self.barrier_param * barrier_inc, initial_barrier)
                    if new_barrier > self.barrier_param:
                        consecutive_rejections = 0
                        if comm_rank == 0:
                            print(
                                f"  Barrier increased: "
                                f"{self.barrier_param:.2e} -> "
                                f"{new_barrier:.2e}"
                            )
                        self.barrier_param = new_barrier
                    else:
                        consecutive_rejections = 0
                        if comm_rank == 0:
                            print(
                                f"  Barrier at max "
                                f"({self.barrier_param:.2e}), "
                                f"cannot increase further"
                            )
                continue

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
                    pass  # solve accuracy check (silent)

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
                    # Eq. 16: reset bound multipliers to prevent divergence.
                    # Clamps z_i to [mu/(kappa*gap), kappa*mu/gap] with kappa=1e10.
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
                    else:
                        pass  # comp at floor, skip monotone

                # Handle rejection escape: increase barrier after too many rejections
                if consecutive_rejections >= max_rejections:
                    new_barrier = min(self.barrier_param * barrier_inc, initial_barrier)
                    if new_barrier > self.barrier_param:
                        consecutive_rejections = 0
                        if comm_rank == 0:
                            print(
                                f"  Barrier increased: {self.barrier_param:.2e} -> "
                                f"{new_barrier:.2e}"
                            )
                        self.barrier_param = new_barrier
                        if quality_func and not qf_free_mode:
                            qf_monotone_mu = new_barrier
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
        if comm_rank == 0 and not opt_data.get("converged", False):
            print(f"\n{'='*70}")
            print(f"  Amigo did NOT converge (max iterations: {max_iters})")
            print(f"{'='*70}")
            print(f"  Residual                {res_norm:>20.10e}")
            print(f"  Barrier parameter       {self.barrier_param:>20.10e}")
            print(f"{'='*70}")

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
        self.solver.factor(
            self._obj_scale,
            x,
            self.diag,
            post_hessian=self._hessian_scaling_fn,
        )

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
