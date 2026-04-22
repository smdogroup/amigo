from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import csr_matrix


class LinearSolver(ABC):
    """
    Minimal contract: any KKT solver.
    """

    @abstractmethod
    def factor(self, alpha, x, diag):
        pass

    @abstractmethod
    def solve(self, bx, px):
        pass

    supports_inertia = False

    def get_inertia(self):
        raise NotImplementedError("This solver does not support inertia queries")


class DirectSparseSolver(LinearSolver):
    """Base class for direct solvers with an assembled CSR KKT matrix.

    Subclasses must set in __init__ (via _init_sparse_structure):
        self.problem         # the optimization problem
        self.hess            # CSR matrix handle
        self.rowp, self.cols # CSR structure arrays
        self._diag_indices   # diagonal data-array indices

    Subclasses must implement _do_factor() (the numerical factorization).
    """

    verbose = False  # gate debug prints

    @abstractmethod
    def _do_factor(self):
        """Numerical factorization of self.hess.

        Called after the base class has assembled H, applied any post-hessian
        scaling, added the diagonal, and copied self.hess to host.  Subclass
        should factor whatever is now in self.hess and raise on failure.
        """
        pass

    def factor(self, alpha, x, diag, post_hessian=None):
        """Assemble Hessian, add diag, and factor (used for inertia retries)."""
        self.problem.hessian(alpha, x, self.hess)
        if post_hessian is not None:
            self.hess.copy_data_device_to_host()
            post_hessian(self.hess)
        self.problem.add_diagonal(diag, self.hess)
        self.hess.copy_data_device_to_host()
        self._do_factor()

    def add_diagonal_and_factor(self, diag):
        """Add diag to already-assembled hess and factor in place."""
        self.problem.add_diagonal(diag, self.hess)
        self.hess.copy_data_device_to_host()
        self._do_factor()

    def _init_sparse_structure(self, problem, loc=None):
        """Allocate the KKT matrix and cache its CSR structure."""
        from amigo import MemoryLocation

        self.problem = problem
        self.hess = problem.create_matrix(
            loc if loc is not None else MemoryLocation.HOST_AND_DEVICE
        )
        self.nrows, self.ncols, self.nnz, self.rowp, self.cols = (
            self.hess.get_nonzero_structure()
        )
        self._diag_indices = self._find_diag_indices(self.rowp, self.cols, self.nrows)

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

    @staticmethod
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

    def iterative_refinement(
        self,
        mu,
        mult_ind,
        rhs_saved,
        vars_,
        grad,
        lbx,
        ubx,
        px,
        res,
        corr,
        max_steps=10,
    ):
        """Iterative refinement on the full 8-block unreduced Newton system.

        Computes residuals on the full system (x, s, y_c, y_d, z_L, z_U,
        v_L, v_U) and corrects the solution by condensing the residual
        into the augmented system, solving with the existing factorization
        (via self.solve), and back-substituting for bound duals.

        Requires that factor() has been called and solve() already
        produced the initial px.  The res vector is reused for each
        correction solve; corr receives the correction.

        Parameters
        ----------
        mu : float
            Barrier parameter.
        mult_ind : np.ndarray[bool]
            True where the row corresponds to a multiplier (constraint).
        rhs_saved : np.ndarray
            The condensed RHS saved before the first solve.
        vars_ : optimizer variables
            Exposes get_zl(), get_zu(), get_sl(), get_su().
        grad : Vector
            Current gradient.
        lbx, ubx : np.ndarray
            Relaxed variable bounds consistent with KKT matrix assembly.
        px : Vector
            Direction to refine (modified in place).
        res, corr : Vector
            Scratch vectors; res is reused for each correction solve.
        max_steps : int
            Maximum refinement iterations.
        """
        px_arr = px.get_array()
        n = len(px_arr)

        # Current-point data (frozen for the entire IR)
        g = np.array(grad.get_array())
        zl = np.array(vars_.get_zl())
        zu = np.array(vars_.get_zu())

        # Index sets
        pi = np.where(~mult_ind)[0]  # primal indices (design + slacks)
        ci = np.where(mult_ind)[0]  # constraint indices (eq + ineq)

        # Stored slacks (always positive, updated incrementally in C++)
        sl = np.array(vars_.get_sl())
        su = np.array(vars_.get_su())
        fl = np.isfinite(lbx)
        fu = np.isfinite(ubx)
        gl = np.where(fl, sl, 1.0)
        gu = np.where(fu, su, 1.0)

        # Sigma = zl/gap_l + zu/gap_u (barrier diagonal, primal-sized)
        sigma = np.where(fl, zl / gl, 0.0) + np.where(fu, zu / gu, 0.0)

        # Initial bound dual steps from back-substitution
        dx = px_arr[pi]
        dzl = np.where(fl, (mu - gl * zl - zl * dx) / gl, 0.0)
        dzu = np.where(fu, (mu - gu * zu + zu * dx) / gu, 0.0)

        # Augmented matrix K for matvec (sparse, includes Sigma on diagonal)
        self.hess.copy_data_device_to_host()
        hdata = np.array(self.hess.get_data())
        K = csr_matrix((hdata, self.cols, self.rowp), shape=(n, n))

        # Full 8-block RHS (constant across IR steps)
        rhs_x = -(g[pi] - zl + zu)
        rhs_c = rhs_saved[ci]
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

        # TODO all of these need to have their GPU equivalent
        for step in range(max_steps):
            Kpx = K.dot(px_arr)

            e_x = rhs_x - Kpx[pi] + sigma * dx + dzl - dzu
            e_c = rhs_c - Kpx[ci]
            e_zl = np.where(fl, rhs_zl - zl * dx - gl * dzl, 0.0)
            e_zu = np.where(fu, rhs_zu + zu * dx - gu * dzu, 0.0)

            nrm_resid = max(
                np.max(np.abs(e_x)) if len(e_x) > 0 else 0.0,
                np.max(np.abs(e_c)) if len(e_c) > 0 else 0.0,
                np.max(np.abs(e_zl)) if len(e_zl) > 0 else 0.0,
                np.max(np.abs(e_zu)) if len(e_zu) > 0 else 0.0,
            )
            nrm_res = max(
                np.max(np.abs(px_arr)) if len(px_arr) > 0 else 0.0,
                np.max(np.abs(dzl)) if len(dzl) > 0 else 0.0,
                np.max(np.abs(dzu)) if len(dzu) > 0 else 0.0,
            )

            residual_ratio = nrm_resid / (min(nrm_res, max_cond * nrm_rhs) + nrm_rhs)

            if residual_ratio < residual_ratio_max and step >= min_refinement_steps:
                break
            if (
                step >= min_refinement_steps
                and residual_ratio > residual_improvement_factor * residual_ratio_old
            ):
                break
            residual_ratio_old = residual_ratio

            # Condense residual for correction solve
            e_cond = np.zeros(n)
            e_cond[pi] = (
                e_x + np.where(fl, e_zl / gl, 0.0) - np.where(fu, e_zu / gu, 0.0)
            )
            e_cond[ci] = e_c

            # Solve correction with existing factorization
            res.get_array()[:] = e_cond
            res.copy_host_to_device()
            self.solve(res, corr)
            corr.copy_device_to_host()
            corr_arr = corr.get_array()

            # Back-substitute for bound dual corrections
            dc = corr_arr[pi]
            dzl += np.where(fl, (e_zl - zl * dc) / gl, 0.0)
            dzu += np.where(fu, (e_zu + zu * dc) / gu, 0.0)

            # Accumulate into solution
            px_arr[:] += corr_arr
            dx = px_arr[pi]

        px.copy_host_to_device()
