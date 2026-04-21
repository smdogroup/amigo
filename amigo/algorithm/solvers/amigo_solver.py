"""Amigo's native LDL direct solver for the KKT system.

Wraps the C++ SparseLDL factorization. Supports inertia queries,
which lets it drive Algorithm IC inertia correction.
"""

import numpy as np

from . import DirectSparseSolver

from amigo import MemoryLocation, SolverType, SparseLDL


class AmigoSolver(DirectSparseSolver):
    """Direct KKT solver using Amigo's native LDL factorization."""

    supports_inertia = True

    def __init__(self, problem, ustab=0.01, pivot_tol=1e-14):

        self._init_sparse_structure(problem)

        stype = SolverType.LDL
        self.ldl = SparseLDL(self.hess, stype, ustab, pivot_tol)

    def add_diagonal_and_factor(self, diag):
        """Add diagonal to the already-assembled Hessian and factorize.

        Must be called after assemble_hessian().  Modifies self.hess
        in-place, so a subsequent retry must use factor() (which
        re-assembles from scratch).
        """
        self.problem.add_diagonal(diag, self.hess)
        self.hess.copy_data_device_to_host()
        flag = self.ldl.factor()
        if flag != 0:
            raise RuntimeError(f"am.SparseLDL factorization failed with flag = {flag}")

    def factor(self, alpha, x, diag, post_hessian=None):
        """Assemble the Hessian, add the diagonal, and factorize in one shot.

        Used during inertia-correction retries where we need a fresh
        assembly before re-factorizing with a larger regularization.
        """
        self.problem.hessian(alpha, x, self.hess)
        if post_hessian is not None:
            self.hess.copy_data_device_to_host()
            post_hessian(self.hess)
        self.problem.add_diagonal(diag, self.hess)
        self.hess.copy_data_device_to_host()
        flag = self.ldl.factor()
        if flag != 0:
            raise RuntimeError(f"am.SparseLDL factorization failed with flag = {flag}")

        # Debug print: factorization success flag + inertia counts.
        # Useful when SparseLDL disagrees with the expected (n_primal+, n_dual-)
        # signature during inertia correction.
        npos, nneg = self.ldl.get_inertia()
        print(
            f"  [LDL] factor flag={flag}, npos={npos}, nneg={nneg}, total={npos+nneg}"
        )

    def solve(self, bx, px):
        """Solve K x = bx using the stored LDL factorization."""
        bx.copy_device_to_host()
        b_arr = bx.get_array().copy()
        px.get_array()[:] = bx.get_array()[:]
        self.ldl.solve(px)
        px.copy_host_to_device()

        # Debug print: RHS and solution infinity norms.
        # A solution magnitude wildly larger than the RHS (or a residual
        # norm close to ||x|| itself) is a sign that the solve is inaccurate.
        px_arr = px.get_array()
        b_nrm = np.max(np.abs(b_arr))
        x_nrm = np.max(np.abs(px_arr))
        print(f"  [LDL solve] ||b||_inf={b_nrm:.3e}, ||x||_inf={x_nrm:.3e}")

    def get_inertia(self):
        """Return (n_positive, n_negative) eigenvalue counts from the factorization."""
        return self.ldl.get_inertia()
