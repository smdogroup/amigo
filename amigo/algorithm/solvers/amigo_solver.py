"""Amigo's native LDL direct solver for the KKT system.

Wraps the C++ SparseLDL factorization. Supports inertia queries,
which lets it drive Algorithm IC inertia correction.
"""

import numpy as np

from . import DirectSparseSolver
from amigo import SolverType, SparseLDL


class AmigoSolver(DirectSparseSolver):
    """Direct KKT solver using Amigo's native LDL factorization."""

    supports_inertia = True
    solver_name = "am.SparseLDL"

    def __init__(self, problem, ustab=0.01, pivot_tol=1e-14):
        self._init_sparse_structure(problem)
        self.ldl = SparseLDL(self.hess, SolverType.LDL, ustab, pivot_tol)

    def _do_factor(self):
        """Run the LDL numerical factorization on self.hess."""
        flag = self.ldl.factor()
        if flag != 0:
            raise RuntimeError(
                f"{self.solver_name} factorization failed with flag = {flag}"
            )

        if self.verbose:
            npos, nneg = self.ldl.get_inertia()
            print(
                f"  [LDL] factor flag={flag}, npos={npos}, nneg={nneg}, total={npos + nneg}"
            )

    def solve(self, bx, px):
        """Solve K x = bx using the stored LDL factorization."""
        bx.copy_device_to_host()
        b_arr = bx.get_array().copy()
        px.get_array()[:] = bx.get_array()[:]
        self.ldl.solve(px)
        px.copy_host_to_device()

        if self.verbose:
            b_nrm = np.max(np.abs(b_arr))
            x_nrm = np.max(np.abs(px.get_array()))
            print(f"  [LDL solve] ||b||_inf={b_nrm:.3e}, ||x||_inf={x_nrm:.3e}")

    def get_inertia(self):
        """Return (n_positive, n_negative) from the factorization."""
        return self.ldl.get_inertia()
