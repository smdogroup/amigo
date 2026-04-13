from . import LinearSolver
from amigo import MemoryLocation, SolverType, SparseLDL


class AmigoSolver(LinearSolver):
    """Use the native Amigo LDL solver"""

    def __init__(self, problem, ustab=0.01, pivot_tol=1e-14):
        self.problem = problem
        loc = MemoryLocation.HOST_AND_DEVICE
        self.hess = self.problem.create_matrix(loc)

        # Create the sparse LDL solver
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
        flag = self.ldl.factor()
        if flag != 0:
            raise RuntimeError(f"am.SparseLDL factorization failed with flag = {flag}")

    def solve(self, bx, px):
        """Solve the KKT system."""
        bx.copy_device_to_host()
        px.get_array()[:] = bx.get_array()[:]
        self.ldl.solve(px)
        px.copy_host_to_device()

    def get_inertia(self):
        """Return (n_positive, n_negative)"""
        return self.ldl.get_inertia()
