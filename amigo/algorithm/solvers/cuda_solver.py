from amigo import MemoryLocation

from . import LinearSolver

# TODO: Current GPU solver path (CSRMatFactorCuda):
#   - No inertia query: cannot drive the IPM inertia-correction (IC) step,
#     so supports_inertia must stay False and we fall back to a heuristic
#     regularization loop instead of a real (n_pos, n_neg, n_zero) count.
#   - Only static pivoting with perturbation: no MUMPS-style delayed
#     pivots across the elimination tree, so near-singular or highly
#     indefinite KKT systems (small mu, rank-deficient constraints) can
#     get silently perturbed and lose accuracy.
#   - No tunable threshold pivoting (no analogue of MUMPS CNTL(1)) and no
#     reporting of perturbed/delayed pivot counts -> hard to diagnose
#     numerical trouble.
#   - No Schur complement, no multi-GPU, no out-of-core.
#   - GPU-resident only.


class DirectCudaSolver(LinearSolver):
    def __init__(self, problem, pivot_eps=1e-12):
        self.problem = problem

        try:
            from amigo.amigo import CSRMatFactorCuda
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
