from . import DirectSparseSolver
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu


class DirectScipySolver(DirectSparseSolver):

    supports_inertia = False
    solver_name = "scipy.splu"

    def __init__(self, problem):
        self._init_sparse_structure(problem)
        self.lu = None

    def _do_factor(self):
        """Factor self.hess via SuperLU (scipy.splu)."""
        H = csr_matrix(
            (self.hess.get_data(), self.cols, self.rowp),
            shape=(self.nrows, self.ncols),
        ).tocsc()
        self.lu = splu(H, permc_spec="COLAMD", diag_pivot_thresh=1.0)

    def solve(self, bx, px):
        """Solve the KKT system."""
        bx.copy_device_to_host()
        px.get_array()[:] = self.lu.solve(bx.get_array())
        px.copy_host_to_device()
