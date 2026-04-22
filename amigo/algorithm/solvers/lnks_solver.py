import numpy as np
from scipy.sparse.linalg import splu
from amigo import MemoryLocation
from amigo.utils import tocsr
from . import LinearSolver


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


class LNKSInexactSolver(LinearSolver):
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
