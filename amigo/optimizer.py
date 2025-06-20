import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from .amigo import QuasidefCholesky


class Optimizer:
    def __init__(self, model, prob, x_init=None):
        self.model = model
        self.prob = prob
        self.x_init = x_init

        self.num_variables = 3421
        self.num_constraints = 1612

        self.x = self.prob.create_vector()
        self.g = self.prob.create_vector()

        self.diag = self.prob.create_vector()
        self.mat_obj = self.prob.create_csr_matrix()

        self.chol = QuasidefCholesky(self.diag, self.mat_obj)
        self.nrows, self.ncols, self.nnz, self.rowp, self.cols = (
            self.mat_obj.get_nonzero_structure()
        )

        self.check()

    def check(self, dh=1e-7):
        # x = self.x.get_array()
        # if self.x_init is not None:
        #     x[:] = self.x_init

        # px = np.random.uniform(size=x.shape)

        # g1 = self.gradient().copy()
        # jac = self.hessian()
        # ans = jac @ px

        # x[:] += dh * px

        # g2 = self.gradient().copy()
        # fd = (g2 - g1) / dh

        # err = fd - ans

        # print("Max absolute error = ", np.max(np.absolute(err)))
        # print("Max relative error = ", np.max(np.absolute(err / fd)))

        return

    def gradient(self):
        self.prob.gradient(self.x, self.g)
        return self.g.get_array()

    def hessian(self):
        self.prob.hessian(self.x, self.mat_obj)
        data = self.mat_obj.get_data()
        jac = csr_matrix((data, self.cols, self.rowp), shape=(self.nrows, self.ncols))
        return jac

    def factor(self, max_try=10):
        # d = self.diag.get_array()
        # d[:] *= 0.25

        for i in range(max_try):
            print("starting factorization")
            info = self.chol.factor()
            print("done factorization")
            if info == 0:
                break
            else:
                print("Failed with infro = ", info)
                d = self.diag.get_array()

                ndof = self.num_variables - self.num_constraints
                for k in range(ndof):
                    d[k] += 1 * (i + 1)
                for k in range(ndof, self.num_variables):
                    d[k] += 1 * (i + 1)
        if info != 0:
            raise RuntimeError("Factorization failed")

    def optimize(self, max_iters=10):
        x = self.x.get_array()
        if self.x_init is not None:
            x[:] = self.x_init

        gnrms = []

        for i in range(max_iters):
            g = self.gradient()

            gnrm = np.linalg.norm(g)
            gnrms.append(gnrm)

            print("||g[%3d]||: " % (i), gnrm)
            if gnrm < 1e-10:
                break

            self.hessian()
            self.factor()
            self.chol.solve(self.g)
            # p = spsolve(H, g)

            if i < 20:
                x[:] -= 0.01 * g[:]
            elif gnrm < 100.0:
                x[:] -= g[:]
            else:
                x[:] -= 0.1 * g[:]

        return x[:], gnrms
