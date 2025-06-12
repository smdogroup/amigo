import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

class Optimizer:
    def __init__(self, model, prob, x_init=None):
        self.model = model
        self.prob = prob
        self.x_init = x_init

        self.x = self.prob.create_vector()
        self.g = self.prob.create_vector()

        self.mat_obj = self.prob.create_csr_matrix()
        self.nrows, self.ncols, self.nnz, self.rowp, self.cols = (
            self.mat_obj.get_nonzero_structure()
        )

        self.check()

    def check(self, dh=1e-7):
        x = self.x.get_array()
        if self.x_init is not None:
            x[:] = self.x_init

        px = np.random.uniform(size=x.shape)

        g1 = self.gradient().copy()
        jac = self.hessian()
        ans = jac @ px

        x[:] += dh * px

        g2 = self.gradient().copy()
        fd = (g2 - g1) / dh

        err = fd - ans

        print("Max absolute error = ", np.max(np.absolute(err)))
        print("Max relative error = ", np.max(np.absolute(err / fd)))

        return

    def gradient(self):
        self.prob.gradient(self.x, self.g)

        return self.g.get_array()

    def hessian(self):
        self.prob.hessian(self.x, self.mat_obj)
        data = self.mat_obj.get_data()
        jac = csr_matrix((data, self.cols, self.rowp), shape=(self.nrows, self.ncols))

        return jac

    def optimize(self):
        x = self.x.get_array()
        if self.x_init is not None:
            x[:] = self.x_init

        gnrms = []

        for i in range(500):
            g = self.gradient()

            gnrm = np.linalg.norm(g)
            gnrms.append(gnrm)

            print("||g[%3d]||: " % (i), gnrm)
            if gnrm < 1e-10:
                break

            H = self.hessian()
            p = spsolve(H, g)

            if i < 20:
                x[:] -= 0.01 * p
            elif gnrm < 100.0:
                x[:] -= p
            else:
                x[:] -= 0.1 * p

        return x[:], gnrms