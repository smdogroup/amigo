import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from mdgo import mdgo
import matplotlib.pylab as plt
import niceplots

# niceplots.set_style("light")


class CartProblem:
    def __init__(self):
        self.N = 201
        self.tf = 2.0
        self.cart = mdgo.CartPoleProblem(self.N, self.tf)

        self.ndof = self.cart.get_num_dof()

        self.mat_obj = self.cart.create_csr_matrix()
        self.nrows, self.ncols, self.nnz, self.rowp, self.cols = (
            self.mat_obj.get_nonzero_structure()
        )

        self.ncomp = 13  # 3 * number of states + 1

        self.xctrl_indices = np.arange(8, self.ncomp * self.N, self.ncomp)

    def get_init_point(self):
        x = np.zeros(self.ndof)

        # Set the initial guess
        x[0 : self.ncomp * self.N : self.ncomp] = np.linspace(0, 2.0, self.N)
        x[1 : self.ncomp * self.N : self.ncomp] = np.linspace(0, np.pi, self.N)
        x[2 : self.ncomp * self.N : self.ncomp] = 1.0
        x[3 : self.ncomp * self.N : self.ncomp] = 1.0

        return x

    def lagrangian(self, x):
        lagrange = self.cart.gradient(x)

        xctrl = x[8 : self.ncomp : self.ncomp * self.N]
        lagrange += 0.5 * np.dot(xctrl, xctrl)

        return lagrange

    def gradient(self, x):
        g = self.cart.gradient(x)

        xctrl = x[self.xctrl_indices]
        g[self.xctrl_indices] += xctrl

        return g

    def hessian(self, x):
        self.cart.hessian(x, self.mat_obj)
        data = self.mat_obj.get_data()

        for row in self.xctrl_indices:
            for jp in range(self.rowp[row], self.rowp[row + 1]):
                if self.cols[jp] == row:
                    data[jp] += 1.0
                    break

        jac = csr_matrix((data, self.cols, self.rowp), shape=(self.nrows, self.ncols))

        return jac

    def test(self, x, p=None, dh=1e-6):
        if p is None:
            p = np.random.uniform(size=x.shape)

        ans = self.hessian(x) @ p
        fd = 0.5 * (self.gradient(x + dh * p) - self.gradient(x - dh * p)) / dh

        rel_err = (ans - fd) / fd
        max_rel_err = np.max(np.absolute(rel_err))
        print("max relative error: ", max_rel_err)

        return

    def optimize(self):

        x = self.get_init_point()

        gnrms = []

        for i in range(500):
            g = self.gradient(x)

            gnrm = np.linalg.norm(g)
            gnrms.append(gnrm)

            if gnrm < 1e-10:
                break

            print("||g[%3d]||: " % (i), gnrm)
            H = self.hessian(x)
            p = spsolve(H, g)

            if i < 10:
                x -= 0.01 * p
            elif gnrm < 100.0:
                x -= p
            else:
                x -= 0.1 * p

        return x, gnrms

    def plot(self, x):

        t = np.linspace(0, self.tf, self.N)

        d = x[0 : self.ncomp * self.N : self.ncomp]
        theta = x[1 : self.ncomp * self.N : self.ncomp]
        v = x[2 : self.ncomp * self.N : self.ncomp]
        xctrl = x[8 : self.ncomp * self.N : self.ncomp]

        with plt.style.context(niceplots.get_style("doumont-light")):
            fig, ax = plt.subplots(3, 1, figsize=(10, 8))
            # plt.subplots_adjust(hspace=1.0)

            ax[0].plot(t, d)
            ax[0].set_ylabel("Cart pos.")

            ax[1].plot(t, (180 / np.pi) * theta)
            ax[1].set_ylabel("Pole angle")

            ax[2].plot(t, xctrl)
            ax[2].set_ylabel("Control force")
            ax[2].set_xlabel("time")

            for axis in ax:
                niceplots.adjust_spines(axis)

        return

    def plot_convergence(self, gnrms):

        with plt.style.context(niceplots.get_style("doumont-light")):
            fig, ax = plt.subplots(1, 1)

            ax.semilogy(gnrms)
            ax.set_ylabel(r"$||\nabla \mathcal{L}||_{2}$")
            ax.set_xlabel("Iteration")

            niceplots.adjust_spines(ax)

    def plot_nnz(self, x):
        H = self.hessian(x)

        plt.spy(H, markersize=1)

        # plt.imshow()


problem = CartProblem()

x, gnrms = problem.optimize()

problem.plot(x)
problem.plot_convergence(gnrms)

plt.show()
