import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from mdgo import mdgo
import matplotlib.pylab as plt
import niceplots


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
        self.xctrl_weights = np.ones(self.xctrl_indices.shape)
        self.xctrl_weights[0] = 0.5
        self.xctrl_weights[-1] = 0.5

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

        xctrl = x[self.xctrl_indices]
        lagrange += 0.5 * np.sum(self.xctrl_weights * xctrl**2)

        return lagrange

    def gradient(self, x):
        g = self.cart.gradient(x)

        xctrl = x[self.xctrl_indices]
        g[self.xctrl_indices] += self.xctrl_weights * xctrl

        return g

    def hessian(self, x):
        self.cart.hessian(x, self.mat_obj)
        data = self.mat_obj.get_data()

        for k, row in enumerate(self.xctrl_indices):
            for jp in range(self.rowp[row], self.rowp[row + 1]):
                if self.cols[jp] == row:
                    data[jp] += self.xctrl_weights[k]
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

            print("||g[%3d]||: " % (i), gnrm)
            if gnrm < 1e-10:
                break

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

        with plt.style.context(niceplots.get_style()):
            data = {}
            data["Cart pos."] = d
            data["Pole angle"] = (180 / np.pi) * theta
            data["Control force"] = xctrl

            fig, ax = niceplots.stacked_plots(
                "Time (s)",
                t,
                [data],
                lines_only=True,
                figsize=(10, 6),
                line_scaler=0.5,
            )

            fontname = "Helvetica"
            for axis in ax:
                axis.xaxis.label.set_fontname(fontname)
                axis.yaxis.label.set_fontname(fontname)

                # Update tick labels
                for tick in axis.get_xticklabels():
                    tick.set_fontname(fontname)

                for tick in axis.get_yticklabels():
                    tick.set_fontname(fontname)

            fig.savefig("cart_stacked.svg")
            fig.savefig("cart_stacked.png")
        return

    def plot_convergence(self, gnrms):

        with plt.style.context(niceplots.get_style()):
            fig, ax = plt.subplots(1, 1)

            ax.semilogy(gnrms, marker="o", clip_on=False, lw=2.0)
            ax.set_ylabel("KKT residual norm")
            ax.set_xlabel("Iteration")

            niceplots.adjust_spines(ax)

            fontname = "Helvetica"
            ax.xaxis.label.set_fontname(fontname)
            ax.yaxis.label.set_fontname(fontname)

            # Update tick labels
            for tick in ax.get_xticklabels():
                tick.set_fontname(fontname)

            for tick in ax.get_yticklabels():
                tick.set_fontname(fontname)

            fig.savefig("cart_residual_norm.svg")
            fig.savefig("cart_residual_norm.png")

    def visualize(self, x, L=0.5):
        with plt.style.context(niceplots.get_style()):

            d = x[0 : self.ncomp * self.N : self.ncomp]
            theta = x[1 : self.ncomp * self.N : self.ncomp]

            # Create the time-lapse visualization
            fig, ax = plt.subplots(1, figsize=(10, 4.5))
            ax.axis("equal")
            ax.axis("off")

            values = np.linspace(0, 1.0, d.shape[0])
            cmap = plt.get_cmap("viridis")

            hx = 0.03
            hy = 0.03
            xpts = []
            ypts = []
            for i in range(0, d.shape[0]):
                color = cmap(values[i])

                x1 = d[i]
                y1 = 0.0
                x2 = d[i] + L * np.sin(theta[i])
                y2 = -L * np.cos(theta[i])

                xpts.append(x2)
                ypts.append(y2)

                if i % 3 == 0:
                    ax.plot([x1, x2], [y1, y2], linewidth=2, color=color)
                    ax.fill(
                        [x1 - hx, x1 + hx, x1 + hx, x1 - hx, x1 - hx],
                        [y1, y1, y1 + hy, y1 + hy, y1],
                        alpha=0.5,
                        linewidth=2,
                        color=color,
                    )

                ax.plot([x2], [y2], color=color, marker="o")

            fig.savefig("cart_pole_history.svg")
            fig.savefig("cart_pole_history.png")

    def plot_nnz(self, x):
        H = self.hessian(x)

        plt.spy(H, markersize=1)

        # plt.imshow()


problem = CartProblem()

x, gnrms = problem.optimize()

problem.plot(x)
problem.plot_convergence(gnrms)
problem.visualize(x)

plt.show()
