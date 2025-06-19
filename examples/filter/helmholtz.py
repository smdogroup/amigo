import amigo as am
import numpy as np  # used for plotting/analysis
import argparse
import time


def eval_shape_funcs(xi, eta):
    N = 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )
    Nxi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
    Neta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])

    return N, Nxi, Neta


def dot(N, u):
    return N[0] * u[0] + N[1] * u[1] + N[2] * u[2] + N[3] * u[3]


def compute_detJ(xi, eta, X, Y, vars):
    N, N_xi, N_ea = eval_shape_funcs(xi, eta)

    x_xi = dot(N_xi, X)
    x_ea = dot(N_ea, X)

    y_xi = dot(N_xi, Y)
    y_ea = dot(N_ea, Y)

    vars["detJ"] = x_xi * y_ea - x_ea * y_xi

    return x_xi, x_ea, y_xi, y_ea


def compute_shape_derivs(xi, eta, X, Y, vars):
    N, N_xi, N_ea = eval_shape_funcs(xi, eta)

    x_xi, x_ea, y_xi, y_ea = compute_detJ(xi, eta, X, Y, vars)
    detJ = vars["detJ"]

    vars["invJ"] = [[y_ea / detJ, -x_ea / detJ], [-y_xi / detJ, x_xi / detJ]]
    invJ = vars["invJ"]

    vars["Nx"] = [
        invJ[0, 0] * N_xi[0] + invJ[1, 0] * N_ea[0],
        invJ[0, 0] * N_xi[1] + invJ[1, 0] * N_ea[1],
        invJ[0, 0] * N_xi[2] + invJ[1, 0] * N_ea[2],
        invJ[0, 0] * N_xi[3] + invJ[1, 0] * N_ea[3],
    ]

    vars["Ny"] = [
        invJ[0, 1] * N_xi[0] + invJ[1, 1] * N_ea[0],
        invJ[0, 1] * N_xi[1] + invJ[1, 1] * N_ea[1],
        invJ[0, 1] * N_xi[2] + invJ[1, 1] * N_ea[2],
        invJ[0, 1] * N_xi[3] + invJ[1, 1] * N_ea[3],
    ]

    return N, N_xi, N_ea


def filter_factory(pt: int):
    qpts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

    def init_func(self):
        am.Component.__init__(self)
        self.xi = qpts[pt % 2]
        self.eta = qpts[pt // 2]

        # The filter radius
        self.add_constant("r_filter", 0.1)

        # The x/y coordinates
        self.add_data("x_coord", shape=(4,))
        self.add_data("y_coord", shape=(4,))

        # The implicit topology input/output
        self.add_input("rho", shape=(4,))

        # Add the residual
        self.add_objective("obj")

        return

    def compute(self):
        r = self.constants["r_filter"]
        rho = self.inputs["rho"]

        X = self.data["x_coord"]
        Y = self.data["y_coord"]

        N, N_xi, N_ea = compute_shape_derivs(self.xi, self.eta, X, Y, self.vars)

        Nx = self.vars["Nx"]
        Ny = self.vars["Ny"]

        self.vars["x"] = dot(N, X)
        self.vars["y"] = dot(N, Y)

        self.vars["rho0"] = dot(N, rho)
        self.vars["rho_x"] = dot(Nx, rho)
        self.vars["rho_y"] = dot(Ny, rho)

        x = self.vars["x"]
        y = self.vars["y"]
        rho0 = self.vars["rho0"]
        rho_x = self.vars["rho_x"]
        rho_y = self.vars["rho_y"]

        detJ = self.vars["detJ"]

        rhs = 1.0 - (x - 1) ** 2 + (y - 1) ** 2
        laplace = r * r * (rho_x * rho_x + rho_y * rho_y)

        self.objective["obj"] = 0.5 * detJ * (rho0 * rho0 - rhs * rho0 + laplace)

        return

    class_name = f"Filter{pt}"
    return type(
        class_name, (am.Component,), {"__init__": init_func, "compute": compute}
    )()


class NodeSource(am.Component):
    def __init__(self):
        super().__init__()

        # Filter input values
        self.add_input("rho")
        self.add_data("x_coord")
        self.add_data("y_coord")

        self.empty = True

    def compute(self):
        pass


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
parser.add_argument(
    "--with-openmp",
    dest="use_openmp",
    action="store_true",
    default=False,
    help="Enable OpenMP",
)
args = parser.parse_args()

nx = 2 * 256
ny = 2 * 256
nnodes = (nx + 1) * (ny + 1)
nelems = nx * ny

nodes = np.arange(nnodes, dtype=int).reshape((nx + 1, ny + 1))

x_coord = np.zeros(nnodes)
y_coord = np.zeros(nnodes)
conn = np.zeros((nelems, 4), dtype=int)

xpts = np.linspace(0, 2, nx + 1)
ypts = np.linspace(0, 2, ny + 1)
for j in range(ny + 1):
    for i in range(nx + 1):
        x_coord[nodes[i, j]] = xpts[i]
        y_coord[nodes[i, j]] = ypts[j]

for i in range(nx):
    for j in range(ny):
        conn[ny * i + j, 0] = nodes[i, j]
        conn[ny * i + j, 1] = nodes[i + 1, j]
        conn[ny * i + j, 2] = nodes[i + 1, j + 1]
        conn[ny * i + j, 3] = nodes[i, j + 1]

module_name = "helmholtz"
model = am.Model(module_name)

node_src = NodeSource()
model.add_component("src", nnodes, node_src)

for n in range(4):
    fltr = filter_factory(n)
    name = f"filter{n}"

    model.add_component(name, nelems, fltr)

    model.link(name + ".x_coord", "src.x_coord", tgt_indices=conn)
    model.link(name + ".y_coord", "src.y_coord", tgt_indices=conn)
    model.link(name + ".rho", "src.rho", tgt_indices=conn)

if args.build:
    model.generate_cpp()

    compile_args = []
    link_args = []
    define_macros = []
    if args.use_openmp:
        compile_args = ["-fopenmp"]
        link_args = ["-fopenmp"]
        define_macros = [("AMIGO_USE_OPENMP", "1")]

    model.build_module(
        compile_args=compile_args, link_args=link_args, define_macros=define_macros
    )

model.initialize(reorder=True)
prob = model.create_opt_problem()

# Set the problem data
data = prob.get_data_vector()
data_array = data.get_array()
data_array[model.get_indices("src.x_coord")] = x_coord
data_array[model.get_indices("src.y_coord")] = y_coord

mat = prob.create_csr_matrix()
x = prob.create_vector()
ans = prob.create_vector()
g = prob.create_vector()
rhs = prob.create_vector()

prob.gradient(x, ans)
prob.gradient(x, g)

prob.hessian(x, mat)
chol = am.QuasidefCholesky(mat)
flag = chol.factor()
print("flag = ", flag)
chol.solve(ans)

mat.mult(ans, rhs)
print("Residual norm: ", np.linalg.norm(rhs.get_array() - g.get_array()))

X, Y = np.meshgrid(xpts, ypts)
vals = ans.get_array()[model.get_indices("src.rho")]
vals = vals.reshape((nx + 1, ny + 1))

import matplotlib.pylab as plt

# Plot using contourf
plt.contourf(X, Y, vals, levels=20, cmap="viridis")
plt.colorbar(label="Z value")
plt.xlabel("x")
plt.ylabel("y")

from scipy.sparse import csr_matrix

nrows, ncols, nnz, rowp, cols = mat.get_nonzero_structure()
data = mat.get_data()
jac = csr_matrix((data, cols, rowp), shape=(nrows, ncols))

plt.figure(figsize=(6, 6))
plt.spy(jac, markersize=0.2)
plt.title("Sparsity pattern of matrix A")
plt.show()

plt.show()
