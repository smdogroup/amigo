import amigo as am
import numpy as np  # used for plotting/analysis
import argparse
import time
import matplotlib.pylab as plt


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


class Helmholtz(am.Component):
    def __init__(self):
        super().__init__()

        # Add keyword arguments for the compute function
        compute_args = []
        for n in range(4):
            compute_args.append({"n": n})
        self.set_compute_args(compute_args)

        # The filter radius
        self.add_constant("r_filter", 0.1)

        # The x/y coordinates
        self.add_data("x_coord", shape=(4,))
        self.add_data("y_coord", shape=(4,))

        # The implicit topology input/constraint
        self.add_input("x", shape=(4,), value=0.5, lower=0.0, upper=1.0)
        self.add_input("rho", shape=(4,), value=0.5, lower=0.0, upper=1.0)

        # Add the residual
        self.add_constraint("rho_res", shape=(4,), value=1.0, lower=0.0, upper=0.0)

        return

    def compute(self, n=None):
        qpts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
        xi = qpts[n % 2]
        eta = qpts[n // 2]

        r = self.constants["r_filter"]
        x = self.inputs["x"]
        rho = self.inputs["rho"]

        X = self.data["x_coord"]
        Y = self.data["y_coord"]

        N, N_xi, N_ea = compute_shape_derivs(xi, eta, X, Y, self.vars)

        Nx = self.vars["Nx"]
        Ny = self.vars["Ny"]

        self.vars["x0"] = dot(N, x)
        self.vars["rho0"] = dot(N, rho)
        self.vars["rho_x"] = dot(Nx, rho)
        self.vars["rho_y"] = dot(Ny, rho)

        x0 = self.vars["x0"]
        rho0 = self.vars["rho0"]
        rho_x = self.vars["rho_x"]
        rho_y = self.vars["rho_y"]

        detJ = self.vars["detJ"]

        self.constraints["rho_res"] = [
            detJ * (N[0] * (rho0 - x0) + r * r * (Nx[0] * rho_x + Ny[0] * rho_y)),
            detJ * (N[1] * (rho0 - x0) + r * r * (Nx[1] * rho_x + Ny[1] * rho_y)),
            detJ * (N[2] * (rho0 - x0) + r * r * (Nx[2] * rho_x + Ny[2] * rho_y)),
            detJ * (N[3] * (rho0 - x0) + r * r * (Nx[3] * rho_x + Ny[3] * rho_y)),
        ]

        return


class Topology(am.Component):
    def __init__(self):
        super().__init__()

        # Add keyword arguments for the compute function
        compute_args = []
        for n in range(4):
            compute_args.append({"n": n})
        self.set_compute_args(compute_args)

        # Constants
        self.add_constant("p", 3.0)
        self.add_constant("E", 1.0)
        self.add_constant("nu", 0.3)
        self.add_constant("kappa", 1e-6)

        # The x/y coordinates
        self.add_data("x_coord", shape=(4,))
        self.add_data("y_coord", shape=(4,))

        # The inputs to the problem
        self.add_input("rho", shape=(4,), value=0.5, lower=0.0, upper=1.0)
        self.add_input("u", shape=(4,), value=0.0)
        self.add_input("v", shape=(4,), value=0.0)

        # Add the residuals
        self.add_constraint("u_res", shape=(4,), value=1.0, lower=0.0, upper=0.0)
        self.add_constraint("v_res", shape=(4,), value=1.0, lower=0.0, upper=0.0)

        # Add the objective
        self.add_objective("compliance")

        return

    def compute(self, n=None):
        qpts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
        xi = qpts[n % 2]
        eta = qpts[n // 2]

        E = self.constants["E"]
        nu = self.constants["nu"]
        kappa = self.constants["kappa"]
        p = self.constants["p"]

        # Extract the input variables
        rho = self.inputs["rho"]
        u = self.inputs["u"]
        v = self.inputs["v"]

        # Extract the input data
        X = self.data["x_coord"]
        Y = self.data["y_coord"]
        compute_shape_derivs(xi, eta, X, Y, self.vars)

        # Set the values of the derivatives of the shape functions
        Nx = self.vars["Nx"]
        Ny = self.vars["Ny"]

        rho0 = 0.25 * (rho[0] + rho[1] + rho[2] + rho[3])
        # self.vars["E0"] = E * (rho0**p + kappa)
        self.vars["E0"] = E * (rho0 + kappa)
        E0 = self.vars["E0"]

        self.vars["Ux"] = [[dot(Nx, u), dot(Ny, u)], [dot(Nx, v), dot(Ny, v)]]
        Ux = self.vars["Ux"]

        self.vars["e"] = [
            Ux[0, 0],
            Ux[1, 1],
            (Ux[0, 1] + Ux[1, 0]),
        ]
        e = self.vars["e"]

        self.vars["s"] = [
            E0 / (1.0 - nu * nu) * (e[0] + nu * e[1]),
            E0 / (1.0 - nu * nu) * (e[1] + nu * e[0]),
            0.5 * E0 / (1.0 + nu) * e[2],
        ]
        s = self.vars["s"]

        detJ = self.vars["detJ"]
        self.constraints["u_res"] = [
            detJ * (Nx[0] * s[0] + Ny[0] * s[2]),
            detJ * (Nx[1] * s[0] + Ny[1] * s[2]),
            detJ * (Nx[2] * s[0] + Ny[2] * s[2]),
            detJ * (Nx[3] * s[0] + Ny[3] * s[2]),
        ]

        self.constraints["v_res"] = [
            detJ * (Nx[0] * s[2] + Ny[0] * s[1]),
            detJ * (Nx[1] * s[2] + Ny[1] * s[1]),
            detJ * (Nx[2] * s[2] + Ny[2] * s[1]),
            detJ * (Nx[3] * s[2] + Ny[3] * s[1]),
        ]

        # Add the objective values
        self.objective["compliance"] = (
            0.5 * detJ * (s[0] * e[0] + s[1] * e[1] + s[2] * e[2])
        )

        return


class MassConstraint(am.Component):
    def __init__(self):
        super().__init__()

        # Add keyword arguments for the compute function
        compute_args = []
        for n in range(4):
            compute_args.append({"n": n})
        self.set_compute_args(compute_args)

        self.add_constant("mass_fraction", value=0.4)

        # The x/y coordinates
        self.add_data("x_coord", shape=(4,))
        self.add_data("y_coord", shape=(4,))

        # The implicit topology input/constraint
        self.add_input("rho", shape=(4,), value=0.5, lower=0.0, upper=1.0)

        # Add the residuals
        self.add_constraint("mass_con", value=1.0, lower=0.0, upper=0.0)

    def compute(self, n=None):
        qpts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        xi = qpts[n % 2]
        eta = qpts[n // 2]

        mass_fraction = self.constants["mass_fraction"]

        # Extract the input variables
        rho = self.inputs["rho"]

        # Extract the input data
        X = self.data["x_coord"]
        Y = self.data["y_coord"]

        compute_detJ(xi, eta, X, Y, self.vars)
        detJ = self.vars["detJ"]
        rho0 = 0.25 * (rho[0] + rho[1] + rho[2] + rho[3])

        self.constraints["mass_con"] = detJ * (rho0 - mass_fraction)

        return


class FixedBoundaryCondition(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("u", value=1.0)
        self.add_input("lam", value=1.0)

        self.add_constraint("disp_res", value=1.0, lower=0.0, upper=0.0)
        self.add_constraint("bc_res", value=1.0, lower=0.0, upper=0.0)

    def compute(self):
        self.constraints["bc_res"] = self.inputs["u"]
        self.constraints["disp_res"] = self.inputs["lam"]


class AppliedLoad(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constraint("u_res", value=1.0, lower=0.0, upper=0.0)
        self.add_constraint("v_res", value=1.0, lower=0.0, upper=0.0)
        self.add_constant("fx", value=-10.0)
        self.add_constant("fy", value=-10.0)
        return

    def compute(self):
        fx = self.constants["fx"]
        fy = self.constants["fy"]
        self.constraints["u_res"] = -fx
        self.constraints["v_res"] = -fy
        return


class NodeSource(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x", value=0.5, lower=0.0, upper=1.0)
        self.add_input("rho", value=0.5, lower=0.0, upper=1.0)
        self.add_input("u", value=0.0, lower=-100, upper=100)
        self.add_input("v", value=0.0, lower=-100, upper=100)

        self.add_constraint("rho_res", value=1.0, lower=0.0, upper=0.0)
        self.add_constraint("u_res", value=1.0, lower=0.0, upper=0.0)
        self.add_constraint("v_res", value=1.0, lower=0.0, upper=0.0)

        self.add_data("x_coord")
        self.add_data("y_coord")


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
parser.add_argument(
    "--order-type",
    choices=["amd", "nd", "natural"],
    default="nd",
    help="Ordering strategy to use (default: amd)",
)
parser.add_argument(
    "--order-for-block",
    dest="order_for_block",
    action="store_true",
    default=False,
    help="Order for 2x2 block KKT matrix",
)
parser.add_argument(
    "--show-sparsity",
    dest="show_sparsity",
    action="store_true",
    default=False,
    help="Show the sparsity pattern",
)
args = parser.parse_args()

# nx = 128
# ny = 64

nx = 64
ny = 32

nnodes = (nx + 1) * (ny + 1)
nelems = nx * ny

nodes = np.arange(nnodes, dtype=int).reshape((nx + 1, ny + 1))

x_coord = np.zeros(nnodes)
y_coord = np.zeros(nnodes)
conn = np.zeros((nelems, 4), dtype=int)

xpts = np.linspace(0, 2, nx + 1)
ypts = np.linspace(0, 1, ny + 1)
for j in range(ny + 1):
    for i in range(nx + 1):
        x_coord[nodes[i, j]] = xpts[i]
        y_coord[nodes[i, j]] = ypts[j]

conn = np.zeros((nelems, 4), dtype=int)
for j in range(ny):
    for i in range(nx):
        conn[ny * i + j, 0] = nodes[i, j]
        conn[ny * i + j, 1] = nodes[i + 1, j]
        conn[ny * i + j, 2] = nodes[i + 1, j + 1]
        conn[ny * i + j, 3] = nodes[i, j + 1]

module_name = "compliance"
model = am.Model(module_name)

node_src = NodeSource()
model.add_component("src", nnodes, node_src)

helmholtz = Helmholtz()
model.add_component("helmholtz", nelems, helmholtz)

# Link the inputs and the constraints
model.link("helmholtz.x_coord", "src.x_coord", tgt_indices=conn)
model.link("helmholtz.y_coord", "src.y_coord", tgt_indices=conn)
model.link("helmholtz.x", "src.x", tgt_indices=conn)
model.link("helmholtz.rho", "src.rho", tgt_indices=conn)
model.link("helmholtz.rho_res", "src.rho_res", tgt_indices=conn)

topo = Topology()
model.add_component("topo", nelems, topo)

# Link the data
model.link("topo.x_coord", "src.x_coord", tgt_indices=conn)
model.link("topo.y_coord", "src.y_coord", tgt_indices=conn)

# Link the inputs and the constraints
model.link("topo.u", "src.u", tgt_indices=conn)
model.link("topo.v", "src.v", tgt_indices=conn)

model.link("topo.u_res", "src.u_res", tgt_indices=conn)
model.link("topo.v_res", "src.v_res", tgt_indices=conn)

# Link the filtered density field
model.link("topo.rho", "src.rho", tgt_indices=conn)

# Add the mass constraint
mass_con = MassConstraint()
model.add_component("mass", nelems, mass_con)

# Link the mass constraint inputs
model.link("mass.x_coord", "src.x_coord", tgt_indices=conn)
model.link("mass.y_coord", "src.y_coord", tgt_indices=conn)
model.link("mass.rho", "src.rho", tgt_indices=conn)

# Set up the mass constraint
model.link("mass.mass_con[1:]", "mass.mass_con[0]")

# Add boundary conditions
bcs_u = FixedBoundaryCondition()
model.add_component("bcs_u", (ny + 1), bcs_u)
model.link("src.u", "bcs_u.u", src_indices=nodes[0, :])
model.link("src.u_res", "bcs_u.disp_res", src_indices=nodes[0, :])

bcs_v = FixedBoundaryCondition()
model.add_component("bcs_v", (ny + 1), bcs_v)
model.link("src.v", "bcs_v.u", src_indices=nodes[0, :])
model.link("src.v_res", "bcs_v.disp_res", src_indices=nodes[0, :])

# Set the applied load
load = AppliedLoad()
model.add_component("load", 1, load)
model.link("src.u_res", "load.u_res", src_indices=nodes[-1, 0])
model.link("src.v_res", "load.v_res", src_indices=nodes[-1, 0])

if args.build:
    compile_args = []
    link_args = ["-lblas", "-llapack"]
    define_macros = []
    if args.use_openmp:
        compile_args = ["-fopenmp"]
        link_args += ["-fopenmp"]
        define_macros = [("AMIGO_USE_OPENMP", "1")]

    model.build_module(
        compile_args=compile_args, link_args=link_args, define_macros=define_macros
    )

start = time.perf_counter()

if args.order_type == "amd":
    order_type = am.OrderingType.AMD
elif args.order_type == "nd":
    order_type = am.OrderingType.NESTED_DISSECTION
elif args.order_type == "natural":
    order_type = am.OrderingType.NATURAL

order_for_block = args.order_for_block
model.initialize(order_type=order_type, order_for_block=order_for_block)

# Initialize the problem
prob = model.get_opt_problem()

end = time.perf_counter()
print(f"Initialization time:        {end - start:.6f} seconds")
print(f"Num variables:              {model.num_variables}")
print(f"Num constraints:            {model.num_constraints}")

# Set the problem data
data = model.get_data_vector()
data["src.x_coord"] = x_coord
data["src.y_coord"] = y_coord

# Set the initial problem variable values
x = model.create_vector()

# Set initial design variable values
x["src.x"] = 0.5
x["src.rho"] = 0.5

# Set initial multiplier values for the constraints
x["src.rho_res"] = 1.0
x["src.u_res"] = 1.0
x["src.v_res"] = 1.0

# Apply lower and upper bound constraints
lower = model.create_vector()
upper = model.create_vector()
lower["src.x"] = 1e-3
upper["src.x"] = 1.0
lower["src.rho"] = 1e-3
upper["src.rho"] = float("inf")

start = time.perf_counter()
mat_obj = prob.create_csr_matrix()
end = time.perf_counter()
print(f"Matrix initialization time: {end - start:.6f} seconds")

start = time.perf_counter()
prob.hessian(x.get_opt_problem_vec(), mat_obj)
end = time.perf_counter()
print(f"Matrix computation time:    {end - start:.6f} seconds")

grad = prob.create_vector()
start = time.perf_counter()
prob.gradient(x.get_opt_problem_vec(), grad)
end = time.perf_counter()
print(f"Residual computation time:  {end - start:.6f} seconds")

opt = am.Optimizer(model, x=x, lower=lower, upper=upper)

options = {
    "max_iterations": 100,
    "initial_barrier_param": 0.1,
    "max_line_search_iterations": 1,
}
opt.optimize(options)

vals = x["src.rho"]
vals = vals.reshape((nx + 1, ny + 1)).T

X, Y = np.meshgrid(xpts, ypts)
plt.figure()
plt.contourf(X, Y, vals, levels=20, cmap="viridis")
plt.colorbar(label="rho value")
plt.xlabel("x")
plt.ylabel("y")

vals = x["src.x"]
vals = vals.reshape((nx + 1, ny + 1)).T

plt.figure()
plt.contourf(X, Y, vals, levels=20, cmap="viridis")
plt.colorbar(label="x value")
plt.xlabel("x")
plt.ylabel("y")

plt.show()
