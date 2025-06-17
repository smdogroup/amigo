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
        self.add_input("x", shape=(4,))
        self.add_input("rho", shape=(4,))

        # Add the residual
        self.add_output("rho_res", shape=(4,))

        return

    def compute(self):
        r = self.constants["r_filter"]
        x = self.inputs["x"]
        rho = self.inputs["rho"]

        X = self.data["x_coord"]
        Y = self.data["y_coord"]

        N, N_xi, N_ea = compute_shape_derivs(self.xi, self.eta, X, Y, self.vars)

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

        self.outputs["rho_res"] = [
            detJ * (N[0] * (rho0 - x0) + r * r * (Nx[0] * rho_x + Ny[0] * rho_y)),
            detJ * (N[1] * (rho0 - x0) + r * r * (Nx[1] * rho_x + Ny[1] * rho_y)),
            detJ * (N[2] * (rho0 - x0) + r * r * (Nx[2] * rho_x + Ny[2] * rho_y)),
            detJ * (N[3] * (rho0 - x0) + r * r * (Nx[3] * rho_x + Ny[3] * rho_y)),
        ]

        return

    class_name = f"Filter{pt}"
    return type(
        class_name, (am.Component,), {"__init__": init_func, "compute": compute}
    )()


def topology_factory(pt: int):
    qpts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

    def init_func(self):
        am.Component.__init__(self)
        self.xi = qpts[pt % 2]
        self.eta = qpts[pt // 2]

        # The filter radius
        self.add_constant("p", 3.0)
        self.add_constant("E", 1.0)
        self.add_constant("nu", 0.3)
        self.add_constant("kappa", 1e-6)

        # The x/y coordinates
        self.add_data("x_coord", shape=(4,))
        self.add_data("y_coord", shape=(4,))

        # The implicit topology input/output
        self.add_input("rho", shape=(4,))
        self.add_input("u", shape=(4,))
        self.add_input("v", shape=(4,))

        # Add the residuals
        self.add_output("u_res", shape=(4,))
        self.add_output("v_res", shape=(4,))

        return

    def compute(self):
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
        compute_shape_derivs(self.xi, self.eta, X, Y, self.vars)

        # Set the values of the derivatives of the shape functions
        Nx = self.vars["Nx"]
        Ny = self.vars["Ny"]

        rho0 = 0.25 * (rho[0] + rho[1] + rho[2] + rho[3])
        self.vars["E0"] = E * (rho0**p + kappa)
        E0 = self.vars["E0"]

        self.vars["Ux"] = [[dot(Nx, u), dot(Ny, u)], [dot(Nx, v), dot(Ny, v)]]
        Ux = self.vars["Ux"]

        self.vars["s"] = [
            E0 / (1.0 - nu * nu) * (Ux[0, 0] + nu * Ux[1, 1]),
            E0 / (1.0 - nu * nu) * (Ux[1, 1] + nu * Ux[0, 0]),
            0.5 * E0 / (1.0 + nu) * (Ux[0, 1] + Ux[1, 0]),
        ]
        s = self.vars["s"]

        detJ = self.vars["detJ"]
        self.outputs["u_res"] = [
            detJ * (Nx[0] * s[0] + Ny[0] * s[2]),
            detJ * (Nx[1] * s[0] + Ny[1] * s[2]),
            detJ * (Nx[2] * s[0] + Ny[2] * s[2]),
            detJ * (Nx[3] * s[0] + Ny[3] * s[2]),
        ]

        self.outputs["v_res"] = [
            detJ * (Nx[0] * s[2] + Ny[0] * s[1]),
            detJ * (Nx[1] * s[2] + Ny[1] * s[1]),
            detJ * (Nx[2] * s[2] + Ny[2] * s[1]),
            detJ * (Nx[3] * s[2] + Ny[3] * s[1]),
        ]

        return

    class_name = f"Topology{pt}"
    return type(
        class_name, (am.Component,), {"__init__": init_func, "compute": compute}
    )()


def mass_factory(pt: int):
    qpts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

    def init_func(self):
        am.Component.__init__(self)
        self.xi = qpts[pt % 2]
        self.eta = qpts[pt // 2]

        self.add_constant("mass_fraction", value=0.4)

        # The x/y coordinates
        self.add_data("x_coord", shape=(4,))
        self.add_data("y_coord", shape=(4,))

        # The implicit topology input/output
        self.add_input("rho", shape=(4,))

        # Add the residuals
        self.add_output("mass_con")

    def compute(self):
        mass_fraction = self.constants["mass_fraction"]

        # Extract the input variables
        rho = self.inputs["rho"]

        # Extract the input data
        X = self.data["x_coord"]
        Y = self.data["y_coord"]

        compute_detJ(self.xi, self.eta, X, Y, self.vars)
        detJ = self.vars["detJ"]
        rho0 = 0.25 * (rho[0] + rho[1] + rho[2] + rho[3])

        self.outputs["mass_con"] = detJ * (rho0 - mass_fraction)

    class_name = f"Mass{pt}"
    return type(
        class_name, (am.Component,), {"__init__": init_func, "compute": compute}
    )()


class FixedBoundaryCondition(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("u")
        self.add_input("v")
        self.add_output("u_res")
        self.add_output("v_res")

    def compute(self):
        self.outputs["u_res"] = self.inputs["u"]
        self.outputs["v_res"] = self.inputs["v"]


class Compliance(am.Component):
    pass


class NodeSource(am.Component):
    def __init__(self):
        super().__init__()

        # Filter input values
        self.add_input("x")
        self.add_input("rho")
        self.add_input("u")
        self.add_input("v")

        self.add_output("rho_res")
        self.add_output("u_res")
        self.add_output("v_res")

        self.add_data("x_coord")
        self.add_data("y_coord")

        self.empty = True

    def compute(self):
        pass


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

nx = 2 * 256
ny = 2 * 128
nnodes = (nx + 1) * (ny + 1)
nelems = nx * ny

nodes = np.arange(nnodes, dtype=int).reshape((nx + 1, ny + 1))

conn = np.zeros((nelems, 4), dtype=int)
for j in range(ny):
    for i in range(nx):
        conn[ny * i + j, 0] = nodes[i, j]
        conn[ny * i + j, 1] = nodes[i + 1, j]
        conn[ny * i + j, 2] = nodes[i + 1, j + 1]
        conn[ny * i + j, 3] = nodes[i, j + 1]

node_src = NodeSource()

module_name = "compliance"
model = am.Model(module_name)

model.add_component("src", nnodes, node_src)

for n in range(4):
    fltr = filter_factory(n)
    name = f"filter{n}"

    model.add_component(name, nelems, fltr)

    # Link the inputs and the outputs
    model.link(name + ".x_coord", "src.x_coord", tgt_indices=conn)
    model.link(name + ".y_coord", "src.y_coord", tgt_indices=conn)

    model.link(name + ".x", "src.x", tgt_indices=conn)
    model.link(name + ".rho", "src.rho", tgt_indices=conn)
    model.link(name + ".rho_res", "src.rho_res", tgt_indices=conn)

for n in range(4):
    topo = topology_factory(n)
    name = f"topo{n}"

    model.add_component(name, nelems, topo)

    # Link the data
    model.link(name + ".x_coord", "src.x_coord", tgt_indices=conn)
    model.link(name + ".y_coord", "src.y_coord", tgt_indices=conn)

    # Link the inputs and the outputs
    model.link(name + ".u", "src.u", tgt_indices=conn)
    model.link(name + ".v", "src.v", tgt_indices=conn)

    model.link(name + ".u_res", "src.u_res", tgt_indices=conn)
    model.link(name + ".v_res", "src.v_res", tgt_indices=conn)

    # Link the filtered density field
    model.link(name + ".rho", "src.rho", tgt_indices=conn)

for n in range(4):
    topo = mass_factory(n)
    name = f"mass{n}"

    model.add_component(name, nelems, topo)

    # Link the data
    model.link(name + ".x_coord", "src.x_coord", tgt_indices=conn)
    model.link(name + ".y_coord", "src.y_coord", tgt_indices=conn)

    # Link the filtered density field
    model.link(name + ".rho", "src.rho", tgt_indices=conn)

    if n == 0:
        model.link(
            name + ".mass_con",
            name + ".mass_con",
            src_indices=np.zeros(nelems - 1, dtype=int),
            tgt_indices=np.arange(1, nelems, dtype=int),
        )
    else:
        model.link(name + ".mass_con", f"mass{n-1}.mass_con")

if args.build:
    model.generate_cpp()
    model.build_module()

start = time.perf_counter()
model.initialize()
end = time.perf_counter()
print(f"Initialization time:        {end - start:.6f} seconds")
print(f"Num variables:              {model.num_variables}")

prob = model.create_opt_problem()

x = prob.create_vector()
x_array = x.get_array()
x_array[model.get_indices("src.x")] = 1.0
x_array[model.get_indices("src.rho")] = 1.0

start = time.perf_counter()
mat_obj = prob.create_csr_matrix()
end = time.perf_counter()
print(f"Matrix initialization time: {end - start:.6f} seconds")

start = time.perf_counter()
prob.hessian(x, mat_obj)
end = time.perf_counter()
print(f"Matrix computation time:    {end - start:.6f} seconds")
