import amigo as am
import numpy as np
import argparse
from scipy.sparse.linalg import spsolve
import matplotlib.pylab as plt
import matplotlib.tri as tri


class Basis:
    def __init__(self):

        V = np.zeros((12, 12))

        nodes = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
        for i, node in enumerate(nodes):
            x0, y0 = node
            V[3 * i : 3 * (i + 1), :] = self._eval_poly(x0, y0)

        self.A = np.linalg.solve(V, np.eye(12))

    def _eval_poly(self, x, y):
        """Evaluate the polynomial basis"""
        P = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [x, 0, 0],
                [0, x, 0.5],
                [y, 0, -0.5],
                [0, y, 0],
                [x * y, 0, -0.5 * x],
                [0, x * y, 0.5 * y],
                [y**2, 0, -y],
                [0, x**2, x],
                [y * (x - y) ** 2, -x * (y - x) ** 2, -2 * (y - x) ** 2],
                [-y * (x + y) ** 2, x * (y + x) ** 2, 2 * (y + x) ** 2],
            ]
        ).T

        return P

    def _eval_poly_deriv(self, x, y):
        """Evaluate the x and y derivatives of the polynomial basis"""

        Px = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
                [y, 0, -0.5],
                [0, y, 0],
                [0, 0, 0],
                [0, 2 * x, 1],
                [2 * y * (x - y), (y - x) * (3 * x - y), 4 * (y - x)],
                [-2 * y * (x + y), (y + x) * (y + 3 * x), 4 * (y + x)],
            ]
        ).T

        Py = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [x, 0, 0],
                [0, x, 0.5],
                [2 * y, 0, -1],
                [0, 0, 0],
                [(x - y) * (x - 3 * y), -2 * x * (y - x), -4 * (y - x)],
                [-(x + y) * (x + 3 * y), 2 * x * (y + x), 4 * (y + x)],
            ]
        ).T

        return Px, Py

    def eval_shape_functions(self, xi, eta):
        """
        Evaluate the shape functions at the specified parametric point
        """

        P = self._eval_poly(xi, eta)

        return P @ self.A

    def eval_shape_gradient(self, xi, eta):
        """
        Evaluate the derivative of the shape functions
        """

        Px, Py = self._eval_poly_deriv(xi, eta)

        Nx = Px @ self.A
        Ny = Py @ self.A

        return Nx, Ny


class Quadrature:
    def __init__(self, n=3):
        if n == 2:
            self.pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
            self.wts = [1.0, 1.0]
        elif n == 3:
            self.pts = [-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)]
            self.wts = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
        elif n == 4:
            # Set the quadrature scheme - 4 point quadrature rule
            a = np.sqrt((3 - 2 * np.sqrt(6.0 / 5.0)) / 7.0)
            wa = (18 + np.sqrt(30)) / 36
            b = np.sqrt((3 + 2 * np.sqrt(6.0 / 5.0)) / 7.0)
            wb = (18 - np.sqrt(30)) / 36

            self.pts = [-a, -b, b, a]
            self.wts = [wa, wb, wb, wa]


class BackgroundElement(am.Component):
    def __init__(self, E=0.01, nu=0.3):
        super().__init__()

        # Add keyword arguments for the compute function
        args = []
        for j in range(3):
            for i in range(3):
                args.append({"i": i, "j": j})
        self.set_args(args)

        self.add_constant("E", E)
        self.add_constant("nu", nu)

        self.add_input("u", shape=(4,), value=0.0)
        self.add_input("v", shape=(4,), value=0.0)
        self.add_input("theta", shape=(4,), value=0.0)

        # The dimension of the element
        self.add_data("D")

        self.add_objective("obj")

        # The basis and quadrature objects
        self.quad = Quadrature(n=3)
        self.basis = Basis()

        return

    def _compute_derivative(self, i, D, N, u, v, t):
        return (
            (
                N[i, 0] * u[0]
                + N[i, 1] * v[0]
                + N[i, 3] * u[1]
                + N[i, 4] * v[1]
                + N[i, 6] * u[2]
                + N[i, 7] * v[2]
                + N[i, 9] * u[3]
                + N[i, 10] * v[3]
            )
            / D
            + N[i, 2] * t[0]
            + N[i, 5] * t[1]
            + N[i, 8] * t[2]
            + N[i, 11] * t[3]
        )

    def compute(self, i=None, j=None):

        D = self.data["D"]
        E = self.constants["E"]
        nu = self.constants["nu"]

        xi = self.quad.pts[i]
        eta = self.quad.pts[j]

        u = self.inputs["u"]
        v = self.inputs["v"]
        t = self.inputs["theta"]

        # Evaluate the derivative
        Nx, Ny = self.basis.eval_shape_gradient(xi, eta)

        ux = self.vars["ux"] = self._compute_derivative(0, D, Nx, u, v, t)
        vx = self.vars["vx"] = self._compute_derivative(1, D, Nx, u, v, t)

        uy = self.vars["uy"] = self._compute_derivative(0, D, Ny, u, v, t)
        vy = self.vars["vy"] = self._compute_derivative(1, D, Ny, u, v, t)

        detJ = self.quad.wts[i] * self.quad.wts[j] * D * D

        s = self.vars["s"] = [
            E / (1.0 - nu * nu) * (ux + nu * vy),
            E / (1.0 - nu * nu) * (vy + nu * ux),
            0.5 * E / (1.0 + nu) * (vx + uy),
        ]

        self.objective["obj"] = 0.5 * detJ * (s[0] * ux + s[1] * vy + s[2] * (vx + uy))


class BeamElement(am.Component):
    def __init__(self, EI=10.0, EA=1.0):
        super().__init__()

        # Add keyword arguments for the compute function
        args = []
        for i in range(4):
            for j in range(4):
                args.append({"i": i, "j": j})

        self.set_args(args)

        self.add_constant("EI", EI)
        self.add_constant("EA", EA)

        self.add_input("u", shape=(4,), value=0.0)
        self.add_input("v", shape=(4,), value=0.0)
        self.add_input("theta", shape=(4,), value=0.0)

        self.add_data("D", value=0.0)
        self.add_data("level_set", shape=(4,), value=0.0)

        self.add_objective("obj")

        # Set the basis and quadrature objects
        self.quad = Quadrature(n=4)
        self.basis = Basis()

        return

    def _compute_derivative(self, i, D, N, u, v, t):
        return (
            (
                N[i, 0] * u[0]
                + N[i, 1] * v[0]
                + N[i, 3] * u[1]
                + N[i, 4] * v[1]
                + N[i, 6] * u[2]
                + N[i, 7] * v[2]
                + N[i, 9] * u[3]
                + N[i, 10] * v[3]
            )
            / D
            + N[i, 2] * t[0]
            + N[i, 5] * t[1]
            + N[i, 8] * t[2]
            + N[i, 11] * t[3]
        )

    def _bilinear_shape_functions(self, x, y):

        N = 0.25 * np.array(
            [
                (1.0 - x) * (1.0 - y),
                (1.0 + x) * (1.0 - y),
                (1.0 + x) * (1.0 + y),
                (1.0 - x) * (1.0 + y),
            ]
        )
        Nx = 0.25 * np.array([-(1.0 - y), (1.0 - y), (1.0 + y), -(1.0 + y)])
        Ny = 0.25 * np.array([-(1.0 - x), -(1.0 + x), (1.0 + x), (1.0 - x)])

        return N, Nx, Ny

    def compute(self, i=None, j=None):

        xi = self.quad.pts[i]
        eta = self.quad.pts[j]

        EI = self.constants["EI"]
        EA = self.constants["EA"]

        D = self.data["D"]
        lsf = self.data["level_set"]

        u = self.inputs["u"]
        v = self.inputs["v"]
        t = self.inputs["theta"]

        N0, Nx0, Ny0 = self._bilinear_shape_functions(xi, eta)

        phi = self.vars["phi"] = (
            N0[0] * lsf[0] + N0[1] * lsf[1] + N0[2] * lsf[2] + N0[3] * lsf[3]
        )

        gx = self.vars["gx"] = (
            Nx0[0] * lsf[0] + Nx0[1] * lsf[1] + Nx0[2] * lsf[2] + Nx0[3] * lsf[3]
        ) / D
        gy = self.vars["gy"] = (
            Ny0[0] * lsf[0] + Ny0[1] * lsf[1] + Ny0[2] * lsf[2] + Ny0[3] * lsf[3]
        ) / D
        g = self.vars["g"] = am.sqrt(gx * gx + gy * gy)

        # Compute the delta value
        eps = 0.1 * D
        delta = self.vars["delta"] = (1.0 / (eps * np.sqrt(np.pi))) * am.exp(
            -((phi * phi) / (eps * eps))
        )

        Nx, Ny = self.basis.eval_shape_gradient(xi, eta)

        ux = self.vars["ux"] = self._compute_derivative(0, D, Nx, u, v, t)
        vx = self.vars["vx"] = self._compute_derivative(1, D, Nx, u, v, t)
        tx = self.vars["tx"] = self._compute_derivative(2, D, Nx, u, v, t)

        uy = self.vars["uy"] = self._compute_derivative(0, D, Ny, u, v, t)
        vy = self.vars["vy"] = self._compute_derivative(1, D, Ny, u, v, t)
        ty = self.vars["ty"] = self._compute_derivative(2, D, Ny, u, v, t)

        # Compute the tangent value
        lx = self.vars["nx"] = 1.0 * gy / g
        ly = self.vars["ny"] = -1.0 * gx / g

        scale = 0.5 * self.quad.wts[i] * self.quad.wts[j] * D * D * delta * g

        epsilon = ux * lx * lx + (uy + vx) * lx * ly + vy * ly * ly

        self.objective["obj"] = scale * (
            (EI * (tx * lx + ty * ly) * (tx * lx + ty * ly)) + (EA * epsilon * epsilon)
        )


class Dirchlet(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("dof")
        self.add_input("lambda")
        self.add_objective("obj")

    def compute(self):
        self.objective["obj"] = self.inputs["dof"] * self.inputs["lambda"]


class PointForce(am.Component):
    def __init__(self, force=1.0):
        super().__init__()
        self.force = force
        self.add_input("dof")
        self.add_objective("obj")

    def compute(self):
        self.objective["obj"] = -self.force * self.inputs["dof"]


class NodeSource(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("u")
        self.add_input("v")
        self.add_input("theta")

        self.add_data("level_set")


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
    "--with-debug",
    dest="use_debug",
    action="store_true",
    default=False,
    help="Enable debug flags",
)
args = parser.parse_args()

nx = 196
ny = 196

nnodes = (nx + 1) * (ny + 1)
nelems = nx * ny
Dx = 0.5 / nx

nodes = np.arange(nnodes, dtype=int).reshape((nx + 1, ny + 1))
conn = np.zeros((nelems, 4), dtype=int)

for j in range(ny):
    for i in range(nx):
        conn[ny * i + j, 0] = nodes[i, j]
        conn[ny * i + j, 1] = nodes[i + 1, j]
        conn[ny * i + j, 2] = nodes[i + 1, j + 1]
        conn[ny * i + j, 3] = nodes[i, j + 1]

x_coord = np.zeros(nnodes)
y_coord = np.zeros(nnodes)

xpts = np.linspace(0, 1, nx + 1)
ypts = np.linspace(0, 1, ny + 1)
for j in range(ny + 1):
    for i in range(nx + 1):
        x_coord[nodes[i, j]] = xpts[i]
        y_coord[nodes[i, j]] = ypts[j]

model = am.Model("beam")

# Allocate the components
model.add_component("src", nnodes, NodeSource())
model.add_component("background", nelems, BackgroundElement())
model.add_component("beam", nelems, BeamElement())
model.add_component("bcs", 3 * (ny + 1), Dirchlet())
model.add_component("force", 1, PointForce())

# Link background element variables
model.link("background.u", "src.u", tgt_indices=conn)
model.link("background.v", "src.v", tgt_indices=conn)
model.link("background.theta", "src.theta", tgt_indices=conn)

# Link the beam element variables
model.link("beam.u", "src.u", tgt_indices=conn)
model.link("beam.v", "src.v", tgt_indices=conn)
model.link("beam.theta", "src.theta", tgt_indices=conn)

# Link the level set data
model.link("beam.level_set", "src.level_set", tgt_indices=conn)

# Link the boundary conditions
model.link(f"bcs.dof[0:{ny + 1}]", "src.u", tgt_indices=nodes[0, :])
model.link(f"bcs.dof[{ny + 1}:{2 * (ny + 1)}]", "src.v", tgt_indices=nodes[0, :])
model.link(f"bcs.dof[{2*(ny + 1)}:]", "src.theta", tgt_indices=nodes[0, :])

# Link the D values
model.link("beam.D", "background.D")
model.link("beam.D[1:]", "beam.D[0]")

# Set the applied loads
model.link("force.dof", f"src.v[{nodes[-1, 0]}]")

if args.build:
    compile_args = []
    link_args = ["-lblas", "-llapack"]
    define_macros = []
    if args.use_openmp:
        compile_args = ["-fopenmp"]
        link_args += ["-fopenmp"]
        define_macros = [("AMIGO_USE_OPENMP", "1")]

    model.build_module(
        compile_args=compile_args,
        link_args=link_args,
        define_macros=define_macros,
        debug=args.use_debug,
    )

model.initialize()

# Set the problem data
data = model.get_data_vector()
data["background.D"] = Dx
data["beam.D"] = Dx

# Set the level set function values for this test case
# r = 0.3
# data["src.level_set"] = (x_coord - 0.5) ** 2 + (y_coord - 0.5) ** 2 - r**2

r = 0.3
d = 10 * ((x_coord - 0.5) ** 2 + (y_coord - 0.25) ** 2 - r**2)

data["src.level_set"] = np.fabs(x_coord + y_coord - 1.0) * np.fabs(y_coord - 0.05) * d

# Set the initial problem variable values
problem = model.get_problem()
mat = problem.create_matrix()

x = problem.create_vector()
grad = problem.create_vector()
ans = problem.create_vector()

problem.gradient(x, grad)
problem.hessian(x, mat)

csr = am.tocsr(mat)
ans.get_array()[:] = spsolve(csr, grad.get_array())

# Extract the solution
u = ans.get_array()[model.get_indices("src.u")][nodes].T
v = ans.get_array()[model.get_indices("src.v")][nodes].T
theta = ans.get_array()[model.get_indices("src.theta")][nodes].T

d = data.get_vector()
lsf = d.get_array()[model.get_indices("src.level_set")][nodes].T

# Plot the u, v and theta values
x = np.linspace(0, 1, nx + 1)
X, Y = np.meshgrid(x, x)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
cs0 = ax[0].contourf(X, Y, u, levels=50)
cs1 = ax[1].contourf(X, Y, v, levels=50)
cs2 = ax[2].contourf(X, Y, theta, levels=50)
ax[0].set_title("u")
ax[1].set_title("v")
ax[2].set_title("theta")

# Plot the deformed shape
scale = 0.05 / np.max((np.max(np.absolute(u)), np.max(np.absolute(v))))

fig, ax = plt.subplots(1, 1)
ax.contourf(X + scale * u, Y + scale * v, theta, levels=50)

plt.show()
