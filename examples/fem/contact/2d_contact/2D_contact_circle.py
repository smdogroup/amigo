# Author: Jack Turbush
# Description: 2D contact problem with a circular geometry using penalty-based contact enforcement.
# Contact is enforced via lower bounds on nodal v-displacement (interior point barrier method).
# Equilibrium is found by minimizing total potential energy: PE = strain energy - work done.

import amigo as am
import numpy as np
import argparse
import matplotlib.pylab as plt

length = 1.0
E = 25000
Fv = -120
Mv = 0.0


class beam_element(am.Component):
    """Euler-Bernoulli beam element strain energy in global coordinates.
    Transforms global DOFs to local frame before computing stiffness contribution."""

    def __init__(self):
        super().__init__()
        self.add_data("x_coord", shape=(2,), value=0.0)
        self.add_data("y_coord", shape=(2,), value=0.0)
        self.add_input("u", shape=(2,), value=0.0)
        self.add_input("v", shape=(2,), value=0.0)
        self.add_input("t", shape=(2,), value=0.0)
        self.add_data("h")
        self.add_objective("obj")

    def compute(self):
        x = self.data["x_coord"]
        y = self.data["y_coord"]
        u = self.inputs["u"]
        v = self.inputs["v"]
        t = self.inputs["t"]
        de_global = np.array([u[0], v[0], t[0], u[1], v[1], t[1]])

        # Rotation matrix from global to local frame
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        L = (dx * dx + dy * dy) ** 0.5
        c = dx / L
        s = dy / L
        T = [
            [c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [0, 0, 0, 0, 0, 1],
        ]
        de_local = T @ de_global

        # Local Euler-Bernoulli stiffness matrix (axial + bending)
        L = length / nelems
        A = 1.0
        I = 1.0
        a1 = E * A / L
        a2 = E * I / (L**3.0)
        Ke = np.empty((6, 6))
        Ke[0, :] = [a1, 0, 0, -a1, 0, 0]
        Ke[1, :] = [0, 12 * a2, 6 * L * a2, 0, -12 * a2, 6 * L * a2]
        Ke[2, :] = [0, 0, 4 * L**2 * a2, 0, -6 * L * a2, 2 * L**2 * a2]
        Ke[3, :] = [0, 0, 0, a1, 0, 0]
        Ke[4, :] = [0, 0, 0, 0, 12 * a2, -6 * L * a2]
        Ke[5, :] = [0, 0, 0, 0, 0, 4 * L**2 * a2]
        Ke = Ke + Ke.T - np.diag(np.diag(Ke))

        self.objective["obj"] = 0.5 * de_local.T @ Ke @ de_local


class BoundaryCondition(am.Component):
    """Pins two nodes on the circle (top of the ring) as fixed supports."""

    def __init__(self):
        super().__init__()
        self.add_input("u", value=1.0)
        self.add_input("v", value=1.0)
        self.add_input("t", value=1.0)
        self.add_constraint("bc_u", value=0.0, lower=0.0, upper=0.0)
        self.add_constraint("bc_v", value=0.0, lower=0.0, upper=0.0)
        self.add_constraint("bc_t", value=0.0, lower=0.0, upper=0.0)

    def compute(self):
        self.constraints["bc_u"] = self.inputs["u"]
        self.constraints["bc_v"] = self.inputs["v"]
        self.constraints["bc_t"] = self.inputs["t"]


class NodeSource(am.Component):
    """Holds nodal coordinates and DOFs (u, v, t) for the circular ring mesh."""

    def __init__(self):
        super().__init__()
        self.add_data("x_coord")
        self.add_data("y_coord")
        self.add_input("u")
        self.add_input("v")
        self.add_input("t")


parser = argparse.ArgumentParser()
parser.add_argument("--build", dest="build", action="store_true", default=False)
args = parser.parse_args()

# Build circular ring mesh with n nodes
n = 100
r = 0.5
X = np.zeros((n, 2))
for i in range(n):
    theta = 2 * np.pi * i / n
    X[i, 0] = r * np.cos(theta)
    X[i, 1] = r * np.sin(theta)

nnodes = n
x_coord = X[:, 0]
y_coord = X[:, 1]

# Closed-loop connectivity
conn = np.zeros((n, 2), dtype=int)
for i in range(n - 1):
    conn[i] = [i, i + 1]
conn[n - 1] = [n - 1, 0]
nelems = n

model = am.Model("beam")
model.add_component("src", nnodes, NodeSource())
model.add_component("beam_element", nelems, beam_element())

model.link("beam_element.x_coord", "src.x_coord", tgt_indices=conn)
model.link("beam_element.y_coord", "src.y_coord", tgt_indices=conn)
model.link("beam_element.u", "src.u", tgt_indices=conn)
model.link("beam_element.v", "src.v", tgt_indices=conn)
model.link("beam_element.t", "src.t", tgt_indices=conn)

# Fix two nodes near the top of the ring as supports
bcs = BoundaryCondition()
model.add_component("bcs", 2, bcs)
bc1 = n // 4 - 1
bc2 = n // 4 + 1
model.link("src.u", "bcs.u", src_indices=[bc1, bc2])
model.link("src.v", "bcs.v", src_indices=[bc1, bc2])
model.link("src.t", "bcs.t", src_indices=[bc1, bc2])

if args.build:
    model.build_module()

model.initialize()

x = model.create_vector()
lower = model.create_vector()
upper = model.create_vector()

x[:] = 1.0

data = model.get_data_vector()
data["src.x_coord"] = x_coord
data["src.y_coord"] = y_coord

lower["src.u"] = -float("inf")
upper["src.u"] = float("inf")
lower["src.v"] = -float("inf")
upper["src.v"] = float("inf")
lower["src.t"] = -float("inf")
upper["src.t"] = float("inf")

# Contact floor at y = -0.4; per-node lower bound = floor - y_coord
y_floor = -0.4
for i, node in enumerate(X):
    lower[f"src.v[{i}]"] = y_floor - y_coord[i]

opt_options = {
    "max_iterations": 200,
    "convergence_tolerance": 1e-6,
    "max_line_search_iterations": 1,
    "initial_barrier_param": 0.1,
}

opt = am.Optimizer(model, x, lower=lower, upper=upper)
opt.optimize(opt_options)


def plot_disp(ax, x, u=None, scale=1.0, color="k"):
    """Plot ring in reference (black) or deformed (red) configuration."""
    if u is not None:
        x = x + scale * u.reshape((-1, 2))
    for index in range(nelems):
        i = conn[index, 0]
        j = conn[index, 1]
        ax.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], f"-{color}o")


def build_global_displacement(u, v):
    """Interleave u and v into a single displacement vector [u0,v0,u1,v1,...]."""
    q = np.zeros(2 * len(u))
    for i in range(len(u)):
        q[2 * i] = u[i]
        q[2 * i + 1] = v[i]
    return q


u = x["src.u"]
v = x["src.v"]
t = x["src.t"]
U = build_global_displacement(u, v)

# Compute contact forces from barrier multipliers: zl = mu / (v - lb)
mu = opt.barrier_param
lbxv = lower["src.v"]
zl = mu / (x["src.v"] - lbxv)
zl_highest = [i for i in zl if i >= 1]
print("zl > epsilon:", zl_highest)

fig, ax = plt.subplots(facecolor="w")
plot_disp(ax, X)
plot_disp(ax, X, U, color="r")
ax.set_aspect("equal")
plt.axhline(y_floor, color="black")
plt.savefig("2d_contact.png")
