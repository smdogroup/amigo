from amigo.fem import Mesh, Problem, SolutionSpace
import amigo as am
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy as np
import argparse


def potential_plane_stress(soln, data=None, geo=None):
    """Strain energy density (integrand of TPE equation)"""
    # Displacement gradients in physical space
    ux_grad = soln["ux"]["grad"]
    uy_grad = soln["uy"]["grad"]

    # Strain components
    e11 = ux_grad[0]
    e22 = uy_grad[1]
    e12 = ux_grad[1] + uy_grad[0]

    # Material properties (hardcoded or pull from data)
    E = 1.0
    nu = 0.3
    c = E / (1.0 - nu**2)  # poisson's ratio

    # Constitutive matrix C acting on [e11, e22, e12]
    # W = 0.5 * eT C e
    W = 0.5 * c * (e11**2 + e22**2 + 2.0 * nu * e11 * e22 + 0.5 * (1.0 - nu) * e12**2)

    fx, fy = 0.0, -1.0
    # W_external = fx * ux + fy * uy
    return W


def potential_traction(soln, data=None, geo=None):
    """External Work Line integral integrand"""
    ux = soln["ux"]["value"]
    uy = soln["uy"]["value"]
    # traction force for element
    # W = uT t
    tx = 0
    ty = -1

    W = ux * tx + uy * ty
    return -W


# Two displacement DOFs per node
soln_space = SolutionSpace({"ux": "H1", "uy": "H1"})
geo_space = SolutionSpace({"x": "H1", "y": "H1"})
data_space = SolutionSpace({})  # empty for now

weakform_map = {
    "plane_stress": {
        "target": ["SURFACE1"],
        "weakform": potential_plane_stress,
    },
    "traction": {
        "target": ["LINE2"],
        "weakform": potential_traction,
    },
}

dirichlet_bc_map = {
    "clamp_x": {
        "type": "dirichlet",
        "target": "LINE4",  # left edge — fix ux
        "input": ["ux"],
        "start": True,
        "end": True,
    },
    "clamp_y": {
        "type": "dirichlet",
        "target": "LINE4",  # same edge — fix uy
        "input": ["uy"],
        "start": True,
        "end": True,
    },
}


mesh = Mesh("plate.inp")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)

args = parser.parse_args()

problem = Problem(
    mesh,
    soln_space,
    data_space,
    geo_space,
    weakform_map=weakform_map,
    dirichlet_bc_map=dirichlet_bc_map,
)

model = problem.create_model("plane_stress")

if args.build:
    model.build_module()

model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)
p = model.get_problem()

# Set geometry data
data = model.get_data_vector()
data["src_geo.x"] = mesh.X[:, 0]
data["src_geo.y"] = mesh.X[:, 1]

# Solve
x = p.create_vector()
mat = p.create_matrix()
g = p.create_vector()
p.hessian(1.0, x, mat)  # assembles K
p.gradient(1.0, x, g)  # assembles f (body force / BC terms)

K = am.tocsr(mat)
u = spsolve(K, g.get_array())

# print(np.max(u))
# Extract displacement fields
ux = u[model.get_indices("src_soln.ux")]
uy = u[model.get_indices("src_soln.uy")]

max_domain = np.max(np.maximum(ux, uy))
min_domain = np.min(np.minimum(ux, uy))


fig, ax = plt.subplots(ncols=2)
mesh.plot(
    ux,
    ax=ax[0],
)
mesh.plot(
    uy,
    ax=ax[1],
)

plt.show()
