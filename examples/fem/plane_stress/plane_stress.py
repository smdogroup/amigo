from amigo.fem import Mesh, Problem, SolutionSpace
import amigo as am
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy as np
import argparse


def potential_plane_stress(soln, data=None, geo=None):
    """Strain energy density (integrand of TPE equation)"""
    # Displacement gradients in physical space
    u_grad = soln["u"]["grad"]
    v_grad = soln["v"]["grad"]

    # Strain components
    exx = u_grad[0]
    eyy = v_grad[1]
    exy = u_grad[1] + v_grad[0]

    # Material properties (hardcoded or pull from data)
    E = 1.0
    nu = 0.3
    c = E / (1.0 - nu**2)  # poisson's ratio

    # Constitutive matrix C acting on [e11, e22, e12]
    # W = 0.5 * eT C e
    W = 0.5 * c * (exx**2 + eyy**2 + 2.0 * nu * exx * eyy + 0.5 * (1.0 - nu) * exy**2)

    return W


def potential_traction(soln, data=None, geo=None):
    """External Work Line integral integrand"""
    u = soln["u"]["value"]
    v = soln["v"]["value"]
    # traction force for element
    # W = uT t
    tx = 0
    ty = -1

    W = u * tx + v * ty
    return -W


# Two displacement DOFs per node
soln_space = SolutionSpace({"u": "H1", "v": "H1"})
geo_space = SolutionSpace({"x": "H1", "y": "H1"})
data_space = SolutionSpace({})  # empty for now

potential_map = {
    "plane_stress": {
        "target": ["SURFACE1"],
        "potential": potential_plane_stress,
    },
    "traction": {
        "target": ["LINE2"],
        "potential": potential_traction,
    },
}

bc_map = {
    "clamp_x": {
        "type": "dirichlet",
        "target": ["LINE4"],  # left edge — fix ux
        "input": ["u", "v"],
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
    potential_map=potential_map,
    bc_map=bc_map,
)

model = problem.create_model("plane_stress")

if args.build:
    model.build_module()

model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)
p = model.get_problem()

# Solve
x = p.create_vector()
mat = p.create_matrix()
g = p.create_vector()
p.hessian(1.0, x, mat)  # assembles K
p.gradient(1.0, x, g)  # assembles f (body force / BC terms)

K = am.tocsr(mat)
soln = spsolve(K, g.get_array())

# Extract displacement fields
u = soln[model.get_indices("src_soln.u")]
v = soln[model.get_indices("src_soln.v")]

fig, ax = plt.subplots(nrows=2)
mesh.plot(u, ax=ax[0])
mesh.plot(v, ax=ax[1])

plt.show()
