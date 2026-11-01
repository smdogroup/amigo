import argparse
import amigo as am
import numpy as np
from amigo.fem import SolutionSpace, Mesh, Problem, IntegrandFormulation
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from amigo.fem.visualization import build_grid


def integrand(test, soln, data=None, geo=None):
    v = test["u"].value
    vx = test["u"].grad

    u = soln["u"].value
    ux = soln["u"].grad

    x = geo["x"].value
    y = geo["y"].value

    f = -2 * np.pi**2 * am.sin(np.pi * (x + 0.5)) * am.sin(np.pi * (y + 0.5))

    return vx[0] * ux[0] + vx[1] * ux[1] - v * f


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

# Create the solution spaces
soln_space = SolutionSpace({"soln": {"u": "H1"}}, degree=1)
geo_space = SolutionSpace({"geo": {("x", "y"): "H1"}}, degree=1)

integrand_map = {
    "domain": {
        "target": ["SURFACE1"],
        "integrand": integrand,
    },
}
bc_map = {
    "bc": {
        "type": "dirichlet",
        "input": ["u"],
        "target": ["LINE1", "LINE2", "LINE3", "LINE4"],
    },
}

# Load the plate
# mesh = Mesh("plate.inp")
mesh = Mesh("plate.inp")

problem = Problem(
    mesh,
    soln_space,
    geo_space,
    integrand_map=integrand_map,
    integrand_formulation=IntegrandFormulation.WEAK,
    bc_map=bc_map,
)
model = problem.create_model("poisson")

if args.build:
    model.build_module()

model.initialize()

x = model.create_vector()
g = model.create_vector()
mat = model.create_matrix()

print("Evaluating the Hessian...")
model.eval_gradient(x, g)
model.eval_hessian(x, mat)

# Solve the equations
print("Solving...")
K = am.tocsr(mat)
x[:] = spsolve(K, g[:])

print("Plotting...")
data = model.get_data_vector()
xpts = data["geo.x"]
ypts = data["geo.y"]

u = x["soln.u"]
u_exact = np.sin(np.pi * (xpts + 0.5)) * np.sin(np.pi * (ypts + 0.5))

# print(np.max(np.absolute(u - u_exact)) / np.max(u_exact))
grid = build_grid(problem, "u", x)
grid.plot(scalars="u", cmap="coolwarm", show_edges=True)
