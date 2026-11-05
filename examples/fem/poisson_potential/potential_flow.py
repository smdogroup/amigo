"""Solve 2D source-sink potential flow with a regularized source pair.

The velocity is grad(phi), so it is irrotational. Its divergence is nonzero
inside the smooth source/sink regions and zero away from them.
"""

import argparse
import math

import amigo as am
from amigo.fem import Mesh, Problem, basis, build_grid, dot_product


SOURCE_STRENGTH = 1.0
SOURCE_CENTER = (0.35, 0.5)
SINK_CENTER = (0.65, 0.5)
SOURCE_WIDTH = 0.04


def gaussian_source(x, y, center, strength):
    dx = x - center[0]
    dy = y - center[1]
    radius_squared = dx**2 + dy**2
    return (
        strength
        / (2.0 * math.pi * SOURCE_WIDTH**2)
        * am.exp(-radius_squared / (2.0 * SOURCE_WIDTH**2))
    )


def potential(soln, data=None, geo=None):
    phi = soln["phi"].value
    grad_phi = soln["phi"].grad
    x = geo["x"].value
    y = geo["y"].value

    source = gaussian_source(x, y, SOURCE_CENTER, SOURCE_STRENGTH)
    sink = gaussian_source(x, y, SINK_CENTER, -SOURCE_STRENGTH)
    return 0.5 * dot_product(grad_phi, grad_phi, n=2) + (source + sink) * phi


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

mesh = Mesh("plate.inp")

integrand_map = {
    "fluid": {"target": "SURFACE1", "integrand": potential},
}
boundary_lines = [
    "LINE1",
    "LINE2",
    "LINE3",
    "LINE4",
    "LINE5",
    "LINE6",
    "LINE7",
    "LINE8",
]
bc_map = {
    "boundary": {
        "type": "dirichlet",
        "target": boundary_lines,
        "input": ["phi"],
    },
}

soln_space = basis.SolutionSpace({"soln": {"phi": "H1"}})
geo_space = basis.SolutionSpace({"geo": {("x", "y"): "H1"}})

problem = Problem(
    mesh,
    soln_space,
    geo_space,
    integrand_map=integrand_map,
    bc_map=bc_map,
)
model = problem.create_model("potential_flow")

if args.build:
    model.build_module()

model.initialize()

mat = model.create_matrix()
x = model.create_vector()
g = model.create_vector()

model.eval_hessian(x, mat)
model.eval_gradient(x, g)

solver = am.SparseLDL(mat, am.SolverType.CHOLESKY)
solver.factor()
x[:] = -g[:]
solver.solve(x.get_vector())

grid = build_grid(problem, "phi", x)
flow_grid = grid.compute_derivative(
    scalars="phi", gradient="velocity", vorticity=False
)
grid.plot(scalars="phi", cmap="coolwarm", show_edges=True)

velocity_glyphs = flow_grid.glyph(orient="velocity", scale=False, factor=0.04)
velocity_glyphs.plot(show_edges=True)
