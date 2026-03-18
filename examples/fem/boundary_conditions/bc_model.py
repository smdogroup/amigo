import numpy as np
from amigo.fem import dot_product, Problem, Mesh, basis
import amigo as am
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import argparse


def weakform1(soln, data=None, geo=None):
    u = soln["u"]
    uvalue = u["value"]
    ugrad = u["grad"]
    wf = 0.5 * dot_product(ugrad, ugrad, n=2)
    return wf


def weakform2(soln, data=None, geo=None):
    u = soln["u"]
    uvalue = u["value"]
    ugrad = u["grad"]

    x = geo["x"]["value"]
    y = geo["y"]["value"]

    f = data["Jz"]["value"]
    wf = 0.5 * dot_product(ugrad, ugrad, n=2) + f * uvalue
    return wf


# Set arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

# Define mesh objects
meshes = {"Mesh0": Mesh("plate.inp")}

weakform_map = {
    "Mesh0": {
        "air": {"target": ["SURFACE1"], "weakform": weakform2},
    }
}

bc_map_mesh0 = {
    "DirichletLines": {
        "type": "dirichlet",
        "target": [
            "LINE1",
            "LINE2",
            "LINE5",
            "LINE6",
        ],
        "input": ["u"],
        "start": False,
        "end": False,
    },
    "SymmMesh0": {
        "type": "symmetric",
        "input": ["u"],
        "start": False,
        "end": False,
        "target": [["LINE8", "LINE7"], ["LINE3", "LINE4"]],
        "flip": [False, True],
        "scale": [1.0, -1.0],
    },
}
bc_map = {"Mesh0": bc_map_mesh0}

# Initialize the spaces (same for all domains)
soln_space = basis.SolutionSpace({"u": "H1"})
data_space = basis.SolutionSpace({"Jz": "const"})
geo_space = basis.SolutionSpace({"x": "H1", "y": "H1"})

# Define the global amigo model
model = am.Model("mms_module")


# Create an amigo model for each mesh
for mesh_name, mesh in meshes.items():
    problem = Problem(
        mesh,
        soln_space,
        data_space,
        geo_space,
        weakform_map=weakform_map[mesh_name],
        bc_map=bc_map[mesh_name],
    )
    sub_model = problem.create_model(mesh_name)
    model.add_model(mesh_name, sub_model)

# Build the model
if args.build:
    model.build_module()
model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)
p = model.get_problem()

# Set the problem data
data = model.get_data_vector()
data["Mesh0.src_geo.x"] = mesh.X[:, 0]
data["Mesh0.src_geo.y"] = mesh.X[:, 1]
data["Mesh0.src_data.Jz[0]"] = 10.0

mat = p.create_matrix()
alpha = 1.0
x = p.create_vector()
ans = p.create_vector()
g = p.create_vector()
rhs = p.create_vector()
p.hessian(alpha, x, mat)
p.gradient(alpha, x, g)
csr_mat = am.tocsr(mat)

ans.get_array()[:] = spsolve(csr_mat, g.get_array())
ans_local = ans
u = ans_local.get_array()[model.get_indices("Mesh0.src_soln.u")]

# Plot the solution field
mesh.plot(u)
plt.show()
