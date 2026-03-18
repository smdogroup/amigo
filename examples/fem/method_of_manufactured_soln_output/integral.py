import numpy as np
from amigo.fem import dot_product, Problem, Mesh, basis
import amigo as am
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import argparse


# Method of Manufactured solutions
def output(soln, data=None, geo=None):
    u = soln["u"]
    uvalue = u["value"]
    return {"integrand": uvalue}


def weakform(soln, data=None, geo=None):
    u = soln["u"]
    uvalue = u["value"]
    ugrad = u["grad"]
    x = geo["x"]["value"]
    y = geo["y"]["value"]

    f = -2 * np.pi**2 * am.sin(np.pi * x) * am.sin(np.pi * y)

    wf = 0.5 * dot_product(ugrad, ugrad, n=2) - f * uvalue
    return wf


def exact(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


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
        "air": {"target": ["SURFACE1"], "weakform": weakform},
    }
}

bc_map_mesh0 = {
    "DirichletBC": {
        "type": "dirichlet",
        "target": [
            "LINE1",
            "LINE2",
            "LINE3",
            "LINE4",
            "LINE5",
            "LINE6",
            "LINE7",
            "LINE8",
        ],
        "input": ["u"],
        "start": True,
        "end": True,
    }
}
bc_map = {"Mesh0": bc_map_mesh0}

output_map = {
    "Mesh0": {
        "integral": {
            "names": ["integrand"],
            "target": ["SURFACE1"],
            "function": output,
        },
    }
}

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
        output_map=output_map[mesh_name],
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

# Compute the exact solution field
u_exact = exact(mesh.X[:, 0], mesh.X[:, 1])

# Get the output
output = model.create_output_vector()
p.compute_output(ans, output.get_vector())
exact_integral = 4 / np.pi**2
amigo_integral = output["Mesh0.outputs.integrand[0]"]
rel_err = np.abs(exact_integral - amigo_integral) / exact_integral
norm = np.linalg.norm(u_exact - u)
print(f"Amigo Integral: {exact_integral:.4e}")
print(f"Analytic Integral: {amigo_integral:.4e}")
print(f"Integral Rel. Err.: {rel_err:.4e}")
print(f"||u_exact - u||: {norm:.4e}")

# Plot the solution field
# fig, ax = plt.subplots(ncols=2)
# mesh.plot(u, ax=ax[0])
# mesh.plot(u_exact, ax=ax[1])
# plt.show()
