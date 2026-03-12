import numpy as np
from fem import (
    Mesh,
    weakform_air,
    weakform_coils,
    weakform_NS_Magnet,
    weakform_SN_Magnet,
    Problem,
)
import basis
import amigo as am
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


# Define mesh objects
meshes = {
    "Mesh0": Mesh("weakform_test_mesh.inp"),
    "Mesh1": Mesh("weakform_test_mesh.inp"),
}

# Define Dirichlet BCs for each mesh
dirichlet_bc_meshes = {
    "Mesh0": {
        "DirichletLine3": {
            "type": "dirichlet",
            "target": "LINE3",
            "input": ["u"],
            "start": True,
            "end": True,
        },
    },
    "Mesh1": {
        "DirichletLine1": {
            "type": "dirichlet",
            "target": "LINE1",
            "input": ["u"],
            "start": True,
            "end": True,
        },
    },
}

# Symmetric BCs mapping for each mesh
symm_bc_meshes = {
    "Mesh0": {
        "SymmMesh0": {
            "input": ["u"],
            "start": False,
            "end": False,
            "target": ["LINE2", "LINE4"],
            "flip": [False, False],
            "scale": [1.0, -1.0],
        },
    },
    "Mesh1": {
        "SymmMesh1": {
            "input": ["u"],
            "start": False,
            "end": False,
            "target": ["LINE2", "LINE4"],
            "flip": [False, False],
            "scale": [1.0, -1.0],
        },
    },
}

# Weak form mapping for each mesh
weakform_map = {
    "Mesh0": {
        "name": "Mesh0_weak_forms",
        "SURFACE1": weakform_air,
        "SURFACE2": weakform_coils,
        "SURFACE3": weakform_coils,
    },
    "Mesh1": {
        "name": "Mesh1_weak_forms",
        "SURFACE1": weakform_air,
        "SURFACE2": weakform_coils,
        "SURFACE3": weakform_coils,
    },
}

# Initialize the spaces (same for all domains)
soln_space = basis.SolutionSpace({"u": "H1"})
data_space = basis.SolutionSpace({"Jz": "const"})
geo_space = basis.SolutionSpace({"x": "H1", "y": "H1"})

# Define the global amigo model
main = am.Model("main")

# Create an amigo model for each mesh
for mesh_name, mesh in meshes.items():
    problem = Problem(
        mesh,
        soln_space,
        weakform_map[mesh_name],
        data_space=data_space,
        geo_space=geo_space,
        dirichlet_bc_map=dirichlet_bc_meshes[mesh_name],
        sym_bc_map=symm_bc_meshes[mesh_name],
        ndim=2,
    )
    model = problem.create_model(mesh_name)
    main.add_model(mesh_name, model)

# Extract the shared edge between the meshes
mesh0 = meshes["Mesh0"]
mesh1 = meshes["Mesh1"]
nodes_line_1 = mesh0.get_bc_nodes("LINE1", "T3D2")
nodes_line_3 = mesh1.get_bc_nodes("LINE3", "T3D2")
nodes_line_3 = np.flip(nodes_line_3)

# Know number of points along shared edge
npts_shared = 20

# Domain1 slides to the right by an integer value
# 5 is the length of the shared edge
slide_number = 0
x_offset = slide_number * (5.0 / npts_shared)


# Add continuity BCs to the global model
nodes_line_1_shared = nodes_line_1[slide_number:]
nodes_line_3_shared = (
    nodes_line_3[:] if slide_number == 0 else nodes_line_3[0:-slide_number]
)
for i in range(len(nodes_line_1_shared)):
    main.link(
        f"Mesh0.src_soln.u[{nodes_line_1_shared[i]}]",
        f"Mesh1.src_soln.u[{nodes_line_3_shared[i]}]",
    )

# BCs for the hanging edges
nodes_line_1_hanging = (
    nodes_line_1[:] if slide_number == 0 else nodes_line_1[0:slide_number]
)
nodes_line_3_hanging = nodes_line_3[-slide_number:]
for i in range(len(nodes_line_1_hanging)):
    main.link(
        f"Mesh0.src_soln.u[{nodes_line_1_hanging[i]}]",
        f"Mesh1.src_soln.u[{nodes_line_3_hanging[i]}]",
    )


# Build the model
main.build_module()
main.initialize(order_type=am.OrderingType.NESTED_DISSECTION)
p = main.get_problem()

# Set the problem data
data = main.get_data_vector()
data["Mesh0.src_geo.x"] = mesh0.X[:, 0]
data["Mesh0.src_geo.y"] = mesh0.X[:, 1]

data["Mesh1.src_geo.x"] = mesh1.X[:, 0]
data["Mesh1.src_geo.y"] = mesh1.X[:, 1]

data["Mesh0.src_data.Jz[0]"] = 0.0  # SURFACE1
data["Mesh0.src_data.Jz[1]"] = 0.0  # SURFACE2
data["Mesh0.src_data.Jz[2]"] = 0.0  # SURFACE3

data["Mesh1.src_data.Jz[0]"] = 0.0  # SURFACE1
data["Mesh1.src_data.Jz[1]"] = 0.0  # SURFACE2
data["Mesh1.src_data.Jz[2]"] = 10.0  # SURFACE3

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
u_domain0 = ans_local.get_array()[main.get_indices("Mesh0.src_soln.u")]
u_domain1 = ans_local.get_array()[main.get_indices("Mesh1.src_soln.u")]

max_domain = np.max(np.maximum(u_domain0, u_domain1))
min_domain = np.min(np.minimum(u_domain0, u_domain1))
print(max_domain, min_domain)

fig, ax = plt.subplots()
mesh.plot(
    u_domain0,
    ax=ax,
    x_offset=0.0,
    y_offset=0.0,
    max_level=max_domain,
    min_level=min_domain,
)
mesh.plot(
    u_domain1,
    ax=ax,
    x_offset=x_offset,
    y_offset=-5.0,
    max_level=max_domain,
    min_level=min_domain,
)
plt.show()

# x = problem.create_vector()
# mat = problem.create_matrix()
# rhs = model.create_vector()
# problem.gradient(1.0, x, rhs.get_vector())
# problem.hessian(1.0, x, mat)

# chol = am.SparseCholesky(mat)
# flag = chol.factor()
# print("flag = ", flag)
# chol.solve(rhs.get_vector())
