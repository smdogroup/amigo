import numpy as np
from amigo.fem import Mesh, Problem, SolutionSpace
import amigo as am
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import argparse
import matplotlib.tri as tri
import amigo as am
import argparse


def plot_mesh(mesh: Mesh, dx, dy):
    fig, ax = plt.subplots()

    # Get the domains in the mesh
    domains = mesh.get_domains()
    etypes = ["CPS3"]
    for name in domains:
        for etype in domains[name]:
            if etype not in etypes:
                continue
            conn = mesh.get_conn(name, etype)
            triang = tri.Triangulation(mesh.X[:, 0], mesh.X[:, 1], conn)
            triang_deformed = tri.Triangulation(
                mesh.X[:, 0] + dx, mesh.X[:, 1] + dy, conn
            )
            ax.triplot(triang, color="grey", linestyle="--", linewidth=1.0)
            ax.triplot(triang_deformed, color="blue", linestyle="-", linewidth=1.0)

    ax.set_aspect("equal")
    ax.set_axis_off()
    return


def strain_energy_integrand_nonlinear(soln, data=None, geo=None):
    # Gradients of the solution field
    dx_grad = soln["dx"]["grad"]
    dy_grad = soln["dy"]["grad"]

    # Constitutive model
    E = 1.0
    nu = 0.33
    mu = E / (2 * (1 + nu))
    kappa = E / (3 * (1 - 2 * nu))

    # Deformation Gradient Matrix (F)
    F11 = 1.0 + dx_grad[0]
    F12 = dx_grad[1]
    F21 = dy_grad[0]
    F22 = 1.0 + dy_grad[1]

    # Determinant of the deformation gradient (det(F))
    J = F11 * F22 - F12 * F21

    # Left Cauchy-Green Stress Tensor = (F)(F^T)
    B11 = F11**2 + F21 * F21
    B22 = F22**2 + F12 * F12
    traceB = B11 + B22 + 1.0

    # Gent Model strain energy density
    # Source: https://www.sandia.gov/files/sierra/SM_LAME_5_24/main/models/gent.html
    Jm = 200  # As Jm -> inf, the model resolves to the neo-Hookean model
    W_gent = 0.5 * kappa * ((0.5 * (J**2 - 1) - am.log(J))) - 0.5 * mu * Jm * am.log(
        1 - (traceB - 3) / Jm
    )
    W_total = W_gent

    return W_total


def strain_energy_integrand_linear(soln, data=None, geo=None):
    # Gradients of the solution field
    dx_grad = soln["dx"]["grad"]
    dy_grad = soln["dy"]["grad"]

    # Strains
    exx = dx_grad[0]
    eyy = dy_grad[1]
    exy = dx_grad[1] + dy_grad[0]

    # Constitutive model
    E = 1.0
    nu = 0.3

    # Assume uniform thickness
    t = 1.0

    # Plane strain
    alpha_plane_strain = E / ((1 + nu) * (1 - 2 * nu))
    C11 = 1.0 - nu
    C12 = nu
    C13 = 0.0
    C21 = nu
    C22 = 1.0 - nu
    C23 = 0.0
    C31 = 0.0
    C32 = 0.0
    C33 = 0.5 * (1 - 2 * nu)

    # Constant strain elements total internal strain energy
    alpha = alpha_plane_strain
    Wx = exx * (C11 * exx + C12 * eyy + C13 * exy)
    Wy = eyy * (C21 * exx + C22 * eyy + C23 * exy)
    Wxy = exy * (C31 * exx + C32 * eyy + C33 * exy)

    # Total strain potential energy integrand
    W_linear = 0.5 * alpha * (Wx + Wy + Wxy) * t

    # Total strain energy
    W_total = W_linear

    return W_total


# Set arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--build",
    dest="build",
    action="store_true",
    default=False,
    help="Enable building",
)

parser.add_argument(
    "--nl",
    action="store_true",
    default=False,
    help="Solve the nonlinear problem",
)

args = parser.parse_args()

# Define the spaces for the solutions
soln_space = SolutionSpace({"dx": "H1", "dy": "H1"})
geo_space = SolutionSpace({"x": "H1", "y": "H1"})
data_space = SolutionSpace({})

if args.nl == True:
    integrand_map = {
        "plane_stress": {
            "target": ["SURFACE1", "SURFACE2"],
            "integrand": strain_energy_integrand_nonlinear,
        },
    }

elif args.nl == False:
    integrand_map = {
        "plane_stress": {
            "target": ["SURFACE1", "SURFACE2"],
            "integrand": strain_energy_integrand_linear,
        },
    }

outer_plate_boundary = ["LINE1", "LINE2", "LINE3", "LINE4"]
inner_plate_boundary = ["LINE5", "LINE6", "LINE7", "LINE8"]
bc_map = {
    "clamp_outer": {
        "type": "dirichlet",
        "target": outer_plate_boundary,
        "input": ["dx", "dy"],
    },
    "clamp_inner": {
        "type": "dirichlet",
        "target": inner_plate_boundary,
        "input": ["dx", "dy"],
    },
}

# Create a global model
model = am.Model("model")

# Initialize the mesh object
mesh = Mesh("mesh.inp")

# Create the submodel for the plane stress problem
problem = Problem(
    mesh,
    soln_space,
    data_space,
    geo_space,
    integrand_map=integrand_map,
    bc_map=bc_map,
)
submodel = problem.create_model("mesh_morph")

# Add the submodel to the global model
model.add_model("submodel", submodel)

# Build the module
if args.build:
    model.build_module()

# Initilize the model
model.initialize()

# Define Nonezero Dirichlet BCs
num_nodes = mesh.get_num_nodes()
num_dims = 2
morph_bc = np.zeros((num_nodes, num_dims))
morph_bc[np.array(mesh.get_nodes_in_domain("LINE5")), 0] = 1.0
morph_bc[np.array(mesh.get_nodes_in_domain("LINE6")), 1] = -0.1
morph_bc[np.array(mesh.get_nodes_in_domain("LINE7")), 0] = -0.5
morph_bc[np.array(mesh.get_nodes_in_domain("LINE8")), 1] = -0.1

# Create the vectors and matrices for the model
x = model.create_vector()
g = model.create_vector()
mat = model.create_matrix()

if args.nl == True:
    print("\nNon-Linear Problem\n")
    max_iter = 100
    for i in range(100):
        # Store the prev state
        x_prev = np.copy(x[:])

        # Make sure x is updated with the boundary conditions
        for bc_line in inner_plate_boundary:
            nodes = mesh.get_nodes_in_domain(bc_line)
            for k in nodes:
                x[f"submodel.soln.dx[{k}]"] = morph_bc[k][0]
                x[f"submodel.soln.dy[{k}]"] = morph_bc[k][1]

        # Compute gradient and the Hessian
        model.eval_gradient(x, g)
        model.eval_hessian(x, mat)
        H = am.tocsr(mat)

        # Newton Update: Hess * x -= g
        x[:] -= spsolve(H, g[:])

        # Termination Criteria
        res_norm = np.linalg.norm(g[:])
        step_norm = np.linalg.norm(x_prev - x[:])
        print(f"Iteration: {i}, step norm: {step_norm:.4e}, res_norm: {res_norm:.4e}")
        if step_norm <= 1e-6 or res_norm <= 1e-6:
            print("\nExit Newton:")
            print(f"\tstep norm: {step_norm:.4e}")
            print(f"\tres. norm: {res_norm:.4e}")
            break

if args.nl == False:
    print("\nLinear Problem\n")

    # Make sure x is updated with the boundary conditions
    for bc_line in inner_plate_boundary:
        nodes = mesh.get_nodes_in_domain(bc_line)
        for k in nodes:
            x[f"submodel.soln.dx[{k}]"] = morph_bc[k][0]
            x[f"submodel.soln.dy[{k}]"] = morph_bc[k][1]

    # Evaluate the gradient and the Hessian
    model.eval_gradient(x, g)
    model.eval_hessian(x, mat)
    H = am.tocsr(mat)

    # Only a single Newton Iteration is required for the linear problem
    x[:] -= spsolve(H, g[:])


# Extract displacement fields
dx = x["submodel.soln.dx"]
dy = x["submodel.soln.dy"]

# Plot the mesh
plot_mesh(mesh, dx, dy)
plt.show()
