from amigo.fem import Mesh, Problem, SolutionSpace
import amigo as am
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy as np
import argparse

E = 1.0  # Young's modulus
nu = 0.3  # Poisson's ratio
t = 0.005  # Plate thickness
k_s = 5.0 / 6.0  # Shear correction factor
D = E * t**3 / (12.0 * (1.0 - nu**2))  # Bending stiffness
G = E / (2.0 * (1.0 + nu))  # Shear stiffness  (= k_s * G * t)
A44 = k_s * G * t
A55 = A44
q0 = 1  # N/m^2, load intensity


def potential_bending(soln, data=None, geo=None):
    """Strain energy density (integrand of TPE equation)"""
    w_grad = soln["w"]["grad"]
    tx_grad = soln["tx"]["grad"]
    ty_grad = soln["ty"]["grad"]

    w_val = soln["w"]["value"]
    tx_val = soln["tx"]["value"]
    ty_val = soln["ty"]["value"]

    # components of kappa vector
    k1 = tx_grad[0]
    k2 = ty_grad[1]
    k3 = tx_grad[1] + ty_grad[0]

    # bending strain energy
    U_b = (
        0.5 * D * (k2 * (k1 * nu + k2) + k1 * (k1 + k2 * nu) + 1 / 2 * k3**2 * (1 - nu))
    )
    Work_ext = -q0 * w_val

    return U_b - Work_ext


def potential_shear(soln, data=None, geo=None):
    w_grad = soln["w"]["grad"]
    tx_grad = soln["tx"]["grad"]
    ty_grad = soln["ty"]["grad"]

    w_val = soln["w"]["value"]
    tx_val = soln["tx"]["value"]
    ty_val = soln["ty"]["value"]

    # shear strain energy
    U_s = 0.5 * (A44 * (w_grad[0] + tx_val) ** 2 + A55 * (w_grad[1] + ty_val) ** 2)

    return U_s


# Two displacement DOFs per node
soln_space = SolutionSpace({"w": "H1", "tx": "H1", "ty": "H1"})
geo_space = SolutionSpace({"x": "H1", "y": "H1"})
data_space = SolutionSpace({})  # empty for now

weakform_map = {
    "bending_potential": {
        "target": ["SURFACE1"],
        "weakform": potential_bending,
    },
    "shear_potential": {
        "target": ["SURFACE1"],
        "weakform": potential_shear,
    },
}


bc_map = {
    "clamp_w": {
        "type": "dirichlet",
        "input": ["w", "tx", "ty"],
        "target": ["LINE1", "LINE2", "LINE3", "LINE4"],  # left edge — fix ux
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
    bc_map=bc_map,
)

model = problem.create_model("plate")

if args.build:
    model.build_module()

model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)
p = model.get_problem()

# Set geometry data
data = model.get_data_vector()
data["src_geo.x"] = mesh.X[:, 0]
data["src_geo.y"] = mesh.X[:, 1]

xm = model.create_vector()

# Solve
x = xm.get_vector()
mat = p.create_matrix()
g = p.create_vector()

print("Evaluating the Hessian...")
p.hessian(1.0, x, mat)  # assembles K
p.gradient(1.0, x, g)  # assembles f (body force / BC terms)

print("Solving...")
K = am.tocsr(mat)
x.get_array()[:] = spsolve(K, g.get_array())

print("Plotting...")

w = xm["src_soln.w"]
tx = xm["src_soln.tx"]
ty = xm["src_soln.ty"]

np.save("w_reducedshear.npy", w)
w_shearlocked = np.load("w_shearlocked.npy")
mesh.plot_3d(w)
print(np.max(np.abs(((w - w_shearlocked)))))
plt.show()
