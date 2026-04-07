from amigo.fem import Mesh, Problem, SolutionSpace, FiniteElement, QuadQuadrature
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


class GapConstraint(am.Component):
    """Floor contact: gap = y_coord + v - y_floor >= 0"""

    def __init__(self):
        super().__init__()
        # self.add_data("y_coord")
        self.add_input("w")
        self.add_constraint("gap", lower=0.0, upper=float("inf"))

    def compute(self):
        # y = self.data["y_coord"]
        z = 0.0
        w = self.inputs["w"]
        z_contact = 75.0e3  # ceiling
        self.constraints["gap"] = -(w - z_contact)


def potential_bending(soln, data=None, geo=None):
    """Strain energy density (integrand of TPE equation)"""
    tx_grad = soln["tx"]["grad"]
    ty_grad = soln["ty"]["grad"]

    w_val = soln["w"]["value"]

    # components of kappa vector
    k1 = tx_grad[0]
    k2 = ty_grad[1]
    k3 = tx_grad[1] + ty_grad[0]

    # bending strain energy
    U_b = (
        0.5 * D * (k2 * (k1 * nu + k2) + k1 * (k1 + k2 * nu) + 1 / 2 * k3**2 * (1 - nu))
    )
    Work_ext = q0 * w_val

    return U_b - Work_ext


def potential_shear(soln, data=None, geo=None):
    w_grad = soln["w"]["grad"]
    tx_val = soln["tx"]["value"]
    ty_val = soln["ty"]["value"]

    # shear strain energy
    U_s = 0.5 * (A44 * (w_grad[0] + tx_val) ** 2 + A55 * (w_grad[1] + ty_val) ** 2)

    return U_s


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

# Create the solution spaces
soln_space = SolutionSpace({"w": "H1", "tx": "H1", "ty": "H1"})
geo_space = SolutionSpace({"x": "H1", "y": "H1"})
data_space = SolutionSpace({})

integrand_map = {
    "bending_potential": {
        "target": ["SURFACE1"],
        "integrand": potential_bending,
    },
    "shear_potential": {
        "target": ["SURFACE1"],
        "integrand": potential_shear,
        "rule": ["1dof"],  # reduced quadrature
    },
}
bc_map = {
    "clamp_w": {
        "type": "dirichlet",
        "input": ["w", "tx", "ty"],
        "target": ["LINE1", "LINE2", "LINE3", "LINE4"],
    },
}
model = am.Model("model")
# Load the plate
mesh = Mesh("plate.inp")

element_objs = {}
for etype in ["CPS4"]:
    # Get the basis and quadrature objects needed for the element
    soln_basis = mesh.get_basis(soln_space, etype, kind="input")
    geo_basis = mesh.get_basis(geo_space, etype, kind="data")
    data_basis = mesh.get_basis(data_space, etype, kind="data")
    quadrature = QuadQuadrature(1)

    # Create the shear element
    elem_name = f"PlateShear{etype}"
    elem = FiniteElement(
        elem_name, soln_basis, data_basis, geo_basis, quadrature, potential_shear
    )

    # Set the element objects
    element_objs[("plate", etype)] = elem

problem = Problem(
    mesh,
    soln_space,
    data_space,
    geo_space,
    integrand_map=integrand_map,
    bc_map=bc_map,
    element_objs=element_objs,
)
submodel = problem.create_model("plate")
model.add_model("fem", submodel)
# Link nodes of constraint model to fem submodel

# Floor contact constraint (only on non-BC nodes to avoid rank-deficient Jacobian)
# non_bc_nodes = [i for i in range(n) if i not in bc_nodes]
nodes = mesh.get_nodes_in_domain("SURFACE1")
nodes_edge1 = mesh.get_nodes_in_domain("LINE1")
nodes_edge2 = mesh.get_nodes_in_domain("LINE2")
nodes_edge3 = mesh.get_nodes_in_domain("LINE3")
nodes_edge4 = mesh.get_nodes_in_domain("LINE4")
non_bc_nodes = []

for i in nodes:
    if i not in nodes_edge1:
        non_bc_nodes.append(i)
    if i not in nodes_edge2:
        non_bc_nodes.append(i)
    if i not in nodes_edge3:
        non_bc_nodes.append(i)
    if i not in nodes_edge4:
        non_bc_nodes.append(i)

model.add_component("contact", len(non_bc_nodes), GapConstraint())
# model.link("fem.geo.y", "contact.y_coord", src_indices=non_bc_nodes)
model.link("fem.soln.w", "contact.w", src_indices=non_bc_nodes)

if args.build:
    model.build_module()

model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)
p = model.get_problem()

xm = model.create_vector()

# Solve
x = xm.get_vector()
mat = p.create_matrix()
g = p.create_vector()

# print("Evaluating the Hessian...")
# p.hessian(1.0, x, mat)  # assembles K
# p.gradient(1.0, x, g)  # assembles f (body force / BC terms)

# print("Solving...")
# K = am.tocsr(mat)
# x.get_array()[:] = spsolve(K, g.get_array())


opt_options = {
    "max_iterations": 200,
    "convergence_tolerance": 1e-6,
    "max_line_search_iterations": 1,
    "initial_barrier_param": 0.1,
    # "initial_barrier_param": 1.0,
    # "max_iterations": 2000,
    # "fraction_to_boundary": 0.995,
    # "max_line_search_iterations": 40,
    # "init_least_squares_multipliers": False,
    # "filter_line_search": True,
    # "second_order_correction": True,
    # "check_update_step": True,
}

# indiv. Amigo problem:
opt = am.Optimizer(
    model,
    x,
)
opt.optimize(opt_options)

print("Plotting...")

w = xm["fem.soln.w"]
tx = xm["fem.soln.tx"]
ty = xm["fem.soln.ty"]

w_normalized = 1000 * D * w / (q0 * 1.0)
print("max normalized plate deflection", np.max(np.abs(w_normalized)))
print("max plate deflection", np.max(w), np.min(w))

fig, ax = plt.subplots(1, 3, figsize=(8, 3))
for index, soln in enumerate([w, tx, ty]):
    mesh.plot(soln, ax=ax[index])
plt.show()
