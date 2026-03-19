import argparse
import amigo as am
from amigo.fem import MITCTyingStrain, MITCElement, SolutionSpace, Mesh, Problem
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


class MITC4PlateTying(MITCTyingStrain):
    def __init__(self):
        # Two tying points each for g23 and g13, respectively
        self.tying_points = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def get_tying_points(self):
        return self.tying_points

    def eval_tying_strain(self, idx, geo, soln):
        tx = soln["tx"]["value"]
        ty = soln["ty"]["value"]

        # Get the derivatives in the computational coordinates
        w_1 = soln["w"]["grad"][0]
        w_2 = soln["w"]["grad"][1]

        # Derivatives wrt computational coordinates
        x_1 = geo["x"]["grad"][0]
        y_1 = geo["y"]["grad"][0]
        x_2 = geo["x"]["grad"][1]
        y_2 = geo["y"]["grad"][1]

        if idx < 2:
            g23 = w_2 + tx * x_2 + ty * y_2
            return g23
        elif idx < 4:
            g13 = w_1 + tx * x_1 + ty * y_1
            return g13

        raise ValueError("Tying point index out of range")

    def interp_and_transform(self, pt, Jinv, e):
        # Interpolate the tensorial components of the tying strains
        g23 = 0.5 * ((1.0 - pt[0]) * e[0] + (1.0 + pt[0]) * e[1])
        g13 = 0.5 * ((1.0 - pt[1]) * e[2] + (1.0 + pt[1]) * e[3])

        gxz = Jinv[0][0] * g13 + Jinv[1][0] * g23
        gyz = Jinv[0][1] * g13 + Jinv[1][1] * g23

        # Convert from tensorial to real strain
        out = {}
        out["gxz"] = {"value": gxz}
        out["gyz"] = {"value": gyz}

        return out


def potential(soln, data=None, geo=None):
    """Strain energy density (integrand of TPE equation)"""

    E = 1.0  # Young's modulus
    nu = 0.3  # Poisson's ratio
    t = 0.01  # Plate thickness
    G = 0.5 * E / (1.0 + nu)  # Shear stiffness
    ks = 5.0 / 6.0  # Shear correction factor
    D = (E * t**3) / (12 * (1 - nu**2))  # Bending stiffness

    tx_grad = soln["tx"]["grad"]
    ty_grad = soln["ty"]["grad"]
    w_val = soln["w"]["value"]
    gxz = soln["gxz"]["value"]
    gyz = soln["gyz"]["value"]

    # components of kappa vector
    kxx = tx_grad[0]
    kyy = ty_grad[1]
    kxy = tx_grad[1] + ty_grad[0]

    # bending strain energy
    Ub = 0.5 * D * (kxx**2 + kyy**2 + 2 * nu * kxx * kyy + 0.5 * kxy**2 * (1 - nu))

    # shear strain energy
    Us = 0.5 * ks * G * t * (gyz**2 + gxz**2)

    W = -1 * w_val

    return Ub + Us - W


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

# Two displacement DOFs per node
soln_space = SolutionSpace({"w": "H1", "tx": "H1", "ty": "H1"})
geo_space = SolutionSpace({"x": "H1", "y": "H1"})
data_space = SolutionSpace({})

potential_map = {
    "plate": {
        "target": ["SURFACE1"],
        "potential": potential,
    },
}
bc_map = {
    "clamp_w": {
        "type": "dirichlet",
        "input": ["w", "tx", "ty"],
        "target": ["LINE1", "LINE2", "LINE3", "LINE4"],
        "start": True,
        "end": True,
    },
}

# Load the plate
mesh = Mesh("plate.inp")

# Get the basis and quadrature objects needed for the element
etype = "CPS4"
soln_basis = mesh.get_basis(soln_space, etype, kind="input")
geo_basis = mesh.get_basis(geo_space, etype, kind="data")
data_basis = mesh.get_basis(data_space, etype, kind="data")
quadrature = mesh.get_quadrature(etype)
mitc = MITC4PlateTying()

# Create the MITC4 element
quad_elem = MITCElement(
    "MITC4", soln_basis, data_basis, geo_basis, quadrature, mitc, potential
)

# Set the element objects
element_objs = {("plate", etype): quad_elem}

# Create the finite element problem
problem = Problem(
    mesh,
    soln_space,
    data_space,
    geo_space,
    potential_map=potential_map,
    bc_map=bc_map,
    element_objs=element_objs,
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

fig, ax = plt.subplots(1, 3, figsize=(8, 3))
for index, soln in enumerate([w, tx, ty]):
    mesh.plot(soln, ax=ax[index])

plt.show()
