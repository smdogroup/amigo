import argparse
import amigo as am
from amigo.fem import MITCTyingStrain, MITCElement, SolutionSpace, Mesh, Problem
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import time


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


def integrand(soln, data=None, geo=None):
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
parser.add_argument(
    "--solver",
    dest="solver",
    choices=["cholesky", "cholesky_left", "ldl", "scipy"],
    default="cholesky",
)
args = parser.parse_args()

# Create the solution spaces
soln_space = SolutionSpace({"w": "H1", "tx": "H1", "ty": "H1"})
geo_space = SolutionSpace({"x": "H1", "y": "H1"})
data_space = SolutionSpace({})

integrand_map = {
    "plate": {
        "target": ["SURFACE1"],
        "integrand": integrand,
    },
}
bc_map = {
    "clamped": {
        "type": "dirichlet",
        "input": ["w", "tx", "ty"],
        "target": ["LINE1", "LINE2", "LINE3", "LINE4"],
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
    "MITC4", soln_basis, data_basis, geo_basis, quadrature, mitc, integrand
)

# Set the element objects
element_objs = {("plate", etype): quad_elem}

# Create the finite element problem
problem = Problem(
    mesh,
    soln_space,
    data_space,
    geo_space,
    integrand_map=integrand_map,
    bc_map=bc_map,
    element_objs=element_objs,
)
model = problem.create_model("plate")

if args.build:
    model.build_module()

model.initialize()

print("Number of variables... ", model.num_variables)

# Create the vectors and matrices for the model
x = model.create_vector()
g = model.create_vector()
mat = model.create_matrix()

print("Evaluating the Hessian...")
model.eval_gradient(x, g)
model.eval_hessian(x, mat)

# Solve the equations
print("Solving...")

if args.solver == "cholesky" or args.solver == "ldl":
    stype = am.SolverType.CHOLESKY
    if args.solver == "ldl":
        stype = am.SolverType.LDL

    ldl = am.SparseLDL(mat, stype, ustab=0.05)
    start_time = time.perf_counter()
    flag = ldl.factor()
    end_time = time.perf_counter()
    if flag != 0:
        print(f"LDL factor flag {flag}")

    x[:] = g[:]
    ldl.solve(x.get_vector())
    if stype == am.SolverType.LDL:
        print("Inertia: ", ldl.get_inertia())
elif args.solver == "cholesky_left":
    chol = am.SparseCholesky(mat)
    start_time = time.perf_counter()
    flag = chol.factor()
    end_time = time.perf_counter()
    if flag != 0:
        print(f"Cholesky factor flag {flag}")

    x[:] = g[:]
    chol.solve(x.get_vector())
elif args.solver == "scipy":
    csr = am.tocsr(mat)

    # This isn't a completely fair comparison
    start_time = time.perf_counter()
    x[:] = spsolve(csr, g[:])
    end_time = time.perf_counter()

print(f"Solve time... {end_time - start_time:.6f} seconds")

print("Plotting...")
w = x["soln.w"]
tx = x["soln.tx"]
ty = x["soln.ty"]

fig, ax = plt.subplots(1, 3, figsize=(8, 3))
for index, soln in enumerate([w, tx, ty]):
    mesh.plot(soln, ax=ax[index])

plt.show()
