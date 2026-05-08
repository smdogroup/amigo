"""
MITC4 shell FEM on a cylinder — end traction variant.
Load applied as axial traction integrated along top ring (LINE1, T3D2),
instead of a distributed surface pressure.
"""

import argparse
import numpy as np
import amigo as am
from amigo.fem import MITCTyingStrain, MITCElement, SolutionSpace, Mesh, Problem
from amigo.fem.basis import QuadLagrangeBasis
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
from utils import write_vtu


class ShellGeoBasis(QuadLagrangeBasis):
    def __init__(self, names, kind="data"):
        super().__init__(1, names, kind=kind)

    def compute_transform(self, geo):
        x1, x2 = geo["x"]["grad"]
        y1, y2 = geo["y"]["grad"]
        z1, z2 = geo["z"]["grad"]

        # Surface normal via cross product a1 x a2
        nx = y1 * z2 - z1 * y2
        ny = z1 * x2 - x1 * z2
        nz = x1 * y2 - y1 * x2
        detJ = am.sqrt(nx**2 + ny**2 + nz**2)

        # Covariant metric tensor
        g11 = x1 * x1 + y1 * y1 + z1 * z1
        g12 = x1 * x2 + y1 * y2 + z1 * z2
        g22 = x2 * x2 + y2 * y2 + z2 * z2
        inv_det_g = 1.0 / (g11 * g22 - g12 * g12)

        # Contravariant metric
        g11c = g22 * inv_det_g
        g12c = -g12 * inv_det_g
        g22c = g11 * inv_det_g

        # 2x3 left-pseudoinverse: Jinv = G^{-1} * J^T
        Jinv = [
            [g11c * x1 + g12c * x2, g11c * y1 + g12c * y2, g11c * z1 + g12c * z2],
            [g12c * x1 + g22c * x2, g12c * y1 + g22c * y2, g12c * z1 + g22c * z2],
        ]
        return detJ, Jinv

    def transform(self, detJ, Jinv, orig):
        # Pass through parametric grads unchanged — integrand uses covariant components
        return {
            name: {"value": orig[name]["value"], "grad": orig[name]["grad"]}
            for name in orig
        }


class ShellSolnBasis(QuadLagrangeBasis):
    def __init__(self, names, kind="input"):
        super().__init__(1, names, kind=kind)

    def transform(self, detJ, Jinv, orig):
        return {
            name: {"value": orig[name]["value"], "grad": orig[name]["grad"]}
            for name in orig
        }


class MITC4ShellTying(MITCTyingStrain):
    def get_tying_points(self):
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def eval_tying_strain(self, idx, geo, soln):
        x1, x2 = geo["x"]["grad"]
        y1, y2 = geo["y"]["grad"]
        z1, z2 = geo["z"]["grad"]

        # Unit normal
        nx = y1 * z2 - z1 * y2
        ny = z1 * x2 - x1 * z2
        nz = x1 * y2 - y1 * x2
        nmag = am.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / nmag, ny / nmag, nz / nmag

        rx, ry, rz = soln["rx"]["value"], soln["ry"]["value"], soln["rz"]["value"]
        u1, u2 = soln["u"]["grad"]
        v1, v2 = soln["v"]["grad"]
        w1, w2 = soln["w"]["grad"]

        # d = theta x n = (rx,ry,rz) x (nx,ny,nz)
        dx = ry * nz - rz * ny
        dy = rz * nx - rx * nz
        dz = rx * ny - ry * nx

        if idx < 2:
            # gamma_23
            return u2 * nx + v2 * ny + w2 * nz + dx * x2 + dy * y2 + dz * z2
        else:
            # gamma_13
            return u1 * nx + v1 * ny + w1 * nz + dx * x1 + dy * y1 + dz * z1

    def interp_and_transform(self, pt, Jinv, e):
        xi, eta = pt
        gamma_2 = 0.5 * ((1.0 - xi) * e[0] + (1.0 + xi) * e[1])
        gamma_1 = 0.5 * ((1.0 - eta) * e[2] + (1.0 + eta) * e[3])
        return {"gs1": {"value": gamma_1}, "gs2": {"value": gamma_2}}


def shell_integrand(soln, data=None, geo=None):
    """
    Shell Potential Energy Integrand
    Membrane energy + Bending energy + Shear energy + Drill Energy
    """

    E = 70e9
    nu = 0.3
    t = 0.002
    ks = 5.0 / 6.0
    G = E / (2.0 * (1.0 + nu))

    x1, x2 = geo["x"]["grad"]
    y1, y2 = geo["y"]["grad"]
    z1, z2 = geo["z"]["grad"]

    g11 = x1 * x1 + y1 * y1 + z1 * z1
    g12 = x1 * x2 + y1 * y2 + z1 * z2
    g22 = x2 * x2 + y2 * y2 + z2 * z2
    inv_det_g = 1.0 / (g11 * g22 - g12 * g12)
    g11c = g22 * inv_det_g
    g12c = -g12 * inv_det_g
    g22c = g11 * inv_det_g

    u1, u2 = soln["u"]["grad"]
    v1, v2 = soln["v"]["grad"]
    w1, w2 = soln["w"]["grad"]
    rx1, rx2 = soln["rx"]["grad"]
    ry1, ry2 = soln["ry"]["grad"]
    rz1, rz2 = soln["rz"]["grad"]

    # Unit normal
    nx = y1 * z2 - z1 * y2
    ny = z1 * x2 - x1 * z2
    nz = x1 * y2 - y1 * x2
    nmag = am.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / nmag, ny / nmag, nz / nmag

    eps11 = u1 * x1 + v1 * y1 + w1 * z1
    eps22 = u2 * x2 + v2 * y2 + w2 * z2
    eps12 = 0.5 * (u1 * x2 + v1 * y2 + w1 * z2 + u2 * x1 + v2 * y1 + w2 * z1)

    tr_eps = g11c * eps11 + 2.0 * g12c * eps12 + g22c * eps22
    eps11c = g11c * g11c * eps11 + 2.0 * g11c * g12c * eps12 + g12c * g12c * eps22
    eps12c = (
        g11c * g12c * eps11 + (g11c * g22c + g12c * g12c) * eps12 + g12c * g22c * eps22
    )
    eps22c = g12c * g12c * eps11 + 2.0 * g12c * g22c * eps12 + g22c * g22c * eps22

    fac_m = E * t / (1.0 - nu**2)
    U_membrane = (
        0.5
        * fac_m
        * (
            nu * tr_eps * tr_eps
            + (1.0 - nu) * (eps11c * eps11 + 2.0 * eps12c * eps12 + eps22c * eps22)
        )
    )

    # Bending strains: kappa_ab = 0.5*(d,a · a_b + d,b · a_a)
    # d = theta x n
    # d,1 = (rx1, ry1, rz1) x (nx, ny, nz)
    d1x = ry1 * nz - rz1 * ny
    d1y = rz1 * nx - rx1 * nz
    d1z = rx1 * ny - ry1 * nx
    # d,2 = (rx2, ry2, rz2) x (nx, ny, nz)
    d2x = ry2 * nz - rz2 * ny
    d2y = rz2 * nx - rx2 * nz
    d2z = rx2 * ny - ry2 * nx

    kap11 = d1x * x1 + d1y * y1 + d1z * z1
    kap22 = d2x * x2 + d2y * y2 + d2z * z2
    kap12 = 0.5 * (d1x * x2 + d1y * y2 + d1z * z2 + d2x * x1 + d2y * y1 + d2z * z1)

    tr_kap = g11c * kap11 + 2.0 * g12c * kap12 + g22c * kap22
    kap11c = g11c * g11c * kap11 + 2.0 * g11c * g12c * kap12 + g12c * g12c * kap22
    kap12c = (
        g11c * g12c * kap11 + (g11c * g22c + g12c * g12c) * kap12 + g12c * g22c * kap22
    )
    kap22c = g12c * g12c * kap11 + 2.0 * g12c * g22c * kap12 + g22c * g22c * kap22

    fac_b = E * t**3 / (12.0 * (1.0 - nu**2))
    U_bending = (
        0.5
        * fac_b
        * (
            nu * tr_kap * tr_kap
            + (1.0 - nu) * (kap11c * kap11 + 2.0 * kap12c * kap12 + kap22c * kap22)
        )
    )
    # Transverse shear (MITC tying, covariant)
    gs1, gs2 = soln["gs1"]["value"], soln["gs2"]["value"]
    gs1c = g11c * gs1 + g12c * gs2
    gs2c = g12c * gs1 + g22c * gs2
    U_shear = 0.5 * ks * G * t * (gs1c * gs1 + gs2c * gs2)

    # Drilling penalty
    # Drilling rotation = dot(theta, n) = rx*nx + ry*ny + rz*nz
    rx, ry, rz = soln["rx"]["value"], soln["ry"]["value"], soln["rz"]["value"]
    drill = rx * nx + ry * ny + rz * nz

    # Also penalize the antisymmetric in-plane displacement gradient
    omega = 0.5 * (u1 * x2 + v1 * y2 + w1 * z2 - u2 * x1 - v2 * y1 - w2 * z1)
    drill_alpha = 1.0e-3 * E * t / (1.0 - nu**2)
    U_drill = 0.5 * drill_alpha * (drill * drill + omega * omega)

    return U_membrane + U_bending + U_shear + U_drill


def end_traction_integrand(soln, data=None, geo=None):
    """Axial traction on top ring: W_ext = tz * w, integrated along LINE1 (T3D2)."""
    tz = 1e6  # N/m — axial load per unit arc length (negative = compression)
    return -tz * soln["w"]["value"]


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

mesh = Mesh("cylinder_coarse.inp")
domains = mesh.get_domains()


# groups made with cylinder_mesh.py
lateral_surfaces = ['Surface1']
bottom_line = 'Line3'
lateral_line = 'Line2'
top_line = 'Line1'
print(f"Lateral: {lateral_surfaces}, bottom: {bottom_line}, top: {top_line}")

soln_space = SolutionSpace(
    {"u": "H1", "v": "H1", "w": "H1", "rx": "H1", "ry": "H1", "rz": "H1"}
)
geo_space = SolutionSpace({"x": "H1", "y": "H1", "z": "H1"})
data_space = SolutionSpace({})

etype = "CPS4"

soln_basis = ShellSolnBasis(["u", "v", "w", "rx", "ry", "rz"], kind="input")
data_basis = mesh.get_basis(data_space, etype, kind="data")
geo_basis = ShellGeoBasis(["x", "y", "z"], kind="data")
quadrature = mesh.get_quadrature(etype)
mitc = MITC4ShellTying()

shell_elem = MITCElement(
    "Shell", soln_basis, data_basis, geo_basis, quadrature, mitc, shell_integrand
)

integrand_map = {
    "shell": {
        "target": lateral_surfaces,
        "integrand": shell_integrand,
    },
    "end_traction": {
        "target": [top_line],  # LINE1: top ring, T3D2 elements
        "integrand": end_traction_integrand,
    },
}
bc_map = {
    "pinned_bottom": {
        "type": "dirichlet",
        "input": ["u", "v", "w", "rx", "ry", "rz"],
        "target": [bottom_line],
    },
}

problem = Problem(
    mesh,
    soln_space,
    data_space,
    geo_space,
    integrand_map=integrand_map,
    bc_map=bc_map,
    element_objs={("shell", etype): shell_elem},
)

model = problem.create_model("cylinder_shell")

if args.build:
    model.build_module()

model.initialize()

x = model.create_vector()
g = model.create_vector()
mat = model.create_matrix()

model.eval_gradient(x, g)
model.eval_hessian(x, mat)

K = am.tocsr(mat)
diag = K.diagonal()
K = K + diags(np.where(np.abs(diag) < 1e-10, 1.0, 0.0))

x.get_vector().get_array()[:] = spsolve(K, g.get_vector().get_array())

u = x["soln.u"]
v = x["soln.v"]
w = x["soln.w"]

disp_mag = np.sqrt(u**2 + v**2 + w**2)
print(f"Max displacement magnitude: {disp_mag.max():.6e}")
print(f"u: [{u.min():.3e}, {u.max():.3e}]")
print(f"v: [{v.min():.3e}, {v.max():.3e}]")
print(f"w: [{w.min():.3e}, {w.max():.3e}]")

conn = np.vstack([mesh.get_conn(s, "CPS4") for s in lateral_surfaces])
write_vtu(mesh, conn, u, v, w, filename="cylinder_shell_endtraction.vtu")
print("Wrote cylinder_shell_endtraction.vtu")
