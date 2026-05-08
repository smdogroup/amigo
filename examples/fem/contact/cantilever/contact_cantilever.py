# Author: Jack Turbush
# Description: Contact problem for a cantilever beam with contact constraint enforcement.
# Contact floor at v = -1e-3 enforced via lower bounds (interior point barrier method).
# Equilibrium via potential energy minimization: PE = strain energy - work done by loads.

import amigo as am
import numpy as np
import argparse
import matplotlib.pylab as plt

length = 1.0
nelems = 50
E = 1e5
Fv = -10000
Mv = 0.0


class beam_element(am.Component):
    """Euler-Bernoulli beam element strain energy (bending only, no axial)."""

    def __init__(self):
        super().__init__()
        self.add_input("v", shape=(2,), value=0.0)
        self.add_input("t", shape=(2,), value=0.0)
        self.add_data("h")
        self.add_objective("obj")

    def compute(self):
        v = self.inputs["v"]
        t = self.inputs["t"]
        L0 = length / nelems
        I = 1.0  # uniform cross-section; set h-dependent I here for thickness opt

        Ke = np.empty((4, 4))
        Ke[0, :] = [12, 6 * L0, -12, 6 * L0]
        Ke[1, :] = [6 * L0, 4 * L0**2, -6 * L0, 2 * L0**2]
        Ke[2, :] = [-12, -6 * L0, 12, -6 * L0]
        Ke[3, :] = [6 * L0, 2 * L0**2, -6 * L0, 4 * L0**2]
        Ke = Ke * (E * I) / L0**3

        de = np.array([v[0], t[0], v[1], t[1]])
        self.objective["obj"] = 0.5 * de.T @ Ke @ de


class BoundaryCondition(am.Component):
    """Fixed (clamped) boundary condition at the cantilever root."""

    def __init__(self):
        super().__init__()
        self.add_input("v", value=1.0)
        self.add_input("t", value=1.0)
        self.add_constraint("bc_v", value=0.0, lower=0.0, upper=0.0)
        self.add_constraint("bc_t", value=0.0, lower=0.0, upper=0.0)

    def compute(self):
        self.constraints["bc_v"] = self.inputs["v"]
        self.constraints["bc_t"] = self.inputs["t"]


class AppliedLoad(am.Component):
    """Distributed transverse load contribution to potential energy (work = -F·u)."""

    def __init__(self):
        super().__init__()
        self.add_input("v", shape=(2,), value=0.0)
        self.add_input("t", shape=(2,), value=0.0)
        self.add_objective("work")

    def compute(self):
        v = self.inputs["v"]
        t = self.inputs["t"]
        Le = length / nelems
        # Consistent nodal load vector for uniform distributed load Fv
        fe = [
            Fv * Le * 0.5,
            Fv * Le * (Le / 12),
            Fv * Le * 0.5,
            -Fv * Le * (Le / 12),
        ]
        de = np.array([v[0], t[0], v[1], t[1]])
        self.objective["work"] = -(
            fe[0] * de[0] + fe[1] * de[1] + fe[2] * de[2] + fe[3] * de[3]
        )


class NodeSource(am.Component):
    """Holds nodal coordinates and DOFs (v, t) for the beam mesh."""

    def __init__(self):
        super().__init__()
        self.add_data("x_coord")
        self.add_data("y_coord")
        self.add_input("v")
        self.add_input("t")


class Compliance(am.Component):
    """Computes compliance (work done by load) — useful for post-opt sensitivity."""

    def __init__(self):
        super().__init__()
        self.add_input("v", shape=(2,))
        self.add_input("t", shape=(2,))
        self.add_output("c")

    def compute_output(self):
        v = self.inputs["v"]
        t = self.inputs["t"]
        Le = length / nelems
        fe = [
            Fv * Le * 0.5,
            Fv * Le * (Le / 12),
            Fv * Le * 0.5,
            -Fv * Le * (Le / 12),
        ]
        de = np.array([v[0], t[0], v[1], t[1]])
        self.outputs["c"] = (
            fe[0] * de[0] + fe[1] * de[1] + fe[2] * de[2] + fe[3] * de[3]
        )


class VolumeConstraint(am.Component):
    """Per-element volume = h * b * L0; sum over elements for total volume constraint."""

    def __init__(self):
        super().__init__()
        self.add_data("h")
        self.add_output("con")

    def compute_output(self):
        self.outputs["con"] = self.data["h"] * 0.33 * length / nelems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", dest="build", action="store_true", default=False)
    args = parser.parse_args()

    x_coords = np.linspace(0, length, nelems + 1)
    nodes = np.arange(nelems + 1, dtype=int)
    conn = np.array([[i, i + 1] for i in range(nelems)], dtype=int)

    model = am.Model("beam")

    model.add_component("src", nelems + 1, NodeSource())
    model.add_component("beam_element", nelems, beam_element())
    model.link("beam_element.v", "src.v", tgt_indices=conn)
    model.link("beam_element.t", "src.t", tgt_indices=conn)

    # Clamped root BC
    bcs = BoundaryCondition()
    model.add_component("bcs", 1, bcs)
    model.link("src.v", "bcs.v", src_indices=[nodes[0]])
    model.link("src.t", "bcs.t", src_indices=[nodes[0]])

    # Distributed load
    load = AppliedLoad()
    model.add_component("load", nelems, load)
    model.link("src.v", "load.v", src_indices=conn)
    model.link("src.t", "load.t", src_indices=conn)

    # Compliance and volume (for post-opt sensitivity or thickness optimization)
    compliance = Compliance()
    model.add_component("comp", nelems, compliance)
    model.link("src.v", "comp.v", src_indices=conn)
    model.link("src.t", "comp.t", src_indices=conn)

    vol_con = VolumeConstraint()
    model.add_component("vol_con", nelems, vol_con)
    model.link("beam_element.h", "vol_con.h")
    model.link("comp.c[1:]", "comp.c[0]")
    model.link("vol_con.con[1:]", "vol_con.con[0]")

    if args.build:
        model.build_module()

    model.initialize()

    x = model.create_vector()
    lower = model.create_vector()
    upper = model.create_vector()

    x[:] = 0.0
    lower["src.v"] = -1e-3  # contact floor
    upper["src.v"] = float("inf")
    lower["src.t"] = -float("inf")
    upper["src.t"] = float("inf")

    opt_options = {
        "max_iterations": 200,
        "convergence_tolerance": 1e-10,
        "max_line_search_iterations": 1,
        "initial_barrier_param": 0.1,
    }

    opt = am.Optimizer(model, x, lower=lower, upper=upper)
    opt.optimize(opt_options)

    # Contact forces from barrier multipliers: zl = mu / (v - lb)
    x_v = x["src.v"]
    mu = opt.barrier_param
    zl = mu / (x_v - lower["src.v"])
    print("zl:", zl)

    fig, ax = plt.subplots()
    plt.plot(x_coords, zl)
    plt.show()

    return x, x_coords


def plot(v, x_c):
    x_ref_arr = np.linspace(0, 1, 51)
    EI = E
    lam = -75
    v_ref = np.empty_like(x_ref_arr)
    v_nocontact = np.empty_like(x_ref_arr)
    for i in range(len(x_ref_arr)):
        x = x_ref_arr[i]
        v_ref[i] = (Fv * x**2 / (24 * EI)) * (x**2 + 6 - 4 * x) - (lam * x**2) / (
            6 * EI
        ) * (3 - x)
        v_nocontact[i] = (Fv * x**2 / (24 * EI)) * (x**2 + 6 - 4 * x)

    fig, ax = plt.subplots()
    ax.plot(x_c, v)
    # ax.plot(x_c,v_ref)
    # ax.plot(x_c,v_nocontact)
    ax.legend([r"$v_{\text{amigo}}$", r"$v_{\text{amigo}}$", r"$v_\text{free}$"])
    print("max relative error: ", np.max((v - v_ref)) / np.linalg.norm(v))
    ax.grid(True)
    plt.axhline(-1e-3, color="black")
    ax.set_ylabel("$vertical displacement (m)$")
    ax.set_xlabel("$x(m)$")
    plt.savefig("contact.png")


if __name__ == "__main__":
    x, x_c = main()
    plot(x["src.v"], x_c)
