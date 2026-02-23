import amigo as am
import numpy as np
import argparse
import json
import math
import matplotlib.pyplot as plt
import niceplots
from pathlib import Path

"""
Space Shuttle Re-entry Trajectory (Maximum Cross-Range)
=====================================================
Based on Betts' book and JuMP reference.

Formulation (trapezoidal direct collocation):
  State q (scaled): [h/scaling["h"], phi, theta, v/scaling["v"], gamma, psi]
  Control u:        [alpha, beta]

  ShuttleCollocation: q2 - q1 - 0.5*dt*(f(q1,u1) + f(q2,u2)) = 0  (per interval)
    _dynamics(q, u) isolates the physics: returns the 6 scaled derivatives.
"""

num_intervals = 100
tf_fixed = 2000.0
N = num_intervals + 1

# Physical constants
Re = 20902900.0  # Earth radius [ft]
mu = 0.14076539e17  # Gravitational parameter [ft^3/s^2]
rho0 = 0.002378  # Sea-level density [slug/ft^3]
h_r = 23800.0  # Density scale height [ft]
S_ref = 2690.0  # Reference area [ft^2]
mass = 203000.0 / 32.174  # Vehicle mass [slug]

# Aero coefficients
a0, a1 = -0.20704, 0.029244
b0, b1, b2 = 0.07854, -0.61592e-2, 0.621408e-3

# Heating coefficients
c0, c1, c2, c3 = 1.0672181, -0.19213774e-1, 0.21286289e-3, -0.10117249e-5
qU = 70.0  # Heating limit [BTU/ft^2/s]

# Scaling factors: q[i] = q_physical[i] / scaling[i]
scaling = {
    "h": 1e5,  # altitude [ft]
    "phi": 1.0,  # longitude [rad]
    "theta": 1.0,  # latitude [rad]
    "v": 1e4,  # velocity [ft/s]
    "gamma": 1.0,  # flight path angle [rad]
    "psi": 1.0,  # heading angle [rad]
}

# Boundary conditions
h0, phi0, theta0 = 260000.0, 0.0, 0.0
v0, gamma0, psi0 = 25600.0, math.radians(-1.0), math.radians(90.0)
hf, vf, gamma_f = 80000.0, 2500.0, math.radians(-5.0)


class ShuttleState(am.Component):
    """Variable container: holds state q and control u at each trajectory node."""

    def __init__(self):
        super().__init__()
        self.add_input("q", shape=6)  # scaled state [h_s, phi, theta, v_s, gamma, psi]
        self.add_input("u", shape=2)  # controls [alpha, beta]


class ShuttleCollocation(am.Component):
    """Trapezoidal direct collocation for one shuttle interval.

    Residual (shape=6): q2 - q1 - 0.5*dt*(f(q1,u1) + f(q2,u2)) = 0

    _dynamics(q, u) isolates the physics and returns the 6 scaled time derivatives.
    Note: phi (longitude) does not appear in the dynamics equations.
    """

    def __init__(self, scaling, dt):
        super().__init__()

        self.scaling = scaling
        self.add_constant("dt", value=dt)
        self.add_constant("Re", value=Re)
        self.add_constant("mu", value=mu)
        self.add_constant("rho0", value=rho0)
        self.add_constant("h_r", value=h_r)
        self.add_constant("S", value=S_ref)
        self.add_constant("mass", value=mass)
        self.add_constant("a0", value=a0)
        self.add_constant("a1", value=a1)
        self.add_constant("b0", value=b0)
        self.add_constant("b1", value=b1)
        self.add_constant("b2", value=b2)

        self.add_input("q1", shape=6)  # scaled state at left node
        self.add_input("q2", shape=6)  # scaled state at right node
        self.add_input("u1", shape=2)  # control at left node
        self.add_input("u2", shape=2)  # control at right node

        self.add_constraint("res", shape=6, label="residual")

    def _dynamics(self, q, u):
        """Scaled time derivatives f(q, u).

        q (scaled): [h_s, phi, theta, v_s, gamma, psi]
        u:          [alpha, beta]
        returns:    [hdot_s, phidot, thetadot, vdot_s, gammadot, psidot]
        """
        h = self.scaling["h"] * q[0]
        theta = q[2]
        v = self.scaling["v"] * q[3]
        gamma = q[4]
        psi = q[5]
        alpha = u[0]
        beta = u[1]

        # Atmospheric density and aero coefficients
        rho = self.constants["rho0"] * am.exp(-h / self.constants["h_r"])
        alpha_deg = alpha * (180.0 / math.pi)
        CL = self.constants["a0"] + self.constants["a1"] * alpha_deg
        CD = (
            self.constants["b0"]
            + self.constants["b1"] * alpha_deg
            + self.constants["b2"] * alpha_deg * alpha_deg
        )

        # Lift, drag, gravity
        q_dyn = 0.5 * rho * v * v
        L = q_dyn * self.constants["S"] * CL
        D = q_dyn * self.constants["S"] * CD
        r = self.constants["Re"] + h
        g = self.constants["mu"] / (r * r)

        # Physical derivatives
        fh = v * am.sin(gamma)
        fphi = (v / r) * am.cos(gamma) * am.sin(psi) / am.cos(theta)
        ftheta = (v / r) * am.cos(gamma) * am.cos(psi)
        fv = -(D / self.constants["mass"]) - g * am.sin(gamma)
        fgamma = (L / (self.constants["mass"] * v)) * am.cos(beta) + am.cos(gamma) * (
            v / r - g / v
        )
        fpsi = (L * am.sin(beta)) / (self.constants["mass"] * v * am.cos(gamma)) + (
            v / (r * am.cos(theta))
        ) * am.cos(gamma) * am.sin(psi) * am.sin(theta)

        return [
            fh / self.scaling["h"],
            fphi / self.scaling["phi"],
            ftheta / self.scaling["theta"],
            fv / self.scaling["v"],
            fgamma / self.scaling["gamma"],
            fpsi / self.scaling["psi"],
        ]

    def compute(self):
        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        dt = self.constants["dt"]

        f1 = self._dynamics(q1, self.inputs["u1"])
        f2 = self._dynamics(q2, self.inputs["u2"])

        self.constraints["res"] = [
            q2[i] - q1[i] - 0.5 * dt * (f1[i] + f2[i]) for i in range(6)
        ]


class InitialConditions(am.Component):
    """Fix all 6 states at t=0."""

    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("q", shape=6)
        self.add_constraint("res", shape=6)

    def compute(self):
        q = self.inputs["q"]
        self.constraints["res"] = [
            q[0] - h0 / self.scaling["h"],
            q[1] - phi0 / self.scaling["phi"],
            q[2] - theta0 / self.scaling["theta"],
            q[3] - v0 / self.scaling["v"],
            q[4] - gamma0 / self.scaling["gamma"],
            q[5] - psi0 / self.scaling["psi"],
        ]


class FinalConditions(am.Component):
    """Fix h, v, gamma at final time (theta is free — it is the objective)."""

    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("q", shape=6)
        self.add_constraint("res", shape=3)

    def compute(self):
        q = self.inputs["q"]
        self.constraints["res"] = [
            q[0] - hf / self.scaling["h"],
            q[3] - vf / self.scaling["v"],
            q[4] - gamma_f / self.scaling["gamma"],
        ]


class HeatingConstraint(am.Component):
    """Path constraint: q_heat(h, v, alpha) / qU = slack, slack in [0, 1].

    Normalizing by qU keeps the residual O(1).
    """

    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_constant("rho0", value=rho0)
        self.add_constant("h_r", value=h_r)
        self.add_constant("c0", value=c0)
        self.add_constant("c1", value=c1)
        self.add_constant("c2", value=c2)
        self.add_constant("c3", value=c3)
        self.add_constant("qU", value=qU)

        self.add_input("h")  # scaled altitude
        self.add_input("v")  # scaled velocity
        self.add_input("alpha")  # angle of attack [rad]
        self.add_input("slack")  # slack in [0, 1]

        self.add_constraint("res", shape=1)

    def compute(self):
        h = self.scaling["h"] * self.inputs["h"]
        v = self.scaling["v"] * self.inputs["v"]

        alpha_deg = self.inputs["alpha"] * (180.0 / math.pi)
        q_a = (
            self.constants["c0"]
            + self.constants["c1"] * alpha_deg
            + self.constants["c2"] * alpha_deg * alpha_deg
            + self.constants["c3"] * alpha_deg * alpha_deg * alpha_deg
        )
        q_r = (
            17700.0
            * am.sqrt(self.constants["rho0"])
            * am.exp(-h / (2.0 * self.constants["h_r"]))
            * (0.0001 * v) ** 3.07
        )

        self.constraints["res"] = [
            q_a * q_r / self.constants["qU"] - self.inputs["slack"]
        ]


class Objective(am.Component):
    """Maximize final cross-range (theta): minimize -theta."""

    def __init__(self):
        super().__init__()
        self.add_input("theta_final")
        self.add_objective("obj")

    def compute(self):
        self.objective["obj"] = -self.inputs["theta_final"]


def create_shuttle_model(
    module_name="spaceshuttle_mod", num_intervals=100, tf_fixed=2000.0
):
    """Build and return the space shuttle trajectory optimization model.

    Components:
      shuttle (N nodes):           ShuttleState  — holds q (shape=6), u (shape=2)
      colloc  (num_intervals):     ShuttleCollocation — trapezoidal collocation residual
      ic      (1):                 InitialConditions
      fc      (1):                 FinalConditions (h, v, gamma fixed)
      heat    (N nodes):           HeatingConstraint
      obj     (1):                 Objective (maximize theta)
    """
    dt = tf_fixed / num_intervals
    N = num_intervals + 1

    shuttle = ShuttleState()
    colloc = ShuttleCollocation(scaling, dt)
    ic = InitialConditions(scaling)
    fc = FinalConditions(scaling)
    heat = HeatingConstraint(scaling)
    obj = Objective()

    model = am.Model(module_name)

    model.add_component("shuttle", N, shuttle)
    model.add_component("colloc", num_intervals, colloc)
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)
    model.add_component("heat", N, heat)
    model.add_component("obj", 1, obj)

    # Link adjacent state/control nodes into each collocation interval
    for i in range(6):
        model.link(f"shuttle.q[:{num_intervals}, {i}]", f"colloc.q1[:, {i}]")
        model.link(f"shuttle.q[1:, {i}]", f"colloc.q2[:, {i}]")
    for i in range(2):
        model.link(f"shuttle.u[:{num_intervals}, {i}]", f"colloc.u1[:, {i}]")
        model.link(f"shuttle.u[1:, {i}]", f"colloc.u2[:, {i}]")

    # Boundary conditions
    model.link("shuttle.q[0, :]", "ic.q[0, :]")
    model.link(f"shuttle.q[{num_intervals}, :]", "fc.q[0, :]")

    # Objective: final latitude (theta = q[:, 2])
    model.link(f"shuttle.q[{num_intervals}, 2]", "obj.theta_final[0]")

    # Heating constraint at all N nodes
    model.link("shuttle.q[:, 0]", "heat.h[:]")
    model.link("shuttle.q[:, 3]", "heat.v[:]")
    model.link("shuttle.u[:, 0]", "heat.alpha[:]")

    return model


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", help="Build C++ module")
args = parser.parse_args()

model = create_shuttle_model(
    module_name="spaceshuttle_mod",
    num_intervals=num_intervals,
    tf_fixed=tf_fixed,
)

if args.build:
    source_dir = Path(__file__).resolve().parent
    model.build_module(source_dir=source_dir)

model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)

print(f"Num variables:   {model.num_variables}")
print(f"Num constraints: {model.num_constraints}")
print(f"Time: {tf_fixed:.0f}s, {N} nodes, dt={tf_fixed/num_intervals:.1f}s")

# Initial guess
x = model.create_vector()

# Linear interpolation between boundary conditions
h_init = np.linspace(h0, hf, N)
v_init = np.linspace(v0, vf, N)
gamma_init = np.linspace(gamma0, gamma_f, N)
phi_init = np.zeros(N)
theta_init = np.zeros(N)
psi_init = np.full(N, psi0)

# Controls: alpha ramps 30→10 deg; beta fixed at -75 deg
alpha_init = np.radians(np.linspace(30.0, 10.0, N))
beta_init = math.radians(-75.0)

x["shuttle.q[:, 0]"] = h_init / scaling["h"]
x["shuttle.q[:, 1]"] = phi_init / scaling["phi"]
x["shuttle.q[:, 2]"] = theta_init / scaling["theta"]
x["shuttle.q[:, 3]"] = v_init / scaling["v"]
x["shuttle.q[:, 4]"] = gamma_init / scaling["gamma"]
x["shuttle.q[:, 5]"] = psi_init / scaling["psi"]

x["shuttle.u[:, 0]"] = alpha_init
x["shuttle.u[:, 1]"] = beta_init

# Heating slack initial guess
alpha_deg_init = np.degrees(alpha_init)
q_a_init = c0 + c1 * alpha_deg_init + c2 * alpha_deg_init**2 + c3 * alpha_deg_init**3
q_r_init = (
    17700.0 * np.sqrt(rho0) * np.exp(-h_init / (2.0 * h_r)) * (0.0001 * v_init) ** 3.07
)
x["heat.slack"] = np.clip(q_a_init * q_r_init / qU, 0.01, 0.99)

print(f"\nInitial guess (linear interpolation):")
print(f"  h:     [{h_init.min():.0f}, {h_init.max():.0f}] ft")
print(f"  v:     [{v_init.min():.0f}, {v_init.max():.0f}] ft/s")
print(
    f"  alpha: [{np.degrees(alpha_init.min()):.1f}, {np.degrees(alpha_init.max()):.1f}] deg"
)
print(
    f"  q_heat range: [{(q_a_init * q_r_init).min():.2f}, {(q_a_init * q_r_init).max():.2f}] BTU/ft^2/s (limit: {qU})"
)

# Bounds
lower = model.create_vector()
upper = model.create_vector()

lower["shuttle.q"] = -float("inf")
upper["shuttle.q"] = float("inf")

lower["shuttle.u[:, 0]"] = math.radians(-90.0)  # alpha: [-90, 90] deg
upper["shuttle.u[:, 0]"] = math.radians(90.0)
lower["shuttle.u[:, 1]"] = math.radians(-89.0)  # beta:  [-89, 1] deg
upper["shuttle.u[:, 1]"] = math.radians(1.0)

lower["heat.slack"] = 0.0  # q_heat in [0, qU]
upper["heat.slack"] = 1.0

# Optimize
opt = am.Optimizer(model, x, lower=lower, upper=upper)

print(f"\nOptimization (heating constraint q_heat <= {qU} BTU/ft^2/s)")
data = opt.optimize(
    {
        "barrier_strategy": "quality_function",
        "initial_barrier_param": 0.1,
        "monotone_barrier_fraction": 0.2,
        "max_iterations": 1000,
        "fraction_to_boundary": 0.995,
        "use_armijo_line_search": False,
        "curvature_probe_convexification": True,
        "nonconvex_constraints": ["colloc.res", "heat.res"],
    }
)

# Results
print(f"\nOptimization Results")
print(f"Converged:      {data['converged']}")
print(f"Iterations:     {len(data['iterations'])}")
print(f"Final residual: {data['iterations'][-1]['residual']:.6e}")

h_sol = np.array(x["shuttle.q[:, 0]"]) * scaling["h"]
phi_sol = np.array(x["shuttle.q[:, 1]"]) * scaling["phi"]
theta_sol = np.array(x["shuttle.q[:, 2]"]) * scaling["theta"]
v_sol = np.array(x["shuttle.q[:, 3]"]) * scaling["v"]
gamma_sol = np.array(x["shuttle.q[:, 4]"]) * scaling["gamma"]
psi_sol = np.array(x["shuttle.q[:, 5]"]) * scaling["psi"]
alpha_sol = np.array(x["shuttle.u[:, 0]"])
beta_sol = np.array(x["shuttle.u[:, 1]"])

print(
    f"\nFinal cross-range (theta): {np.degrees(theta_sol[-1]):.4f} deg  ({theta_sol[-1]:.4f} rad)"
)
print(f"\nFinal conditions:")
print(f"  h     = {h_sol[-1]:.0f} ft    (target: {hf:.0f})")
print(f"  v     = {v_sol[-1]:.0f} ft/s  (target: {vf:.0f})")
print(
    f"  gamma = {np.degrees(gamma_sol[-1]):.2f} deg (target: {np.degrees(gamma_f):.1f})"
)

alpha_deg_sol = np.degrees(alpha_sol)
q_a_sol = c0 + c1 * alpha_deg_sol + c2 * alpha_deg_sol**2 + c3 * alpha_deg_sol**3
q_r_sol = (
    17700.0 * np.sqrt(rho0) * np.exp(-h_sol / (2.0 * h_r)) * (0.0001 * v_sol) ** 3.07
)
q_heat_sol = q_a_sol * q_r_sol
slack_sol = np.array(x["heat.slack"])

print(f"\nHeating constraint (q_heat <= {qU} BTU/ft^2/s):")
print(
    f"  Max q_heat = {q_heat_sol.max():.2f}  {'VIOLATED' if q_heat_sol.max() > qU else 'ok'}"
)
print(f"  Slack range: [{slack_sol.min():.4f}, {slack_sol.max():.4f}]")
print(f"\nControl ranges:")
print(
    f"  alpha: [{np.degrees(alpha_sol.min()):.2f}, {np.degrees(alpha_sol.max()):.2f}] deg"
)
print(
    f"  beta:  [{np.degrees(beta_sol.min()):.2f}, {np.degrees(beta_sol.max()):.2f}] deg"
)

with open("spaceshuttle_opt_data.json", "w") as fp:
    json.dump(data, fp, indent=2)
print("\nOptimization data saved to spaceshuttle_opt_data.json")

# Plot
t = np.linspace(0, tf_fixed, N)

with plt.style.context(niceplots.get_style()):
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    axes[0, 0].plot(t, h_sol / 1000)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Altitude (1000 ft)")
    axes[0, 0].grid(True)

    axes[0, 1].plot(t, v_sol / 1000)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Velocity (1000 ft/s)")
    axes[0, 1].grid(True)

    axes[1, 0].plot(t, np.degrees(gamma_sol))
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("FPA (deg)")
    axes[1, 0].grid(True)

    axes[1, 1].plot(t, np.degrees(theta_sol))
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Latitude (deg)")
    axes[1, 1].grid(True)

    axes[2, 0].plot(t, np.degrees(alpha_sol))
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("alpha (deg)")
    axes[2, 0].grid(True)

    axes[2, 1].plot(t, np.degrees(beta_sol))
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].set_ylabel("beta (deg)")
    axes[2, 1].grid(True)

    axes[3, 0].plot(t, q_heat_sol)
    axes[3, 0].axhline(y=qU, ls="--", label=f"Limit ({qU})")
    axes[3, 0].set_xlabel("Time (s)")
    axes[3, 0].set_ylabel("BTU/ft^2/s")
    axes[3, 0].legend()
    axes[3, 0].grid(True)

    axes[3, 1].plot(np.degrees(phi_sol), np.degrees(theta_sol))
    axes[3, 1].set_xlabel("Longitude (deg)")
    axes[3, 1].set_ylabel("Latitude (deg)")
    axes[3, 1].grid(True)

    fig.savefig("spaceshuttle_solution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Convergence history
    nrms = [it["residual"] for it in data["iterations"]]
    fig2, ax2 = plt.subplots(1, 1)

    ax2.semilogy(nrms, marker="o", clip_on=False, lw=2.0)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("KKT residual norm")
    ax2.grid(True)

    niceplots.adjust_spines(ax2)
    fig2.savefig("spaceshuttle_convergence.png", dpi=300, bbox_inches="tight")
    plt.show()

print("Plots saved to spaceshuttle_solution.png, spaceshuttle_convergence.png")
