import amigo as am
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Dynamic Soaring - Stage 1: Simplified Non-Periodic Version
===========================================================

This solves a SIMPLER problem without periodic constraints:
- Fixed start and end points
- Just optimize the trajectory and controls
- Find a feasible soaring maneuver

Once this converges, we'll use it as initial guess for the full periodic problem.
"""

num_time_steps = 100  # Moderate resolution


class TrapezoidRule(am.Component):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("tf")
        self.add_input("q1")
        self.add_input("q2")
        self.add_input("q1dot")
        self.add_input("q2dot")
        self.add_constraint("res")

    def compute(self):
        tf = self.scaling["time"] * self.inputs["tf"]
        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        q1dot = self.inputs["q1dot"]
        q2dot = self.inputs["q2dot"]
        dt = tf / num_time_steps
        self.constraints["res"] = q2 - q1 - 0.5 * dt * (q1dot + q2dot)


class SoaringDynamics(am.Component):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling

        # Physical constants
        self.add_constant("g", value=32.2)
        self.add_constant("m", value=5.6)
        self.add_constant("S", value=45.09703)
        self.add_constant("rho", value=0.002378)
        self.add_constant("CD0", value=0.00873)
        self.add_constant("K", value=0.045)
        self.add_constant("beta", value=0.08)  # Fixed wind gradient

        # Control inputs
        self.add_input("CL", label="control")
        self.add_input("phi", label="control")

        # State variables
        self.add_input("q", shape=6, label="state")
        self.add_input("qdot", shape=6, label="rate")

        # Dynamics constraints
        self.add_constraint("res", shape=6, label="residual")

    def compute(self):
        g = self.constants["g"]
        m = self.constants["m"]
        S = self.constants["S"]
        rho = self.constants["rho"]
        CD0 = self.constants["CD0"]
        K = self.constants["K"]
        beta = self.constants["beta"]

        CL = self.inputs["CL"]
        phi = self.inputs["phi"]
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        # Unscale states
        x = self.scaling["distance"] * q[0]
        y = self.scaling["distance"] * q[1]
        h = self.scaling["altitude"] * q[2]
        v = self.scaling["velocity"] * q[3]
        gamma = self.scaling["angle"] * q[4]
        psi = self.scaling["angle"] * q[5]

        # Trig functions
        sin_gamma = am.sin(gamma)
        cos_gamma = am.cos(gamma)
        sin_psi = am.sin(psi)
        cos_psi = am.cos(psi)
        sin_phi = am.sin(phi)
        cos_phi = am.cos(phi)

        # Wind model
        Wx = beta * h
        dWx = beta
        hdot = v * sin_gamma

        # Aerodynamics
        q_dyn = 0.5 * rho * v * v
        CD = CD0 + K * CL * CL
        L = q_dyn * S * CL
        D = q_dyn * S * CD

        # Dynamics
        res = 6 * [None]
        res[0] = qdot[0] - (v * cos_gamma * sin_psi + Wx) / self.scaling["distance"]
        res[1] = qdot[1] - (v * cos_gamma * cos_psi) / self.scaling["distance"]
        res[2] = qdot[2] - (v * sin_gamma) / self.scaling["altitude"]
        res[3] = (
            qdot[3]
            - (-D / m - g * sin_gamma - dWx * hdot * cos_gamma * sin_psi)
            / self.scaling["velocity"]
        )
        res[4] = (
            qdot[4]
            - (
                (L * cos_phi - m * g * cos_gamma + m * dWx * hdot * sin_gamma * sin_psi)
                / (m * v)
            )
            / self.scaling["angle"]
        )
        res[5] = (
            qdot[5]
            - ((L * sin_phi - m * dWx * hdot * cos_psi) / (m * v * cos_gamma))
            / self.scaling["angle"]
        )

        self.constraints["res"] = res


class Objective(am.Component):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("tf", label="final_time")
        self.add_objective("obj")

    def compute(self):
        tf = self.inputs["tf"]
        self.objective["obj"] = tf  # Minimize time


class InitialConditions(am.Component):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("q", shape=6)
        self.add_constraint("res", shape=6)

    def compute(self):
        q = self.inputs["q"]
        # Start at origin with reasonable velocity
        self.constraints["res"] = [
            q[0],  # x = 0
            q[1],  # y = 0
            q[2] - 100.0 / self.scaling["altitude"],  # h = 100 ft
            q[3] - 150.0 / self.scaling["velocity"],  # v = 150 ft/s
            q[4],  # gamma = 0
            q[5],  # psi = 0
        ]


class FinalConditions(am.Component):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("q", shape=6)
        self.add_constraint("res", shape=6)

    def compute(self):
        q = self.inputs["q"]
        # End at origin with same velocity (simplified periodic-like)
        self.constraints["res"] = [
            q[0],  # x = 0
            q[1],  # y = 0
            q[2] - 100.0 / self.scaling["altitude"],  # h = 100 ft (same as start)
            q[3] - 150.0 / self.scaling["velocity"],  # v = 150 ft/s (same as start)
            q[4],  # gamma = 0 (same as start)
            q[5] - 2.0 * np.pi / self.scaling["angle"],  # psi = 2π (one full rotation)
        ]


def create_simple_model(scaling, module_name="dynamic_soaring_simple"):
    dynamics = SoaringDynamics(scaling)
    trap = TrapezoidRule(scaling)
    obj = Objective(scaling)
    ic = InitialConditions(scaling)
    fc = FinalConditions(scaling)

    model = am.Model(module_name)

    model.add_component("dynamics", num_time_steps + 1, dynamics)
    model.add_component("trap", 6 * num_time_steps, trap)
    model.add_component("obj", 1, obj)
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)

    # Link trapezoidal rule
    for i in range(6):
        start = i * num_time_steps
        end = (i + 1) * num_time_steps
        model.link(f"dynamics.q[:{num_time_steps}, {i}]", f"trap.q1[{start}:{end}]")
        model.link(f"dynamics.q[1:, {i}]", f"trap.q2[{start}:{end}]")
        model.link(f"dynamics.qdot[:-1, {i}]", f"trap.q1dot[{start}:{end}]")
        model.link(f"dynamics.qdot[1:, {i}]", f"trap.q2dot[{start}:{end}]")

    # Boundary conditions
    model.link("dynamics.q[0, :]", "ic.q[0, :]")
    model.link(f"dynamics.q[{num_time_steps}, :]", "fc.q[0, :]")

    # Link final time
    model.link("obj.tf[0]", "trap.tf")

    return model


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", help="Build module")
parser.add_argument("--with-openmp", action="store_true", help="Use OpenMP")
args = parser.parse_args()

# Scaling
scaling = {
    "distance": 1000.0,
    "altitude": 100.0,  # Smaller scale for altitude
    "velocity": 100.0,
    "angle": 1.0,
    "time": 100.0,
    "beta": 10.0,
}

model = create_simple_model(scaling)

if args.build:
    compile_args = []
    link_args = []
    define_macros = []
    if args.with_openmp:
        compile_args = ["-fopenmp"]
        link_args = ["-fopenmp"]
        define_macros = [("AMIGO_USE_OPENMP", "1")]

    model.build_module(
        compile_args=compile_args, link_args=link_args, define_macros=define_macros
    )

model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)

print(f"Num variables:   {model.num_variables}")
print(f"Num constraints: {model.num_constraints}")

# Create design vector
x = model.create_vector()
x[:] = 0.0

# Initial guess - smooth circular trajectory
N = num_time_steps + 1
tf_guess = 100.0 / scaling["time"]
t_frac = np.linspace(0, 1, N)
theta = 2.0 * np.pi * t_frac

radius = 600.0
x_north = radius * (np.cos(theta) - 1.0)
y_east = radius * np.sin(theta)
altitude = 100.0 + 50.0 * np.sin(theta)  # Oscillate 50-150 ft

x["dynamics.q[:, 0]"] = x_north / scaling["distance"]
x["dynamics.q[:, 1]"] = y_east / scaling["distance"]
x["dynamics.q[:, 2]"] = altitude / scaling["altitude"]
x["dynamics.q[:, 3]"] = 150.0 / scaling["velocity"]
x["dynamics.q[:, 4]"] = np.radians(5.0) * np.sin(2 * theta) / scaling["angle"]
x["dynamics.q[:, 5]"] = theta / scaling["angle"]

# Controls - large bank angle for turning
x["dynamics.CL"] = 1.0
x["dynamics.phi"] = np.radians(50.0)

x["obj.tf"] = tf_guess

# Derivatives
omega = 2.0 * np.pi / (tf_guess * scaling["time"])
x["dynamics.qdot[:, 0]"] = (-radius * omega * np.sin(theta)) / scaling["distance"]
x["dynamics.qdot[:, 1]"] = (radius * omega * np.cos(theta)) / scaling["distance"]
x["dynamics.qdot[:, 2]"] = (50.0 * omega * np.cos(theta)) / scaling["altitude"]
x["dynamics.qdot[:, 3]"] = 0.0
x["dynamics.qdot[:, 4]"] = (2 * np.radians(5.0) * omega * np.cos(2 * theta)) / scaling[
    "angle"
]
x["dynamics.qdot[:, 5]"] = omega / scaling["angle"]

# Bounds
lower = model.create_vector()
upper = model.create_vector()

# Wide state bounds
lower["dynamics.q[:, 0]"] = -2000.0 / scaling["distance"]
upper["dynamics.q[:, 0]"] = 2000.0 / scaling["distance"]

lower["dynamics.q[:, 1]"] = -2000.0 / scaling["distance"]
upper["dynamics.q[:, 1]"] = 2000.0 / scaling["distance"]

lower["dynamics.q[:, 2]"] = 10.0 / scaling["altitude"]
upper["dynamics.q[:, 2]"] = 500.0 / scaling["altitude"]

lower["dynamics.q[:, 3]"] = 50.0 / scaling["velocity"]
upper["dynamics.q[:, 3]"] = 300.0 / scaling["velocity"]

lower["dynamics.q[:, 4]"] = np.radians(-45.0) / scaling["angle"]
upper["dynamics.q[:, 4]"] = np.radians(45.0) / scaling["angle"]

lower["dynamics.q[:, 5]"] = np.radians(-10.0) / scaling["angle"]
upper["dynamics.q[:, 5]"] = np.radians(370.0) / scaling["angle"]

lower["dynamics.qdot"] = -float("inf")
upper["dynamics.qdot"] = float("inf")

# Control bounds
lower["dynamics.CL"] = 0.1  # Minimum lift to avoid stall
upper["dynamics.CL"] = 1.5

lower["dynamics.phi"] = np.radians(-89.0)
upper["dynamics.phi"] = np.radians(89.0)

# Time bounds
lower["obj.tf"] = 20.0 / scaling["time"]
upper["obj.tf"] = 300.0 / scaling["time"]

# Optimize
print("\n" + "=" * 60)
print("STAGE 1: Solving simplified (non-periodic) problem")
print("=" * 60)

opt = am.Optimizer(model, x, lower=lower, upper=upper)
data = opt.optimize(
    {
        "max_iterations": 200,
        "initial_barrier_param": 10.0,
        "monotone_barrier_fraction": 0.5,
        "barrier_strategy": "monotone",
        "convergence_tolerance": 1e-5,
        "max_line_search_iterations": 10,
        "init_affine_step_multipliers": False,
    }
)

with open("dynamic_soaring_simple_opt_data.json", "w") as fp:
    json.dump(data, fp, indent=2)

# Extract solution
tf_opt = x["obj.tf"][0] * scaling["time"]
print(f"\n{'='*60}")
print(f"Stage 1 COMPLETE!")
print(f"{'='*60}")
print(f"Optimal time: {tf_opt:.2f} s")
print(f"Converged: {data.get('converged', False)}")
print(f"Final residual: {data['iterations'][-1]['residual']:.3e}")

# Save solution for use as initial guess in Stage 2
q_sol = x["dynamics.q"]
qdot_sol = x["dynamics.qdot"]
CL_sol = x["dynamics.CL"]
phi_sol = x["dynamics.phi"]

solution_data = {
    "q": q_sol.tolist() if hasattr(q_sol, "tolist") else list(q_sol),
    "qdot": qdot_sol.tolist() if hasattr(qdot_sol, "tolist") else list(qdot_sol),
    "CL": CL_sol.tolist() if hasattr(CL_sol, "tolist") else list(CL_sol),
    "phi": phi_sol.tolist() if hasattr(phi_sol, "tolist") else list(phi_sol),
    "tf": tf_opt,
    "scaling": scaling,
}

with open("stage1_solution.json", "w") as fp:
    json.dump(solution_data, fp, indent=2)

print("\nStage 1 solution saved to 'stage1_solution.json'")
print("Use this as initial guess for the full periodic problem!")

# Plot results
t = np.linspace(0, tf_opt, num_time_steps + 1)
q_unscaled = np.zeros((N, 6))
q_unscaled[:, 0] = q_sol[:, 0] * scaling["distance"]
q_unscaled[:, 1] = q_sol[:, 1] * scaling["distance"]
q_unscaled[:, 2] = q_sol[:, 2] * scaling["altitude"]
q_unscaled[:, 3] = q_sol[:, 3] * scaling["velocity"]
q_unscaled[:, 4] = q_sol[:, 4] * scaling["angle"]
q_unscaled[:, 5] = q_sol[:, 5] * scaling["angle"]

fig = plt.figure(figsize=(12, 8))

# 3D trajectory
ax1 = fig.add_subplot(2, 3, 1, projection="3d")
ax1.plot(q_unscaled[:, 0], q_unscaled[:, 1], q_unscaled[:, 2], "b-", linewidth=2)
ax1.scatter([0], [0], [100], c="g", s=100, marker="o", label="Start")
ax1.scatter([0], [0], [100], c="r", s=100, marker="x", label="End")
ax1.set_xlabel("x (ft)")
ax1.set_ylabel("y (ft)")
ax1.set_zlabel("h (ft)")
ax1.set_title("3D Trajectory")
ax1.legend()

ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(t, q_unscaled[:, 2], "b-", linewidth=2)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Altitude (ft)")
ax2.set_title("Altitude vs Time")
ax2.grid(True)

ax3 = fig.add_subplot(2, 3, 3)
ax2.plot(t, q_unscaled[:, 3], "b-", linewidth=2)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Velocity (ft/s)")
ax3.set_title("Velocity vs Time")
ax3.grid(True)

ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(t, np.rad2deg(q_unscaled[:, 4]), "b-", linewidth=2)
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("γ (deg)")
ax4.set_title("Flight Path Angle")
ax4.grid(True)

ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(t, np.rad2deg(q_unscaled[:, 5]), "b-", linewidth=2)
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("ψ (deg)")
ax5.set_title("Heading Angle")
ax5.grid(True)

ax6 = fig.add_subplot(2, 3, 6)
ax6.plot(t, CL_sol, "b-", linewidth=1, label="CL")
ax6.plot(t, np.rad2deg(phi_sol), "r-", linewidth=1, label="φ (deg)")
ax6.set_xlabel("Time (s)")
ax6.set_ylabel("Controls")
ax6.set_title("Control History")
ax6.legend()
ax6.grid(True)

plt.suptitle(
    f"Stage 1: Simplified Soaring (tf={tf_opt:.1f}s, β=0.08 1/ft)", fontsize=14
)
plt.tight_layout()
plt.savefig("dynamic_soaring_stage1.png", dpi=300)
plt.show()

print("\nPlot saved as 'dynamic_soaring_stage1.png'")

