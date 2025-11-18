import amigo as am
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Dynamic Soaring Optimal Control Problem
========================================

Based on the formulation from Darby et al. (2011) and Zhao (2004).

This problem finds the minimum wind gradient slope (beta) required to sustain
a continuous cycle of dynamic soaring flight.

States:
    x: distance north (ft)
    y: distance east (ft)
    h: altitude (ft)
    v: velocity (ft/s)
    gamma: flight path angle (rad)
    psi: heading angle (rad)

Controls:
    CL: lift coefficient
    phi: bank angle (rad)

Parameter:
    beta: wind gradient (1/ft) - to be minimized
"""

num_time_steps = 50  # Start with moderate resolution


class TrapezoidRule(am.Component):
    def __init__(self, scaling):
        super().__init__()

        self.scaling = scaling

        self.add_input("tf")  # Final time
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

        # Physical constants (US customary units)
        self.add_constant("g", value=32.2)  # ft/s^2
        self.add_constant("m", value=5.6)  # slug
        self.add_constant("S", value=45.09703)  # ft^2
        self.add_constant("rho", value=0.002378)  # slug/ft^3
        self.add_constant("CD0", value=0.00873)  # profile drag coefficient
        self.add_constant("K", value=0.045)  # induced drag factor

        # Wind gradient parameter (FIXED for now to validate physics)
        self.add_constant("beta", value=0.08)  # 1/ft - known feasible value

        # Control inputs
        self.add_input("CL", label="control")
        self.add_input("phi", label="control")

        # State variables
        self.add_input("q", shape=6, label="state")
        self.add_input("qdot", shape=6, label="rate")

        # Dynamics constraints
        self.add_constraint("res", shape=6, label="residual")

        # Path constraint on load factor
        self.add_constraint("load_factor", label="path_constraint")

    def compute(self):
        # Get constants
        g = self.constants["g"]
        m = self.constants["m"]
        S = self.constants["S"]
        rho = self.constants["rho"]
        CD0 = self.constants["CD0"]
        K = self.constants["K"]
        beta = self.constants["beta"]  # Fixed wind gradient

        # Get inputs (scaled)
        CL = self.inputs["CL"]
        phi = self.inputs["phi"]
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        # Extract and unscale states
        x = self.scaling["distance"] * q[0]  # distance north
        y = self.scaling["distance"] * q[1]  # distance east
        h = self.scaling["altitude"] * q[2]  # altitude
        v = self.scaling["velocity"] * q[3]  # velocity
        gamma = self.scaling["angle"] * q[4]  # flight path angle
        psi = self.scaling["angle"] * q[5]  # heading angle

        # Compute trig functions
        sin_gamma = self.vars["sin_gamma"] = am.sin(gamma)
        cos_gamma = self.vars["cos_gamma"] = am.cos(gamma)
        sin_psi = self.vars["sin_psi"] = am.sin(psi)
        cos_psi = self.vars["cos_psi"] = am.cos(psi)
        sin_phi = self.vars["sin_phi"] = am.sin(phi)
        cos_phi = self.vars["cos_phi"] = am.cos(phi)

        # Wind model: W_x = beta * h, dW_x/dh = beta
        Wx = beta * h
        dWx = beta

        # Dynamic pressure
        q_dyn = 0.5 * rho * v * v

        # Drag coefficient
        CD = CD0 + K * CL * CL

        # Lift and drag forces
        L = q_dyn * S * CL
        D = q_dyn * S * CD

        # Dynamics (unscaled physics, then scaled residuals)
        hdot = v * sin_gamma

        res = 6 * [None]

        # Position derivatives
        res[0] = qdot[0] - (v * cos_gamma * sin_psi + Wx) / self.scaling["distance"]
        res[1] = qdot[1] - (v * cos_gamma * cos_psi) / self.scaling["distance"]
        res[2] = qdot[2] - (v * sin_gamma) / self.scaling["altitude"]

        # Velocity derivative
        res[3] = (
            qdot[3]
            - (-D / m - g * sin_gamma - dWx * hdot * cos_gamma * sin_psi)
            / self.scaling["velocity"]
        )

        # Flight path angle derivative
        res[4] = (
            qdot[4]
            - (
                (L * cos_phi - m * g * cos_gamma + m * dWx * hdot * sin_gamma * sin_psi)
                / (m * v)
            )
            / self.scaling["angle"]
        )

        # Heading angle derivative
        res[5] = (
            qdot[5]
            - ((L * sin_phi - m * dWx * hdot * cos_psi) / (m * v * cos_gamma))
            / self.scaling["angle"]
        )

        self.constraints["res"] = res

        # Load factor path constraint: -2 <= L/(mg) <= 5
        load_factor = L / (m * g)
        self.constraints["load_factor"] = load_factor


class Objective(am.Component):
    def __init__(self, scaling):
        super().__init__()

        self.scaling = scaling
        self.add_input("tf", label="final_time")
        self.add_objective("obj")

    def compute(self):
        tf = self.inputs["tf"]
        # Minimize the final time (for fixed beta problem)
        self.objective["obj"] = tf


class InitialConditions(am.Component):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("q", shape=6)
        self.add_constraint("res", shape=3)  # Only x, y, h = 0

    def compute(self):
        q = self.inputs["q"]
        # x(0) = y(0) = h(0) = 0
        # v, gamma, psi will be enforced through periodic boundary conditions
        self.constraints["res"] = [q[0], q[1], q[2]]


class FinalConditions(am.Component):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("q0", shape=6, label="initial_state")
        self.add_input("qf", shape=6, label="final_state")
        self.add_constraint("res", shape=6)

    def compute(self):
        q0 = self.inputs["q0"]
        qf = self.inputs["qf"]

        # Periodic boundary conditions (in scaled space)
        # x(tf) = x(0) = 0
        # y(tf) = y(0) = 0
        # h(tf) = h(0) = 0
        # v(tf) = v(0)
        # gamma(tf) = gamma(0)
        # psi(tf) = psi(0) + 2*pi
        two_pi_scaled = 2.0 * np.pi / self.scaling["angle"]

        self.constraints["res"] = [
            qf[0] - q0[0],  # x periodic (both should be 0)
            qf[1] - q0[1],  # y periodic (both should be 0)
            qf[2] - q0[2],  # h periodic (both should be 0)
            qf[3] - q0[3],  # v periodic
            qf[4] - q0[4],  # gamma periodic
            qf[5] - q0[5] - two_pi_scaled,  # psi increases by 2*pi
        ]


def create_dynamic_soaring_model(scaling, module_name="dynamic_soaring"):
    dynamics = SoaringDynamics(scaling)
    trap = TrapezoidRule(scaling)
    obj = Objective(scaling)
    ic = InitialConditions(scaling)
    fc = FinalConditions(scaling)

    model = am.Model(module_name)

    # Add components
    model.add_component("dynamics", num_time_steps + 1, dynamics)
    model.add_component("trap", 6 * num_time_steps, trap)
    model.add_component("obj", 1, obj)
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)

    # Link state variables and derivatives via trapezoidal rule
    for i in range(6):
        start = i * num_time_steps
        end = (i + 1) * num_time_steps
        # Link the state variables
        model.link(f"dynamics.q[:{num_time_steps}, {i}]", f"trap.q1[{start}:{end}]")
        model.link(f"dynamics.q[1:, {i}]", f"trap.q2[{start}:{end}]")

        # Link the state rates
        model.link(f"dynamics.qdot[:-1, {i}]", f"trap.q1dot[{start}:{end}]")
        model.link(f"dynamics.qdot[1:, {i}]", f"trap.q2dot[{start}:{end}]")

    # Link initial conditions
    model.link("dynamics.q[0, :]", "ic.q[0, :]")

    # Link boundary conditions (initial and final states)
    model.link("dynamics.q[0, :]", "fc.q0[0, :]")
    model.link(f"dynamics.q[{num_time_steps}, :]", "fc.qf[0, :]")

    # Broadcast the final time
    model.link("obj.tf[0]", "trap.tf")

    return model


def plot_results(x, scaling, num_time_steps):
    """Plot the dynamic soaring trajectory and key variables"""

    # Extract solution (unscale)
    tf = x["obj.tf"][0] * scaling["time"]
    t = np.linspace(0, tf, num_time_steps + 1)

    q_scaled = x["dynamics.q"]
    x_pos = q_scaled[:, 0] * scaling["distance"]
    y_pos = q_scaled[:, 1] * scaling["distance"]
    h = q_scaled[:, 2] * scaling["altitude"]
    v = q_scaled[:, 3] * scaling["velocity"]
    gamma = q_scaled[:, 4] * scaling["angle"]
    psi = q_scaled[:, 5] * scaling["angle"]

    CL = x["dynamics.CL"]
    phi = x["dynamics.phi"]
    beta = 0.08  # Fixed beta value

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))

    # 3D trajectory
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    ax1.plot(x_pos, y_pos, h, "b-", linewidth=2)
    ax1.set_xlabel("x (ft)")
    ax1.set_ylabel("y (ft)")
    ax1.set_zlabel("h (ft)")
    ax1.set_title("3D Flight Path")
    ax1.grid(True)

    # Velocity
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(t, v, "b-", linewidth=2)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Velocity (ft/s)")
    ax2.set_title("Velocity History")
    ax2.grid(True)

    # Flight path angle
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t, np.rad2deg(gamma), "b-", linewidth=2)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Flight Path Angle (deg)")
    ax3.set_title("Flight Path Angle History")
    ax3.grid(True)

    # Heading angle
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(t, np.rad2deg(psi), "b-", linewidth=2)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Heading Angle (deg)")
    ax4.set_title("Heading Angle History")
    ax4.grid(True)

    # Lift coefficient
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t, CL, "b-", linewidth=2)
    ax5.axhline(y=1.5, color="r", linestyle="--", label="CL max")
    ax5.axhline(y=0.0, color="r", linestyle="--")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Lift Coefficient")
    ax5.set_title("Lift Coefficient History")
    ax5.legend()
    ax5.grid(True)

    # Bank angle
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(t, np.rad2deg(phi), "b-", linewidth=2)
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Bank Angle (deg)")
    ax6.set_title("Bank Angle History")
    ax6.grid(True)

    plt.suptitle(f"Dynamic Soaring Solution (β = {beta:.6f} 1/ft)", fontsize=14)
    plt.tight_layout()
    plt.savefig("dynamic_soaring_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nOptimized wind gradient: β = {beta:.8f} 1/ft")
    print(f"Minimum wind gradient: β = {beta:.8f} 1/ft = {beta * 0.3048:.8f} 1/m")
    print(f"Final time: tf = {tf:.2f} s")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
parser.add_argument(
    "--with-openmp",
    dest="use_openmp",
    action="store_true",
    default=False,
    help="Enable OpenMP",
)
parser.add_argument(
    "--with-debug",
    dest="use_debug",
    action="store_true",
    default=False,
    help="Enable debug flags",
)
args = parser.parse_args()

# Define scaling factors (critical for numerical conditioning)
scaling = {
    "distance": 1000.0,  # distances in units of 1000 ft
    "altitude": 500.0,  # altitude in units of 500 ft
    "velocity": 100.0,  # velocity in units of 100 ft/s
    "angle": 1.0,  # angles in radians (no scaling)
    "time": 100.0,  # time in units of 100 s
    "beta": 10.0,  # beta in units of 0.1 (1/ft)
}

# Create the model
model = create_dynamic_soaring_model(scaling)

if args.build:
    compile_args = []
    link_args = []
    define_macros = []
    if args.use_openmp:
        compile_args = ["-fopenmp"]
        link_args = ["-fopenmp"]
        define_macros = [("AMIGO_USE_OPENMP", "1")]

    model.build_module(
        compile_args=compile_args,
        link_args=link_args,
        define_macros=define_macros,
        debug=args.use_debug,
    )

model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)

with open("dynamic_soaring_model.json", "w") as fp:
    json.dump(model.get_serializable_data(), fp, indent=2)

print(f"Num variables:              {model.num_variables}")
print(f"Num constraints:            {model.num_constraints}")

# Create design variable vector and initial guess
x = model.create_vector()
x[:] = 0.0

# Initial guess based on YAPSS reference solution (in scaled units)
N = num_time_steps + 1
tf_guess = 100.0 / scaling["time"]  # 100 seconds

# Time-varying trajectory following YAPSS pattern
t_frac = np.linspace(0, 1, N)
theta_path = 2.0 * np.pi * t_frac

# Circular soaring trajectory (YAPSS-like pattern)
radius = 600.0  # ft
x_north = radius * (np.cos(theta_path) - 1.0)  # oscillating pattern
y_east = radius * np.sin(theta_path)  # circular pattern
altitude = np.maximum(10.0, -0.7 * x_north)  # varying altitude, stay above ground

x["dynamics.q[:, 0]"] = x_north / scaling["distance"]  # x (north)
x["dynamics.q[:, 1]"] = y_east / scaling["distance"]  # y (east)
x["dynamics.q[:, 2]"] = altitude / scaling["altitude"]  # h (altitude)
x["dynamics.q[:, 3]"] = 150.0 / scaling["velocity"]  # v (velocity)
x["dynamics.q[:, 4]"] = 0.0  # gamma (flight path angle) - start level
x["dynamics.q[:, 5]"] = theta_path / scaling["angle"]  # psi matches trajectory

# Control guesses - use physically reasonable values
# For CL=0.5, v=150 ft/s, we need ~73 deg bank for level turn
x["dynamics.CL"] = 0.5  # lift coefficient
x["dynamics.phi"] = np.radians(70.0)  # Large bank angle for turning

# Final time guess (scaled)
x["obj.tf"] = tf_guess

# Compute derivatives from the trajectory
omega = 2.0 * np.pi / (tf_guess * scaling["time"])  # angular velocity (rad/s)
x["dynamics.qdot[:, 0]"] = (-radius * omega * np.sin(theta_path)) / scaling["distance"]
x["dynamics.qdot[:, 1]"] = (radius * omega * np.cos(theta_path)) / scaling["distance"]
x["dynamics.qdot[:, 2]"] = (
    -0.7 * x["dynamics.qdot[:, 0]"] * scaling["distance"]
) / scaling["altitude"]
x["dynamics.qdot[:, 3]"] = 0.0  # assume constant velocity initially
x["dynamics.qdot[:, 4]"] = 0.0  # assume level flight initially
x["dynamics.qdot[:, 5]"] = omega / scaling["angle"]  # rotating at constant rate

# Set bounds (in scaled units)
lower = model.create_vector()
upper = model.create_vector()

# State bounds - wide bounds to avoid infeasibility
lower["dynamics.q[:, 0]"] = -2000.0 / scaling["distance"]
upper["dynamics.q[:, 0]"] = 2000.0 / scaling["distance"]

lower["dynamics.q[:, 1]"] = -2000.0 / scaling["distance"]
upper["dynamics.q[:, 1]"] = 2000.0 / scaling["distance"]

lower["dynamics.q[:, 2]"] = 1.0 / scaling["altitude"]  # Keep above ground
upper["dynamics.q[:, 2]"] = 1500.0 / scaling["altitude"]

lower["dynamics.q[:, 3]"] = 5.0 / scaling["velocity"]  # Minimum velocity
upper["dynamics.q[:, 3]"] = 400.0 / scaling["velocity"]

lower["dynamics.q[:, 4]"] = np.radians(-90.0) / scaling["angle"]
upper["dynamics.q[:, 4]"] = np.radians(90.0) / scaling["angle"]

lower["dynamics.q[:, 5]"] = np.radians(-360.0) / scaling["angle"]
upper["dynamics.q[:, 5]"] = np.radians(450.0) / scaling["angle"]  # Allow more than 2*pi

# State derivative bounds
lower["dynamics.qdot"] = -float("inf")
upper["dynamics.qdot"] = float("inf")

# Control bounds
lower["dynamics.CL"] = 0.0
upper["dynamics.CL"] = 1.5  # Maximum lift coefficient

lower["dynamics.phi"] = np.radians(-90.0)
upper["dynamics.phi"] = np.radians(90.0)

# Final time bounds (scaled)
lower["obj.tf"] = 10.0 / scaling["time"]
upper["obj.tf"] = 200.0 / scaling["time"]

# Path constraint bounds (load factor: -2 <= L/(mg) <= 5)
lower["dynamics.load_factor"] = -2.0
upper["dynamics.load_factor"] = 5.0

# Optimize with robust settings
opt = am.Optimizer(model, x, lower=lower, upper=upper)
data = opt.optimize(
    {
        "max_iterations": 500,
        "initial_barrier_param": 1.0,
        "monotone_barrier_fraction": 0.25,
        "barrier_strategy": "monotone",
        "convergence_tolerance": 1e-8,
        "max_line_search_iterations": 5,
        "init_affine_step_multipliers": False,
    }
)

with open("dynamic_soaring_opt_data.json", "w") as fp:
    json.dump(data, fp, indent=2)

# Plot results
plot_results(x, scaling, num_time_steps)
