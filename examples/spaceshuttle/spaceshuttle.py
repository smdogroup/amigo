import amigo as am
import numpy as np
import argparse
import json
import math
import matplotlib.pylab as plt
import niceplots

# Problem parameters
num_time_steps = 100

"""
Space Shuttle Re-entry Trajectory (Maximum Cross-Range)
======================================================
This example reproduces the optimal-control problem described in
Betts, *Practical Methods for Optimal Control and Estimation Using Non-linear Programming*,
Example 8.1. The goal is to maximise the final latitude (cross-range) of a shuttle-like
vehicle subject to six-DoF point-mass dynamics with lift/drag and optional
heat-rate limits.

State vector (English units, radians):
    h      altitude            [ft]
    phi    longitude           [rad]
    theta  latitude            [rad]
    v      speed               [ft/s]
    gamma  flight-path angle   [rad]
    psi    heading             [rad]

Control vector:
    alpha  angle of attack     [rad]
    beta   bank angle          [rad] 

"""

# Constants List:
aero_const = {
    # Atmosphere
    "rho0": 0.002378,  # [slug/ft^3] sea-level density
    "h_r": 23800.0,  # [ft] density scale height
    # Earth & gravity
    "Re": 20902900.0,  # [ft] Earth radius
    "mu": 0.14076539e17,  # [ft^3/s^2] gravitational parameter (G*M)
    # Shuttle geometry
    "S": 2690.0,  # [ft^2] reference area
    # Lift curve coefficients c_L = a0 + a1*alpha
    "a0": -0.20704,
    "a1": 0.029244,
    # Drag polar c_D = b0 + b1*alpha + b2*alpha^2
    "b0": 0.07854,
    "b1": -0.61592e-2,
    "b2": 0.621408e-3,
    # other coefficients
    "c0": 1.0672181,
    "c1": -0.19213774e-1,
    "c2": 0.21286289e-3,
    "c3": -0.10117249e-5,
}


# Direct Collocation Method
class TrapezoidRule(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("tf_fixed", value=2000.0)  # Fixed final time
        self.add_input("q1")
        self.add_input("q2")
        self.add_input("q1dot")
        self.add_input("q2dot")

        self.add_constraint("res")

        return

    def compute(self):
        tf = self.constants["tf_fixed"]
        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        q1dot = self.inputs["q1dot"]
        q2dot = self.inputs["q2dot"]

        dt = tf / num_time_steps  # Fixed time step
        self.constraints["res"] = q2 - q1 - 0.5 * dt * (q1dot + q2dot)

        return


class ShuttleDynamics(am.Component):
    def __init__(self, scaling):
        super().__init__()

        self.scaling = scaling

        # Physical constants
        for k, v in aero_const.items():
            self.add_constant(k, value=v)
        self.add_constant("mass", value=203000 / 32.174)
        self.add_constant("qU", value=70.0)  # heating rate limit

        # Add Inputs
        self.add_input("q", shape=(6), label="state")  # [h, phi, theta, v, gamma, psi]
        self.add_input("qdot", shape=(6), label="rate")  # time derivatives
        self.add_input("u", shape=(2), label="ctrl")  # [alpha, beta]

        # Constraint residuals
        self.add_constraint("res", shape=(6), label="dynamics residuals")

        # Heating constraint: q_heat <= qU
        self.add_constraint(
            "heat_cons", lower=float("-inf"), upper=0.0, label="heating constraint"
        )

    def compute(self):
        # Get constants
        S = self.constants["S"]
        a0 = self.constants["a0"]
        a1 = self.constants["a1"]
        b0 = self.constants["b0"]
        b1 = self.constants["b1"]
        b2 = self.constants["b2"]
        Re = self.constants["Re"]
        mu = self.constants["mu"]
        mass = self.constants["mass"]
        rho0 = self.constants["rho0"]
        h_r = self.constants["h_r"]
        c0 = self.constants["c0"]
        c1 = self.constants["c1"]
        c2 = self.constants["c2"]
        c3 = self.constants["c3"]
        qU = self.constants["qU"]

        # Get inputs
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]
        u = self.inputs["u"]

        # State variables (unscaled for physics calculations)
        h = self.scaling["altitude"] * q[0]  # altitude [ft]
        phi = self.scaling["longitude"] * q[1]  # longitude [rad]
        theta = self.scaling["latitude"] * q[2]  # latitude [rad]
        v = self.scaling["velocity"] * q[3]  # velocity [ft/s]
        gamma = self.scaling["gamma"] * q[4]  # flight path angle [rad]
        psi = self.scaling["psi"] * q[5]  # heading [rad]

        # Control variables (unscaled for physics calculations)
        alpha = self.scaling["alpha"] * u[0]  # angle of attack [rad]
        beta = self.scaling["beta"] * u[1]  # bank angle [rad]

        # Atmospheric properties:
        rho = rho0 * am.exp(-h / h_r)

        # Aerodynamic Coefficients:
        conv = 180 / np.pi
        CL = a0 + a1 * conv * alpha
        CD = b0 + b1 * conv * alpha + b2 * (conv * alpha) ** 2

        # Aerodynamic Forces:
        q_dyn = 0.5 * rho * v**2
        L = q_dyn * S * CL
        D = q_dyn * S * CD  # fixed typo

        # Other calculations to complete the dynamics
        r = Re + h
        g = mu / r**2

        # Heating rate calculation
        q_alpha = c0 + c1 * conv * alpha + c2 * conv * alpha**2 + c3 * conv * alpha**3
        q_r = 17700 * am.sqrt(rho) * (0.0001 * v) ** 3.07
        q_heat = q_r * q_alpha

        # Equations of motion (scaled residuals):
        res = [None] * 6
        res[0] = qdot[0] - (v * am.sin(gamma)) / self.scaling["altitude"]
        res[1] = (
            qdot[1]
            - ((v / r) * am.cos(gamma) * am.sin(psi) / am.cos(theta))
            / self.scaling["longitude"]
        )
        res[2] = (
            qdot[2] - ((v / r) * am.cos(gamma) * am.cos(psi)) / self.scaling["latitude"]
        )
        res[3] = qdot[3] - (-(D / mass) - g * am.sin(gamma)) / self.scaling["velocity"]
        res[4] = (
            qdot[4]
            - ((L / (mass * v)) * am.cos(beta) + am.cos(gamma) * (v / r - g / v))
            / self.scaling["gamma"]
        )
        res[5] = (
            qdot[5]
            - (
                (L * am.sin(beta)) / (mass * v * am.cos(gamma))
                + (v / r * am.cos(theta)) * am.cos(gamma) * am.sin(psi) * am.sin(theta)
            )
            / self.scaling["psi"]
        )

        self.constraints["res"] = res

        # Heating constraint: q_heat <= qU (q_heat - qU <= 0)
        self.constraints["heat_cons"] = q_heat - qU

        return


class Objective(am.Component):
    def __init__(self):
        super().__init__()
        self.add_constant("tf_fixed", value=2000.0)  # Fixed final time
        self.add_input("theta", label="final latitude")
        self.add_objective("obj")

        return

    def compute(self):
        theta = self.inputs["theta"]
        # Maximize final latitude (cross-range)
        self.objective["obj"] = -theta
        return


class InitialConditions(am.Component):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("q", shape=6)
        self.add_constraint("res", shape=6)

    def compute(self):
        q = self.inputs["q"]

        # Initial Conditions (from Betts Example 8.1) - scaled
        self.constraints["res"] = [
            q[0] - 260000.0 / self.scaling["altitude"],
            q[1] - 0.0 / self.scaling["longitude"],
            q[2] - 0.0 / self.scaling["latitude"],
            q[3] - 25600.0 / self.scaling["velocity"],
            q[4] - (-math.radians(1.0)) / self.scaling["gamma"],
            q[5] - math.radians(90.0) / self.scaling["psi"],
        ]


class FinalConditions(am.Component):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("q", shape=6)
        self.add_constraint("res", shape=3)

    def compute(self):
        q = self.inputs["q"]
        self.constraints["res"] = [
            q[0] - 80000.0 / self.scaling["altitude"],
            q[3] - 2500.0 / self.scaling["velocity"],
            q[4] - (-math.radians(5.0)) / self.scaling["gamma"],
        ]


# Command line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

# Set the scaling factors for the space shuttle problem
scaling = {
    "altitude": 100000.0,
    "longitude": 1.0,
    "latitude": 1.0,
    "velocity": 10000.0,
    "gamma": 1.0,
    "psi": 1.0,
    "time": 1000.0,
    "alpha": 1.0,
    "beta": 1.0,
}

# Create component instances
dyn = ShuttleDynamics(scaling)
trap = TrapezoidRule()
ic = InitialConditions(scaling)
fc = FinalConditions(scaling)
obj = Objective()

# Create the model
module_name = "spaceshuttle"
model = am.Model(module_name)

# Add components to the model
model.add_component("dyn", num_time_steps + 1, dyn)
model.add_component("trap", 6 * num_time_steps, trap)
model.add_component("obj", 1, obj)
model.add_component("ic", 1, ic)
model.add_component("fc", 1, fc)

# Link the trapezoidal rule for each state
for i in range(6):
    start = i * num_time_steps
    end = (i + 1) * num_time_steps

    # Link state variables
    model.link(f"dyn.q[:{num_time_steps}, {i}]", f"trap.q1[{start}:{end}]")
    model.link(f"dyn.q[1:, {i}]", f"trap.q2[{start}:{end}]")

    # Link state derivatives
    model.link(f"dyn.qdot[:-1, {i}]", f"trap.q1dot[{start}:{end}]")
    model.link(f"dyn.qdot[1:, {i}]", f"trap.q2dot[{start}:{end}]")

# Link boundary conditions
model.link("dyn.q[0, :]", "ic.q[0, :]")
model.link(f"dyn.q[{num_time_steps}, :]", "fc.q[0, :]")

# Link objective - final latitude (theta at final time)
model.link(f"dyn.q[{num_time_steps}, 2]", "obj.theta[0]")

# Build the module if requested
if args.build:
    model.build_module()

# Initialize the model
model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)

data = model.get_data_vector()

print(f"Num variables:              {model.num_variables}")
print(f"Num constraints:            {model.num_constraints}")

# Create the design vector
x = model.create_vector()

# Default to zero design variable values
x[:] = 0.0

# Set initial guess for design variables (scaled)
x["obj.theta[0]"] = 30.0 / scaling["latitude"]  # initial guess for final latitude

# Set initial guess for states
h0, phi0, theta0 = 260000.0, 0.0, 0.0
v0, gamma0, psi0 = 25600.0, math.radians(-1.0), math.radians(90.0)

# Final conditions
hf, vf, gamma_f = 80000.0, 2500.0, math.radians(-5.0)

N = num_time_steps + 1

t_norm = np.linspace(0, 1, N)

# Scale the initial guess values
x["dyn.q[:,0]"] = (h0 + (hf - h0) * t_norm) / scaling["altitude"]
x["dyn.q[:,1]"] = phi0 / scaling["longitude"]  # phi (constant)
x["dyn.q[:,2]"] = theta0 / scaling["latitude"]  # theta (start at zero)
x["dyn.q[:,3]"] = (v0 + (vf - v0) * t_norm) / scaling["velocity"]
x["dyn.q[:,4]"] = (gamma0 + (gamma_f - gamma0) * t_norm) / scaling["gamma"]  # gamma
x["dyn.q[:,5]"] = psi0 / scaling["psi"]  # psi (constant)

# Controls:
# alpha
alpha_profile = 20.0 * np.exp(-2 * t_norm) + 5.0
x["dyn.u[:,0]"] = np.radians(alpha_profile) / scaling["alpha"]

# Bank angle profile for cross-range:
beta_profile = 30.0 * np.sin(np.pi * t_norm)
x["dyn.u[:,1]"] = np.radians(beta_profile) / scaling["beta"]

# Set up bounds
lower = model.create_vector()
upper = model.create_vector()

# State bounds
lower["dyn.q"] = -float("inf")
upper["dyn.q"] = float("inf")

lower["dyn.qdot"] = -float("inf")
upper["dyn.qdot"] = float("inf")

# Control bounds
# Angle of attack bounds
lower["dyn.u[:, 0]"] = np.radians(-20.0) / scaling["alpha"]
upper["dyn.u[:, 0]"] = np.radians(40.0) / scaling["alpha"]

# Bank angle bounds
lower["dyn.u[:, 1]"] = np.radians(-90.0) / scaling["beta"]
upper["dyn.u[:, 1]"] = np.radians(90.0) / scaling["beta"]

# Create optimizer and solve
opt = am.Optimizer(model, x, lower=lower, upper=upper)
data = opt.optimize(
    {
        "initial_barrier_param": 0.1,
        "convergence_tolerance": 1e-10,
        "max_line_search_iterations": 4,  # 30,  # Reasonable for intermediate problem
        "max_iterations": 500,  # Sufficient iterations
        "init_affine_step_multipliers": True,  # Enable for better scaling
        # Use the new heuristic barrier parameter update
        "barrier_strategy": "heuristic",
        "heuristic_barrier_gamma": 0.1,  # Scale factor γ
        "heuristic_barrier_r": 0.95,  # Steplength parameter r
        "verbose_barrier": True,  # Show ξ and complementarity values
    }
)

# Save optimization data
with open("spaceshuttle_opt_data.json", "w") as fp:
    json.dump(data, fp, indent=2)

# Extract results (unscaled)
tf_opt = 2000.0  # Fixed final time
q_scaled = x["dyn.q"]
u_scaled = x["dyn.u"]
t = np.linspace(0, tf_opt, num_time_steps + 1)

# Unscale the state and control variables for output
q = np.zeros_like(q_scaled)
q[:, 0] = q_scaled[:, 0] * scaling["altitude"]  # altitude
q[:, 1] = q_scaled[:, 1] * scaling["longitude"]  # longitude
q[:, 2] = q_scaled[:, 2] * scaling["latitude"]  # latitude
q[:, 3] = q_scaled[:, 3] * scaling["velocity"]  # velocity
q[:, 4] = q_scaled[:, 4] * scaling["gamma"]  # flight path angle
q[:, 5] = q_scaled[:, 5] * scaling["psi"]  # heading

u = np.zeros_like(u_scaled)
u[:, 0] = u_scaled[:, 0] * scaling["alpha"]  # angle of attack
u[:, 1] = u_scaled[:, 1] * scaling["beta"]  # bank angle

# Print barrier parameter evolution
print("\nBarrier parameter evolution (heuristic method):")
print("Iteration | Barrier Param | Residual")
print("-" * 40)
for i, iter_data in enumerate(data["iterations"]):
    if i < 20 or i % 10 == 0:  # Show first 20 iterations, then every 10th
        print(
            f"{i:8d} | {iter_data['barrier_param']:12.6e} | {iter_data['residual']:12.6e}"
        )
if len(data["iterations"]) > 20:
    print(f"... (showing every 10th iteration after 20)")
    print(
        f"{len(data['iterations'])-1:8d} | {data['iterations'][-1]['barrier_param']:12.6e} | {data['iterations'][-1]['residual']:12.6e}"
    )

# Print results
print(f"\nOptimization Results:")
print(f"Optimal time:         {tf_opt:.1f} seconds")
print(f"Final altitude:       {q[-1, 0]:.0f} ft")
print(f"Final latitude:       {np.degrees(q[-1, 2]):.2f} degrees")
print(f"Final velocity:       {q[-1, 3]:.0f} ft/s")
print(f"Maximum cross-range:  {np.degrees(np.max(q[:, 2])):.2f} degrees")

print(f"\nFinal conditions check:")
print(f"h_final = {q[-1, 0]:.0f} ft (target: 80,000 ft)")
print(f"v_final = {q[-1, 3]:.0f} ft/s (target: 2,500 ft/s)")
print(f"gamma_final = {np.degrees(q[-1, 4]):.1f} deg (target: -5.0 deg)")
