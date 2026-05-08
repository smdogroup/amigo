import amigo as am
from amigo.interp import BSpline
import numpy as np
import sys, os
import matplotlib.pylab as plt
import niceplots
import argparse
import json

# Problem parameters
num_time_steps = 500

"""
Min Time to Climb 
==============================

This example is from Betts' book "Practical Methods for Optimal Control
Using NonlinearProgramming", 3rd edition, Chapter 10: Test Problems.

This is a non-scaled problem with a single component dynamics code
"""


class AircraftDynamics(am.TrajectoryComponent):
    def __init__(self, scaling):
        # super().__init__() will automatically create the inputs and constraint
        super().__init__(num_time_steps, state_size=5, aux_inputs=[{"name": "alpha"}])

        self.scaling = scaling

        # Physical constants matching the original code
        self.add_constant("S", value=49.2386)  # m^2
        self.add_constant("CL_alpha", value=3.44)
        self.add_constant("CD0", value=0.013)
        self.add_constant("kappa", value=0.54)
        self.add_constant("Isp", value=1600.0)  # s
        self.add_constant("TtoW", value=0.9)
        self.add_constant("m0", value=19030.0)  # kg
        self.add_constant("gamma_air", value=1.4)
        self.add_constant("R", value=287.058)
        self.add_constant("g", value=9.81)
        self.add_constant("conv", value=np.pi / 180.0)

        return

    def dynamics(self, q, alpha):
        # Get constants
        S = self.constants["S"]
        CL_alpha = self.constants["CL_alpha"]
        CD0 = self.constants["CD0"]
        kappa = self.constants["kappa"]
        Isp = self.constants["Isp"]
        TtoW = self.constants["TtoW"]
        m0 = self.constants["m0"]
        g = self.constants["g"]
        conv = self.constants["conv"]

        # State variables
        v = 100.0 * q[0]  # Convert velocity to [m/s]
        gamma = q[1]  # flight path angle [degrees]
        m = 1000.0 * q[4]  # Convert mass to [kg]

        # Atmospheric properties (simplified)
        T_atm = 288.15  # K
        h_m = q[2] * self.scaling["altitude"]
        rho = 1.225 * am.exp(-h_m / 8500.0)  # kg/m^3

        # Aerodynamics matching original code
        CL = CL_alpha * conv * alpha  # Convert alpha from degrees to radians

        # Drag coefficient (simplified - no compressibility for now)
        CD = CD0 + kappa * CL**2

        # Dynamic pressure and forces
        qinfty = 0.5 * rho * v**2
        D = qinfty * S * CD
        L = qinfty * S * CL

        # Thrust
        T = TtoW * m0 * g

        # Convert angles to radians for dynamics
        alpha_rad = conv * alpha
        gamma_rad = conv * gamma

        # Set intermediate varaibles
        sin_alpha = am.sin(alpha_rad)
        cos_alpha = am.cos(alpha_rad)
        sin_gamma = am.sin(gamma_rad)
        cos_gamma = am.cos(gamma_rad)

        # Aircraft dynamics equations (matching original computeSystemResidual)
        qdot = [
            ((T / m) * cos_alpha - (D / m) - g * sin_gamma) / self.scaling["velocity"],
            ((T / (m * v) * sin_alpha + (L / (m * v)) - (g / v) * cos_gamma)) / conv,
            v * sin_gamma / self.scaling["altitude"],
            v * cos_gamma / self.scaling["range"],
            -(T / (g * Isp)) / self.scaling["mass"],
        ]

        return qdot


class Objective(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("tf", label="final time")
        self.add_objective("obj")

        return

    def compute(self):
        tf = self.inputs["tf"]
        self.objective["obj"] = tf  # Minimize final time

        return


class InitialConditions(am.Component):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("q", shape=5)
        self.add_constraint("res", shape=5)

    def compute(self):
        q = self.inputs["q"]

        # Initial conditions matching original getInitConditions
        self.constraints["res"] = [
            q[0] - 136.0 / self.scaling["velocity"],  # 136 [m/s]
            q[1] - 0.0,  # flight path angle [degrees]
            q[2] - 100.0 / self.scaling["altitude"],  # 100.0 [m]
            q[3] - 0.0 / self.scaling["range"],  # [m]
            q[4] - 19030.0 / self.scaling["mass"],  # 19030 [kg]
        ]


class FinalConditions(am.Component):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("q", shape=5)
        self.add_constraint("res", shape=3)

    def compute(self):
        q = self.inputs["q"]
        self.constraints["res"] = [
            q[0] - 340.0 / self.scaling["velocity"],  # 340 [m/s]
            q[1] - 0.0,  # Final flight path angle [degrees]
            q[2] - 20000.0 / self.scaling["altitude"],  # 20 [km]
        ]


def plot_results(t, q, alpha):
    """Plot the optimization results"""

    with plt.style.context(niceplots.get_style()):
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))

        # State variables
        axes[0, 0].plot(t, 100 * q[:, 0])
        axes[0, 0].set_ylabel("Velocity (m/s)")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].grid(True)

        axes[0, 1].plot(t, q[:, 1])
        axes[0, 1].set_ylabel("Flight path angle (deg)")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].grid(True)

        axes[1, 0].plot(t, q[:, 2])
        axes[1, 0].set_ylabel("Altitude (km)")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].grid(True)

        axes[1, 1].plot(t, q[:, 3])
        axes[1, 1].set_ylabel("Range (km)")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].grid(True)

        axes[2, 0].plot(t, 1000 * q[:, 4])
        axes[2, 0].set_ylabel("Mass (kg)")
        axes[2, 0].set_xlabel("Time (s)")
        axes[2, 0].grid(True)

        axes[2, 1].plot(t, alpha)
        axes[2, 1].set_ylabel("Angle of attack (deg)")
        axes[2, 1].set_xlabel("Time (s)")
        axes[2, 1].grid(True)

        plt.tight_layout()
        plt.savefig("time_to_climb_results.png", dpi=300, bbox_inches="tight")
        plt.show()


# Command line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

# Set the scaling
scaling = {"velocity": 100.0, "altitude": 1000.0, "range": 1000.0, "mass": 1000.0}

# Create component instances
ac = AircraftDynamics(scaling)
obj = Objective()
ic = InitialConditions(scaling)
fc = FinalConditions(scaling)
# TrajectoryModel creates a submodel that links the dynamics and an automatic trajectory source
traj = am.TrajectoryModel(ac)

# Number of bspline control points
nctrl = 10

# Interpolate from the x control points to the angle of attack
bspline = BSpline(
    input_name="x",
    output_name="alpha",
    num_interp_points=(num_time_steps + 1),
    num_ctrl_points=nctrl,
)

# Create the model
module_name = "time_to_climb"
model = am.Model(module_name)

# Add components to the model
model.add_model("bspline", bspline.create_model())
model.add_component("obj", 1, obj)
model.add_component("ic", 1, ic)
model.add_component("fc", 1, fc)
# Instead of adding the dynamics and trapezoid integration separately, add the submodel
model.add_model("traj", traj.create_model())

# Link the alpha values - ac.alpha to bspline output
model.link("traj.source.alpha", "bspline.interp_values.alpha")

# Link final time from objective to all trapezoidal rule components
model.link("obj.tf[0]", f"traj.kernel.tf[:]")

# Link boundary conditions using new utility
traj.link_boundary_conditions(model, "traj", "ic", "fc")

# Build the module if requested
if args.build:
    model.build_module()

# Initialize the model
model.initialize()

print(f"Num variables:              {model.num_variables}")
print(f"Num constraints:            {model.num_constraints}")

# Create the design vector
x = model.create_vector()

# Default to zero design variable values
x[:] = 0.0

# Set initial guess for final time
tf_guess = 200.0
x["obj.tf"] = tf_guess

# Set initial guess for states (reasonable trajectory)
t_guess = np.linspace(0, tf_guess, num_time_steps + 1)

# Set the initial guess
x["traj.source.q[:, 0]"] = 1.36 + (3.40 - 1.36) * t_guess / tf_guess  # velocity
x["traj.source.q[:, 1]"] = 5.0 * np.sin(np.pi * t_guess / tf_guess)  # flight path angle
x["traj.source.q[:, 2]"] = 1.0 + (20.0 - 1.0) * t_guess / tf_guess  # altitude
x["traj.source.q[:, 3]"] = 1.36 * t_guess  # range
x["traj.source.q[:, 4]"] = 19.03 - 0.2 * t_guess / tf_guess  # mass decrease

# Set initial guess for control (constant small angle)
x["traj.source.alpha"] = 1.0

# Set up bounds
lower = model.create_vector()
upper = model.create_vector()

# Final time bounds
lower["obj.tf"] = 1.0
upper["obj.tf"] = float("inf")

# Control bounds (matching original -5 to 5 degrees)
lower["traj.source.alpha"] = -float("inf")
upper["traj.source.alpha"] = float("inf")

# Unconstrain the state bounds by default
lower["traj.source.q"] = -float("inf")
upper["traj.source.q"] = float("inf")

# State bounds for the flight path angle
lower["traj.source.q[:, 1]"] = -90.0
upper["traj.source.q[:, 1]"] = 90.0

# State bounds for the altitude
lower["traj.source.q[:, 2]"] = 0.0
upper["traj.source.q[:, 2]"] = 25.0

# Set the anlge of attack between an lower and an upper bound
lower["bspline.control_points.x"] = -25.0
upper["bspline.control_points.x"] = 25.0

# Optimize
opt = am.Optimizer(model, x, lower=lower, upper=upper, solver="mumps")
data = opt.optimize(
    {
        "initial_barrier_param": 1.0,
        "monotone_barrier_fraction": 0.25,
        "barrier_strategy": "monotone",
        "convergence_tolerance": 1e-10,
        "max_line_search_iterations": 5,
        "max_iterations": 1000,
        "init_affine_step_multipliers": False,
    }
)

# Save optimization data
with open("time_to_climb_opt_data.json", "w") as fp:
    json.dump(data, fp, indent=2)

# Extract results
tf_opt = x["obj.tf"][0]  # Extract scalar from array
q = x["traj.source.q"]
alpha = x["traj.source.alpha"]
t = np.linspace(0, tf_opt, num_time_steps + 1)

# Print the optimized results
print(f"\nOptimization Results:")
print(f"Optimal time:   {tf_opt:.2f} seconds")
print(f"Final altitude: {q[-1, 2]:.0f} m")
print(f"Final velocity: {q[-1, 0]:.1f} m/s")
print(f"Final mass:     {q[-1, 4]:.0f} kg")

# Plot results
plot_results(t, q, alpha)
