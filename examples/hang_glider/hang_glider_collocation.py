import amigo as am
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt


"""
Maximum Range of a Hang Glider
==============================

This is example from Betts' book "Practical Methods for Optimal Control
Using NonlinearProgramming", 3rd edition, Chapter 10: Test Problems.

**States**

- :math:`x, y, v_x, v_y` : state variables

**Controls**

- :math:`C_L` : control variable
"""


num_time_steps = 200


class TrapezoidRule(am.Component):
    def __init__(self, scaling):
        super().__init__()

        self.scaling = scaling

        self.add_input("tf")  # Final time as input
        self.add_input("q1")
        self.add_input("q2")
        self.add_input("q1dot")
        self.add_input("q2dot")

        self.add_constraint("res")

        return

    def compute(self):
        tf = self.scaling["time"] * self.inputs["tf"]

        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        q1dot = self.inputs["q1dot"]
        q2dot = self.inputs["q2dot"]

        dt = tf / num_time_steps
        self.constraints["res"] = q2 - q1 - 0.5 * dt * (q1dot + q2dot)

        return


class GliderDynamics(am.Component):
    def __init__(self, scaling):
        super().__init__()

        self.scaling = scaling

        # Add constants (SI units)
        self.add_constant("uM", value=2.5)
        self.add_constant("m", value=100.0)
        self.add_constant("R", value=100.0)
        self.add_constant("S", value=14.0)
        self.add_constant("C0", value=0.034)
        self.add_constant("rho", value=1.13)
        self.add_constant("k", value=0.069662)
        self.add_constant("g", value=9.80665)

        # Inputs
        self.add_input("CL", label="Control")
        self.add_input("q", shape=4, label="state variables")
        self.add_input("qdot", shape=4, label="state derivatives")

        # Constraint residuals
        self.add_constraint("res", shape=4, label="dynamics of the hang glider")

        return

    def compute(self):
        # Get constants
        uM = self.constants["uM"]
        m = self.constants["m"]
        R = self.constants["R"]
        S = self.constants["S"]
        C0 = self.constants["C0"]
        rho = self.constants["rho"]
        k = self.constants["k"]
        g = self.constants["g"]

        # Get inputs
        CL = self.inputs["CL"]
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        # State variables
        x = self.scaling["distance"] * q[0]  # horizontal distance
        y = self.scaling["distance"] * q[1]  # altitude
        vx = self.scaling["velocity"] * q[2]  # horizontal velocity
        vy = self.scaling["velocity"] * q[3]  # vertical velocity

        # Get the required values to complete the dynamics
        # X = (x/R - 2.5)^2
        x_over_R = x / R
        X_term = x_over_R - 2.5
        X = X_term * X_term

        # ua = uM * (1 - X) * exp(-X)
        ua = uM * (1.0 - X) * am.exp(-X)

        # Vy = vy - ua
        Vy = vy - ua

        # vr = sqrt(vx^2 + Vy^2)
        vx_sq = vx * vx
        Vy_sq = Vy * Vy
        vr_sq = vx_sq + Vy_sq
        vr = vr_sq**0.5  # TODO amigo doesn't support the sqrt operation

        # sin_eta = Vy / vr, cos_eta = vx / vr
        sin_eta = Vy / vr
        cos_eta = vx / vr

        # Get the quadratic drag polar
        CD = C0 + k * CL * CL

        # Aero forces
        half_rho_S = 0.5 * rho * S
        D = half_rho_S * CD * vr_sq
        L = half_rho_S * CL * vr_sq

        # Gravitational force
        W = m * g

        # Dynamics describing the planar motion for the hang glider:
        res = 4 * [None]

        # Position derivatives
        res[0] = qdot[0] - vx / self.scaling["distance"]
        res[1] = qdot[1] - vy / self.scaling["distance"]

        # Velocity derivatives (accelerations)
        vxdot = (-L * sin_eta - D * cos_eta) / m
        vydot = (L * cos_eta - D * sin_eta - W) / m
        res[2] = qdot[2] - vxdot / self.scaling["velocity"]
        res[3] = qdot[3] - vydot / self.scaling["velocity"]

        self.constraints["res"] = res

        return


class RangeObjective(am.Component):
    def __init__(self, scaling):
        super().__init__()

        self.scaling = scaling
        self.add_input("xf", label="Horizontal distance at tf")
        self.add_objective("obj")

        return

    def compute(self):
        xf = self.inputs["xf"]
        # Only the final x (range) should be maximized
        self.objective["obj"] = -xf  # / self.scaling["distance"]


class InitialConditions(am.Component):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("q", shape=4)
        self.add_constraint("res", shape=4)

    def compute(self):
        # Initial conditions from the problem specification
        x0, y0, vx0, vy0 = 0.0, 1000.0, 13.227567500, -1.2875005200
        q = self.inputs["q"]
        self.constraints["res"] = [
            q[0] - x0 / self.scaling["distance"],
            q[1] - y0 / self.scaling["distance"],
            q[2] - vx0 / self.scaling["velocity"],
            q[3] - vy0 / self.scaling["velocity"],
        ]


class FinalConditions(am.Component):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("q", shape=4)
        self.add_constraint("res", shape=3)

    def compute(self):
        # Final conditions from the problem specification
        yF, vxF, vyF = 900.0, 13.227567500, -1.2875005200
        q = self.inputs["q"]
        self.constraints["res"] = [
            q[1] - yF / self.scaling["distance"],  # altitude constraint
            q[2] - vxF / self.scaling["velocity"],  # horizontal velocity constraint
            q[3] - vyF / self.scaling["velocity"],  # vertical velocity constraint
        ]


# Build and link the model
def create_hang_glide_model(scaling, num_time_steps=200, model_name="max_range_glide"):
    # Build the components
    gd = GliderDynamics(scaling)
    trap = TrapezoidRule(scaling)
    obj = RangeObjective(scaling)
    ic = InitialConditions(scaling)
    fc = FinalConditions(scaling)

    model = am.Model(model_name)

    model.add_component("gd", num_time_steps + 1, gd)
    model.add_component("trap", 4 * num_time_steps, trap)
    model.add_component("obj", 1, obj)
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)

    # Add the links
    for i in range(4):
        start = i * num_time_steps
        end = (i + 1) * num_time_steps

        # Link the state variables
        model.link(f"gd.q[:{num_time_steps}, {i}]", f"trap.q1[{start}:{end}]")
        model.link(f"gd.q[1:, {i}]", f"trap.q2[{start}:{end}]")

        # Link the state rates
        model.link(f"gd.qdot[:-1, {i}]", f"trap.q1dot[{start}:{end}]")
        model.link(f"gd.qdot[1:, {i}]", f"trap.q2dot[{start}:{end}]")

    # Link the initial and final conditions with the Dynamics
    model.link("gd.q[0, :]", "ic.q[0, :]")
    model.link(f"gd.q[{num_time_steps}, :]", "fc.q[0, :]")

    model.link(f"gd.q[{num_time_steps},0]", "obj.xf[0]")  # pass xf

    return model


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
    "--show-sparsity",
    dest="show_sparsity",
    action="store_true",
    default=False,
    help="Show the sparsity pattern",
)
parser.add_argument(
    "--single-mesh",
    dest="single_mesh",
    action="store_true",
    default=False,
    help="Use single mesh instead of mesh refinement",
)
args = parser.parse_args()

# Set the scaling
scaling = {"velocity": 10.0, "distance": 100.0, "time": 10.0}
model = create_hang_glide_model(scaling)

if args.build:
    compile_args = []
    link_args = []
    define_macros = []
    if args.use_openmp:
        compile_args = ["-fopenmp"]
        link_args = ["-fopenmp"]
        define_macros = [("AMIGO_USE_OPENMP", "1")]

    model.build_module(
        compile_args=compile_args, link_args=link_args, define_macros=define_macros
    )

model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)

with open("glider_model.json", "w") as fp:
    json.dump(model.get_serializable_data(), fp, indent=2)

print(f"Num variables:              {model.num_variables}")
print(f"Num constraints:            {model.num_constraints}")

# Get the design variables
x = model.create_vector()
x[:] = 0.0

# Set initial guess following book recommendations:
tf_guess = 100.0
t_guess = np.linspace(0, tf_guess, num_time_steps + 1)
t_frac = t_guess / tf_guess
x[f"gd.q[:, 0]"] = (0.0 + t_frac * 1250.0) / scaling["distance"]

# Interpolate y from 1000 to 900
x[f"gd.q[:, 1]"] = (1000.0 + t_frac * (900.0 - 1000.0)) / scaling["distance"]

# Keep velocities constant as per boundary conditions
x[f"gd.q[:, 2]"] = 13.227567500 / scaling["velocity"]
x[f"gd.q[:, 3]"] = -1.2875005200 / scaling["velocity"]

# Set initial control values (as recommended in the book: CL(0) = CL(tF) = 1)
x["gd.CL"] = 1.0

# Set initial final time (from the book's plots, optimal time is around 100 seconds)
x["trap.tf"] = tf_guess / scaling["time"]

# Set bounds for the optimization problem:
lower = model.create_vector()
upper = model.create_vector()

# Control bounds (from problem specification)
lower["gd.CL"] = -4.0
upper["gd.CL"] = 4.0  # 1.4

# Time bounds
lower["trap.tf"] = 50.0 / scaling["time"]
# upper["trap.tf"] = 150.0 / scaling["time"]
upper["trap.tf"] = float("inf")

# State bounds - add reasonable physical bounds
# Horizontal position (x)
lower["gd.q[:, 0]"] = -2000.0 / scaling["distance"]
upper["gd.q[:, 0]"] = 2000.0 / scaling["distance"]

# Altitude (y)
lower["gd.q[:, 1]"] = -2000.0 / scaling["distance"]
upper["gd.q[:, 1]"] = 2000.0 / scaling["distance"]

# Horizontal velocity (vx)
lower["gd.q[:, 2]"] = -200.0 / scaling["velocity"]
upper["gd.q[:, 2]"] = 200.0 / scaling["velocity"]

# Vertical velocity (vy)
lower["gd.q[:, 3]"] = -200.0 / scaling["velocity"]
upper["gd.q[:, 3]"] = 200.0 / scaling["velocity"]

lower["gd.qdot"] = -float("inf")
upper["gd.qdot"] = float("inf")

opt = am.Optimizer(model, x, lower=lower, upper=upper)
data = opt.optimize(
    {
        "initial_barrier_param": 0.1,
        "max_line_search_iterations": 1,
        "max_iterations": 500,
        "init_affine_step_multipliers": False,
    }
)

with open("hang_glider_opt_data.json", "w") as fp:
    json.dump(data, fp, indent=2)
