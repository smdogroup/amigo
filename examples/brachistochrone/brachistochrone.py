import amigo as am
import numpy as np
import argparse
import json
import matplotlib.pylab as plt
import niceplots

num_time_steps = 100

# Assumptions:
# Only gravity is acting on the particle


class TrapezoidRule(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("tf")  # Final time (design variable)
        self.add_input("q1")
        self.add_input("q2")
        self.add_input("q1dot")
        self.add_input("q2dot")

        self.add_constraint("res")

        return

    def compute(self):
        tf = self.inputs["tf"]
        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        q1dot = self.inputs["q1dot"]
        q2dot = self.inputs["q2dot"]

        dt = tf / num_time_steps
        self.constraints["res"] = q2 - q1 - 0.5 * dt * (q1dot + q2dot)


class ParticleDynamics(am.Component):
    def __init__(self):
        super().__init__()

        # Declare constants
        self.add_constant("g", value=9.80655, label="gravitational acceleration")

        # Declare control
        self.add_input("theta", label="control")

        # Declare inputs
        self.add_input("q", shape=(3), label="state")
        self.add_input("qdot", shape=(3), label="rate")

        # Delcare the dynamics constraints
        self.add_constraint("res", shape=(3), label="residual")

    def compute(self):
        g = self.constants["g"]
        theta = self.inputs["theta"]
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        # Dynamic constraints equation
        res = 3 * [None]

        # Position derivatives equal velocities
        # Compute the declared variable values
        sint = self.vars["sint"] = am.sin(theta)
        cost = self.vars["cost"] = am.cos(theta)
        res[0] = qdot[0] - q[2] * sint
        res[1] = qdot[1] + q[2] * cost

        # Tangential acceleration component due to gravity
        res[2] = qdot[2] - g * cost

        self.constraints["res"] = res

        return


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
    def __init__(self):
        super().__init__()
        self.add_input("q", shape=3)
        self.add_constraint("res", shape=3)

    def compute(self):
        q = self.inputs["q"]
        self.constraints["res"] = [q[0], q[1] - 10.0, q[2]]  # q(0) = [0, 10, 0]


class FinalConditions(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("q", shape=3)
        self.add_constraint("res", shape=2)  # Only constrain x and y

    def compute(self):
        q = self.inputs["q"]
        self.constraints["res"] = [q[0] - 10.0, q[1] - 5.0]  # Target: (x=10, y=5)


def plot_result(x):
    t = np.linspace(0, x["obj.tf"], num_time_steps + 1)
    xvals = x["dynamics.q[:, 0]"]
    yvals = x["dynamics.q[:, 1]"]
    vvals = x["dynamics.q[:, 2]"]
    theta = x["dynamics.theta"]

    with plt.style.context(niceplots.get_style()):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # State variables
        axes[0, 0].plot(xvals, yvals)
        axes[0, 0].set_ylabel("x Position (m)")
        axes[0, 0].set_ylabel("y Position (m)")
        axes[0, 0].grid(True)

        axes[0, 1].plot(t, yvals)
        axes[0, 1].plot(t, xvals)
        axes[0, 1].set_ylabel("Position (m)")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].grid(True)

        axes[1, 0].plot(t, vvals)
        axes[1, 0].set_ylabel("Velocity (m/s)")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].grid(True)

        axes[1, 1].plot(t, theta)
        axes[1, 1].set_ylabel("Theta (rad)")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig("brachistochrone.png", dpi=300, bbox_inches="tight")
        plt.show()


def create_brachistochrone_model(module_name="brachistochrone"):
    dynamics = ParticleDynamics()
    trap = TrapezoidRule()
    obj = Objective()
    ic = InitialConditions()
    fc = FinalConditions()

    model = am.Model(module_name)

    # Add components
    model.add_component("dynamics", num_time_steps + 1, dynamics)
    model.add_component("trap", 3 * num_time_steps, trap)
    model.add_component("obj", 1, obj)
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)

    # Link state variables and derivatives
    for i in range(3):
        start = i * num_time_steps
        end = (i + 1) * num_time_steps
        # Link the state variables
        model.link(f"dynamics.q[:{num_time_steps}, {i}]", f"trap.q1[{start}:{end}]")
        model.link(f"dynamics.q[1:, {i}]", f"trap.q2[{start}:{end}]")

        # Link the state rates
        model.link(f"dynamics.qdot[:-1, {i}]", f"trap.q1dot[{start}:{end}]")
        model.link(f"dynamics.qdot[1:, {i}]", f"trap.q2dot[{start}:{end}]")

    # Link initial/final conditions and objective
    model.link("dynamics.q[0, :]", "ic.q[0, :]")
    model.link(f"dynamics.q[{num_time_steps}, :]", "fc.q[0, :]")

    # Broadcast the scalar final time from the objective component to every trapezoid instance
    model.link("obj.tf[0]", "trap.tf")

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
args = parser.parse_args()

model = create_brachistochrone_model()

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

with open("brachistochrone_model.json", "w") as fp:
    json.dump(model.get_serializable_data(), fp, indent=2)

print(f"Num variables:              {model.num_variables}")
print(f"Num constraints:            {model.num_constraints}")

prob = model.get_opt_problem()

# Create the design variable vector and provide an initial guess
x = model.create_vector()
x[:] = 0.0

# Linear initial guess for the path (straight line) and constant speed
N = num_time_steps + 1
x["dynamics.q[:, 0]"] = np.linspace(0.0, 10.0, N)  # x
x["dynamics.q[:, 1]"] = np.linspace(10.0, 5.0, N)  # y
x["dynamics.q[:, 2]"] = np.linspace(0.0, 9.9, N)  # v

# Set the variable values
x["dynamics.qdot"] = np.outer(np.linspace(0.0, 1.0, N), [1, 1, 1])

# Initial guess for the control (angle):
tetai = 0.1
tetaf = 0.1 * np.pi
x["dynamics.theta"] = np.linspace(tetai, tetaf, N)  # teta in radians

# Initial guess for final time (seconds)
x["obj.tf"] = 3.0

# Lower and upper bounds for selected variables
lower = model.create_vector()
upper = model.create_vector()

# Final time bounds
lower["obj.tf"] = 1.0
upper["obj.tf"] = float("inf")

lower["dynamics.q"] = -float("inf")
upper["dynamics.q"] = float("inf")

lower["dynamics.qdot"] = -float("inf")
upper["dynamics.qdot"] = float("inf")

# # Position x and y bounds
# # Bounds on x
# lower["dynamics.q[:, 0]"] = -1.0
# upper["dynamics.q[:, 0]"] = 20.0

# # # Bounds on y
# lower["dynamics.q[:, 1]"] = -1.0
# upper["dynamics.q[:, 1]"] = 20.0

# # Bounds on the velocity
# lower["dynamics.q[:, 2]"] = -1.0
# upper["dynamics.q[:, 2]"] = 50.0

# # Bounds on the control angle:
lower["dynamics.theta"] = 0.0
upper["dynamics.theta"] = np.pi

# lower["dynamics.theta"] = -float("inf")
# upper["dynamics.theta"] = float("inf")

opt = am.Optimizer(model, x, lower=lower, upper=upper)
data = opt.optimize(
    {
        "max_iterations": 500,
        "initial_barrier_param": 1.0,
        "max_line_search_iterations": 5,
        # "check_update_step": True,
    }
)

with open("brachistochrone_opt_data.json", "w") as fp:
    json.dump(data, fp, indent=2)

# Simple print-out of the optimized final time
print(f"Optimized final time: {x['obj.tf'][0]:.6f} s")

plot_result(x)
