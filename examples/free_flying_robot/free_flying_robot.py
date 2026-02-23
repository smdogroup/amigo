import amigo as am
import numpy as np
import argparse
import matplotlib.pyplot as plt
import niceplots

"""
Free Flying Robot Optimal Control Problem
==========================================

This is example from Betts' book "Practical Methods for Optimal Control
Using Nonlinear Programming", 3rd edition, Chapter 8, Section 8.45.

The problem is also known as the Sakawa problem, featuring a free-flying
robot with two thrusters. The objective is to minimize fuel consumption
(sum of absolute thrust values) while transferring the robot from initial
to final state.

State variables:
    y1, y2, y3 - position (x, y) and angle
    y4, y5, y6 - velocity (vx, vy) and angular velocity

Control variables:
    u1, u2, u3, u4 - positive/negative components of two thrusters
    T1 = u1 - u2, T2 = u3 - u4
    |T1| = u1 + u2, |T2| = u3 + u4

The controls are split to avoid absolute values in the objective.

Reference: Betts (2010), Chapter 8, Example 8.45
"""

# Problem parameters
final_time = 12.0
num_time_steps = 100

# Physical constants
alpha = 0.2  # Distance parameter for thruster 1
beta = 0.2  # Distance parameter for thruster 2


class TrapezoidRule(am.Component):
    """
    Trapezoidal collocation for numerical integration.
    Enforces: q2 - q1 = 0.5 * dt * (q1dot + q2dot)
    """

    def __init__(self):
        super().__init__()

        self.add_constant("dt", value=final_time / num_time_steps)

        self.add_input("q1")
        self.add_input("q2")
        self.add_input("q1dot")
        self.add_input("q2dot")

        self.add_constraint("res")

    def compute(self):
        dt = self.constants["dt"]
        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        q1dot = self.inputs["q1dot"]
        q2dot = self.inputs["q2dot"]

        self.constraints["res"] = q2 - q1 - 0.5 * dt * (q1dot + q2dot)


class FreeFlyingRobotDynamics(am.Component):
    """
    Free flying robot dynamics with two thrusters.

    State equations:
        dy1/dt = y4                                    (position x)
        dy2/dt = y5                                    (position y)
        dy3/dt = y6                                    (angle)
        dy4/dt = [u1 - u2 + u3 - u4] * cos(y3)        (velocity x)
        dy5/dt = [u1 - u2 + u3 - u4] * sin(y3)        (velocity y)
        dy6/dt = alpha*(u1 - u2) - beta*(u3 - u4)     (angular velocity)

    Constraints:
        u1 + u2 <= 1
        u3 + u4 <= 1
        uk >= 0 for k = 1,2,3,4
    """

    def __init__(self):
        super().__init__()

        self.add_constant("alpha", value=alpha)
        self.add_constant("beta", value=beta)

        # State and control inputs
        self.add_input("q", shape=6, label="state [y1,y2,y3,y4,y5,y6]")
        self.add_input("qdot", shape=6, label="state derivatives")
        self.add_input("u", shape=4, label="control [u1,u2,u3,u4]")

        # Constraints
        self.add_constraint("res", shape=6, label="dynamics residuals")

    def compute(self):
        # Get constants
        alpha = self.constants["alpha"]
        beta = self.constants["beta"]

        # Get inputs
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]
        u = self.inputs["u"]

        # State variables
        y1, y2, y3, y4, y5, y6 = q[0], q[1], q[2], q[3], q[4], q[5]

        # Control variables
        u1, u2, u3, u4 = u[0], u[1], u[2], u[3]

        # Thruster forces (net thrust)
        T1 = u1 - u2  # Thruster 1
        T2 = u3 - u4  # Thruster 2
        T_total = T1 + T2  # Total thrust

        # Trigonometric terms
        cos_y3 = self.vars["cos_y3"] = am.cos(y3)
        sin_y3 = self.vars["sin_y3"] = am.sin(y3)

        # Dynamics equations (from Betts eq. 8.230-8.235)
        res = [None] * 6
        res[0] = qdot[0] - y4
        res[1] = qdot[1] - y5
        res[2] = qdot[2] - y6
        res[3] = qdot[3] - T_total * cos_y3
        res[4] = qdot[4] - T_total * sin_y3
        # Angular dynamics: dω/dt = α*T1 - β*T2
        # Thrusters on opposite sides of CoM create opposite torques
        res[5] = qdot[5] - (alpha * T1 - beta * T2)

        self.constraints["res"] = res


class Objective(am.Component):
    """
    Objective: Minimize fuel consumption

    Implemented as: minimize ∫(u1 + u2 + u3 + u4) dt
    Using trapezoidal rule for integration.
    """

    def __init__(self):
        super().__init__()

        self.add_constant("dt", value=final_time / num_time_steps)

        self.add_input("u1", shape=4, label="control at time i")
        self.add_input("u2", shape=4, label="control at time i+1")

        self.add_objective("obj")

    def compute(self):
        u1 = self.inputs["u1"]  # shape (4,) - controls at time i
        u2 = self.inputs["u2"]  # shape (4,) - controls at time i+1
        dt = self.constants["dt"]

        # Sum of all 4 controls at each time point
        sum_u1 = u1[0] + u1[1] + u1[2] + u1[3]
        sum_u2 = u2[0] + u2[1] + u2[2] + u2[3]

        # Trapezoidal integration: 0.5 * dt * (f(t_i) + f(t_{i+1}))
        self.objective["obj"] = 0.5 * dt * (sum_u1 + sum_u2)


class InitialConditions(am.Component):
    """
    Initial conditions (from GPOPS-II Free Flying Robot problem)
    Starting position: (-10, -10) with angle π/2, all velocities zero
        x(0) = -10, y(0) = -10, θ(0) = π/2
        vx(0) = 0, vy(0) = 0, ω(0) = 0
    """

    def __init__(self):
        super().__init__()
        self.add_constant("pi", value=np.pi)
        self.add_input("q", shape=6)
        self.add_constraint("res", shape=6)

    def compute(self):
        pi = self.constants["pi"]
        q = self.inputs["q"]
        # Initial conditions: start at (-10, -10) with angle π/2
        self.constraints["res"] = [
            q[0] + 10.0,  # x(0) = -10
            q[1] + 10.0,  # y(0) = -10
            q[2] - pi / 2.0,  # Use constant, not np.pi!
            q[3] - 0.0,  # vx(0) = 0
            q[4] - 0.0,  # vy(0) = 0
            q[5] - 0.0,  # ω(0) = 0
        ]


class FinalConditions(am.Component):
    """
    Final conditions (from GPOPS-II Free Flying Robot problem)
    Target: reach origin (0, 0) with angle 0, all velocities zero
        x(tf) = 0, y(tf) = 0, θ(tf) = 0
        vx(tf) = 0, vy(tf) = 0, ω(tf) = 0
    """

    def __init__(self):
        super().__init__()
        self.add_input("q", shape=6)
        self.add_constraint("res", shape=6)

    def compute(self):
        q = self.inputs["q"]

        # Final state conditions: reach origin with zero angle and velocities
        self.constraints["res"] = [
            q[0] - 0.0,  # x(tf) = 0
            q[1] - 0.0,  # y(tf) = 0
            q[2] - 0.0,  # θ(tf) = 0
            q[3] - 0.0,  # vx(tf) = 0
            q[4] - 0.0,  # vy(tf) = 0
            q[5] - 0.0,  # ω(tf) = 0
        ]


def plot_results(t, q, u):
    """Plot the optimization results"""

    with plt.style.context(niceplots.get_style()):
        fig, axes = plt.subplots(4, 3, figsize=(15, 16))

        # State variables - position
        axes[0, 0].plot(t, q[:, 0])
        axes[0, 0].set_ylabel("x (position)")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].grid(True)

        axes[0, 1].plot(t, q[:, 1])
        axes[0, 1].set_ylabel("y (position)")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].grid(True)

        axes[0, 2].plot(t, q[:, 2])
        axes[0, 2].set_ylabel("θ (angle, rad)")
        axes[0, 2].set_xlabel("Time (s)")
        axes[0, 2].grid(True)

        # State variables - velocity
        axes[1, 0].plot(t, q[:, 3])
        axes[1, 0].set_ylabel("vx (velocity x)")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].grid(True)

        axes[1, 1].plot(t, q[:, 4])
        axes[1, 1].set_ylabel("vy (velocity y)")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].grid(True)

        axes[1, 2].plot(t, q[:, 5])
        axes[1, 2].set_ylabel("ω (angular velocity)")
        axes[1, 2].set_xlabel("Time (s)")
        axes[1, 2].grid(True)

        # Individual control variables
        axes[2, 0].plot(t, u[:, 0])
        axes[2, 0].set_ylabel("u1")
        axes[2, 0].set_xlabel("Time (s)")
        axes[2, 0].grid(True)

        axes[2, 1].plot(t, u[:, 1])
        axes[2, 1].set_ylabel("u2")
        axes[2, 1].set_xlabel("Time (s)")
        axes[2, 1].grid(True)

        axes[2, 2].plot(t, u[:, 2])
        axes[2, 2].set_ylabel("u3")
        axes[2, 2].set_xlabel("Time (s)")
        axes[2, 2].grid(True)

        # u4 and net thrusts
        axes[3, 0].plot(t, u[:, 3])
        axes[3, 0].set_ylabel("u4")
        axes[3, 0].set_xlabel("Time (s)")
        axes[3, 0].grid(True)

        axes[3, 1].plot(t, u[:, 0] - u[:, 1])
        axes[3, 1].set_ylabel("T1 = u1 - u2")
        axes[3, 1].set_xlabel("Time (s)")
        axes[3, 1].grid(True)

        axes[3, 2].plot(t, u[:, 2] - u[:, 3])
        axes[3, 2].set_ylabel("T2 = u3 - u4")
        axes[3, 2].set_xlabel("Time (s)")
        axes[3, 2].grid(True)

        plt.tight_layout()
        plt.savefig("freeflyingrobot_results.png", dpi=300, bbox_inches="tight")
        plt.show()


def plot_convergence(nrms):
    """Plot convergence history"""
    with plt.style.context(niceplots.get_style()):
        fig, ax = plt.subplots(1, 1)

        ax.semilogy(nrms, marker="o", clip_on=False, lw=2.0)
        ax.set_ylabel("KKT residual norm")
        ax.set_xlabel("Iteration")
        ax.grid(True)

        niceplots.adjust_spines(ax)
        fig.savefig("freeflyingrobot_convergence.png", dpi=300, bbox_inches="tight")


def create_freeflyingrobot_model(module_name="freeflyingrobot"):
    """Create the free flying robot optimization model"""
    robot = FreeFlyingRobotDynamics()
    trap = TrapezoidRule()
    obj = Objective()
    ic = InitialConditions()
    fc = FinalConditions()

    model = am.Model(module_name)

    # Add components
    model.add_component("robot", num_time_steps + 1, robot)
    model.add_component("trap", 6 * num_time_steps, trap)
    model.add_component("obj", num_time_steps, obj)
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)

    # Link state variables and derivatives through trapezoidal rule
    for i in range(6):  # 6 states
        start = i * num_time_steps
        end = (i + 1) * num_time_steps

        # Link state variables: q[0:N-1] -> q1, q[1:N] -> q2
        model.link(f"robot.q[:{num_time_steps}, {i}]", f"trap.q1[{start}:{end}]")
        model.link(f"robot.q[1:, {i}]", f"trap.q2[{start}:{end}]")

        # Link state derivatives: qdot[0:N-1] -> q1dot, qdot[1:N] -> q2dot
        model.link(f"robot.qdot[:-1, {i}]", f"trap.q1dot[{start}:{end}]")
        model.link(f"robot.qdot[1:, {i}]", f"trap.q2dot[{start}:{end}]")

    # Link controls to objective (for integration)
    model.link("robot.u[:-1, :]", "obj.u1[:, :]")
    model.link("robot.u[1:, :]", "obj.u2[:, :]")

    # Link boundary conditions
    model.link("robot.q[0, :]", "ic.q[0, :]")
    model.link(f"robot.q[{num_time_steps}, :]", "fc.q[0, :]")

    return model


# Command line argument parsing
parser = argparse.ArgumentParser(
    description="Free Flying Robot Optimal Control Problem"
)
parser.add_argument(
    "--build",
    dest="build",
    action="store_true",
    default=False,
    help="Build the C++ module",
)
args = parser.parse_args()

model = create_freeflyingrobot_model()

if args.build:
    model.build_module()

model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)

print(f"Number of variables:     {model.num_variables}")
print(f"Number of constraints:   {model.num_constraints}")

# Create design variable vector
x = model.create_vector()
x[:] = 0.0

# Initial guess for states - linear interpolation from initial to final
t_normalized = np.linspace(0, 1, num_time_steps + 1)

# Position states: linear interpolation (smooth S-curve would be better)
# x: -10 -> 0, y: -10 -> 0, θ: π/2 -> 0
# Use smooth polynomial that satisfies BCs: x(0)=-10, x(1)=0, x'(0)=0, x'(1)=0
s = t_normalized
pos_profile = 3 * s**2 - 2 * s**3  # Smooth transition from 0 to 1
x["robot.q[:, 0]"] = -10.0 + 10.0 * pos_profile  # x: -10 -> 0
x["robot.q[:, 1]"] = -10.0 + 10.0 * pos_profile  # y: -10 -> 0
x["robot.q[:, 2]"] = np.pi / 2.0 * (1.0 - pos_profile)  # θ: π/2 -> 0

# Velocity states: derivative of position profile
# d/dt(pos_profile) = (6s - 6s^2) * (1/tf)
vel_profile = (6 * s - 6 * s**2) / final_time
x["robot.q[:, 3]"] = 10.0 * vel_profile  # vx (scaled by position change)
x["robot.q[:, 4]"] = 10.0 * vel_profile  # vy
x["robot.q[:, 5]"] = -np.pi / 2.0 * vel_profile  # ω (angle rate)

# Initialize qdot consistent with the trajectory using numerical differentiation
dt = final_time / num_time_steps
for i in range(6):
    x[f"robot.qdot[:, {i}]"] = np.gradient(x[f"robot.q[:, {i}]"], dt)

# Control initial guess: bang-bang pattern for minimum fuel
# Use smooth transition to help optimizer
t_frac = np.linspace(0, 1, num_time_steps + 1)
switch = 0.5 * (1 + np.tanh(10 * (t_frac - 0.5)))

# First half: positive thrust, second half: negative thrust
x["robot.u[:, 0]"] = 0.4 * (1 - switch)  # u1
x["robot.u[:, 1]"] = 0.4 * switch  # u2
x["robot.u[:, 2]"] = 0.4 * (1 - switch)  # u3
x["robot.u[:, 3]"] = 0.4 * switch  # u4

# Set bounds
lower = model.create_vector()
upper = model.create_vector()

# State bounds (reasonable physical limits)
lower["robot.q"] = -float("inf")
upper["robot.q"] = float("inf")

# State derivative bounds (free)
lower["robot.qdot"] = -float("inf")
upper["robot.qdot"] = float("inf")

# Control bounds: 0 <= uk <= 1 (since u1+u2 <= 1 and u3+u4 <= 1)
lower["robot.u"] = 0.0
upper["robot.u"] = 1.0

opt = am.Optimizer(model, x, lower=lower, upper=upper)
opt_data = opt.optimize(
    {
        "barrier_strategy": "monotone",
        "initial_barrier_param": 10.0,
        "monotone_barrier_fraction": 0.1,
        "max_line_search_iterations": 50,
        "max_iterations": 1000,
        "convergence_tolerance": 1e-6,
        "init_least_squares_multipliers": True,
        "init_affine_step_multipliers": False,
        "use_armijo_line_search": True,
        "regularization_eps_x": 1e-6,
        "regularization_eps_z": 1e-6,
        "adaptive_regularization": True,
        "fraction_to_boundary": 0.99,
    }
)

print(f"\nConverged: {opt_data.get('converged', False)}")
print(f"Final residual: {opt_data['iterations'][-1]['residual']:.2e}")
print(f"Iterations: {len(opt_data['iterations'])}")

# Compute total fuel consumption
total_fuel = 0.0
u_vals = x["robot.u"]
for i in range(num_time_steps):
    sum_u1 = np.sum(u_vals[i])
    sum_u2 = np.sum(u_vals[i + 1])
    total_fuel += 0.5 * dt * (sum_u1 + sum_u2)
print(f"  Total fuel (objective): {total_fuel:.4f}")

# Extract results
q = x["robot.q"]
u = x["robot.u"]
t = np.linspace(0, final_time, num_time_steps + 1)

# Plot results
plot_results(t, q, u)
