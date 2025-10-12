import amigo as am
import numpy as np
import argparse
import json
import matplotlib.pylab as plt
import matplotlib.animation as animation
import niceplots


# Assumptions:
# unit gravity
# cannon is a point mass (no rotational inertia)
"""
Optimal trajectory problem for aiming a cannon with non-negligable quadratic drag on the cannon ball
==============================

**States**

- :math:`x, y, v_x, v_y` : state variables

**Controls**

- :math:`c` : control variable

**Objective**
minimize the amount of powder used for the shot
"""


num_time_steps = 50
final_time = 3.0


class TrapezoidRule(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("dt", value=final_time / num_time_steps)

        self.add_input("q1")
        self.add_input("q2")

        self.add_input("q1dot")
        self.add_input("q2dot")

        self.add_constraint("res")

        return

    def compute(self):
        dt = self.constants["dt"]

        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        q1dot = self.inputs["q1dot"]
        q2dot = self.inputs["q2dot"]

        self.constraints["res"] = (q2 - q1) - 0.5 * dt * (q1dot + q2dot)

        return


class CannonDynamics(am.Component):
    def __init__(self):
        super().__init__()

        # Declare Constant Values
        self.add_constant("c", value=0.4, label="drag")

        # Declare inputs
        self.add_input("q", shape=(4), label="state")
        self.add_input("qdot", shape=(4), label="rate")

        # Delcare the dynamics constraints
        self.add_constraint("res", shape=(4), label="residual")

    def compute(self):
        # g = self.constants["g"]
        c = self.constants["c"]
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        # Dynamic constraints equations
        res = 4 * [None]
        # v = (q[2] * q[2] + q[3] * q[3]) ** 0.5
        v = self.vars["v"] = am.sqrt(q[2] * q[2] + q[3] * q[3])

        # Position derivatives equal velocities
        res[0] = qdot[0] - q[2]  # xdot = vx
        res[1] = qdot[1] - q[3]  # ydot = vy

        # Velocity derivatives (accelerations) with drag
        res[2] = qdot[2] + c * q[2] * v  # vxdot = -c*vx*v
        res[3] = qdot[3] + c * q[3] * v + 1  # vydot = -c*vy*v - g # assume unit gravity

        self.constraints["res"] = res
        return


class Objective(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("q", shape=(4), label="state")
        self.add_objective("obj")

    def compute(self):
        q = self.inputs["q"]
        # Only the initial state's velocities
        self.objective["obj"] = q[2] * q[2] + q[3] * q[3]


# Initial conditions are q = 0
class InitialConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("q", shape=4)
        self.add_constraint("res", shape=2)

    def compute(self):
        q = self.inputs["q"]
        self.constraints["res"] = [q[0], q[1]]


# Final conditions
class FinalConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("q", shape=4)
        self.add_constraint("res", shape=2)

    def compute(self):
        q = self.inputs["q"]
        # Final position at target (xT=6, yT=0)
        self.constraints["res"] = [q[0] - 6.0, q[1]]


def create_cannon_model(module_name="cannon"):
    cannon = CannonDynamics()
    trap = TrapezoidRule()
    obj = Objective()
    ic = InitialConditions()
    fc = FinalConditions()

    model = am.Model(module_name)

    model.add_component("cannon", num_time_steps + 1, cannon)
    model.add_component("trap", 4 * num_time_steps, trap)
    model.add_component("obj", 1, obj)
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)

    # Add the links
    for i in range(4):
        start = i * num_time_steps
        end = (i + 1) * num_time_steps

        # Link the state variables
        model.link(f"cannon.q[:{num_time_steps}, {i}]", f"trap.q1[{start}:{end}]")
        model.link(f"cannon.q[1:, {i}]", f"trap.q2[{start}:{end}]")

        # Link the state rates
        model.link(f"cannon.qdot[:-1, {i}]", f"trap.q1dot[{start}:{end}]")
        model.link(f"cannon.qdot[1:, {i}]", f"trap.q2dot[{start}:{end}]")

    # Link the initial and final conditions with the Dynamics
    model.link("cannon.q[0, :]", "ic.q[0, :]")
    model.link(f"cannon.q[{num_time_steps}, :]", "fc.q[0, :]")

    model.link("cannon.q[0,:]", "obj.q[0,:]")  # pass (vx0, vy0)

    return model


def plot_trajectory(x_opt, title="Cannon Trajectory", save_name="cannon_trajectory"):
    """Plot the final optimized trajectory"""
    t = np.linspace(0, final_time, num_time_steps + 1)

    # Extract positions and velocities
    x_pos = x_opt["cannon.q[:,0]"]
    y_pos = x_opt["cannon.q[:,1]"]
    vx = x_opt["cannon.q[:,2]"]
    vy = x_opt["cannon.q[:,3]"]

    with plt.style.context(niceplots.get_style()):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Trajectory plot
        ax1.plot(x_pos, y_pos, "b-", linewidth=2, label="Trajectory")
        ax1.plot(x_pos[0], y_pos[0], "go", markersize=8, label="Start")
        ax1.plot(x_pos[-1], y_pos[-1], "ro", markersize=8, label="End")
        ax1.plot(6.0, 0.0, "r*", markersize=12, label="Target")
        ax1.set_xlabel("X Position", fontsize=10)
        ax1.set_ylabel("Y Position", fontsize=10)
        ax1.set_title("Trajectory", fontsize=10)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.axis("equal")

        # X position vs time
        ax2.plot(t, x_pos, "b-", linewidth=2)
        ax2.set_xlabel("Time (s)", fontsize=10)
        ax2.set_ylabel("X Position", fontsize=10)
        ax2.set_title("X Position vs Time", fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Y position vs time
        ax3.plot(t, y_pos, "r-", linewidth=2)
        ax3.set_xlabel("Time (s)", fontsize=10)
        ax3.set_ylabel("Y Position", fontsize=10)
        ax3.set_title("Y Position vs Time", fontsize=10)
        ax3.grid(True, alpha=0.3)

        # Velocities vs time
        ax4.plot(t, vx, "b-", linewidth=2, label="V_x")
        ax4.plot(t, vy, "r-", linewidth=2, label="V_y")
        ax4.set_xlabel("Time (s)", fontsize=10)
        ax4.set_ylabel("Velocity", fontsize=10)
        ax4.set_title("Velocities vs Time", fontsize=10)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(f"{save_name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{save_name}.svg", bbox_inches="tight")
        plt.show()

    return fig


def animate_optimization_progress(trajectory_history, save_name="cannon_optimization"):
    """Create animation showing optimization progress"""
    with plt.style.context(niceplots.get_style()):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Set up the plot
        ax.set_xlim(-1, 7)
        ax.set_ylim(-1, 4)
        ax.set_xlabel("X Position", fontsize=10)
        ax.set_ylabel("Y Position", fontsize=10)
        ax.set_title("Cannon Trajectory Optimization Progress", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Plot target
        ax.plot(6.0, 0.0, "r*", markersize=12, label="Target")
        ax.plot(0.0, 0.0, "go", markersize=8, label="Start")

        # Initialize trajectory line
        (line,) = ax.plot([], [], "b-", linewidth=2, alpha=0.7)
        text = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        def animate(frame):
            if frame < len(trajectory_history):
                traj = trajectory_history[frame]
                line.set_data(traj["x"], traj["y"])
                text.set_text(f"Iteration: {traj['iteration']}")
            return line, text

        # Create animation
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=len(trajectory_history),
            interval=500,
            blit=True,
            repeat=True,
        )

        ax.legend(fontsize=8)

        # Save animation
        try:
            anim.save(f"{save_name}.gif", writer="pillow", fps=2)
            print(f"Animation saved as {save_name}.gif")
        except:
            print("Could not save animation (pillow writer not available)")

        plt.show()

    return anim


def plot_results(t, q):
    t = np.linspace(0, final_time, num_time_steps + 1)

    with plt.style.context(niceplots.get_style()):
        data = {}
        data["Cannon Horizontal Position"] = q[:, 0]
        data["Cannon Vertical Position"] = q[:, 1]

        fig, ax = niceplots.stacked_plots(
            "Time (s)",
            t,
            [data],
            lines_only=True,
            figsize=(10, 6),
            line_scaler=0.5,
        )

    fontname = "Arial"

    for axis in ax:
        # Update tick labels
        for tick in axis.get_xticklabels():
            tick.set_fontname(fontname)

        for tick in axis.get_yticklabels():
            tick.set_fontname(fontname)

    fig.savefig("cart_stacked.svg")
    fig.savefig("cart_stacked.png")

    return


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

model = create_cannon_model()

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

with open("cannon_model.json", "w") as fp:
    json.dump(model.get_serializable_data(), fp, indent=2)


print(f"Num variables:              {model.num_variables}")
print(f"Num constraints:            {model.num_constraints}")

# Get the design variables
x = model.create_vector()
x[:] = 0.0

# Set the initial conditions based on the variables
# Calculate initial velocities for ballistic trajectory to reach target (6m, 0m) in 4s
target_x = 6.0
target_y = 0.0
g = 1

vx_init = target_x / final_time
vy_init = target_y / final_time + 0.5 * g * final_time

N = num_time_steps + 1
t = np.linspace(0, final_time, N)

# Ballistic trajectory (no drag) initial guess
x0 = vx_init * t
y0 = vy_init * t - 0.5 * g * t * t

# Ballistic velocities initial guess
vx0 = np.full(N, vx_init)
vy0 = vy_init - g * t

x["cannon.q[:,0]"] = x0
x["cannon.q[:,1]"] = y0
x["cannon.q[:,2]"] = vx0
x["cannon.q[:,3]"] = vy0

# Set Bounds
lower = model.create_vector()
upper = model.create_vector()
# Position bounds
lower["cannon.q[:,0]"] = -10  # x position
upper["cannon.q[:,0]"] = 50
lower["cannon.q[:,1]"] = -50  # y position
upper["cannon.q[:,1]"] = 50

# Velocity bounds
lower["cannon.q[:,2]"] = -50  # vx bounds
upper["cannon.q[:,2]"] = 50
lower["cannon.q[:,3]"] = -50  # vy bounds
upper["cannon.q[:,3]"] = 50

# Set the qdot values
lower["cannon.qdot"] = -float("inf")
upper["cannon.qdot"] = float("inf")

# Optimization with trajectory recording
opt = am.Optimizer(model, x, lower=lower, upper=upper)
try:
    # Record intermediate trajectories
    opt_options = {"max_iterations": 100, "record_components": ["cannon.q"]}
    data = opt.optimize(opt_options)
    print("Optimization completed successfully!")
except Exception as e:
    print(f"Optimization failed with error: {e}")
    if np.any(np.isnan(x[:])):
        print("NaN values detected in solution vector!")
    raise

# Save optimization data
with open("cannon_opt_data.json", "w") as fp:
    json.dump(data, fp, indent=2)

# Extract trajectory history from optimization data
trajectory_history = []
for i, iter_data in enumerate(data["iterations"]):
    if i % 5 == 0 and "x" in iter_data and "cannon.q" in iter_data["x"]:
        q_data = np.array(iter_data["x"]["cannon.q"])
        trajectory_history.append(
            {
                "iteration": iter_data["iteration"],
                "x": q_data[:, 0],  # x positions
                "y": q_data[:, 1],  # y positions
            }
        )

# Visualize results
print("Creating trajectory plots...")
plot_trajectory(x, title="Optimized Cannon Trajectory")

# Create optimization animation
if len(trajectory_history) > 1:
    print("Creating optimization animation:")
    animate_optimization_progress(trajectory_history)
else:
    print("Not enough trajectory data collected for animation")

print(f"Initial velocity: vx₀={x['cannon.q[0,2]']:.3f}, vy₀={x['cannon.q[0,3]']:.3f}")
print(f"Final position: x={x['cannon.q[-1,0]']:.3f}, y={x['cannon.q[-1,1]']:.3f}")
print(f"Target distance error: {abs(x['cannon.q[-1,0]'] - 6.0):.6f}")
