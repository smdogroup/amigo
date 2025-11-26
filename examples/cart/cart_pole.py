import argparse
import json
from pathlib import Path
import amigo as am
import numpy as np
import matplotlib.pylab as plt
import niceplots

try:
    from mpi4py import MPI

    COMM_WORLD = MPI.COMM_WORLD
except:
    COMM_WORLD = None


final_time = 2.0
num_time_steps = 100


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

        self.constraints["res"] = q2 - q1 - 0.5 * dt * (q1dot + q2dot)

        return


class CartComponent(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("g", value=9.81)
        self.add_data("L", value=0.5)
        self.add_data("m1", value=1.0)
        self.add_data("m2", value=0.3)

        self.add_input("x", label="control")
        self.add_input("q", shape=(4), label="state")
        self.add_input("qdot", shape=(4), label="rate")

        self.add_constraint("res", shape=(4), label="residual")

        return

    def compute(self):
        g = self.constants["g"]
        L0 = self.data["L"]
        m1 = self.data["m1"]
        m2 = self.data["m2"]

        x = self.inputs["x"]
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        # Compute the declared variable values
        sint = self.vars["sint"] = am.sin(q[1])
        cost = self.vars["cost"] = am.cos(q[1])

        res = 4 * [None]
        res[0] = q[2] - qdot[0]
        res[1] = q[3] - qdot[1]
        res[2] = (m1 + m2 * (1.0 - cost * cost)) * qdot[2] - (
            L0 * m2 * sint * q[3] * q[3] * x + m2 * g * cost * sint
        )
        res[3] = L0 * (m1 + m2 * (1.0 - cost * cost)) * qdot[3] + (
            L0 * m2 * cost * sint * q[3] * q[3] + x * cost + (m1 + m2) * g * sint
        )

        self.constraints["res"] = res

        return


class KineticEnergyOutput(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("dt", value=final_time / num_time_steps)
        self.add_data("m1")

        self.add_input("u1dot")
        self.add_input("u2dot")

        self.add_output("ke")

        return

    def compute_output(self):
        dt = self.constants["dt"]
        m1 = self.data["m1"]

        u1dot = self.inputs["u1dot"]
        u2dot = self.inputs["u2dot"]

        self.outputs["ke"] = m1 * dt * (u1dot * u1dot + u2dot * u2dot)


class Objective(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x1", label="control")
        self.add_input("x2", label="control")

        self.add_objective("obj")

        return

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]

        self.objective["obj"] = (x1 * x1 + x2 * x2) / 2

        return


# Initial conditions are q = 0
class InitialConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("q", shape=4)
        self.add_constraint("res", shape=4)

    def compute(self):
        q = self.inputs["q"]
        self.constraints["res"] = [q[0], q[1], q[2], q[3]]


# Set the final conditions
class FinalConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("pi", value=np.pi)

        self.add_input("q", shape=4)
        self.add_constraint("res", shape=4)

    def compute(self):
        pi = self.constants["pi"]
        q = self.inputs["q"]
        self.constraints["res"] = [q[0] - 2.0, q[1] - pi, q[2], q[3]]


def plot(d, theta, xctrl):
    t = np.linspace(0, final_time, num_time_steps + 1)

    with plt.style.context(niceplots.get_style()):
        data = {}
        data["Cart pos."] = d
        data["Pole angle"] = (180 / np.pi) * theta
        data["Control force"] = xctrl

        fig, ax = niceplots.stacked_plots(
            "Time (s)",
            t,
            [data],
            lines_only=True,
            figsize=(10, 6),
            line_scaler=0.5,
        )

        fontname = "Helvetica"
        for axis in ax:
            axis.xaxis.label.set_fontname(fontname)
            axis.yaxis.label.set_fontname(fontname)

            # Update tick labels
            for tick in axis.get_xticklabels():
                tick.set_fontname(fontname)

            for tick in axis.get_yticklabels():
                tick.set_fontname(fontname)

        fig.savefig("cart_stacked.svg")
        fig.savefig("cart_stacked.png")

    return


def plot_for_documentation(x, final_time=2.0, num_time_steps=100):
    """
    Create documentation-style plots matching control-toolbox format
    2x2 grid for states, separate panel for control with clean styling
    """
    # Extract solution
    time = np.linspace(0, final_time, num_time_steps + 1)
    q1 = x["cart.q[:, 0]"]  # Cart position
    q2 = x["cart.q[:, 1]"]  # Pole angle
    q1dot = x["cart.q[:, 2]"]  # Cart velocity
    q2dot = x["cart.q[:, 3]"]  # Angular velocity
    u = x["cart.x[:]"]  # Control force

    # Professional blue color
    blue_color = "#4063D8"

    # Create figure with more vertical spacing
    fig = plt.figure(figsize=(14, 11))

    # Add centered main title for states
    fig.text(0.5, 0.95, "State", ha="center", fontsize=14, fontweight="bold")

    # State variables in 2x2 grid (top 2/3 of figure)
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(time, q1, color=blue_color, linewidth=2.5)
    ax1.set_ylabel(r"$q_1$ [m]", fontsize=12)
    ax1.grid(True, alpha=0.25, linewidth=0.5)
    ax1.tick_params(labelsize=10)

    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(time, q2, color=blue_color, linewidth=2.5)
    ax2.set_ylabel(r"$q_2$ [rad]", fontsize=12)
    ax2.axhline(y=np.pi, color="#888888", linestyle="--", alpha=0.6, linewidth=1.5)
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    ax2.tick_params(labelsize=10)

    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(time, q1dot, color=blue_color, linewidth=2.5)
    ax3.set_ylabel(r"$\dot{q}_1$ [m/s]", fontsize=12)
    ax3.set_xlabel("Time [s]", fontsize=11)
    ax3.grid(True, alpha=0.25, linewidth=0.5)
    ax3.tick_params(labelsize=10)

    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(time, q2dot, color=blue_color, linewidth=2.5)
    ax4.set_ylabel(r"$\dot{q}_2$ [rad/s]", fontsize=12)
    ax4.set_xlabel("Time [s]", fontsize=11)
    ax4.grid(True, alpha=0.25, linewidth=0.5)
    ax4.tick_params(labelsize=10)

    # Add centered title for control
    fig.text(0.5, 0.31, "Control", ha="center", fontsize=14, fontweight="bold")

    # Control plot in bottom section
    ax5 = plt.subplot(3, 1, 3)
    ax5.plot(time, u, color=blue_color, linewidth=2.5)
    ax5.set_xlabel("Time [s]", fontsize=11)
    ax5.set_ylabel(r"$x$ [N]", fontsize=12)
    ax5.grid(True, alpha=0.25, linewidth=0.5)
    ax5.tick_params(labelsize=10)

    # Set font for all axes
    fontname = "Helvetica"
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.xaxis.label.set_fontname(fontname)
        ax.yaxis.label.set_fontname(fontname)
        for tick in ax.get_xticklabels():
            tick.set_fontname(fontname)
        for tick in ax.get_yticklabels():
            tick.set_fontname(fontname)

    plt.subplots_adjust(top=0.94, bottom=0.06, hspace=0.35, wspace=0.25)
    fig.savefig(
        "cart_pole_solution.png", dpi=300, bbox_inches="tight", facecolor="white"
    )
    fig.savefig("cart_pole_solution.svg", bbox_inches="tight", facecolor="white")
    print("Saved cart_pole_solution.png/svg for documentation")

    return


def plot_convergence(nrms):
    with plt.style.context(niceplots.get_style()):
        fig, ax = plt.subplots(1, 1)

        ax.semilogy(nrms, marker="o", clip_on=False, lw=2.0)
        ax.set_ylabel("KKT residual norm")
        ax.set_xlabel("Iteration")

        niceplots.adjust_spines(ax)

        fontname = "Helvetica"
        ax.xaxis.label.set_fontname(fontname)
        ax.yaxis.label.set_fontname(fontname)

        # Update tick labels
        for tick in ax.get_xticklabels():
            tick.set_fontname(fontname)

        for tick in ax.get_yticklabels():
            tick.set_fontname(fontname)

        fig.savefig("cart_residual_norm.svg")
        fig.savefig("cart_residual_norm.png")


def visualize(d, theta, L=0.5):
    with plt.style.context(niceplots.get_style()):
        # Create the time-lapse visualization
        fig, ax = plt.subplots(1, figsize=(10, 4.5))
        ax.axis("equal")
        ax.axis("off")

        values = np.linspace(0, 1.0, d.shape[0])
        cmap = plt.get_cmap("viridis")

        hx = 0.03
        hy = 0.03
        xpts = []
        ypts = []
        for i in range(0, d.shape[0]):
            color = cmap(values[i])

            x1 = d[i]
            y1 = 0.0
            x2 = d[i] + L * np.sin(theta[i])
            y2 = -L * np.cos(theta[i])

            xpts.append(x2)
            ypts.append(y2)

            if i % 3 == 0:
                ax.plot([x1, x2], [y1, y2], linewidth=2, color=color)
                ax.fill(
                    [x1 - hx, x1 + hx, x1 + hx, x1 - hx, x1 - hx],
                    [y1, y1, y1 + hy, y1 + hy, y1],
                    alpha=0.5,
                    linewidth=2,
                    color=color,
                )

            ax.plot([x2], [y2], color=color, marker="o")

        fig.savefig("cart_pole_history.svg")
        fig.savefig("cart_pole_history.png")

    return


def create_cart_model(module_name="cart_pole"):
    cart = CartComponent()
    trap = TrapezoidRule()
    obj = Objective()
    ic = InitialConditions()
    fc = FinalConditions()
    kin = KineticEnergyOutput()

    model = am.Model(module_name)

    model.add_component("cart", num_time_steps + 1, cart)
    model.add_component("trap", 4 * num_time_steps, trap)
    model.add_component("obj", num_time_steps, obj)
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)
    model.add_component("kin", num_time_steps, kin)

    # Add the links
    for i in range(4):
        start = i * num_time_steps
        end = (i + 1) * num_time_steps
        # Link the state variables
        model.link(f"cart.q[:{num_time_steps}, {i}]", f"trap.q1[{start}:{end}]")
        model.link(f"cart.q[1:, {i}]", f"trap.q2[{start}:{end}]")

        # Link the state rates
        model.link(f"cart.qdot[:-1, {i}]", f"trap.q1dot[{start}:{end}]")
        model.link(f"cart.qdot[1:, {i}]", f"trap.q2dot[{start}:{end}]")

    model.link(f"cart.x[:-1]", f"obj.x1[:]")
    model.link(f"cart.x[1:]", f"obj.x2[:]")

    model.link("cart.q[0, :]", "ic.q[0, :]")
    model.link(f"cart.q[{num_time_steps}, :]", "fc.q[0, :]")

    # Link the data
    model.link("cart.m1[1:]", "cart.m1[0]")
    model.link("cart.m2[1:]", "cart.m2[0]")
    model.link("cart.L[1:]", "cart.L[0]")

    # Link the kinetic energy computation
    model.link("kin.m1", "cart.m1[0]")
    model.link("kin.u1dot", "cart.qdot[:-1, 0]")
    model.link("kin.u2dot", "cart.qdot[1:, 0]")

    # Link the outputs
    model.link("kin.ke[1:]", "kin.ke[0]")

    return model


def check_post_optimality_derivative(x, opt, opt_options={}, dh=1e-7):
    # Set the initial conditions based on the varaibles
    x[:] = 0.0
    x["cart.q[:, 0]"] = np.linspace(0, 2.0, num_time_steps + 1)
    x["cart.q[:, 1]"] = np.linspace(0, np.pi, num_time_steps + 1)
    x["cart.q[:, 2]"] = 1.0
    x["cart.q[:, 3]"] = 1.0

    opt.optimize(opt_options)
    output = opt.compute_output()

    dfdx, of_map, wrt_map = opt.compute_post_opt_derivatives(
        of="kin.ke[0]", wrt=["cart.m1[0]", "cart.m2[0]", "cart.L[0]"], method="adjoint"
    )

    # Set the initial conditions based on the varaibles
    x[:] = 0.0
    x["cart.q[:, 0]"] = np.linspace(0, 2.0, num_time_steps + 1)
    x["cart.q[:, 1]"] = np.linspace(0, np.pi, num_time_steps + 1)
    x["cart.q[:, 2]"] = 1.0
    x["cart.q[:, 3]"] = 1.0

    # Compute the derivative wrt L
    data["cart.L[0]"] += dh

    opt.optimize(opt_options)
    output2 = opt.compute_output()

    ans = dfdx[0, wrt_map["cart.L[0]"]]
    fd = (output2["kin.ke[0]"] - output["kin.ke[0]"]) / dh
    print(ans, fd, (ans - fd) / fd)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
parser.add_argument(
    "--show-sparsity",
    dest="show_sparsity",
    action="store_true",
    default=False,
    help="Show the sparsity pattern",
)
parser.add_argument(
    "--show-graph",
    dest="show_graph",
    action="store_true",
    default=False,
    help="Show the graph",
)
parser.add_argument(
    "--graph-timestep",
    dest="graph_timestep",
    type=str,
    default=None,
    help="Show graph for timesteps. Can be a single int (e.g., 0) or a list (e.g., '[0,5,6]')",
)
parser.add_argument(
    "--distribute",
    dest="distribute",
    action="store_true",
    default=False,
    help="Distribute the problem",
)
parser.add_argument(
    "--post-optimality",
    dest="post_optimality",
    action="store_true",
    default=False,
    help="Compute the post-optimality",
)

args = parser.parse_args()

model = create_cart_model()

if args.build:
    source_dir = Path(__file__).resolve().parent
    model.build_module(source_dir=source_dir)

comm = COMM_WORLD
model.initialize(comm=comm)

comm_rank = 0
distribute = False
if comm is not None:
    comm_rank = comm.rank
    if comm.size > 1:
        distribute = True

if comm_rank == 0:
    with open("cart_pole_model.json", "w") as fp:
        json.dump(model.get_serializable_data(), fp, indent=2)

    print(f"Num variables:              {model.num_variables}")
    print(f"Num constraints:            {model.num_constraints}")

# Set the data
data = model.get_data_vector()

data["cart.L"] = 0.5
data["cart.m1"] = 1.0
data["cart.m2"] = 0.3

# Get the design variables
x = model.create_vector()
lower = model.create_vector()
upper = model.create_vector()

if comm_rank == 0:
    # Set the initial conditions based on the varaibles
    x["cart.q[:, 0]"] = np.linspace(0, 2.0, num_time_steps + 1)
    x["cart.q[:, 1]"] = np.linspace(0, np.pi, num_time_steps + 1)
    x["cart.q[:, 2]"] = 1.0
    x["cart.q[:, 3]"] = 1.0

    # Apply lower and upper bound constraints
    lower["cart.q"] = -float("inf")
    lower["cart.qdot"] = -float("inf")
    lower["cart.x"] = -50

    upper["cart.q"] = float("inf")
    upper["cart.qdot"] = float("inf")
    upper["cart.x"] = 50


opt_options = {
    "initial_barrier_param": 0.1,
    "convergence_tolerance": 1e-10,
    "max_line_search_iterations": 4,  # 30,  # Reasonable for intermediate problem
    "max_iterations": 500,  # Sufficient iterations
    # "init_affine_step_multipliers": True,  # Enable for better scaling
    # Use the new heuristic barrier parameter update
    # "barrier_strategy": "heuristic",
    # "verbose_barrier": True,  # Show ξ and complementarity values
}

# Set up the optimizer
opt = am.Optimizer(model, x, lower=lower, upper=upper, comm=comm, distribute=distribute)

opt_data = opt.optimize(opt_options)

with open("cart_opt_data.json", "w") as fp:
    json.dump(opt_data, fp, indent=2)

if args.post_optimality:
    check_post_optimality_derivative(x, opt, opt_options)


if comm_rank == 0:
    d = x["cart.q[:, 0]"]
    theta = x["cart.q[:, 1]"]
    xctrl = x["cart.x"]

    norms = []
    for iter_data in opt_data["iterations"]:
        norms.append(iter_data["residual"])

    plot(d, theta, xctrl)
    plot_convergence(norms)
    visualize(d, theta)

    # Generate documentation-style plots
    plot_for_documentation(x, final_time, num_time_steps)

    if args.show_sparsity:
        H = am.tocsr(opt.solver.hess)
        plt.figure(figsize=(6, 6))
        plt.spy(H, markersize=0.2)
        plt.title("Sparsity pattern of matrix A")
        plt.show()

    if args.show_graph:
        from pyvis.network import Network

        # Parse timestep argument - can be None, single int, or list
        t = args.graph_timestep
        if t is not None:
            if t.startswith("[") and t.endswith("]"):
                # Parse list format like "[0,5,6]"
                import ast

                try:
                    t = ast.literal_eval(t)
                except (ValueError, SyntaxError):
                    print(
                        f"Warning: Could not parse timestep list '{t}', using all timesteps"
                    )
                    t = None
            else:
                # Try to parse as single integer
                try:
                    t = int(t)
                except ValueError:
                    print(
                        f"Warning: Could not parse timestep '{t}', using all timesteps"
                    )
                    t = None

        graph = model.create_graph(timestep=t)
        net = Network(
            notebook=True,
            height="1000px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
        )

        net.from_nx(graph)
        # net.show_buttons(filter_=["physics"])
        net.set_options(
            """
            var options = {
            "interaction": {
                "dragNodes": false
            }
            }
            """
        )

        net.show("cart_pole_graph.html")
