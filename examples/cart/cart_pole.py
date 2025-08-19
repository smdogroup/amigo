import amigo as am
import numpy as np
import argparse
import json
import matplotlib.pylab as plt
import niceplots

try:
    from mpi4py import MPI

    COMM_WORLD = MPI.COMM_WORLD
except:
    COMM_WORLD = None


final_time = 2.0
num_time_steps = 1000


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
        self.add_constant("L", value=0.5)
        self.add_constant("m1", value=1.0)
        self.add_constant("m2", value=0.3)

        self.add_input("x", label="control")
        self.add_input("q", shape=(4), label="state")
        self.add_input("qdot", shape=(4), label="rate")

        self.add_constraint("res", shape=(4), label="residual")

        return

    def compute(self):
        g = self.constants["g"]
        L = self.constants["L"]
        m1 = self.constants["m1"]
        m2 = self.constants["m2"]

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
            L * m2 * sint * q[3] * q[3] * x + m2 * g * cost * sint
        )
        res[3] = L * (m1 + m2 * (1.0 - cost * cost)) * qdot[3] + (
            L * m2 * cost * sint * q[3] * q[3] + x * cost + (m1 + m2) * g * sint
        )

        self.constraints["res"] = res

        return


class Objective(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("dt", value=final_time / num_time_steps)

        self.add_input("x1", label="control")
        self.add_input("x2", label="control")

        self.add_objective("obj")

        return

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        dt = self.constants["dt"]

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

    model = am.Model(module_name)

    model.add_component("cart", num_time_steps + 1, cart)
    model.add_component("trap", 4 * num_time_steps, trap)
    model.add_component("obj", num_time_steps, obj)
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)

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
    "--with-debug",
    dest="use_debug",
    action="store_true",
    default=False,
    help="Enable debug flags",
)
parser.add_argument(
    "--show-sparsity",
    dest="show_sparsity",
    action="store_true",
    default=False,
    help="Show the sparsity pattern",
)
parser.add_argument(
    "--distribute",
    dest="distribute",
    action="store_true",
    default=False,
    help="Distribute the problem",
)
args = parser.parse_args()

model = create_cart_model()

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

# Set up the optimizer
opt = am.Optimizer(model, x, lower=lower, upper=upper, comm=comm, distribute=distribute)

data = opt.optimize(
    {
        "max_iterations": 100,
        "record_components": ["cart.x[-1]"],
        "max_line_search_iterations": 10,
        # "check_update_step": True,
    }
)
with open("cart_opt_data.json", "w") as fp:
    json.dump(data, fp, indent=2)

if comm_rank == 0:
    d = x["cart.q[:, 0]"]
    theta = x["cart.q[:, 1]"]
    xctrl = x["cart.x"]

    norms = []
    for iter_data in data["iterations"]:
        norms.append(iter_data["residual"])

    plot(d, theta, xctrl)
    plot_convergence(norms)
    visualize(d, theta)

    if args.show_sparsity:
        H = am.tocsr(opt.solver.hess)
        plt.figure(figsize=(6, 6))
        plt.spy(H, markersize=0.2)
        plt.title("Sparsity pattern of matrix A")
        plt.show()
