import argparse
import json
from pathlib import Path
import amigo as am
import numpy as np
import cart_plots
import matplotlib.pylab as plt

try:
    from mpi4py import MPI

    COMM_WORLD = MPI.COMM_WORLD
except:
    COMM_WORLD = None


final_time = 2.0


class TrapezoidRule(am.Component):
    def __init__(self, final_time=2.0, num_time_steps=100):
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
        # Kinematic constraints
        res[0] = q[2] - qdot[0]
        res[1] = q[3] - qdot[1]

        res[2] = (m1 + m2 * sint * sint) * qdot[2] - (
            x + m2 * L0 * sint * q[3] * q[3] + m2 * g * cost * sint
        )

        # Pole acceleration:
        res[3] = L0 * (m1 + m2 * sint * sint) * qdot[3] - (
            -x * cost - m2 * L0 * cost * sint * q[3] * q[3] - (m1 + m2) * g * sint
        )

        self.constraints["res"] = res

        return


class KineticEnergyOutput(am.Component):
    def __init__(self, final_time=2.0, num_time_steps=100):
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

        self.objective["obj"] = (x1**2 + x2**2) / 2

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


def create_cart_model(module_name="cart_pole", final_time=2.0, num_time_steps=100):
    cart = CartComponent()
    trap = TrapezoidRule(final_time=final_time, num_time_steps=num_time_steps)
    obj = Objective()
    ic = InitialConditions()
    fc = FinalConditions()
    kin = KineticEnergyOutput(final_time=final_time, num_time_steps=num_time_steps)

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


def check_post_optimality_derivative(
    x, opt, opt_options={}, dh=1e-7, num_time_steps=100
):
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
    "--with-cuda",
    dest="with_cuda",
    action="store_true",
    default=False,
    help="Use the CUDSS factorization on the GPU",
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
parser.add_argument(
    "--num-time-steps",
    dest="num_time_steps",
    type=int,
    default=100,
    help="Number of time steps",
)
parser.add_argument(
    "--opt-filename",
    dest="opt_filename",
    type=str,
    default="cart_opt_data.json",
    help="Number of time steps",
)

args = parser.parse_args()

model = create_cart_model(num_time_steps=args.num_time_steps)

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

opt_options = {
    "initial_barrier_param": 0.1,
    "convergence_tolerance": 5e-7,
    "max_line_search_iterations": 4,  # 30,  # Reasonable for intermediate problem
    "max_iterations": 500,  # Sufficient iterations
    # "init_affine_step_multipliers": True,  # Enable for better scaling
    # Use the new heuristic barrier parameter update
    # "barrier_strategy": "heuristic",
    # "verbose_barrier": True,  # Show ξ and complementarity values
}

solver = None
if args.with_cuda:
    solver = am.DirectCudaSolver(model.get_problem())

for opt_iter in range(4):
    # Get the design variables
    x = model.create_vector()
    lower = model.create_vector()
    upper = model.create_vector()

    if comm_rank == 0:
        # Set the initial conditions based on the varaibles
        x["cart.q[:, 0]"] = np.linspace(0, 2.0, args.num_time_steps + 1)
        x["cart.q[:, 1]"] = np.linspace(0, np.pi, args.num_time_steps + 1)
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
    opt = am.Optimizer(
        model,
        x,
        lower=lower,
        upper=upper,
        comm=comm,
        distribute=distribute,
        solver=solver,
    )

    # Optimize
    opt_data = opt.optimize(opt_options)

    opt_data["num_time_steps"] = args.num_time_steps
    opt_data["num_variables"] = model.num_variables
    opt_data["num_constraints"] = model.num_constraints

# Copy the solution from the device to host
x.get_vector().copy_device_to_host()

# Print objective value
u = x["cart.x[:]"]  # control force at each timestep
obj_value = 0.0
dt = final_time / args.num_time_steps
for i in range(args.num_time_steps):
    obj_value += (u[i] ** 2 + u[i + 1] ** 2) / 2 * dt

print(obj_value)

with open(args.opt_filename, "w") as fp:
    json.dump(opt_data, fp, indent=2)

if args.post_optimality:
    check_post_optimality_derivative(
        x, opt, opt_options, num_time_steps=args.num_time_steps
    )


if comm_rank == 0:
    d = x["cart.q[:, 0]"]
    theta = x["cart.q[:, 1]"]
    xctrl = x["cart.x"]

    norms = []
    for iter_data in opt_data["iterations"]:
        norms.append(iter_data["residual"])

    cart_plots.plot_solution(d, theta, xctrl, num_time_steps=args.num_time_steps)
    cart_plots.plot_convergence(norms)
    cart_plots.visualize(d, theta)

    # Generate documentation-style plots
    cart_plots.plot_for_documentation(x, final_time, args.num_time_steps)

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
