import argparse
import json
import amigo as am
import numpy as np
import matplotlib.pylab as plt
import niceplots

num_cells = 200


class RoeFlux(am.Component):
    def __init__(self, gamma=1.4):
        super().__init__()

        # Add the constants
        self.add_constant("gamma", value=gamma)
        self.add_constant("gam1", value=(gamma - 1.0))
        self.add_constant("ggam1", value=gamma / (gamma - 1.0))

        self.add_input("A")
        self.add_input("QL", shape=3)
        self.add_input("QR", shape=3)
        self.add_input("F", shape=3)
        self.add_constraint("res", shape=3)

    def compute(self):
        gam1 = self.constants["gam1"]
        ggam1 = self.constants["ggam1"]

        A = self.inputs["A"]
        QL = self.inputs["QL"]
        QR = self.inputs["QR"]
        F = self.inputs["F"]

        # Compute the weights
        rhoL = QL[0]
        rhoR = QR[0]
        r = self.vars["r"] = am.sqrt(rhoR / rhoL)
        wL = self.vars["wL"] = r / (r + 1.0)
        wR = self.vars["wR"] = 1.0 - wL

        # Compute the left and right states
        uL = self.vars["uL"] = QL[1] / QL[0]
        pL = self.vars["pL"] = gam1 * (QL[2] - 0.5 * rhoL * uL * uL)
        HL = ggam1 * (pL / rhoL) + 0.5 * uL * uL

        uR = self.vars["uR"] = QR[1] / QR[0]
        pR = self.vars["pR"] = gam1 * (QR[2] - 0.5 * rhoR * uR * uR)
        HR = ggam1 * (pR / rhoR) + 0.5 * uR * uR

        # Compute the left and right fluxes
        FL = [rhoL * uL, rhoL * uL * uL + pL, rhoL * HL * uL]
        FR = [rhoR * uR, rhoR * uR * uR + pR, rhoR * HR * uR]

        # Compute the Roe averages
        rho = self.vars["rho"] = r * rhoL
        u = self.vars["u"] = wL * uL + wR * uR
        H = self.vars["H"] = wL * HL + wR * HR

        a = self.vars["a"] = am.sqrt(gam1 * (H - 0.5 * u * u))
        ainv = self.vars["ainv"] = 1.0 / a

        fp = self.vars["fp"] = ainv * ainv * (pR - pL)
        fu = self.vars["fu"] = (uR - uL) * rho * ainv

        # Compute the weights
        w0 = self.vars["w0"] = ((rhoR - rhoL) - fp) * am.abs(u)
        w1 = self.vars["w1"] = (0.5 * (fp + fu)) * am.abs(u + a)
        w2 = self.vars["w2"] = (0.5 * (fp - fu)) * am.abs(u - a)

        Fr = 0.5 * w0 * u * u + w1 * (H + u * a) + w2 * (H - u * a)
        self.constraints["res"] = [
            F[0] - 0.5 * A * ((FL[0] + FR[0]) - (w0 + w1 + w2)),
            F[1] - 0.5 * A * ((FL[1] + FR[1]) - (w0 * u + w1 * (u + a) + w2 * (u - a))),
            F[2] - 0.5 * A * ((FL[2] + FR[2]) - Fr),
        ]


class Nozzle(am.Component):
    def __init__(self, gamma=1.4, dx=1.0):
        super().__init__()

        self.add_constant("gam1", value=(gamma - 1.0))
        self.add_constant("dx", value=dx)

        self.add_input("Q", shape=3)
        self.add_input("FL", shape=3)
        self.add_input("FR", shape=3)
        self.add_input("dAdx")

        self.add_constraint("res", shape=3)

    def compute(self):
        gam1 = self.constants["gam1"]
        dx = self.constants["dx"]

        Q = self.inputs["Q"]
        FL = self.inputs["FL"]
        FR = self.inputs["FR"]
        dAdx = self.inputs["dAdx"]

        # Compute the pressure
        rho = Q[0]
        u = self.vars["u"] = Q[1] / Q[0]
        p = self.vars["p"] = gam1 * (Q[2] - 0.5 * rho * u * u)

        self.constraints["res"] = [
            (FR[0] - FL[0]) / dx,
            (FR[1] - FL[1]) / dx - dAdx * p,
            (FR[2] - FL[2]) / dx,
        ]


class InletFlux(am.Component):
    def __init__(self, gamma=1.4, T_res=1.0, p_res=1.0):
        super().__init__()

        self.add_constant("gamma", value=gamma)
        self.add_constant("gam1", value=(gamma - 1.0))
        self.add_constant("ggam1", value=gamma / (gamma - 1.0))

        rho_res = gamma * p_res / T_res
        S_res = rho_res**gamma / p_res

        self.add_constant("T_res", value=T_res)
        self.add_constant("p_res", value=p_res)
        self.add_constant("rho_res", value=rho_res)
        self.add_constant("S_res", value=S_res)

        self.add_input("A")
        self.add_input("Q", shape=3)
        self.add_input("F", shape=3)
        self.add_constraint("res", shape=3)

    def compute(self):
        gamma = self.constants["gamma"]
        gam1 = self.constants["gam1"]
        ggam1 = self.constants["ggam1"]

        T_res = self.constants["T_res"]
        S_res = self.constants["S_res"]

        A = self.inputs["A"]
        Q = self.inputs["Q"]
        F = self.inputs["F"]

        # Compute the velocity and speed of sound at the input
        rho_int = Q[0]
        u_int = self.vars["u_int"] = Q[1] / Q[0]
        p_int = self.vars["p_int"] = gam1 * (Q[2] - 0.5 * rho_int * u_int * u_int)
        a_int = self.vars["a_int"] = am.sqrt(gamma * p_int / rho_int)

        # Compute the two invariants
        a_res = self.vars["a_res"] = am.sqrt(T_res)
        invar_int = self.vars["invar_int"] = u_int - 2.0 * a_int / gam1
        invar_res = self.vars["invar_res"] = 2 * a_res / gam1

        # Based on the invariants, compute the velocity and speed of sound
        u_b = self.vars["u_b"] = 0.5 * (invar_int + invar_res)
        a_b = self.vars["a_b"] = 0.25 * gam1 * (invar_int - invar_res)
        rho_b = self.vars["rho_b"] = (a_b * a_b * S_res / gamma) ** (1.0 / gam1)

        # Compute the remaining states
        p_b = self.vars["p_b"] = rho_b * a_b**2 / gamma
        H_b = self.vars["H_b"] = ggam1 * p_b / rho_b + 0.5 * u_b * u_b

        # Compute the constraints
        self.constraints["res"] = [
            F[0] - A * rho_b * u_b,
            F[1] - A * (rho_b * u_b**2 + p_b),
            F[2] - A * rho_b * u_b * H_b,
        ]


class OutletFlux(am.Component):
    def __init__(self, gamma=1.4, p_back=1.0):
        super().__init__()

        self.add_constant("gamma", value=gamma)
        self.add_constant("gam1", value=(gamma - 1.0))
        self.add_constant("ggam1", value=gamma / (gamma - 1.0))

        self.add_constant("p_back", value=p_back)

        self.add_input("A")
        self.add_input("Q", shape=3)
        self.add_input("F", shape=3)
        self.add_constraint("res", shape=3)

    def compute(self):
        gamma = self.constants["gamma"]
        gam1 = self.constants["gam1"]
        ggam1 = self.constants["ggam1"]

        p_back = self.constants["p_back"]

        A = self.inputs["A"]
        Q = self.inputs["Q"]
        F = self.inputs["F"]

        # Compute the velocity and speed of sound at the input
        rho_int = Q[0]
        u_int = self.vars["u_int"] = Q[1] / Q[0]
        p_int = self.vars["p_int"] = gam1 * (Q[2] - 0.5 * rho_int * u_int * u_int)
        a_int = self.vars["a_int"] = am.sqrt(gamma * p_int / rho_int)

        # Compute the two invariants
        invar_int = self.vars["invar_int"] = u_int + 2.0 * a_int / gam1
        S_int = self.vars["S_int"] = rho_int**gamma / p_int

        # Set the back pressure
        p_b = p_back

        # Keep the entropy from the interior S_int = rho_b**gamma / p_b
        rho_b = self.vars["rho_b"] = (p_b * S_int) ** (1.0 / gamma)
        a_b = self.vars["a_b"] = am.sqrt(gamma * p_b / rho_b)
        u_b = self.vars["u_b"] = invar_int - 2 * a_b / gam1

        # Set the enthalpy
        H_b = self.vars["H_b"] = ggam1 * p_b / rho_b + 0.5 * u_b * u_b

        self.constraints["res"] = [
            F[0] - A * rho_b * u_b,
            F[1] - A * (rho_b * u_b**2 + p_b),
            F[2] - A * rho_b * u_b * H_b,
        ]


class Objective(am.Component):
    def __init__(self, gamma=1.4):
        super().__init__()

        self.add_constant("gam1", value=(gamma - 1.0))

        # Add the objective function
        self.add_objective("obj")

        # Add the pressure data
        self.add_data("p0")

        # Add the input conservative variables
        self.add_input("Q", shape=3)

    def compute(self):
        gam1 = self.constants["gam1"]
        Q = self.inputs["Q"]
        p0 = self.data["p0"]

        # Compute the pressure
        rho = Q[0]
        u = self.vars["u"] = Q[1] / Q[0]
        p = self.vars["p"] = gam1 * (Q[2] - 0.5 * rho * u * u)

        self.objective["obj"] = (p - p0) ** 2


class AreaControlPoints(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("area")


def plot_solution(rho, u, p, p_target, num_cells, length):

    dx = length / num_cells
    xloc = np.linspace(0.5 * dx, length - 0.5 * dx, num_cells)

    with plt.style.context(niceplots.get_style()):
        fig, ax = plt.subplots(3, 1, figsize=(10, 6))
        colors = niceplots.get_colors_list()

        labels = [r"$\rho$", "$u$", "$p$"]
        xlabel = "location"
        xticks = [0, 2, 4, 6, 8, 10]

        for i, label in enumerate(labels):
            ax[i].set_ylabel(label, rotation="horizontal", horizontalalignment="right")

        line_scaler = 1.0

        indices = [0, 1, 2, 2]
        cindices = [0, 0, 0, 1]
        data = [rho, u, p, p_target]
        label = [None, None, "solution", "target"]

        for i, (index, c, y) in enumerate(zip(indices, cindices, data)):
            ax[index].plot(
                xloc,
                y,
                clip_on=False,
                lw=3 * line_scaler,
                color=colors[c],
                label=label[i],
            )

        fontname = "Helvetica"
        ax[-1].legend(loc="lower right", prop={"family": fontname, "size": 12})

        for axis in ax:
            niceplots.adjust_spines(axis, outward=True)

        for axis in ax[:-1]:
            axis.get_xaxis().set_visible(False)
            axis.spines["bottom"].set_visible(False)
            axis.get_xaxis().set_ticks([])

        fig.align_labels()
        ax[-1].set_xlabel(xlabel)
        ax[-1].set_xticks(xticks)

        for axis in ax:
            axis.xaxis.label.set_fontname(fontname)
            axis.yaxis.label.set_fontname(fontname)

            # Update tick labels
            for tick in axis.get_xticklabels():
                tick.set_fontname(fontname)

            for tick in axis.get_yticklabels():
                tick.set_fontname(fontname)

        fig.savefig("nozzle_stacked.svg")
        fig.savefig("nozzle_stacked.png")

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

        fig.savefig("nozzle_residual_norm.svg")
        fig.savefig("nozzle_residual_norm.png")


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
    "--with-debug",
    dest="use_debug",
    action="store_true",
    default=False,
    help="Enable debug flags",
)
args = parser.parse_args()

# Set reference and reservoir temperature, pressure in physical units
gamma = 1.4
R = 287.0  # J/(kg K) Gas constant
T_reservoir = 300.0  # degK
p_reservoir = 100.0e3  # Pa
rho_reservoir = p_reservoir / (R * T_reservoir)  # kg / m^3

# Set the reference temperature
T_ref = T_reservoir
rho_ref = rho_reservoir

# Compute the remaining reference values
a_ref = np.sqrt(gamma * R * T_ref)
p_ref = rho_ref * a_ref**2

# Compute the non-dimensional output values
T_res = T_reservoir / T_ref
p_res = p_reservoir / p_ref

# Set the back-pressure
p_back = 0.9 * p_res

# Set values for the length
length = 10.0

# Set the constants needed for the inputs/outputs
dx = length / num_cells

# Set the pressure values
p_inlet = 0.95 * p_res
p_min = 0.5 * p_res
p_outlet = 0.9 * p_res

# Number of control points
nctrl = 5

# Create the model
model = am.Model("nozzle")

# Add the flux computations at the interior points
model.add_component("flux", num_cells - 1, RoeFlux(gamma=gamma))

# Add the flux computations at the end points
model.add_component("inlet", 1, InletFlux(gamma=gamma, T_res=T_res, p_res=p_res))
model.add_component("outlet", 1, OutletFlux(gamma=gamma, p_back=p_back))

# Add the 1D nozzle
model.add_component("nozzle", num_cells, Nozzle(gamma=gamma, dx=(length / num_cells)))

# Add the objective function
model.add_component("objective", num_cells, Objective())

# Add the area source
model.add_component("area_ctrl", nctrl, AreaControlPoints())

# Add the Bspline components that interpolate from the same area
xi_interface = np.linspace(0, 1.0, num_cells + 1)
area = am.BSplineInterpolant(xi=xi_interface, k=4, n=nctrl, deriv=0, length=length)
model.add_component("area", num_cells + 1, area)

xi_cell_center = np.linspace(0.5 / num_cells, 1.0 - 0.5 / num_cells, num_cells)
area_derivative = am.BSplineInterpolant(
    xi=xi_cell_center, k=4, n=nctrl, deriv=1, length=length
)
model.add_component("area_derivative", num_cells, area_derivative)

# Link the states
model.link(f"nozzle.Q[:-1, :]", "flux.QL")
model.link(f"nozzle.Q[1:, :]", "flux.QR")

# Link the fluxes
model.link(f"nozzle.FL[1:, :]", "flux.F")
model.link(f"nozzle.FR[:-1, :]", "flux.F")

# Link the area evaluations to the fluxes and boundary conditions
model.link("area.output[1:-1]", "flux.A")

# Link the Area and dAdx values to the nozzle equations
model.link("nozzle.dAdx", "area_derivative.output")

# Set the control points
area.add_links("area", model, "area_ctrl.area")
area_derivative.add_links("area_derivative", model, "area_ctrl.area")

# Link the input boundary states and fluxes
model.link("area.output[0]", "inlet.A")
model.link("nozzle.Q[0, :]", "inlet.Q[0, :]")
model.link("nozzle.FL[0, :]", "inlet.F[0, :]")

# Link the output boundary states and fluxes
model.link("area.output[-1]", "outlet.A")
model.link("nozzle.Q[-1, :]", "outlet.Q[0, :]")
model.link("nozzle.FR[-1, :]", "outlet.F[0, :]")

# Link the objective states
model.link("nozzle.Q", "objective.Q")

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

model.initialize(order_type=am.OrderingType.NATURAL)

# Get the data and set the target pressure distribution
data = model.get_data_vector()

# Quadratic interpolation with p_inlet, p_min and p_out
b = 4 * p_min - p_outlet - 3 * p_inlet
a = 2 * p_inlet - 4 * p_min + 2 * p_outlet
data["objective.p0"] = p_inlet + b * xi_cell_center + a * xi_cell_center**2

# Set the area derivative data
area.set_data("area", data)
area_derivative.set_data("area_derivative", data)

# Get the design variables
x = model.create_vector()
lower = model.create_vector()
upper = model.create_vector()

# Set the initial solution guess
rho = 1.0
u = 0.01
p = 1.0 / gamma
E = (p / rho) / (gamma - 1.0) + 0.5 * u**2
H = E + p / rho
x["nozzle.Q[:, 0]"] = rho
x["nozzle.Q[:, 1]"] = rho * u
x["nozzle.Q[:, 2]"] = rho * E

x["nozzle.FL[:, 0]"] = rho * u
x["nozzle.FL[:, 1]"] = rho * u**2 + p
x["nozzle.FL[:, 2]"] = rho * u * H

x["nozzle.FR[:, 0]"] = rho * u
x["nozzle.FR[:, 1]"] = rho * u**2 + p
x["nozzle.FR[:, 2]"] = rho * u * H

# Set the lower and upper bounds for Q
lower["nozzle.Q"] = float("-inf")
upper["nozzle.Q"] = float("inf")

# Set a lower variable bound on the density
lower["nozzle.Q[:, 0]"] = 1e-3

# Set the initial values
x["nozzle.dAdx"] = 0.0
x["area.output"] = 1.0

# Set the bounds on the the areas
x["area_ctrl.area"] = 1.0
lower["area_ctrl.area"] = 0.1
upper["area_ctrl.area"] = 2.0

# Set the remaining variable lower and upper bounds
lower["area.output"] = -float("inf")
upper["area.output"] = float("inf")
lower["area_derivative.output"] = float("-inf")
upper["area_derivative.output"] = float("inf")
lower["flux.F"] = float("-inf")
upper["flux.F"] = float("inf")
lower["inlet.F"] = float("-inf")
upper["inlet.F"] = float("inf")
lower["outlet.F"] = float("-inf")
upper["outlet.F"] = float("inf")

# Set up the optimizer
opt = am.Optimizer(model, x, lower=lower, upper=upper)

opt_history = opt.optimize(
    {
        "max_iterations": 1500,
        "record_components": ["area_ctrl.area"],
        "max_line_search_iterations": 10,
        "convergence_tolerance": 1e-12,
    }
)

with open("nozzle_history.json", "w") as fp:
    json.dump(opt_history, fp, indent=2)

inputs, cons, _, _ = model.get_names()

res = model.create_vector()
model.problem.gradient(x.get_vector(), res.get_vector())
inputs.extend(cons)

print("Variable summary")
for name in inputs:
    print(f"{name:<30} {np.linalg.norm(x[name])}")

print("Residual summary")
for name in inputs:
    print(f"{name:<30} {np.linalg.norm(res[name])}")

rho = x["nozzle.Q[:, 0]"]
u = x["nozzle.Q[:, 1]"] / rho
p = (gamma - 1.0) * (x["nozzle.Q[:, 2]"] - 0.5 * rho * u**2)
p_target = data["objective.p0"]
plot_solution(rho, u, p, p_target, num_cells, length)

norms = []
for iter_data in opt_history["iterations"]:
    norms.append(iter_data["residual"])

plot_convergence(norms)

plt.show()
