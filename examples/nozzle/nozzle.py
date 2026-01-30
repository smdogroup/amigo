import argparse
import json
from pathlib import Path
import amigo as am
import numpy as np
import nozzle_plots


class RoeFlux(am.Component):
    def __init__(self, gamma=1.4):
        super().__init__()

        # Add the constants
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

        r = am.sqrt(rhoR / rhoL)
        wL = r / (r + 1.0)
        wR = 1.0 - wL

        # Compute the left and right states
        uL = QL[1] / QL[0]
        pL = gam1 * (QL[2] - 0.5 * rhoL * uL * uL)
        HL = ggam1 * (pL / rhoL) + 0.5 * uL * uL

        uR = QR[1] / QR[0]
        pR = gam1 * (QR[2] - 0.5 * rhoR * uR * uR)
        HR = ggam1 * (pR / rhoR) + 0.5 * uR * uR

        # Compute the left and right fluxes
        FL = [rhoL * uL, rhoL * uL * uL + pL, rhoL * HL * uL]
        FR = [rhoR * uR, rhoR * uR * uR + pR, rhoR * HR * uR]

        # Compute the Roe averages
        rho = r * rhoL
        u = wL * uL + wR * uR
        H = wL * HL + wR * HR

        a = am.sqrt(gam1 * (H - 0.5 * u * u))
        ainv = 1.0 / a

        fp = ainv * ainv * (pR - pL)
        fu = (uR - uL) * rho * ainv

        # Entropy fix with a square root function
        h = 0.05 * (am.abs(u) + a)
        lam1 = am.sqrt(u * u + h * h / 4)
        lam2 = am.sqrt((u + a) * (u + a) + h * h / 4)
        lam3 = am.sqrt((u - a) * (u - a) + h * h / 4)

        w0 = ((rhoR - rhoL) - fp) * lam1
        w1 = (0.5 * (fp + fu)) * lam2
        w2 = (0.5 * (fp - fu)) * lam3

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
        u = Q[1] / Q[0]
        p = gam1 * (Q[2] - 0.5 * rho * u * u)

        self.constraints["res"] = [
            (FR[0] - FL[0]) / dx,
            (FR[1] - FL[1]) / dx - dAdx * p,
            (FR[2] - FL[2]) / dx,
        ]


class ExactNozzleMachCalc(am.Component):
    def __init__(self, gamma=1.4, Astar=1.0):
        super().__init__()

        self.add_constant("gamma", value=gamma)
        self.add_constant("gam1", value=(gamma - 1.0))
        self.add_constant("Astar", value=Astar)

        self.add_input("A")
        self.add_input("M")
        self.add_constraint("res")

    def compute(self):
        gamma = self.constants["gamma"]
        gam1 = self.constants["gam1"]
        Astar = self.constants["Astar"]
        M = self.inputs["M"]
        A = self.inputs["A"]

        fact = 1.0 + 0.5 * gam1 * M**2
        ratio = ((2.0 / (gamma + 1.0)) * fact) ** ((gamma + 1.0) / (2 * gam1))

        self.constraints["res"] = ratio - (A / Astar) * M


class SubsonicInletFlux(am.Component):
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

        self.add_input("A_inlet")
        self.add_input("M_inlet")
        self.add_input("Q", shape=3)
        self.add_input("F", shape=3)
        self.add_constraint("res", shape=3)

    def compute(self):
        gamma = self.constants["gamma"]
        gam1 = self.constants["gam1"]
        ggam1 = self.constants["ggam1"]
        T_res = self.constants["T_res"]
        S_res = self.constants["S_res"]

        A_inlet = self.inputs["A_inlet"]
        M_inlet = self.inputs["M_inlet"]
        Q = self.inputs["Q"]
        F = self.inputs["F"]

        # Compute the velocity and speed of sound at the input
        rho_int = Q[0]
        u_int = Q[1] / Q[0]
        p_int = gam1 * (Q[2] - 0.5 * rho_int * u_int * u_int)
        a_int = am.sqrt(gamma * p_int / rho_int)

        # Compute the isentropic factor
        fact = 1.0 + 0.5 * gam1 * M_inlet * M_inlet

        # Compute the inlet speed of sound
        a_inlet = am.sqrt(T_res / fact)

        # Compute the inlet velocity
        u_inlet = M_inlet * a_inlet

        # Compute the two invariants
        invar_int = u_int - 2.0 * a_int / gam1
        invar_inlet = u_inlet + 2.0 * a_inlet / gam1

        # Based on the invariants, compute the velocity and speed of sound
        u_b = 0.5 * (invar_int + invar_inlet)
        a_b = 0.25 * gam1 * (invar_inlet - invar_int)
        rho_b = (a_b * a_b * S_res / gamma) ** (1.0 / gam1)

        # Compute the remaining states
        p_b = rho_b * a_b * a_b / gamma
        H_b = ggam1 * p_b / rho_b + 0.5 * u_b * u_b

        # Compute the constraints
        self.constraints["res"] = [
            F[0] - A_inlet * rho_b * u_b,
            F[1] - A_inlet * (rho_b * u_b**2 + p_b),
            F[2] - A_inlet * rho_b * u_b * H_b,
        ]


class SubsonicOutletFlux(am.Component):
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

        self.add_input("A_outlet")
        self.add_input("M_outlet")
        self.add_input("Q", shape=3)
        self.add_input("F", shape=3)
        self.add_constraint("res", shape=3)

    def compute(self):
        gamma = self.constants["gamma"]
        gam1 = self.constants["gam1"]
        ggam1 = self.constants["ggam1"]

        M_outlet = self.inputs["M_outlet"]
        A_outlet = self.inputs["A_outlet"]
        Q = self.inputs["Q"]
        F = self.inputs["F"]

        # Compute the velocity and speed of sound at the input
        rho_int = Q[0]
        u_int = Q[1] / Q[0]
        p_int = gam1 * (Q[2] - 0.5 * rho_int * u_int * u_int)
        a_int = am.sqrt(gamma * p_int / rho_int)
        S_int = rho_int**gamma / p_int

        # Compute the isentropic factor
        fact = 1.0 + 0.5 * gam1 * M_outlet * M_outlet

        # Compute the outlet speed of sound and velocity
        a_outlet = am.sqrt(T_res / fact)
        u_outlet = M_outlet * a_outlet

        # Compute the two invariants
        invar_int = u_int + 2.0 * a_int / gam1
        S_int = rho_int**gamma / p_int

        # Compute the invariants
        invar_outlet = u_outlet - 2.0 * a_outlet / gam1

        # Based on the invariants, compute the velocity and speed of sound
        u_b = 0.5 * (invar_int + invar_outlet)
        a_b = 0.25 * gam1 * (invar_outlet - invar_int)
        rho_b = (a_b * a_b * S_int / gamma) ** (1.0 / gam1)

        # Compute the remaining states
        p_b = rho_b * a_b * a_b / gamma
        H_b = ggam1 * p_b / rho_b + 0.5 * u_b * u_b

        self.constraints["res"] = [
            F[0] - A_outlet * rho_b * u_b,
            F[1] - A_outlet * (rho_b * u_b**2 + p_b),
            F[2] - A_outlet * rho_b * u_b * H_b,
        ]


class Objective(am.Component):
    def __init__(self, gamma=1.4, dx=1.0):
        super().__init__()

        self.add_constant("gam1", value=(gamma - 1.0))
        self.add_constant("dx", value=dx)

        # Add the objective function
        self.add_objective("obj")

        # Add the pressure data
        self.add_data("p0")

        # Add the input conservative variables
        self.add_input("Q", shape=3)

    def compute(self):
        gam1 = self.constants["gam1"]
        dx = self.constants["dx"]
        Q = self.inputs["Q"]
        p0 = self.data["p0"]

        # Compute the pressure
        rho = Q[0]
        u = Q[1] / Q[0]
        p = gam1 * (Q[2] - 0.5 * rho * u * u)

        self.objective["obj"] = dx * (p - p0) ** 2


class AreaControlPoints(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("area")


class PseudoTransient(am.Component):
    def __init__(self, gamma=1.4, dx=1.0):
        super().__init__()

        self.add_constant("gamma", value=gamma)
        self.add_constant("gam1", value=(gamma - 1.0))
        self.add_constant("dx", value=dx)

        # Add the inputs and constraints
        self.add_input("Q", shape=3)
        self.add_constraint("res", shape=3)

        # Set the CFL as input data
        self.add_data("CFL")

        # This is a continuation component. These component contributions are only
        # added for the Hessian contribution. This is used for continuation methods.
        self.set_continuation_component(True)

        return

    def compute(self):
        gamma = self.constants["gamma"]
        gam1 = self.constants["gam1"]
        dx = self.constants["dx"]

        Q = self.inputs["Q"]
        CFL = self.data["CFL"]

        # Extract passive values - this won't be differentiated
        Qp = am.passive(Q)
        rho = Qp[0]
        q1 = Qp[1]
        q2 = Qp[2]

        # Compute the time step
        u = q1 / rho
        p = gam1 * (q2 - 0.5 * rho * u * u)
        a = am.sqrt(gamma * p / rho)
        V = a + am.fabs(u)
        dt = dx * CFL / V

        self.constraints["res"] = [Q[0] / dt, Q[1] / dt, Q[2] / dt]

        return


class DesignContinuation(am.Component):
    def __init__(self):
        super().__init__()

        self.add_data("diagonal_weight")
        self.add_input("area_ctrl")
        self.add_objective("obj")

        self.set_continuation_component(True)
        return

    def compute(self):
        diagonal_weight = self.data["diagonal_weight"]
        area_ctrl = self.inputs["area_ctrl"]

        self.objective["obj"] = 0.5 * diagonal_weight * area_ctrl**2
        return


def compute_area_target(eta, A_inlet=2.0, A_outlet=1.5, A_min=0.8, eta_sigma=0.3):
    """
    Compute the target area distribution
    """

    # Compute the magnitude of the exponential term
    A_amplitude = A_min - 0.5 * (A_inlet + A_outlet)

    # Evaluate the target area
    A_target = A_inlet * (1.0 - eta) + A_outlet * eta
    A_target += A_amplitude * np.exp(-(((eta - 0.5) / eta_sigma) ** 2))

    return A_target


def compute_pressure_target(
    A_target, p_res, gamma, Astar, newton_iterations=20, tol=1e-12
):
    """
    Given a target area distribution, compute the exact target pressure distribution
    """

    M_target = np.zeros(len(A_target))
    p_target = np.zeros(len(A_target))
    for i in range(len(A_target)):

        # Solve for the Mach number
        M = 0.01
        for j in range(newton_iterations):
            fact = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * M**2)
            ratio = fact ** ((gamma + 1.0) / (2 * (gamma - 1.0)))

            # Solve the equation
            R = ratio / M - A_target[i] / Astar

            # Check the convergence tolerance
            if np.fabs(R) < tol:
                break

            dRdM = fact ** ((3.0 - gamma) / (2.0 * (gamma - 1.0)))
            dRdM -= ratio / M**2

            M -= R / dRdM

        # Compute the pressure
        M_target[i] = M
        fact = 1.0 + 0.5 * (gamma - 1.0) * M**2
        p_target[i] = p_res * (fact ** (-gamma / (gamma - 1)))

    return M_target, p_target


def get_initial_point_and_bounds(model):
    # Get the design variables
    x = model.create_vector()
    lower = model.create_vector()
    upper = model.create_vector()

    # Set the initial solution guess
    rho = 0.5
    u = 0.2
    p = 0.5 / gamma

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

    # Set the bounds for the inlet and outlet Mach number
    x["inlet.M_inlet"] = 0.01
    lower["inlet.M_inlet"] = 0.0
    upper["inlet.M_inlet"] = 1.0

    x["outlet.M_outlet"] = 0.01
    lower["outlet.M_outlet"] = 0.0
    upper["outlet.M_outlet"] = 1.0

    # Set the lower and upper bounds for Q
    lower["nozzle.Q"] = float("-inf")
    upper["nozzle.Q"] = float("inf")

    # Set a lower variable bound on the density
    lower["nozzle.Q[:, 0]"] = 1e-3

    # Set the initial values
    x["nozzle.dAdx"] = 0.0
    x["area.output"] = 1.0

    # Set the bounds on the the areas
    x["area_ctrl.area"] = 1.5
    lower["area_ctrl.area"] = -3.0
    upper["area_ctrl.area"] = 3.0

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

    return x, lower, upper


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
    "--with-lnks",
    dest="use_lnks",
    action="store_true",
    default=False,
    help="Enable the Largrange-Newton-Krylov-Schur inexact solver",
)
parser.add_argument(
    "--with-cuda",
    dest="use_cuda",
    action="store_true",
    default=False,
    help="Enable the CUDA solver",
)
parser.add_argument(
    "--num-cells", dest="num_cells", default=400, type=int, help="Number of cells"
)
parser.add_argument(
    "--opt-filename",
    dest="opt_filename",
    type=str,
    default="nozzle_history.json",
    help="Number of time steps",
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

# Set the reference area
Astar = 0.7

# Set values for the length
length = 10.0

# Set the constants needed for the inputs/outputs
dx = length / args.num_cells

# Number of control points
nctrl = 10

# Create the model
model = am.Model("nozzle_module")

# Add the flux computations at the interior points
model.add_component("flux", args.num_cells - 1, RoeFlux(gamma=gamma))

# Add the flux computations at the end points
inlet = SubsonicInletFlux(gamma=gamma, T_res=T_res, p_res=p_res)
model.add_component("inlet", 1, inlet)

outlet = SubsonicOutletFlux(gamma=gamma, T_res=T_res, p_res=p_res)
model.add_component("outlet", 1, outlet)

# Add the 1D nozzle
model.add_component(
    "nozzle", args.num_cells, Nozzle(gamma=gamma, dx=(length / args.num_cells))
)

# Add the nozzle boundary condition calculation
model.add_component("calc", 2, ExactNozzleMachCalc(gamma=gamma, Astar=Astar))

# Add the objective function
model.add_component(
    "objective", args.num_cells, Objective(gamma=gamma, dx=(length / args.num_cells))
)

# Add the area source
model.add_component("area_ctrl", nctrl, AreaControlPoints())

# Add the Bspline area calculation component
xi_interface = np.linspace(0, length, args.num_cells + 1)
area = am.BSplineInterpolant(xi=xi_interface, k=4, n=nctrl, deriv=0, length=length)
model.add_component("area", args.num_cells + 1, area)

# Add the Bspline derivative evaluation component
first_cell = 0.5 * dx
last_cell = length - 0.5 * dx
xi_cell_center = np.linspace(first_cell, last_cell, args.num_cells)
area_derivative = am.BSplineInterpolant(
    xi=xi_cell_center, k=4, n=nctrl, deriv=1, length=length
)
model.add_component("area_derivative", args.num_cells, area_derivative)

# Add the continuation components
pseudo_transient = PseudoTransient(gamma=gamma, dx=(length / args.num_cells))
model.add_component("pseudo_transient", args.num_cells, pseudo_transient)

design_continuation = DesignContinuation()
model.add_component("design_continuation", nctrl, design_continuation)

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
model.link("area.output[0]", "inlet.A_inlet")
model.link("nozzle.Q[0, :]", "inlet.Q[0, :]")
model.link("nozzle.FL[0, :]", "inlet.F[0, :]")

# Link the output boundary states and fluxes
model.link("area.output[-1]", "outlet.A_outlet")
model.link("nozzle.Q[-1, :]", "outlet.Q[0, :]")
model.link("nozzle.FR[-1, :]", "outlet.F[0, :]")

# Link the objective states
model.link("nozzle.Q", "objective.Q")

# Connect the nozzle boundary condition calculations
model.link("area.output[0]", "calc.A[0]")
model.link("inlet.M_inlet", "calc.M[0]")

model.link("area.output[-1]", "calc.A[1]")
model.link("outlet.M_outlet", "calc.M[1]")

# Link the pseudo-transient continuation values
model.link("pseudo_transient.Q", "nozzle.Q")
model.link("pseudo_transient.res", "nozzle.res")
model.link("pseudo_transient.CFL[1:]", "pseudo_transient.CFL[0]")

# Link the design continuation parameters
model.link("design_continuation.area_ctrl", "area_ctrl.area")
model.link(
    "design_continuation.diagonal_weight[1:]", "design_continuation.diagonal_weight[0]"
)

if args.build:
    source_dir = Path(__file__).resolve().parent
    model.build_module(source_dir=source_dir)

model.initialize(order_type=am.OrderingType.NATURAL)

# Get the data and set the target pressure distribution
data = model.get_data_vector()

# Compute the non-dimensional locations to evaluate the area
eta = xi_cell_center / length

# Compute the target area distribution
A_target = compute_area_target(eta)

solver = None
if args.use_lnks:
    problem = model.get_problem()

    state_vars = [
        "nozzle.Q",
        "flux.F",
        "inlet.F",
        "outlet.F",
        "calc.M",
    ]
    residuals = ["nozzle.res", "flux.res", "inlet.res", "outlet.res", "calc.res"]

    solver = am.LNKSInexactSolver(
        problem,
        model=model,
        state_vars=state_vars,
        residuals=residuals,
    )
elif args.use_cuda:
    solver = am.DirectCudaSolver(model.get_problem(), pivot_eps=1e-8)


def continuation_control(iteration, res_norm):
    data["pseudo_transient.CFL[0]"] = np.min((10000.0, 10.0 + 10.0 * iteration))
    data["design_continuation.diagonal_weight[0]"] = np.max((0.0, 100.0 - iteration))
    data.get_vector().copy_host_to_device()
    return


# Print the size of the model
print(f"Num variables:              {model.num_variables}")
print(f"Num constraints:            {model.num_constraints}")

for opt_iter in range(2):
    # Evaluate the target pressure distribution
    M_target, p_target = compute_pressure_target(A_target, p_res, gamma, Astar)

    # Set the pressure target distribution
    data["objective.p0"] = p_target

    # Set the initial continuation parameters
    data["pseudo_transient.CFL[0]"] = 10.0
    data["design_continuation.diagonal_weight[0]"] = 100.0

    # Set the area derivative data
    area.set_data("area", data)
    area_derivative.set_data("area_derivative", data)
    data.get_vector().copy_host_to_device()

    # Set the initial point and bounds
    x, lower, upper = get_initial_point_and_bounds(model)

    # Set up the optimizer
    opt = am.Optimizer(model, x, lower=lower, upper=upper, solver=solver)

    opt_history = opt.optimize(
        {
            "max_iterations": 1000,
            "record_components": ["area_ctrl.area"],
            "max_line_search_iterations": 4,
            "convergence_tolerance": 1e-9,
            "monotone_barrier_fraction": 0.1,
            "continuation_control": continuation_control,
        }
    )

opt_history["num_cells"] = args.num_cells
opt_history["num_variables"] = model.num_variables
opt_history["num_constraints"] = model.num_constraints

with open(args.opt_filename, "w") as fp:
    json.dump(opt_history, fp, indent=2, default=lambda obj: "")

inputs, cons, _, _ = model.get_names()

res = model.create_vector()
model.problem.gradient(1.0, x.get_vector(), res.get_vector())
inputs.extend(cons)

# Copy the info to the host
x.get_vector().copy_device_to_host()
res.get_vector().copy_device_to_host()

print("Variable summary")
for name in inputs:
    print(f"{name:<30} {np.linalg.norm(x[name])}")

print("Residual summary")
for name in inputs:
    print(f"{name:<30} {np.linalg.norm(res[name])}")

# Plot the solution
rho = x["nozzle.Q[:, 0]"]
u = x["nozzle.Q[:, 1]"] / rho
p = (gamma - 1.0) * (x["nozzle.Q[:, 2]"] - 0.5 * rho * u**2)
nozzle_plots.plot_solution(rho, u, p, M_target, p_target, args.num_cells, length)

# Plot the nozzle problem solution
nozzle_plots.plot_nozzle(
    x["area.output"], x["area_derivative.output"], A_target, args.num_cells, length
)

norms = []
for iter_data in opt_history["iterations"]:
    norms.append(iter_data["residual"])

nozzle_plots.plot_convergence(norms)
