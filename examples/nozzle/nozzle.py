import amigo as am
import argparse
import numpy as np

num_cells = 200


class RoeFlux(am.Component):
    def __init__(self, gamma=1.4):
        super().__init__()

        # Add the constants
        self.add_constant("gamma", value=gamma)
        self.add_constant("gam1", value=(gamma - 1.0))
        self.add_constant("ggam1", value=gamma / (gamma - 1.0))

        self.add_input("QL", shape=3)
        self.add_input("QR", shape=3)
        self.add_input("F", shape=3)
        self.add_constraint("res", shape=3)

    def compute(self):
        gam1 = self.constants["gam1"]
        ggam1 = self.constants["ggam1"]

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
            F[0] - 0.5 * ((FL[0] + FR[0]) - (w0 + w1 + w2)),
            F[1] - 0.5 * ((FL[1] + FR[1]) - (w0 * u + w1 * (u + a) + w2 * (u - a))),
            F[2] - 0.5 * ((FL[2] + FR[2]) - Fr),
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
            (FL[0] - FR[0]) / dx,
            (FL[1] - FR[1]) / dx - dAdx * p,
            (FL[2] - FR[2]) / dx,
        ]


class InletFlux(am.Component):
    def __init__(self, gamma=1.4, T_res=1.0, p_res=1.0):
        super().__init__()

        self.add_constant("gamma", value=gamma)
        self.add_constant("gam1", value=(gamma - 1.0))
        self.add_constant("ggam1", value=gamma / (gamma - 1.0))

        self.add_constant("T_res", value=T_res)
        self.add_constant("p_res", value=p_res)

        self.add_input("Q", shape=3)
        self.add_input("F", shape=3)
        self.add_constraint("res", shape=3)

    def compute(self):
        gamma = self.constants["gamma"]
        gam1 = self.constants["gam1"]
        ggam1 = self.constants["ggam1"]

        T_res = self.constants["T_res"]
        p_res = self.constants["p_res"]

        Q = self.inputs["Q"]
        F = self.inputs["F"]

        # Compute the velocity and speed of sound at the input
        rho_int = Q[0]
        u_int = self.vars["u_int"] = Q[1] / Q[0]
        p_int = self.vars["p_int"] = gam1 * (Q[2] - 0.5 * rho_int * u_int * u_int)
        a_int = self.vars["a_int"] = am.sqrt(gamma * p_int / rho_int)

        # Compute the two invariants
        a_res = self.vars["a_res"] = am.sqrt(T_res)
        invar_int = self.vars["invar_int"] = u_int + 2.0 * a_int / (gamma - 1.0)
        invar_res = self.vars["invar_res"] = 2 * a_res / (gamma - 1.0)

        # Based on the invariants, compute the velocity and speed of sound
        u_b = self.vars["u_b"] = 0.5 * (invar_int + invar_res)
        a_b = self.vars["a_b"] = 0.25 * (gamma - 1.0) * (invar_int - invar_res)

        # Compute the Mach number
        M_b = self.vars["M_b"] = u_b / a_b
        p_b = self.vars["p_b"] = p_res / (1.0 + 0.5 * gam1 * M_b**2) ** (ggam1)

        rho_b = self.vars["rho_b"] = gamma * p_b / a_b**2
        H_b = self.vars["H_b"] = ggam1 * p_b / rho_b + 0.5 * u_b * u_b

        # Compute the constraints
        self.constraints["res"] = [
            F[0] - rho_b * u_b,
            F[1] - (rho_b * u_b**2 + p_b),
            F[2] - rho_b * u_b * H_b,
        ]


class OutletFlux(am.Component):
    def __init__(self, gamma=1.4, p_back=1.0):
        super().__init__()

        self.add_constant("gamma", value=gamma)
        self.add_constant("gam1", value=(gamma - 1.0))
        self.add_constant("ggam1", value=gamma / (gamma - 1.0))

        self.add_constant("p_back", value=p_back)

        self.add_input("Q", shape=3)
        self.add_input("F", shape=3)
        self.add_constraint("res", shape=3)

    def compute(self):
        gamma = self.constants["gamma"]
        gam1 = self.constants["gam1"]
        ggam1 = self.constants["ggam1"]

        p_back = self.constants["p_back"]

        Q = self.inputs["Q"]
        F = self.inputs["F"]

        # Compute the velocity and speed of sound at the input
        rho_int = Q[0]
        u_int = self.vars["u_int"] = Q[1] / Q[0]
        p_int = self.vars["p_int"] = gam1 * (Q[2] - 0.5 * rho_int * u_int * u_int)
        a_int = self.vars["a_int"] = am.sqrt(gamma * p_int / rho_int)

        # Compute the two invariants
        invar_int = self.vars["invar_int"] = u_int + 2.0 * a_int / (gamma - 1.0)
        S_int = self.vars["S_int"] = rho_int**gamma / p_int

        # Set the back pressure
        p_b = p_back

        # Keep the entropy from the interior S_int = rho_b**gamma / p_b
        rho_b = self.vars["rho_b"] = (p_b * S_int) ** (1.0 / gamma)
        a_b = self.vars["a_b"] = am.sqrt(gamma * p_b / rho_b)
        u_b = self.vars["u_b"] = invar_int - 2 * a_b / (gamma - 1.0)

        # Set the enthalpy
        H_b = self.vars["H_b"] = ggam1 * p_b / rho_b + 0.5 * u_b * u_b

        self.constraints["res"] = [
            F[0] - rho_b * u_b,
            F[1] - (rho_b * u_b**2 + p_b),
            F[2] - rho_b * u_b * H_b,
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

# Set reference and reservoir temperature, pressure in physical units
gamma = 1.4
R = 287.0  # J/(kg K) Gas constant
T_ref = 300.0  # degK
p_ref = 100.0e3  # Pa
rho_ref = p_ref / (R * T_ref)  # kg / m^3
a_ref = np.sqrt(gamma * R * T_ref)

# Set values for the length
length = 10.0

# Set the constants needed for the inputs/outputs
dx = length / num_cells

# Set the non-dimensional input parameters
T_res = 1.0  # Reservoir temperature
p_res = 1.0  # Reservoir pressure

# Set the non-dimenional output parameters
p_out = 0.5 * p_ref
p_min = 0.2 * p_ref

# Number of control points
nctrl = 5

# Create the model
model = am.Model("nozzle")

# Add the flux computations at the interior points
model.add_component("flux", num_cells - 1, RoeFlux(gamma=gamma))

# Add the flux computations at the end points
model.add_component("inlet", 1, InletFlux(gamma=gamma, T_res=T_res, p_res=p_res))
model.add_component("outlet", 1, OutletFlux(gamma=gamma, p_back=p_out))

# Add the 1D nozzle
model.add_component("nozzle", num_cells, Nozzle(gamma=gamma, dx=(length / num_cells)))

# Add the Bspline component
xi = np.linspace(0.5 / num_cells, 1.0 - 0.5 / num_cells, num_cells)
bspline = am.BSplineInterpolant(xi=xi, k=4, n=nctrl, deriv=1)

# Set the bspline coefficients
model.add_component("bspline", num_cells, bspline)

# Add the objective function
model.add_component("objective", num_cells, Objective())

# Link the states
model.link(f"nozzle.Q[:-1, :]", "flux.QL")
model.link(f"nozzle.Q[1:, :]", "flux.QR")

# Link the fluxes
model.link(f"nozzle.FL[1:, :]", "flux.F")
model.link(f"nozzle.FR[:-1, :]", "flux.F")

# Link the dAdx values
model.link("nozzle.dAdx", "bspline.output")

# Link the input boundary states and fluxes
model.link("nozzle.Q[0, :]", "inlet.Q[0, :]")
model.link("nozzle.FL[0, :]", "inlet.F[0, :]")

# Link the output boundary states and fluxes
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
        compile_args=compile_args, link_args=link_args, define_macros=define_macros
    )


model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)

# Get the data and set the target pressure distribution
data = model.get_data_vector()

# Quadratic interpolation with p_res, p_min and p_out
b = 4 * p_min - p_out - 3 * p_res
a = 2 * p_res - 4 * p_min + 2 * p_out
data["objective.p0"] = p_res + b * (xi / length) + a * (xi / length) ** 2

# Get the design variables
x = model.create_vector()
lower = model.create_vector()
upper = model.create_vector()

# Set the lower bounds for
x["nozzle.Q"] = 1.0
x["nozzle.Q"] = 1.0
x["nozzle.Q"] = 1.0
x["nozzle.Q"] = 1.0
