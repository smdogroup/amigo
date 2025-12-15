import amigo as am
import numpy as np
import argparse
from scipy.sparse.linalg import spsolve
import matplotlib.pylab as plt
import matplotlib.tri as tri
import json
import openmdao.api as om


def original_om_problem():
    from openmdao.test_suite.test_examples.beam_optimization.beam_group import BeamGroup

    # import openmdao.api as om
    # import matplotlib.pyplot as plt

    E = 1.0
    L = 1.0
    b = 0.1
    volume = 0.01

    num_elements = 50

    prob = om.Problem(
        model=BeamGroup(E=E, L=L, b=b, volume=volume, num_elements=num_elements)
    )

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"] = 1e-9
    prob.driver.options["disp"] = True

    prob.setup()
    prob.run_model()
    totals_obj = prob.check_totals(of="compliance_comp.compliance", wrt="h")
    totals_con = prob.check_totals(of="volume_comp.volume", wrt="h")
    # exit()
    prob.run_driver()

    print(prob["h"])
    h_target = prob["h"]
    displacements = prob["compliance_comp.displacements"]
    # plt.plot(prob['h'])
    # print(np.allclose(h_results, h_omexample))
    # # plt.gca().invert_xaxis()  # if needed to match OpenMDAO’s plotting convention
    # plt.show()
    return h_target, displacements, totals_obj, totals_con


length = 1.0
nelems = 50


class beam_element(am.Component):
    """
    Element Strain Energy for potential energy minimization.
    Equilibrium is obtained by minimizing total PE = Strain Energy - Work.
    No residuals needed - equilibrium comes from dPE/du = 0.
    """

    def __init__(self):
        super().__init__()
        self.add_input("v", shape=(2,), value=0.0)
        self.add_input("t", shape=(2,), value=0.0)
        self.add_data("h")
        self.add_constant("E", value=1.0)
        self.add_objective("obj")

    def compute(self):
        v = self.inputs["v"]
        t = self.inputs["t"]
        E = self.constants["E"]
        L0 = length / nelems

        # thickness Optimization changes I
        h = self.data["h"]
        I = 1.0 / 12.0 * 0.1 * h**3

        # Create beam element stiffness matrix
        Ke = np.empty((4, 4))
        Ke[0, :] = [12, 6 * L0, -12, 6 * L0]
        Ke[1, :] = [6 * L0, 4 * L0**2, -6 * L0, 2 * L0**2]
        Ke[2, :] = [-12, -6 * L0, 12, -6 * L0]
        Ke[3, :] = [6 * L0, 2 * L0**2, -6 * L0, 4 * L0**2]
        Ke = Ke * (E * I) / L0**3

        # Assemble element displacement vector
        de = np.array([v[0], t[0], v[1], t[1]])

        # Calculate strain energy U for element
        self.objective["obj"] = 0.5 * de.T @ Ke @ de


class BoundaryCondition(am.Component):
    """
    Fixed boundary condition for energy minimization.
    Simply enforces displacement and rotation to be zero.
    """

    def __init__(self):
        super().__init__()

        self.add_input("v", value=1.0)
        self.add_input("t", value=1.0)

        self.add_constraint("bc_v", value=0.0, lower=0.0, upper=0.0)
        self.add_constraint("bc_t", value=0.0, lower=0.0, upper=0.0)

    def compute(self):
        # Enforce zero displacement and rotation at fixed end
        self.constraints["bc_v"] = self.inputs["v"]
        self.constraints["bc_t"] = self.inputs["t"]


class AppliedLoad(am.Component):
    """
    For potential energy minimization: contributes work done by external forces
    Total PE = Strain Energy - Work
    Work = F Â· u (force dot displacement)
    """

    def __init__(self):
        super().__init__()

        self.add_input("v", value=0.0)
        self.add_input("t", value=0.0)
        self.add_constant("Fv", value=-1.0)  # Force in v direction
        self.add_constant("Mt", value=0.0)  # Moment (if any)
        self.add_objective("work")
        return

    def compute(self):
        v = self.inputs["v"]
        t = self.inputs["t"]
        Fv = self.constants["Fv"]
        Mt = self.constants["Mt"]
        # Work done by external forces (negative contributes to total PE)
        self.objective["work"] = -(Fv * v + Mt * t)
        return


class NodeSource(am.Component):
    def __init__(self):
        super().__init__()

        # Mesh coordinates
        self.add_data("x_coord")
        self.add_data("y_coord")

        # Displacement degrees of freedom
        self.add_input("v")
        self.add_input("t")


class Compliance(am.Component):
    """Compliance for an applied load condition at the tip"""

    def __init__(self):
        super().__init__()

        self.add_input("v")
        self.add_input("t")

        self.add_constant("Fv", value=-1.0)  # Force in v direction
        self.add_constant("Mt", value=0.0)  # Moment (if any)
        self.add_output("c")

    def compute_output(self):
        v = self.inputs["v"]
        t = self.inputs["t"]
        Fv = self.constants["Fv"]
        Mt = self.constants["Mt"]
        self.outputs["c"] = (
            Fv * v + Mt * t
        )  # dot(Fv, v), but only for one nodal v displ.


class VolumeConstraint(am.Component):
    """sum(h)*b*L0 = volume of beam"""

    def __init__(self):
        super().__init__()

        self.add_data("h")
        self.add_output("con")

    def compute_output(self):
        h = self.data["h"]
        vol = h * 0.1 * 1.0 / 50.0  # nelems
        self.outputs["con"] = vol


# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
parser.add_argument(
    "--optimizer",
    choices=["SLSQP", "L-BFGS-B", "trust-constr"],
    default="SLSQP",
    help="scipy.Optimize optimizer",
)
args = parser.parse_args()

x_coords = np.linspace(0, length, nelems + 1)

nodes = np.arange(nelems + 1, dtype=int)
conn = np.array([[i, i + 1] for i in range(nelems)], dtype=int)

model = am.Model("beam")

# Allocate the components
model.add_component("src", nelems + 1, NodeSource())
model.add_component("beam_element", nelems, beam_element())

# Link beam element variables to source
model.link("beam_element.v", "src.v", tgt_indices=conn)
model.link("beam_element.t", "src.t", tgt_indices=conn)

# Fixed boundary conditions at cantilever support
bcs = BoundaryCondition()
model.add_component("bcs", 1, bcs)
model.link("src.v", "bcs.v", src_indices=[nodes[0]])
model.link("src.t", "bcs.t", src_indices=[nodes[0]])

# Apply load at free end (contributes to potential energy)
load = AppliedLoad()
model.add_component("load", 1, load)
model.link("src.v", "load.v", src_indices=[nodes[-1]])
model.link("src.t", "load.t", src_indices=[nodes[-1]])

# Link variables for compliance calculation
compliance = Compliance()
model.add_component("comp", 1, compliance)
model.link("src.v", "comp.v", src_indices=[nodes[-1]])
model.link("src.t", "comp.t", src_indices=[nodes[-1]])

# Add Volume Constraint
vol_con = VolumeConstraint()
model.add_component("vol_con", nelems, vol_con)
model.link("beam_element.h", "vol_con.h")

# Summing constraints as output
model.link("vol_con.con[1:]", "vol_con.con[0]")

if args.build:
    model.build_module()

model.initialize()

# Create displacement vector and provide initial guess
x = model.create_vector()
lower = model.create_vector()
upper = model.create_vector()

# Initial guess (start from zero displacements)
x[:] = 0.0

# Set bounds on displacements
lower["src.v"] = -float("inf")
upper["src.v"] = float("inf")

lower["src.t"] = -float("inf")
upper["src.t"] = float("inf")

# OpenMDAO optimization
prob = om.Problem()

# OpenMDAO Independent Variables
indeps = prob.model.add_subsystem("indeps", om.IndepVarComp())

# run openmdao example and extract optimal h to test amigo fem
om_h, om_v, om_totals_obj, om_totals_con = original_om_problem()
om_c_wrt_h = om_totals_obj["compliance_comp.compliance", "h"]["J_rev"]
om_con_wrt_h = om_totals_con["volume_comp.volume", "h"]["J_rev"]

indeps.add_output("h", shape=nelems, val=5 * np.linspace(om_h[0], om_h[-1], len(om_h)))

# Amigo potential energy minimization parameters
opt_options = {
    "max_iterations": 500,
    "convergence_tolerance": 1e-8,
    "max_line_search_iterations": 1,
    "initial_barrier_param": 0.1,
}

# Amigo Optimizer
prob.model.add_subsystem(
    "fea",
    am.ExplicitOpenMDAOPostOptComponent(
        data=["beam_element.h"],
        output=["comp.c", "vol_con.con[0]"],
        data_mapping={"beam_element.h": "h"},
        output_mapping={"comp.c": "c", "vol_con.con[0]": "con"},
        model=model,
        x=x,
        lower=lower,
        upper=upper,
        opt_options=opt_options,
    ),
)

prob.model.connect("indeps.h", "fea.h")

# setup the OpenMDAO optimization
prob.driver = om.ScipyOptimizeDriver()

if args.optimizer == "SLSQP":
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["maxiter"] = 200
    prob.driver.options["tol"] = 1e-9
    prob.driver.options["disp"] = True
else:
    prob.driver.options["optimizer"] = args.optimizer

# perform optimziation
prob.model.add_design_var("indeps.h", lower=1e-2, upper=10)
prob.model.add_objective("fea.c", ref=20000)
prob.model.add_constraint("fea.con", equals=0.01)
prob.setup(check=True)

# Run Optimization Problem
prob.run_driver()
h_results = prob["indeps.h"]
print("vol: ", prob["fea.con"])
print("compliance: ", prob["fea.c"])

data = prob.check_partials(compact_print=False, step=1e-6)

fig, ax = plt.subplots()
# ax.plot(prob["indeps.h"])
ax.plot(prob["indeps.h"], marker="o", label="amigo")
ax.plot(om_h, label="openMDAO example")
ax.legend()
ax.set_ylabel(r"$h*$")
ax.set_xlabel(r"$x$")
ax.set_title("Optimized Thickness Distribution")

plt.show()
