import amigo as am
import numpy as np
import argparse
from scipy.sparse.linalg import spsolve
import matplotlib.pylab as plt
import matplotlib.tri as tri
import json
import openmdao.api as om

"""
                  ↓ 
///|===============

"""

length = 1.0
nelems = 50

# class EulerBernoulli(om.ExplicitComponent):
#     def setup(self):
#         self.add_input("comp_amigo", val=0.0)

#         self.add_output("comp", val=0.0)

#     def setup_partials(self):
#         self.declare_partials("*", "*", method="fd")

#     def compute(self, inputs, outputs):
#         comp = inputs["comp_amigo"]

#         outputs["comp"] = comp


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
        # self.add_input("h", value=0.1)
        self.add_data("h", value=0.1)
        # thickness Optimization
        # I = 1./12.*0.1**3*

        # Square Cross Section
        # I = 1./12.*0.1**4.
        # print(type(I))
        self.add_constant("E", value=200e9)
        # self.add_constant("I", value=I)
        self.add_objective("obj")

    def compute(self):
        v = self.inputs["v"]
        t = self.inputs["t"]
        E = self.constants["E"]
        # I = self.constants["I"]
        L0 = length / nelems

        # thickness Optimization changes I
        # For rectangular cross-section: I = (b * h^3) / 12
        # where b = 0.1 (width) and h is the height
        h = self.data["h"]
        b = 0.1
        I = b * h**3 / 12.0

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
    Work = F · u (force dot displacement)
    """

    def __init__(self):
        super().__init__()

        self.add_input("v", value=0.0)
        self.add_input("t", value=0.0)
        self.add_constant("Fv", value=-1000.0)  # Force in v direction
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

        self.add_constant("Fv", value=-1000.0)  # Force in v direction
        self.add_constant("Mt", value=0.0)  # Moment (if any)
        # self.add_input("f")
        self.add_output("c")

    def compute_output(self):
        v = self.inputs["v"]
        t = self.inputs["t"]
        Fv = self.constants["Fv"]
        Mt = self.constants["Mt"]
        # Compliance: work done by external forces
        # With Fv=-1 (downward force) and v<0 (downward disp), Fv*v is positive
        self.outputs["c"] = Fv * v + Mt * t


class VolumeComp(am.Component):
    """Compute volume contribution for each element: h*b*L0"""

    def __init__(self):
        super().__init__()

        self.add_constant("b", value=0.1)  # beam width
        self.add_data("h")  # single element height
        self.add_output("vol")  # Output: element volume contribution

    def compute_output(self):
        h = self.data["h"]
        b = self.constants["b"]
        L0 = length / nelems

        # Each element contributes h * b * L0 to the total volume
        self.outputs["vol"] = h * b * L0


# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--build", dest="build", action="store_true", default=False, help="Enable building"
# )
# parser.add_argument(
#     "--with-openmp",
#     dest="use_openmp",
#     action="store_true",
#     default=False,
#     help="Enable OpenMP",
# )
# parser.add_argument(
#     "--order-type",
#     choices=["amd", "nd", "natural"],
#     default="nd",
#     help="Ordering strategy to use (default: amd)",
# )
# parser.add_argument(
#     "--with-debug",
#     dest="use_debug",
#     action="store_true",
#     default=False,
#     help="Enable debug flags",
# )
# args = parser.parse_args()

# Initialize cantilever beam with tip load
# nelems = 50
# length = 1.0

x_coords = np.linspace(0, length, nelems + 1)
# nodes = x_coords.reshape(-1,1)

nodes = np.arange(nelems + 1, dtype=int)
conn = np.array([[i, i + 1] for i in range(nelems)], dtype=int)

# Post optimality
h = 0.5

model = am.Model("beam")

# Allocate the components
model.add_component("src", nelems + 1, NodeSource())
model.add_component("beam_element", nelems, beam_element())
# DO NOT USE model.add_component("bcs", 2*(nelems+1), Dirichlet())
# model.add_component("bcs",2,Dirichlet())
# model.add_component("force", 1, PointForce(force=3))

# Link beam element variables to source
model.link("beam_element.v", "src.v", tgt_indices=conn)
model.link("beam_element.t", "src.t", tgt_indices=conn)

# Link the boundary conditions
# bcs.dof[0]

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

# Add volume computation - one per element
vol_comp = VolumeComp()
model.add_component("vol_comp", nelems, vol_comp)
model.link("vol_comp.h", "beam_element.h")

# Sum all volume contributions by linking them together
model.link("vol_comp.vol[1:]", "vol_comp.vol[0]")

# if args.build:
#     compile_args = []
#     link_args = ["-lblas", "-llapack"]
#     define_macros = []
#     if args.use_openmp:
#         compile_args = ["-fopenmp"]
#         link_args += ["-fopenmp"]
#         define_macros = [("AMIGO_USE_OPENMP", "1")]

#     model.build_module(
#         compile_args=compile_args,
#         link_args=link_args,
#         define_macros=define_macros,
#         debug=args.use_debug,
#     )
model.build_module(debug=True)
model.initialize()

# Post optimality
# Get the data vector
data = model.get_data_vector()
data["beam_element.h"] = h

## Optimization alternative to hessian solve

# Create displacement vector and provide initial guess
x = model.create_vector()
lower = model.create_vector()
upper = model.create_vector()

# Initial guess (start from zero displacements)
x[:] = 0.0

# Set bounds on displacements
lower["src.v"] = -10.0
upper["src.v"] = 10.0

lower["src.t"] = -10.0
upper["src.t"] = 10.0

# Start optimization - minimize total potential energy
opt = am.Optimizer(model, x, lower=lower, upper=upper)
opt_data = opt.optimize(  #
    {
        "max_iterations": 500,
        "initial_barrier_param": 1.0,
        "max_line_search_iterations": 5,
        "convergence_tolerance": 1.0e-5,
    }
)

# Obtain Post Optimality Derivatives
dfdx, of_map, wrt_map = opt.compute_post_opt_derivatives(
    of="comp.c", wrt="beam_element.h"
)

with open("eb_opt_data.json", "w") as fp:
    json.dump(opt_data, fp, indent=2)

# Results
print("Results")
print()
print(f'compliance: {x["comp.c"]}')
print(f'thicknesses: {data["beam_element.h"]}')
print(f'optimized vert. displacements: {x["src.v"]}')
print(f'optimized angle theta: {x["src.t"]}')
print("dc/dh: ", dfdx[of_map["comp.c"], wrt_map["beam_element.h"]])

# Plot results
fig, ax = plt.subplots()
plt.plot(x_coords, x["src.v"], label="y-displacements")
plt.legend()
fig, ax = plt.subplots()
plt.plot(x_coords, x["src.t"], label="theta")
plt.legend()
fig, ax = plt.subplots()
# plt.plot(x_coords, np.insert(dfdx[of_map["comp.c"], wrt_map["beam_element.h"]],0,0))
plt.plot(x_coords[0:-1], dfdx[of_map["comp.c"], wrt_map["beam_element.h"]])
ax.set_ylabel(r"$\nabla_\mathbf{h}c$")
ax.set_xlabel(r"$x_{coord}$")
# plt.show()


# OpenMDAO optimization
prob = om.Problem()

# OpenMDAO Independent Variables
indeps = prob.model.add_subsystem("indeps", om.IndepVarComp())
indeps.add_output("beam_element_h", shape=nelems, val=0.1)

# Create fresh vectors for OpenMDAO optimization
x_om = model.create_vector()
lower_om = model.create_vector()
upper_om = model.create_vector()

# Initial guess
x_om[:] = 0.0

# Set bounds
lower_om["src.v"] = -10.0
upper_om["src.v"] = 10.0
lower_om["src.t"] = -10.0
upper_om["src.t"] = 10.0

# Amigo Optimizer - outputs compliance and total volume
prob.model.add_subsystem(
    "fea",
    am.ExplicitOpenMDAOPostOptComponent(
        data=["beam_element.h"],
        output=["comp.c", "vol_comp.vol"],
        model=model,
        x=x_om,
        lower=lower_om,
        upper=upper_om,
    ),
)

# prob.model.add_subsystem("eb", EulerBernoulli())

# Note that I'm adding an underscore here to the amigo names - this is a bit annoying
prob.model.connect("indeps.beam_element_h", "fea.beam_element_h")

# prob.model.connect("fea.comp_c", "eb.comp_amigo")

# Add volume constraint using the volume computed by amigo
# Note: vol_comp_vol has shape (50,) but element [0] contains the sum of all elements
prob.model.add_subsystem(
    "volume_con",
    om.ExecComp("con = vol - target_vol", vol=0.01, target_vol=0.01),
)
prob.model.connect("fea.vol_comp_vol", "volume_con.vol", src_indices=[0])

# setup the optimization
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"

prob.model.add_design_var("indeps.beam_element_h", lower=1e-5, upper=10)
prob.model.add_objective("fea.comp_c")
prob.model.add_constraint("volume_con.con", equals=0.0)

prob.setup()
prob.check_partials(compact_print=True)
prob.run_driver()


fig, ax = plt.subplots()
ax.plot(prob["indeps.beam_element_h"])
ax.set_ylabel(r"$h*$")
ax.set_xlabel(r"$x$")
ax.set_title("Optimized Thickness Distribution")
plt.show()
