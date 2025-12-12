import amigo as am
import numpy as np
import argparse
from scipy.sparse.linalg import spsolve
import matplotlib.pylab as plt
import matplotlib.tri as tri
import json
import openmdao.api as om

"""
                  â†“ 
///|===============

"""


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


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)

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

model.build_module()
model.initialize()


def show_graph():
    from pyvis.network import Network

    graph = model.create_graph()
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

    net.show("eb_mdao.html")


# Post optimality
# Get the data vector
# data = model.get_data_vector()
# # data["beam_element.h"] = 1.0
# # print('h:', data["beam_element.h"])

# print('h: ', data['beam_element.h'])
# x = model.create_vector()
# output = model.create_output_vector()
# problem = model.get_problem()
# problem.compute_output(x.get_vector(), output.get_vector())

# print(output["vol_con.con[0]"])

# exit(0)

## Optimization alternative to hessian solve

# # Create displacement vector and provide initial guess
x = model.create_vector()
lower = model.create_vector()
upper = model.create_vector()

# # Initial guess (start from zero displacements)
x[:] = 0.0

# Set bounds on displacements
lower["src.v"] = -float("inf")
upper["src.v"] = float("inf")

lower["src.t"] = -float("inf")
upper["src.t"] = float("inf")

# inputs, cons, _, _ = model.get_names()
# inputs.extend(cons)

# for name in inputs:
#     print(name, x[name])

# for name in inputs:
#     print(name, lower[name], upper[name])

# exit(0)


# # Start optimization - minimize total potential energy
# opt = am.Optimizer(model, x, lower=lower, upper=upper)
# opt_data = opt.optimize(  #
#     {
#         "max_iterations": 500,
#         # "initial_barrier_param": 1.0,
#         # "max_line_search_iterations": 5,
#         "convergence_tolerance": 1.0e-8,
#     }
# )


# # Obtain Post Optimality Derivatives
# # dfdx, of_map, wrt_map = opt.compute_post_opt_derivatives(
# #     of="comp.c", wrt="beam_element.h"
# # )
# dfdx, of_map, wrt_map = opt.compute_post_opt_derivatives(
#     of="vol_con.con[0]", wrt="beam_element.h"
# )

# with open("eb_opt_data.json", "w") as fp:
#     json.dump(opt_data, fp, indent=2)

# Get results
# comp_1 = x["comp.c"]
# thicknesses_1 = model.get_data_vector()["beam_element.h"]
# v_1 = x["src.v"]
# t_1 = x["src.t"]
# volume_1 = x["vol_con.con"]
# print('h: ', thicknesses_1)
# print('volume computation: ', volume_1)
# print('volume sum', np.sum(volume_1))
# print(dfdx)

# exit()
# # Results
# print("Results")
# print()
# print(f'compliance: {comp_1}')
# print(f'thicknesses: {thicknesses_1}')
# print(f'fem vert. displacements: {v_1}')
# print(f'fem angle theta: {t_1}')
# print('dc/dh: ',dfdx[of_map["comp.c"], wrt_map["beam_element.h"]])

# # Plot results
# fig, ax = plt.subplots()
# plt.plot(x_coords, x["src.v"], label="y-displacements")
# plt.legend()
# fig, ax = plt.subplots()
# plt.plot(x_coords, x["src.t"], label="theta")
# plt.legend()
# fig, ax = plt.subplots()
# # plt.plot(x_coords, np.insert(dfdx[of_map["comp.c"], wrt_map["beam_element.h"]],0,0))
# plt.plot(x_coords[0:-1], dfdx[of_map["comp.c"], wrt_map["beam_element.h"]])
# ax.set_ylabel(r"$\nabla_\mathbf{h}c$")
# ax.set_xlabel(r"$x_{coord}$")
# plt.show()
# exit()

# OpenMDAO optimization
prob = om.Problem()

# OpenMDAO Independent Variables
indeps = prob.model.add_subsystem("indeps", om.IndepVarComp())
# indeps.add_output("h", shape=nelems, val=1.0)

# run openmdao example and extract optimal h to test amigo fem
om_h, om_v, om_totals_obj, om_totals_con = original_om_problem()
om_c_wrt_h = om_totals_obj["compliance_comp.compliance", "h"]["J_rev"]
om_con_wrt_h = om_totals_con["volume_comp.volume", "h"]["J_rev"]
# exit()
# testing if compliance calc is correct for om-beam optimization
indeps.add_output(
    "h", shape=nelems, val=5 * np.linspace(om_h[0], om_h[-1], len(om_h))
)  # substitute openmdao results
# print(np.linspace(om_h[0],om_h[-1],len(om_h)))
# exit()


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

# Add volume constraint using the volume computed by amigo
# Note: vol_comp_vol has shape (50,) but element [0] contains the sum of all elements
# prob.model.add_subsystem(
#     "volume_con",
#     om.ExecComp("con = vol - target_vol", vol=0.1, target_vol=0.01),
# )

# prob.model.connect("fea.con", "volume_con.vol")
# prob.model.add_constraint('volume_con.con', equals=0.0)

# setup the OpenMDAO optimization
prob.driver = om.ScipyOptimizeDriver()
# prob.driver.options["optimizer"] = "SLSQP"
# prob.driver.options["maxiter"] = 200
# prob.driver.options["tol"] = 1e-9
# prob.driver.options['disp'] = True

# prob.driver.options["optimizer"] = "L-BFGS-B"

prob.driver.options["optimizer"] = "trust-constr"

# perform optimziation with
prob.model.add_design_var("indeps.h", lower=1e-2, upper=10)
prob.model.add_objective("fea.c", ref=20000)
prob.model.add_constraint("fea.con", equals=0.01)
prob.setup(check=True)

# # debugging
# prob.model.list_inputs()
# prob.model.list_inputs(units=True, prom_name=True, hierarchical=True)
# prob.model.list_outputs(units=True, prom_name=True, hierarchical=True)
# print(prob.model.get_io_metadata()['fea.vol_con_con']['shape'])
# prob.run_model()

# # derivative computation testing
data = prob.check_partials(compact_print=False, step=1e-5)
# c_wrt_h = data['fea']['c','h']['J_fwd']
# con_wrt_h = data['fea']['con','h']['J_fwd']
# # exit()

# exit()

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

# # derivative computation error comparison graph
# fig,ax = plt.subplots()
# print(c_wrt_h)
# print(om_c_wrt_h)
# print(con_wrt_h)
# print(om_con_wrt_h)
# ax.plot((c_wrt_h[0]-om_c_wrt_h[0])/om_c_wrt_h[0])
# ax.set_title('objective derivative computation error')
# # ax.plot(om_c_wrt_h[0])

# plt.show()
