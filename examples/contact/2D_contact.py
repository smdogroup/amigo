import amigo as am
import numpy as np
import argparse
from scipy.sparse.linalg import spsolve
import matplotlib.pylab as plt
import matplotlib.tri as tri
import json
import openmdao.api as om


length = 1.0
nelems = 5
E = 25000
Fv = -120
Mv = 0.0

def get_transformation(x1,x2,y1,y2):
    phi = np.arctan2(y2-y1,x2-x1)
    T = np.empty(6,6)
    T[0,:] = [np.cos(phi), np.sin(phi), 0, 0, 0, 0]
    T[1,:] = [-np.sin(phi), np.cos(phi), 0, 0, 0, 0]
    T[2,:] = [0, 0, 1, 0, 0, 0]
    T[3,:] = [0, 0, 0, np.cos(phi), np.sin(phi), 0]
    T[4,:] = [0, 0, 0, -np.sin(phi), np.cos(phi), 0]
    T[5,:] = [0, 0, 0, 0, 0, 1]
    return T



class beam_element(am.Component):
    """
    Element Strain Energy for potential energy minimization.
    Equilibrium is obtained by minimizing total PE = Strain Energy - Work.
    No residuals needed - equilibrium comes from dPE/du = 0.
    """

    def __init__(self):
        super().__init__()
        self.add_data("x_coord", shape=(2,), value=0.0)
        self.add_data("y_coord", shape=(2,), value=0.0)
        self.add_input("u", shape=(2,), value=0.0)
        self.add_input("v", shape=(2,), value=0.0)
        self.add_input("t", shape=(2,), value=0.0)
        self.add_data("h")
        self.add_objective("obj")

    def compute(self):
        x = self.data["x_coord"]
        y = self.data["y_coord"]
        u = self.inputs["u"]
        v = self.inputs["v"]
        t = self.inputs["t"]
        L0 = length / nelems
        # Assemble global element displacement vector
        de_global = np.array([u[0], v[0], t[0], u[1], v[1], t[1]])

        # Transform into local displacement vector
        # T = get_transformation(x[0],x[1],y[0],y[1])
        # phi = np.cos(y[1]-y[0]) #,x[1]-x[0])
        # T = np.empty(6,6)
        # T[0,:] = [np.cos(phi), np.sin(phi), 0, 0, 0, 0]
        # T[1,:] = [-np.sin(phi), np.cos(phi), 0, 0, 0, 0]
        # T[2,:] = [0, 0, 1, 0, 0, 0]
        # T[3,:] = [0, 0, 0, np.cos(phi), np.sin(phi), 0]
        # T[4,:] = [0, 0, 0, -np.sin(phi), np.cos(phi), 0]
        # T[5,:] = [0, 0, 0, 0, 0, 1]
        # de_local = T @ de_local
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        L = (dx*dx + dy*dy)**0.5
        c = dx / L
        s = dy / L
        T = [[ c, s, 0, 0, 0, 0],
             [-s, c, 0, 0, 0, 0],
             [ 0, 0, 1, 0, 0, 0],
             [ 0, 0, 0, c, s, 0],
             [ 0, 0, 0,-s, c, 0],
             [ 0, 0, 0, 0, 0, 1]]
        
        de_local = T @ de_global

        # thickness Optimization changes I
        h = self.data["h"]
        I = 1.0 / 12.0 * 0.333 * h**3
        
        # Create local beam element stiffness matrix
        L = length/nelems
        A = 1.0
        I = 1.0
        a1 = E*A/L
        a2 = E*I/(L**3.0)
        Ke = np.empty((6,6))
        Ke[0,:] = [a1, 0, 0, -a1, 0, 0]
        Ke[1,:] = [0, 12*a2, 6*L*a2, 0, -12*a2, 6*L*a2]
        Ke[2,:] = [0, 0, 4*L**2*a2, 0, -6*L*a2, 2*L**2*a2]
        Ke[3,:] = [0, 0, 0, a1, 0, 0]
        Ke[4,:] = [0, 0, 0, 0, 12*a2, -6*L*a2]
        Ke[5,:] = [0, 0, 0, 0, 0, 4*L**2*a2]
        Ke = Ke + Ke.T - np.diag(np.diag(Ke))

        # Calculate strain energy U for element
        self.objective["obj"] = 0.5 * de_local.T @ Ke @ de_local


class BoundaryCondition(am.Component):
    """
    Fixed boundary condition for energy minimization.
    Simply enforces displacement and rotation to be zero.
    """

    def __init__(self):
        super().__init__()

        self.add_input("u", value=1.0)
        self.add_input("v", value=1.0)
        self.add_input("t", value=1.0)

        self.add_constraint("bc_u", value=0.0, lower=0.0, upper=0.0)
        self.add_constraint("bc_v", value=0.0, lower=0.0, upper=0.0)
        self.add_constraint("bc_t", value=0.0, lower=0.0, upper=0.0)

    def compute(self):
        # Enforce zero displacement and rotation at fixed end
        self.constraints["bc_u"] = self.inputs["u"]
        self.constraints["bc_v"] = self.inputs["v"]
        self.constraints["bc_t"] = self.inputs["t"]


class AppliedLoadDist(am.Component):
    """
    For potential energy minimization: contributes work done by external forces
    Total PE = Strain Energy - Work
    Work = F Â· u (force dot displacement)
    """

    def __init__(self):
        super().__init__()

        self.add_input("v", shape=(2,), value=0.0)
        self.add_input("t", shape=(2,), value=0.0)
        # self.add_constant("Fv", value=-1.0)  # Force in v direction
        # self.add_constant("Mt", value=0.0)  # Moment (if any)
        self.add_objective("work")
        return

    def compute(self):
        v = self.inputs["v"]
        t = self.inputs["t"]
        # Fv = self.constants["Fv"]
        # Mt = self.constants["Mt"]
        # Work done by external forces (negative contributes to total PE)
        Le = length/50
        fe = [
            Fv * Le * 0.5,
            Fv * Le * (Le / 12),
            Fv * Le * 0.5,
            -Fv * Le * (Le / 12),
        ]
        de = np.array([v[0], t[0], v[1], t[1]])
        # self.objective["work"] = -1*(fe @ de)
        self.objective["work"] = -(
            fe[0] * de[0] +
            fe[1] * de[1] +
            fe[2] * de[2] +
            fe[3] * de[3]
        )
        return

class AppliedLoad(am.Component):
    """
    For potential energy minimization: contributes work done by external forces
    Total PE = Strain Energy - Work
    Work = F Â· u (force dot displacement)
    """

    def __init__(self):
        super().__init__()

        self.add_input("u", value=0.0)
        self.add_input("v", value=0.0)
        self.add_input("t", value=0.0)
        self.add_constant("Fv", value=-1.0e5)  # Force in v direction
        self.add_constant("Mt", value=0.0)  # Moment (if any)
        self.add_objective("work")
        return

    def compute(self):
        u = self.inputs["u"]
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
        self.add_input("u")
        self.add_input("v")
        self.add_input("t")

        self.add_constraint("gap", upper = 0.0)

    def compute(self):
        y = self.data["y_coord"]
        v = self.inputs["v"]

        self.constraints["gap"] = -(y + v - (-0.4))

# class ContactBoundary(am.Component):
#     def __init__(self):
#         super().__init__()

#         # Mesh coordinates
#         self.add_data("x_coord")
#         self.add_data("y_coord")

#         # Displacement degrees of freedom
#         self.add_input("u")
#         self.add_input("v")
#         self.add_input("t")

#         self.add_constraint("gap", lower=-0.1)

#     def compute(self):
#         v = self.inputs["v"]
#         y_coord = self.data["y_coord"]
#         self.constraints["gap"] = v 


class Compliance(am.Component):
    """Compliance for an applied load condition at the tip"""

    def __init__(self):
        super().__init__()

        self.add_input("u")
        self.add_input("v")
        self.add_input("t")

        self.add_constant("Fv", value=-1.0)  # Force in v direction
        self.add_constant("Mt", value=0.0)  # Moment (if any)
        self.add_output("c")

    def compute_output(self):
        u = self.inputs["u"]
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
        vol = h * 0.33 * 1.0 / 50.0  # nelems
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

# x_coords = np.linspace(0, length, nelems + 1)

# Node locations with L = 1.0
X = np.array([[0.0, 0.0], [1.0, -0.5], [1.0, 1.0], [2.0, -0.5]])
nnodes = 4
x_coord = X[:,0]
y_coord = X[:,1]

conn = np.array([[0, 2], [0, 1], [1, 2], [1, 3], [2, 3]], dtype=int)

model = am.Model("beam")

# Allocate the components
model.add_component("src", nnodes, NodeSource())
model.add_component("beam_element", nelems, beam_element())

# Link nodal coordinates of structure
model.link("beam_element.x_coord", "src.x_coord", tgt_indices=conn)
model.link("beam_element.y_coord", "src.y_coord", tgt_indices=conn)

# Link beam element variables to source
model.link("beam_element.u", "src.u", tgt_indices=conn)
model.link("beam_element.v", "src.v", tgt_indices=conn)
model.link("beam_element.t", "src.t", tgt_indices=conn)

# Fixed boundary conditions at cantilever support
bcs = BoundaryCondition()
model.add_component("bcs", 2, bcs)
model.link("src.u", "bcs.u", src_indices=[0,2])
model.link("src.v", "bcs.v", src_indices=[0,2])
model.link("src.t", "bcs.t", src_indices=[0,2])

# Apply load at free end (contributes to potential energy)
load = AppliedLoad()
model.add_component("load", nelems, load)
model.link("src.u", "load.u", src_indices=[3])
model.link("src.v", "load.v", src_indices=[3])
model.link("src.t", "load.t", src_indices=[3])


# Add contact boundary constraint
# contact = ContactBoundary()
# model.add_component("bound", nnodes, contact)
# model.link("src.u", "bound.u", src_indices=[1,3])
# model.link("src.v", "bound.v", src_indices=[1,3])
# model.link("src.t", "bound.t", src_indices=[1,3])

# Link variables for compliance calculation
compliance = Compliance()
model.add_component("comp", nelems, compliance)
model.link("src.u", "comp.u", src_indices=[0])
model.link("src.v", "comp.v", src_indices=[0])
model.link("src.t", "comp.t", src_indices=[0])

# Add Volume Constraint
vol_con = VolumeConstraint()
model.add_component("vol_con", nelems, vol_con)
model.link("beam_element.h", "vol_con.h")
model.link("comp.c[1:]", "comp.c[0]")
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

# set x and y coords
data = model.get_data_vector()
data["src.x_coord"] = x_coord
data["src.y_coord"] = y_coord

# Set bounds on displacements
lower["src.u"] = -float("inf")
upper["src.u"] = float("inf")

lower["src.v"] = -float("inf")
upper["src.v"] = float("inf")

# y_ground = -0.6
# for i in range(len(y_coord)):
#     lower_bound = y_ground - y_coord[i]
#     lower[f"src.v[{i}]"] = lower_bound

lower["src.t"] = -float("inf")
upper["src.t"] = float("inf")

# Amigo potential energy minimization parameters
opt_options = {
    "max_iterations": 200,
    "convergence_tolerance": 1e-5,
    "max_line_search_iterations": 1,
    "initial_barrier_param": 0.1,
}

# #indiv. Amigo problem:
opt = am.Optimizer(
    model,
    x,
    lower = lower,
    upper = upper,
)
opt.optimize(opt_options)

# dfdx, of_map, wrt_map = opt.compute_post_opt_derivatives(
#     of="comp.c[0]", wrt="beam_element.h", method="adjoint"
# )

def plot_truss(ax, xpts, conn):
    # Show the node numbers
    for i, pt in enumerate(xpts):
        ax.text(pt[0] + 0.02, pt[1] + 0.05, "N%d" %(i + 1))

    # Loop over the elements in the connectivity
    for i, e in enumerate(conn):
        xe = [xpts[e[0], 0], xpts[e[1], 0]]
        ye = [xpts[e[0], 1], xpts[e[1], 1]]
        ax.plot(xe, ye, '-ok')

        xmid = 0.5 * (xe[0] + xe[1])
        ymid = 0.5 * (ye[0] + ye[1])
        ax.text(xmid - 0.02, ymid + 0.02, "E%d" %(i + 1))

    ax.set_aspect("equal", "box")

def plot_disp(x, u=None, scale=1.0):
        """
        Visualize the truss and optionally its deformation.
        """

        if u is not None:
            x = x + scale*u.reshape((-1, 2))
        else:
            x = x

        for index in range(5):
            i = conn[index, 0]
            j = conn[index, 1]
            plt.plot([x[i,0], x[j,0]], [x[i,1], x[j,1]], '-ko')

        # plt.show()

        return

def build_global_displacement(u, v):
    nnodes = len(u)
    q = np.zeros(2 * nnodes)

    for i in range(nnodes):
        q[2*i + 0] = u[i]
        q[2*i + 1] = v[i]

    return q

# fig,ax = plt.subplots()
# plot_truss(ax, X, conn)
# plt.show()
u = x["src.u"]
v = x["src.v"]
t = x["src.t"]
U = build_global_displacement(u,v)
# np.save('U_bounded.npy',U)
U_unbounded = np.load('U_unbounded.npy')

fig = plt.figure(facecolor='w')
plot_disp(X)
# plot_disp(X, U_unbounded)
plot_disp(X, U)

plt.axhline(-0.6, color = 'black')
plt.savefig('2d_contact.png')
# plt.show()
# print(u,v)

exit()


# # OpenMDAO optimization
# prob = om.Problem()

# # OpenMDAO Independent Variables
# yndeps = prob.model.add_subsystem("indeps", om.IndepVarComp())

# # # run openmdao example and extract optimal h to test amigo fem
# # om_h, om_v, om_totals_obj, om_totals_con = original_om_problem()
# # om_c_wrt_h = om_totals_obj["compliance_comp.compliance", "h"]["J_rev"]
# # om_con_wrt_h = om_totals_con["volume_comp.volume", "h"]["J_rev"]

# # indeps.add_output("h", shape=nelems, val=5 * np.linspace(om_h[0], om_h[-1], len(om_h)))
# indeps.add_output("h", shape=nelems, val=0.1)

# # Amigo Optimizer
# prob.model.add_subsystem(
#     "fea",
#     am.ExplicitOpenMDAOPostOptComponent(
#         data=["beam_element.h"],
#         output=["comp.c[0]", "vol_con.con[0]"],
#         data_mapping={"beam_element.h": "h"},
#         output_mapping={"comp.c[0]": "c", "vol_con.con[0]": "con"},
#         model=model,
#         x=x,
#         lower=lower,
#         upper=upper,
#         opt_options=opt_options,
#     ),
# )

# prob.model.connect("indeps.h", "fea.h")

# # setup the OpenMDAO optimization
# prob.driver = om.ScipyOptimizeDriver()

# if args.optimizer == "SLSQP":
#     prob.driver.options["optimizer"] = "SLSQP"
# else:
#     prob.driver.options["optimizer"] = args.optimizer


# prob.driver.options["maxiter"] = 500
# prob.driver.options["tol"] = 1e-6
# prob.driver.options["disp"] = True

# # perform optimziation
# prob.model.add_design_var("indeps.h", lower=1e-6, upper=1.0, ref = 1.0e-1)
# prob.model.add_objective("fea.c", ref=1.0e3)
# prob.model.add_constraint("fea.con", equals=0.03)
# prob.setup(check=True)


# # data = prob.check_partials(compact_print=False, step=1e-6)
# # exit()
# # Run Optimization Problem
# prob.run_driver()
# h_results = prob["indeps.h"]
# print("vol: ", prob["fea.con"])
# print("compliance: ", prob["fea.c"])

# # data = prob.check_partials(compact_print=False, step=1e-6)

# fig, ax = plt.subplots()
# # ax.plot(prob["indeps.h"])
# ax.plot(prob["indeps.h"], marker="o", label="amigo")
# # ax.plot(om_h, label="openMDAO example")
# ax.legend()
# ax.set_ylabel(r"$h*$")
# ax.set_xlabel(r"$x$")
# ax.set_title("Optimized Thickness Distribution")

# fig,ax = plt.subplots()
# ax.plot(x["src.v"])


# plt.show()
