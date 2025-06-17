import amigo as am
import sys
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pylab as plt
import niceplots


num_time_steps = 200


class TrapezoidRule(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("num_time_steps", value=num_time_steps, type=int)

        self.add_input("tf")
        self.add_var("dt")

        self.add_input("q1")
        self.add_input("q2")

        self.add_input("q1dot")
        self.add_input("q2dot")

        self.add_output("res")

        return

    def compute(self):
        tf = self.inputs["tf"]

        self.vars["dt"] = tf / self.constants["num_time_steps"]
        dt = self.vars["dt"]

        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        q1dot = self.inputs["q1dot"]
        q2dot = self.inputs["q2dot"]

        self.outputs["res"] = q2 - q1 - 0.5 * dt * (q1dot + q2dot)

        return


class AircraftDynamics(am.Component):
    def __init__(self):
        m0 = 19030.0
        self.add_constant("g", value=9.81)
        self.add_constant("Isp", value=9.81)
        self.add_constant("T", value=0.9 * m0 * 9.81)

        # Thrust, lift and drag values
        # self.add_input("T")
        self.add_input("L")
        self.add_input("D")

        # Angle of attack control variable
        self.add_input("alpha")

        # The state variables
        self.add_input("q", shape=5)
        self.add_input("qdot", shape=5)

        # The residual outputs
        self.add_output("res", shape=5)

        return

    def compute(self):
        g = self.constants["g"]
        Isp = self.constants["Isp"]

        alpha = self.inputs["alpha"]
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        # T = self.inputs["T"]
        T = self.constants["T"]
        D = self.inputs["D"]
        L = self.inputs["L"]

        v = q[0]
        gamma = q[1]
        h = q[2]
        r = q[3]
        m = q[4]

        self.outputs["res"] = [
            qdot[0] - ((T / m) * am.cos(alpha) - (D / m) - g * am.sin(gamma)),
            qdot[1]
            - (T / (m * v) * am.sin(alpha) + L / (m * v) - (g / v) * am.cos(gamma)),
            qdot[2] - v * am.sin(gamma),
            qdot[3] - v * am.cos(gamma),
            qdot[4] + T / (g * Isp),
        ]


class AeroModel(am.Component):
    def __init__(self):
        self.add_constant("CL_alpha", value=3.44)
        self.add_constant("CD0", value=0.013)
        self.add_constant("kappa", value=0.54)
        self.add_constant("S_ref", value=49.25)

        self.add_input("v")
        self.add_input("alpha")

        self.add_input("L")
        self.add_input("D")

        self.add_output("L_res")
        self.add_output("D_res")

        return

    def compute(self):
        D = self.inputs["D"]
        L = self.inputs["L"]
        v = self.inputs["v"]
        alpha = self.inputs["alpha"]

        CD0 = self.constants["CD0"]
        kappa = self.constants["kappa"]
        CL_alpha = self.constants["CL_alpha"]
        S_ref = self.constants["S_ref"]

        rho = 1.225
        q = 0.5 * rho * v * v
        CL = CL_alpha * alpha
        CD = CD0 + kappa * CL**2

        self.outputs["L_res"] = L - q * S_ref * CL
        self.outputs["D_res"] = D - q * S_ref * CD

        return


class Objective(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("tf", label="final time")
        self.add_objective("obj")

        return

    def compute(self):
        tf = self.inputs["tf"]
        self.objective["obj"] = tf

        return


# Initial conditions are q = 0
class InitialConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("q", shape=4)
        self.add_output("res", shape=4)

    def compute(self):
        q = self.inputs["q"]
        self.outputs["res"] = [q[0], q[1], q[2], q[3]]


# Set the final conditions
class FinalConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("pi", value=np.pi)

        self.add_input("q", shape=4)
        self.add_output("res", shape=4)

    def compute(self):
        pi = self.constants["pi"]
        q = self.inputs["q"]
        self.outputs["res"] = [q[0] - 2.0, q[1] - pi, q[2], q[3]]


ac = AircraftDynamics()
aero = AeroModel()
trap = TrapezoidRule()
obj = Objective()
ic = InitialConditions()
fc = FinalConditions()

module_name = "time_to_climb"
model = am.Model(module_name)
model.add_component("ac", num_time_steps + 1, ac)
model.add_component("aero", num_time_steps + 1, aero)
model.add_component("trap", 5 * num_time_steps, trap)
model.add_component("obj", 1, obj)
model.add_component("ic", 1, ic)
model.add_component("fc", 1, fc)

# Add the connections
model.connect("ac.q[0, :]")

model.connect()


model.initialize()

if "build_ext" in sys.argv:
    model.generate_cpp()
    model.build_module()
