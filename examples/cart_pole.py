import amigo as am


final_time = 2.0
num_time_steps = 200

class TrapezoidRule(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("dt", value=final_time / num_time_steps)

        self.add_input("q1")
        self.add_input("q2")

        self.add_input("q1dot")
        self.add_input("q2dot")

        self.add_output("res")

        return
    
    def compute(self):
        dt = self.constants["dt"]

        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        q1dot = self.inputs["q1dot"]
        q2dot = self.inputs["q2dot"]

        self.outputs["res"] = q2 - q1 - 0.5 * dt * (q1dot + q2dot)

        return

class CartComponent(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("g", value=9.81)
        self.add_constant("L", value=0.5)
        self.add_constant("m1", value=0.5)
        self.add_constant("m2", value=0.3)

        self.add_input("x", label="control")
        self.add_input("q", shape=(4), label="state")
        self.add_input("qdot", shape=(4), label="rate")

        self.add_var("cost")
        self.add_var("sint")

        self.add_output("res", shape=(4), label="residual")

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
        self.vars["sint"] = am.sin(q[1])
        self.vars["cost"] = am.cos(q[1])

        # Extract a reference to the variable values
        sint = self.vars["sint"]
        cost = self.vars["cost"]

        res = 4 * [None]
        res[0] = q[2] - qdot[0]
        res[1] = q[3] - qdot[1]
        res[2] = (m1 + m2 * (1.0 - cost * cost)) * qdot[2] - (
            L * m2 * sint * q[3] * q[3] * x + m2 * g * cost * sint
        )
        res[3] = L * (m1 + m2 * (1.0 - cost * cost)) * qdot[3] + (
            L * m2 * cost * sint * q[3] * q[3] + x * cost + (m1 + m2) * g * sint
        )

        self.outputs["res"] = res

        return


cart = CartComponent()
trap = TrapezoidRule()
print(cart.generate_cpp())
print(cart.generate_pybind11())

print(trap.generate_cpp())
