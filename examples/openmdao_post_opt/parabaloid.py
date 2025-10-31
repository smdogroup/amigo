import amigo as am
import openmdao.api as om


class Paraboloid(om.ExplicitComponent):
    def setup(self):
        self.add_input("x", val=0.0)
        self.add_input("y", val=0.0)
        self.add_input("z", val=0.0)

        self.add_output("f", val=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        x = inputs["x"]
        y = inputs["y"]
        z = inputs["z"]

        outputs["f"] = (x - 3.0) ** 2 + x * y + (y + z) ** 2 - 3.0


class Quadratic(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x1")
        self.add_input("x2")
        self.add_data("a", value=1.0, lower=-float("inf"), upper=float("inf"))
        self.add_data("b", value=1.0, lower=-float("inf"), upper=float("inf"))
        self.add_objective("obj")
        self.add_output("f")

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]

        a = self.data["a"]
        b = self.data["b"]

        self.objective["obj"] = x1**2 + x1 * x2 + x2**2 - a * x1 - b * x2

    def compute_output(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]

        self.outputs["f"] = x1**2 + x2**2


model = am.Model("parabaloid")
model.add_component("quad", 1, Quadratic())

model.build_module(debug=True)
model.initialize()

x = model.create_vector()
lower = model.create_vector()
upper = model.create_vector()
lower[:] = -10.0
upper[:] = 10.0

prob = om.Problem()

indeps = prob.model.add_subsystem("indeps", om.IndepVarComp())
indeps.add_output("x", val=3.0)
indeps.add_output("y", val=-4.0)

prob.model.add_subsystem(
    "optimizer",
    am.ExplicitOpenMDAOPostOptComponent(
        data=["quad.a", "quad.b"],  # The Amigo names
        output=["quad.f"],
        model=model,
        x=x,
        lower=lower,
        upper=upper,
    ),
)

prob.model.add_subsystem("paraboloid", Paraboloid())

# Note that I'm adding an underscore here to the amigo names - this is a bit annoying
prob.model.connect("indeps.x", "optimizer.quad_a")
prob.model.connect("indeps.x", "paraboloid.x")

prob.model.connect("indeps.y", "optimizer.quad_b")
prob.model.connect("indeps.y", "paraboloid.y")

prob.model.connect("optimizer.quad_f", "paraboloid.z")

# setup the optimization
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"

prob.model.add_design_var("indeps.x", lower=-50, upper=50)
prob.model.add_design_var("indeps.y", lower=-50, upper=50)
prob.model.add_objective("paraboloid.f")

prob.setup()
prob.run_driver()
