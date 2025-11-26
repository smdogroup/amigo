import numpy as np
import openmdao.api as om
import amigo as am


class Constraint(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x")
        self.add_input("y")

        self.add_constraint("con")

        return

    def compute(self):
        x = self.inputs["x"]
        y = self.inputs["y"]

        self.constraints["con"] = x + 3.0 * y - 1.0


# Build the OpenMDAO model
prob = om.Problem()

prob.model.add_subsystem("paraboloid", om.ExecComp("f = (x-3)**2 + x*y + (y+4)**2 - 3"))

prob.model.add_design_var("paraboloid.x", lower=-50, upper=50)
prob.model.add_design_var("paraboloid.y", lower=-50, upper=50)
prob.model.add_objective("paraboloid.f")

prob.setup()
prob.run_model()

# Create the amigo model
model = am.Model("paraboloid")
model.add_component("src", 1, Constraint())

inputs = ["src.x", "src.y"]
constraints = []
model.add_external_component(
    "openmdao", am.ExternalOpenMDAOComponent(prob, model), inputs, constraints
)

model.build_module()
model.initialize()

# Create the design variable vector and provide an initial guess
x = model.create_vector()
lower = model.create_vector()
upper = model.create_vector()
x[:] = 0.25
lower[["src.x", "src.y"]] = -50.0
upper[["src.x", "src.y"]] = 50.0

opt = am.Optimizer(model, x, lower=lower, upper=upper)
opt_data = opt.optimize(
    {
        "max_iterations": 500,
        "initial_barrier_param": 1.0,
        "max_line_search_iterations": 5,
        "init_affine_step_multipliers": False,
        "init_least_squares_multipliers": False,
    }
)

print("x = ", x["src.x"])
print("y = ", x["src.y"])
