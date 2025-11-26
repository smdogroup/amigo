import amigo as am
import numpy as np


# Set up a constraint and an input
class Source(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x1")
        self.add_input("x2")
        self.add_input("x3")

        self.add_constraint("c1")
        self.add_constraint("c2")


class Objective(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x1")
        self.add_input("x2")
        self.add_input("x3")

        self.add_objective("obj")

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        x3 = self.inputs["x3"]

        self.objective["obj"] = x1 * x1 + x2 * x2 + x3 * x3


class ExternalQuadratic:
    def get_constraint_jacobian_csr(self):
        ncon = 2
        nvars = 3
        rowp = np.array([0, 3, 6])
        cols = np.array([0, 1, 2, 0, 1, 2])

        return ncon, nvars, rowp, cols

    def evaluate(self, x, con, grad, jac):
        con[0] = x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1.0
        con[1] = x[0] + x[1] + 2.0 * x[2] - 2.0

        jac[0] = 2.0 * x[0]
        jac[1] = 2.0 * x[1]
        jac[2] = 2.0 * x[2]
        jac[3] = 1.0
        jac[4] = 1.0
        jac[5] = 2.0

        return


model = am.Model("quadratic")
model.add_component("src", 1, Source())
model.add_component("obj", 1, Objective())
model.link_by_name()

inputs = ["src.x1", "src.x2", "src.x3"]
constraints = ["src.c1", "src.c2"]
model.add_external_component("extrn", ExternalQuadratic(), inputs, constraints)

model.build_module()
model.initialize()

# Create the design variable vector and provide an initial guess
x = model.create_vector()
lower = model.create_vector()
upper = model.create_vector()
x[:] = 0.25
lower[["src.x1", "src.x2", "src.x3"]] = -10.0
upper[["src.x1", "src.x2", "src.x3"]] = 10.0

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
