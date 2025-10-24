import amigo as am


class Quadratic(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x1")
        self.add_input("x2")
        self.add_data("a", value=1.0)
        self.add_data("b", value=1.0)
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


a = 1.0
b = 3.0

model = am.Model("quadratic")
model.add_component("quad", 1, Quadratic())

model.build_module(debug=True)
model.initialize()

# Get the data vector
data = model.get_data_vector()
data["quad.a"] = a
data["quad.b"] = b

# Create the design variable vector and provide an initial guess
x = model.create_vector()
lower = model.create_vector()
upper = model.create_vector()

x[:] = 0.0
lower["quad.x1"] = -10.0
upper["quad.x1"] = 10.0

lower["quad.x2"] = -10.0
upper["quad.x2"] = 10.0

opt = am.Optimizer(model, x, lower=lower, upper=upper)
opt_data = opt.optimize(
    {
        "max_iterations": 500,
        "initial_barrier_param": 1.0,
        "max_line_search_iterations": 5,
    }
)

dfdx, of_map, wrt_map = opt.compute_post_opt_derivatives(
    of="quad.f", wrt=["quad.a", "quad.b"]
)

# f^{star} = (5 * a**2 - 8 * a * b + 5 * b**2) / 9.0
print(dfdx[of_map["quad.f"], wrt_map["quad.a"]] - (10 * a - 8 * b) / 9)
print(dfdx[of_map["quad.f"], wrt_map["quad.b"]] - (10 * b - 8 * a) / 9)
