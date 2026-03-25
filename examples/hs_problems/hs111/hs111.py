"""
HS111: 10 variables, 3 equalities: larger scale
  min  sum_{j=1}^{10} exp(x_j) * (c_j + x_j - ln(sum_{k=1}^{10} exp(x_k)))
  s.t. exp(x1) + 2*exp(x2) + 2*exp(x3) + exp(x6) + exp(x10) - 2 = 0
       exp(x4) + 2*exp(x5) + exp(x6) + exp(x7) - 1 = 0
       exp(x3) + exp(x7) + exp(x8) + 2*exp(x9) + exp(x10) - 1 = 0
       -100 <= xi <= 100
  x0 = (-2.3, ..., -2.3), f* = -47.7610
"""

import amigo as am
import argparse

c_vals = [
    -6.089,
    -17.164,
    -34.054,
    -5.914,
    -24.721,
    -14.986,
    -24.100,
    -10.708,
    -26.662,
    -22.179,
]


class HS111(am.Component):
    def __init__(self):
        super().__init__()
        for i in range(10):
            self.add_input(f"x{i+1}", value=-2.3, lower=-100.0, upper=100.0)
        self.add_objective("obj")
        self.add_constraint("c1", lower=0.0, upper=0.0)
        self.add_constraint("c2", lower=0.0, upper=0.0)
        self.add_constraint("c3", lower=0.0, upper=0.0)

    def compute(self):
        x = [self.inputs[f"x{i+1}"] for i in range(10)]
        ex = [am.exp(x[i]) for i in range(10)]
        s = sum(ex)
        self.objective["obj"] = sum(
            ex[j] * (c_vals[j] + x[j] - am.log(s)) for j in range(10)
        )
        self.constraints["c1"] = ex[0] + 2 * ex[1] + 2 * ex[2] + ex[5] + ex[9] - 2
        self.constraints["c2"] = ex[3] + 2 * ex[4] + ex[5] + ex[6] - 1
        self.constraints["c3"] = ex[2] + ex[6] + ex[7] + 2 * ex[8] + ex[9] - 1


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs111")
model.add_component("hs111", 1, HS111())
if args.build:
    model.build_module()
model.initialize()

opt = am.Optimizer(model)
opt.optimize(
    {
        "max_iterations": 300,
        "filter_line_search": True,
        "convergence_tolerance": 1e-8,
        "max_line_search_iterations": 30,
    }
)
# f* = -47.7610
