"""
HS106: Heat exchanger design (8 vars, 6 inequality constraints).
  min  x1 + x2 + x3
  s.t. 1 - 0.0025*(x4 + x6) >= 0
       1 - 0.0025*(x5 + x7 - x4) >= 0
       1 - 0.01*(x8 - x5) >= 0
       x1*x6 - 833.33252*x4 - 100*x1 + 83333.333 >= 0
       x2*x7 - 1250*x5 - x2*x4 + 1250*x4 >= 0
       x3*x8 - 1250000 - x3*x5 + 2500*x5 >= 0
  100 <= x1 <= 10000
  1000 <= x2, x3 <= 10000
  10 <= x4..x8 <= 1000
  x0 = (5000, 5000, 5000, 200, 350, 150, 225, 425)
  f* = 7049.3309
"""

import amigo as am
import argparse


class HS106(am.Component):
    def __init__(self):
        super().__init__()
        lb = [100, 1000, 1000, 10, 10, 10, 10, 10]
        ub = [10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000]
        x0 = [5000, 5000, 5000, 200, 350, 150, 225, 425]
        for i in range(8):
            self.add_input(f"x{i+1}", value=x0[i], lower=lb[i], upper=ub[i])
        self.add_objective("obj")
        for i in range(6):
            self.add_constraint(f"c{i+1}", lower=0.0, upper=float("inf"))

    def compute(self):
        x = [self.inputs[f"x{i+1}"] for i in range(8)]

        self.objective["obj"] = x[0] + x[1] + x[2]

        self.constraints["c1"] = 1 - 0.0025 * (x[3] + x[5])
        self.constraints["c2"] = 1 - 0.0025 * (x[4] + x[6] - x[3])
        self.constraints["c3"] = 1 - 0.01 * (x[7] - x[4])
        self.constraints["c4"] = x[0] * x[5] - 833.33252 * x[3] - 100 * x[0] + 83333.333
        self.constraints["c5"] = x[1] * x[6] - 1250 * x[4] - x[1] * x[3] + 1250 * x[3]
        self.constraints["c6"] = x[2] * x[7] - 1250000 - x[2] * x[4] + 2500 * x[4]


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs106")
model.add_component("hs106", 1, HS106())
if args.build:
    model.build_module()
model.initialize()

opt = am.Optimizer(model)
opt.optimize(
    {
        "max_iterations": 200,
        "filter_line_search": True,
        "convergence_tolerance": 1e-8,
        "max_line_search_iterations": 30,
    }
)
# f* = 7049.3309
