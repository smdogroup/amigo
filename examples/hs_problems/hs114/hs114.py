"""
HS114: Alkylation process (10 vars, 8 ineq + 3 eq constraints).
  min  5.04*x1 + 0.035*x2 + 10*x3 + 3.36*x5 - 0.063*x4*x7
  s.t. 8 nonlinear inequalities + 3 nonlinear equalities (see below)
  f* = -1768.80696

  Ref: Hock & Schittkowski, Test Examples for Nonlinear Programming Codes,
       Lecture Notes in Economics and Mathematical Systems, v. 187, p. 123.
"""

import amigo as am
import argparse


class HS114(am.Component):
    def __init__(self):
        super().__init__()
        a = 0.99
        b = 0.9
        self._a = a
        self._b = b

        lb = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 85, 90, 3, 1.2, 145]
        ub = [2000, 16000, 120, 5000, 2000, 93, 95, 12, 4, 162]
        x0 = [1745, 12000, 110, 3048, 1974, 89.2, 92.8, 8, 3.6, 145]
        for i in range(10):
            self.add_input(f"x{i+1}", value=x0[i], lower=lb[i], upper=ub[i])
        self.add_objective("obj")
        # 8 inequality constraints
        for i in range(8):
            self.add_constraint(f"g{i+1}", lower=0.0, upper=float("inf"))
        # 3 equality constraints
        for i in range(3):
            self.add_constraint(f"h{i+1}", lower=0.0, upper=0.0)

    def compute(self):
        x = [self.inputs[f"x{i+1}"] for i in range(10)]
        a = self._a
        b = self._b

        # Objective
        self.objective["obj"] = (
            5.04 * x[0] + 0.035 * x[1] + 10 * x[2] + 3.36 * x[4] - 0.063 * x[3] * x[6]
        )

        # Intermediates
        G1 = 35.82 - 0.222 * x[9] - b * x[8]
        G2 = -133 + 3 * x[6] - a * x[9]
        G5 = 1.12 * x[0] + 0.13167 * x[0] * x[7] - 0.00667 * x[0] * x[7] ** 2 - a * x[3]
        G6 = 57.425 + 1.098 * x[7] - 0.038 * x[7] ** 2 + 0.325 * x[5] - a * x[6]

        # Inequality constraints (>= 0)
        self.constraints["g1"] = G1
        self.constraints["g2"] = G2
        self.constraints["g3"] = -G1 + x[8] * (1 / b - b)
        self.constraints["g4"] = -G2 + (1 / a - a) * x[9]
        self.constraints["g5"] = G5
        self.constraints["g6"] = G6
        self.constraints["g7"] = -G5 + (1 / a - a) * x[3]
        self.constraints["g8"] = -G6 + (1 / a - a) * x[6]

        # Equality constraints (= 0)
        self.constraints["h1"] = 1.22 * x[3] - x[0] - x[4]
        self.constraints["h2"] = 98000 * x[2] / (x[3] * x[8] + 1000 * x[2]) - x[5]
        self.constraints["h3"] = (x[1] + x[4]) / x[0] - x[7]


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs114")
model.add_component("hs114", 1, HS114())
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
# f* = -1768.80696
