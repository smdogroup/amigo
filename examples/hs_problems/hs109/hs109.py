"""
HS109: Structural mechanics (9 vars, 4 ineq + 6 eq constraints).
  min  3*x1 + 1e-6*x1^3 + 2*x2 + 0.522074e-6*x2^3
  s.t. 4 nonlinear inequalities, 6 nonlinear equalities (sin/cos)
  f* = 5362.0681 (approx)

  Challenging: trig functions, infeasible starting point, highly nonlinear
  equalities with products of variables and sin/cos.
"""

import amigo as am
import argparse
import math


class HS109(am.Component):
    def __init__(self):
        super().__init__()
        # Parameters
        self._pa = 50.176
        self._pb = math.sin(0.25)
        self._pc = math.cos(0.25)

        # Bounds and starting points (x0 projected into bounds)
        lb = [0.0, 0.0, -0.55, -0.55, 196, 196, 196, -400, -400]
        ub = [1e20, 1e20, 0.55, 0.55, 252, 252, 252, 800, 800]
        x0 = [0.0, 0.0, 0.0, 0.0, 224, 224, 224, 0.0, 0.0]
        for i in range(9):
            self.add_input(f"x{i+1}", value=x0[i], lower=lb[i], upper=ub[i])
        self.add_objective("obj")
        for i in range(4):
            self.add_constraint(f"g{i+1}", lower=0.0, upper=float("inf"))
        for i in range(6):
            self.add_constraint(f"h{i+1}", lower=0.0, upper=0.0)

    def compute(self):
        x = [self.inputs[f"x{i+1}"] for i in range(9)]
        a = self._pa
        b = self._pb
        c = self._pc

        # Objective
        self.objective["obj"] = (
            3 * x[0] + 1e-6 * x[0] ** 3 + 2 * x[1] + 0.522074e-6 * x[1] ** 3
        )

        # Inequality constraints (>= 0)
        self.constraints["g1"] = x[3] - x[2] + 0.55
        self.constraints["g2"] = x[2] - x[3] + 0.55
        self.constraints["g3"] = 2250000 - x[0] ** 2 - x[7] ** 2
        self.constraints["g4"] = 2250000 - x[1] ** 2 - x[8] ** 2

        # Equality constraints (= 0)
        self.constraints["h1"] = (
            x[4] * x[5] * am.sin(-x[2] - 0.25)
            + x[4] * x[6] * am.sin(-x[3] - 0.25)
            + 2 * b * x[4] ** 2
            - a * x[0]
            + 400 * a
        )
        self.constraints["h2"] = (
            x[4] * x[5] * am.sin(x[2] - 0.25)
            + x[5] * x[6] * am.sin(x[2] - x[3] - 0.25)
            + 2 * b * x[5] ** 2
            - a * x[1]
            + 400 * a
        )
        self.constraints["h3"] = (
            x[4] * x[6] * am.sin(x[3] - 0.25)
            + x[5] * x[6] * am.sin(x[3] - x[2] - 0.25)
            + 2 * b * x[6] ** 2
            + 881.779 * a
        )
        self.constraints["h4"] = (
            a * x[7]
            + x[4] * x[5] * am.cos(-x[2] - 0.25)
            + x[4] * x[6] * am.cos(-x[3] - 0.25)
            - 200 * a
            - 2 * c * x[4] ** 2
            + 0.7533e-3 * a * x[4] ** 2
        )
        self.constraints["h5"] = (
            a * x[8]
            + x[4] * x[5] * am.cos(x[2] - 0.25)
            + x[5] * x[6] * am.cos(x[2] - x[3] - 0.25)
            - 2 * c * x[5] ** 2
            + 0.7533e-3 * a * x[5] ** 2
            - 200 * a
        )
        self.constraints["h6"] = (
            x[4] * x[6] * am.cos(x[3] - 0.25)
            + x[5] * x[6] * am.cos(x[3] - x[2] - 0.25)
            - 2 * c * x[6] ** 2
            + 22.938 * a
            + 0.7533e-3 * a * x[6] ** 2
        )


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs109")
model.add_component("hs109", 1, HS109())
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
# f* = 5362.0681
