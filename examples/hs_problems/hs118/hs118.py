"""
HS118: Large LP-like problem (15 vars, 29 ineq)
  min  sum of quadratic terms in x1..x15
  s.t. 24 two-sided difference constraints + 5 row-sum constraints
       bounds on all variables
  x0 = (20,55,15,20,60,20,20,60,20,20,60,20,20,60,20)
  f* = 664.82045
"""

import amigo as am
import argparse


class HS118(am.Component):
    def __init__(self):
        super().__init__()
        x0 = [20, 55, 15, 20, 60, 20, 20, 60, 20, 20, 60, 20, 20, 60, 20]
        lb = [8, 43, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ub = [21, 57, 16, 90, 120, 60, 90, 120, 60, 90, 120, 60, 90, 120, 60]
        for i in range(15):
            self.add_input(
                f"x{i+1}", value=float(x0[i]), lower=float(lb[i]), upper=float(ub[i])
            )
        self.add_objective("obj")
        # 12 two-sided difference constraints (split into 24 one-sided)
        for i in range(24):
            self.add_constraint(f"d{i+1}", lower=0.0, upper=float("inf"))
        # 5 summation constraints
        for i in range(5):
            self.add_constraint(f"s{i+1}", lower=0.0, upper=float("inf"))

    def compute(self):
        x = [self.inputs[f"x{i+1}"] for i in range(15)]

        self.objective["obj"] = (
            2.3 * x[0]
            + 0.0001 * x[0] ** 2
            + 1.7 * x[1]
            + 0.0001 * x[1] ** 2
            + 2.2 * x[2]
            + 0.00015 * x[2] ** 2
            + 2.3 * x[3]
            + 0.0001 * x[3] ** 2
            + 1.7 * x[4]
            + 0.0001 * x[4] ** 2
            + 2.2 * x[5]
            + 0.00015 * x[5] ** 2
            + 2.3 * x[6]
            + 0.0001 * x[6] ** 2
            + 1.7 * x[7]
            + 0.0001 * x[7] ** 2
            + 2.2 * x[8]
            + 0.00015 * x[8] ** 2
            + 2.3 * x[9]
            + 0.0001 * x[9] ** 2
            + 1.7 * x[10]
            + 0.0001 * x[10] ** 2
            + 2.2 * x[11]
            + 0.00015 * x[11] ** 2
            + 2.3 * x[12]
            + 0.0001 * x[12] ** 2
            + 1.7 * x[13]
            + 0.0001 * x[13] ** 2
            + 2.2 * x[14]
            + 0.00015 * x[14] ** 2
        )

        # Difference constraints: 0 <= x[j] - x[i] + 7 <= upper
        # Column 1 (indices 0,3,6,9,12): upper = 13
        # Column 2 (indices 1,4,7,10,13): upper = 14
        # Column 3 (indices 2,5,8,11,14): upper = 13
        upper = [13, 14, 13]  # per column
        k = 0
        for group in range(4):  # 4 groups of differences
            for col in range(3):
                i = group * 3 + col  # source index
                j = (group + 1) * 3 + col  # target index
                diff = x[j] - x[i] + 7
                self.constraints[f"d{k+1}"] = diff  # >= 0 (lower)
                self.constraints[f"d{k+2}"] = upper[col] - diff  # >= 0 (upper)
                k += 2

        # Summation constraints: sum of each row >= threshold
        thresholds = [60, 50, 70, 85, 100]
        for row in range(5):
            s = x[row * 3] + x[row * 3 + 1] + x[row * 3 + 2]
            self.constraints[f"s{row+1}"] = s - thresholds[row]


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs118")
model.add_component("hs118", 1, HS118())
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
# f* = 664.82045
