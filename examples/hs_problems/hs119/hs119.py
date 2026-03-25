"""
HS119: Largest HS problem (16 vars, 8 equality constraints).
  min  sum of 46 quartic product terms
  s.t. 8 linear equalities
  0 <= x_i <= 5, x0 = (10,...,10) projected to 5
  f* = 244.8997

  Stress test: largest variable count in HS set, dense quartic Hessian.
"""

import amigo as am
import argparse

# Sparse index pairs for the 46 product terms (x_i, x_j), 0-indexed
_PAIRS = [
    (0, 0),
    (0, 3),
    (0, 6),
    (0, 7),
    (0, 15),
    (1, 1),
    (1, 2),
    (1, 6),
    (1, 9),
    (2, 2),
    (2, 6),
    (2, 8),
    (2, 9),
    (2, 13),
    (3, 3),
    (3, 6),
    (3, 10),
    (3, 14),
    (4, 4),
    (4, 5),
    (4, 9),
    (4, 11),
    (4, 15),
    (5, 5),
    (5, 7),
    (5, 14),
    (6, 6),
    (6, 10),
    (6, 12),
    (7, 7),
    (7, 9),
    (7, 14),
    (8, 8),
    (8, 11),
    (8, 15),
    (9, 9),
    (9, 13),
    (10, 10),
    (10, 12),
    (10, 11),
    (11, 13),
    (12, 12),
    (12, 13),
    (13, 13),
    (14, 14),
    (15, 15),
]

# Constraint coefficients: s[k] = sum_j A[k][j]*x[j] + x[k+8] = c[k]
_A = [
    [0.22, 0.20, 0.19, 0.25, 0.15, 0.11, 0.12, 0.13],
    [-1.46, 0, -1.30, 1.82, -1.15, 0, 0.80, 0],
    [1.29, -0.89, 0, 0, -1.16, -0.96, 0, -0.49],
    [-1.10, -1.06, 0.95, -0.54, 0, -1.78, -0.41, 0],
    [0, 0, 0, -1.43, 1.51, 0.59, -0.33, -0.43],
    [0, -1.72, -0.33, 0, 1.62, 1.24, 0.21, -0.26],
    [1.12, 0, 0, 0.31, 0, 0, 1.12, 0],
    [0, 0.45, 0.26, -1.10, 0.58, 0, -1.03, 0.10],
]
_C = [2.5, 1.1, -3.1, -3.5, 1.3, 2.1, 2.3, -1.5]


class HS119(am.Component):
    def __init__(self):
        super().__init__()
        for i in range(16):
            self.add_input(f"x{i+1}", value=5.0, lower=0.0, upper=5.0)
        self.add_objective("obj")
        for i in range(8):
            self.add_constraint(f"h{i+1}", lower=0.0, upper=0.0)

    def compute(self):
        x = [self.inputs[f"x{i+1}"] for i in range(16)]

        # Objective: sum of 46 products (x_i^2 + x_i + 1)*(x_j^2 + x_j + 1)
        def q(xi):
            return xi**2 + xi + 1

        obj = 0
        for i, j in _PAIRS:
            obj = obj + q(x[i]) * q(x[j])
        self.objective["obj"] = obj

        # Equality constraints: A*x[0:8] + x[8+k] = c[k]
        for k in range(8):
            s = sum(_A[k][j] * x[j] for j in range(8)) + x[8 + k]
            self.constraints[f"h{k+1}"] = s - _C[k]


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs119")
model.add_component("hs119", 1, HS119())
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
# f* = 244.8997
