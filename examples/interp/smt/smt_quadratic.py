"""Minimal ExternalSMTComponent verification example.

Problem: minimize x  subject to  KRG(x) = y,  y >= 0.25
where KRG is trained on y = x^2 over x in [0, 2].

Known answer: x* = 0.5, y* = 0.25  (active lower bound on y)

This example isolates ExternalSMTComponent from application complexity
to verify that the surrogate constraint is wired and evaluated correctly.
"""

import amigo as am
import numpy as np
from smt.surrogate_models import KRG
from amigo.interp import ExternalSMTComponent

# ── Train a 1-input KRG surrogate on y = x² ──────────────────────────────────
x_train = np.linspace(0.0, 2.0, 20).reshape(-1, 1)
y_train = x_train**2

sm = KRG(theta0=[1e-2], print_global=False)
sm.set_training_values(x_train, y_train)
sm.train()

# Sanity check: surrogate should predict ≈ 0.25 at x = 0.5
pred = sm.predict_values(np.array([[0.5]]))
print(f"Surrogate check: KRG(0.5) = {pred[0,0]:.6f}  (expect ~0.25)")

# ── ExternalSMTComponent ──────────────────────────────────────────────────────
# num_points=1, num_inputs=1: the x-vector layout inside evaluate() is
#   x[0:1] = surrogate input (src.x)
#   x[1:2] = surrogate output (src.y)
# and the constraint enforced is:  KRG(x[0]) - x[1] = 0
smt_ext = ExternalSMTComponent(num_points=1, num_inputs=1, smt_model=sm)


# ── Amigo components ──────────────────────────────────────────────────────────
class Source(am.Component):
    """Holds the design variable x and the surrogate output y."""

    def __init__(self):
        super().__init__()
        self.add_input("x")
        self.add_input("y")


class SMTConstraintPlaceholder(am.Component):
    """Declares the surrogate residual constraint without a compute() override.

    No compute() method means is_compute_empty() returns True, so Amigo
    generates no C++ group for this component.  The ExternalSMTComponent is
    the sole writer of 'res'.
    """

    def __init__(self):
        super().__init__()
        self.add_constraint("res")


class Objective(am.Component):
    """Minimize x."""

    def __init__(self):
        super().__init__()
        self.add_input("x")
        self.add_objective("obj")

    def compute(self):
        self.objective["obj"] = self.inputs["x"]


# ── Assemble model ────────────────────────────────────────────────────────────
model = am.Model("smt_quadratic")
model.add_component("src", 1, Source())
model.add_component("smt_con", 1, SMTConstraintPlaceholder())
model.add_component("obj", 1, Objective())
model.link("src.x", "obj.x")

# The inputs list must match ExternalSMTComponent's x-vector layout:
#   first num_inputs*num_points entries  → surrogate inputs (column-major by input)
#   last  num_points entries             → surrogate outputs
model.add_external_component(
    "smt",
    smt_ext,
    inputs=["src.x", "src.y"],
    constraints=["smt_con.res"],
)

model.build_module()
model.initialize()

print(f"Num variables:   {model.num_variables}")
print(f"Num constraints: {model.num_constraints}")

# ── Initial guess ─────────────────────────────────────────────────────────────
xvec = model.create_vector()
lower = model.create_vector()
upper = model.create_vector()

xvec["src.x"] = 1.0
xvec["src.y"] = float(
    sm.predict_values(np.array([[1.0]]))[0, 0]
)  # on surrogate → res ≈ 0

lower["src.x"] = 0.0
upper["src.x"] = 2.0
lower["src.y"] = 0.25  # y >= 0.25  drives x >= 0.5 via the surrogate constraint
upper["src.y"] = float("inf")

# ── Optimize ──────────────────────────────────────────────────────────────────
opt = am.Optimizer(model, xvec, lower=lower, upper=upper)
opt.optimize(
    {
        "max_iterations": 200,
        "initial_barrier_param": 1.0,
        "convergence_tolerance": 1e-10,
        "max_line_search_iterations": 10,
        "init_affine_step_multipliers": False,
        "init_least_squares_multipliers": False,
    }
)

x_opt = xvec["src.x"][0]
y_opt = xvec["src.y"][0]
print(f"\nResult:  x = {x_opt:.6f}  (expect 0.5)")
print(f"         y = {y_opt:.6f}  (expect 0.25)")
print(f"  KRG(x*) = {sm.predict_values(np.array([[x_opt]]))[0,0]:.6f}  (expect ~0.25)")
