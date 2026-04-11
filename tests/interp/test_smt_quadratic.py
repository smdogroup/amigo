"""pytest for the ExternalSMTComponent example.

Verifies that a 1-input KRG surrogate is correctly wired into an Amigo
model via ExternalSMTComponent, and that the optimizer recovers the known
solution x* = 0.5, y* = 0.25 for:

    minimize  x
    subject to  KRG(x) = y   (surrogate equality constraint)
                y >= 0.25    (bound drives x to 0.5)
                x in [0, 2]

Run with:
    pytest test_smt_quadratic.py -v
"""

import numpy as np
import pytest

import amigo as am
from amigo.interp import ExternalSMTComponent
from smt.surrogate_models import KRG


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def trained_krg():
    """KRG surrogate trained on y = x^2 over [0, 2]."""
    x_train = np.linspace(0.0, 2.0, 20).reshape(-1, 1)
    y_train = x_train**2
    sm = KRG(theta0=[1e-2], print_global=False)
    sm.set_training_values(x_train, y_train)
    sm.train()
    return sm


@pytest.fixture(scope="module")
def opt_result(trained_krg):
    """Build and solve the model once; return (xvec, model)."""
    sm = trained_krg
    smt_ext = ExternalSMTComponent(num_points=1, num_inputs=1, smt_model=sm)

    class Source(am.Component):
        def __init__(self):
            super().__init__()
            self.add_input("x")
            self.add_input("y")

    class SMTConstraintPlaceholder(am.Component):
        def __init__(self):
            super().__init__()
            self.add_constraint("res")

    class Objective(am.Component):
        def __init__(self):
            super().__init__()
            self.add_input("x")
            self.add_objective("obj")

        def compute(self):
            self.objective["obj"] = self.inputs["x"]

    model = am.Model("smt_quadratic_test")
    model.add_component("src", 1, Source())
    model.add_component("smt_con", 1, SMTConstraintPlaceholder())
    model.add_component("obj", 1, Objective())
    model.link("src.x", "obj.x")
    model.add_external_component(
        "smt",
        smt_ext,
        inputs=["src.x", "src.y"],
        constraints=["smt_con.res"],
    )
    model.build_module()
    model.initialize()

    xvec = model.create_vector()
    lower = model.create_vector()
    upper = model.create_vector()

    xvec["src.x"] = 1.0
    xvec["src.y"] = float(sm.predict_values(np.array([[1.0]]))[0, 0])

    lower["src.x"] = 0.0
    upper["src.x"] = 2.0
    lower["src.y"] = 0.25
    upper["src.y"] = float("inf")

    opt = am.Optimizer(model, xvec, lower=lower, upper=upper)
    data = opt.optimize(
        {
            "max_iterations": 200,
            "initial_barrier_param": 1.0,
            "convergence_tolerance": 1e-10,
            "max_line_search_iterations": 10,
            "init_affine_step_multipliers": False,
            "init_least_squares_multipliers": False,
        }
    )
    return xvec, data


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_surrogate_accuracy(trained_krg):
    """KRG should interpolate y = x^2 to machine precision at training points."""
    sm = trained_krg
    x_check = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
    pred = sm.predict_values(x_check).flatten()
    expected = x_check.flatten() ** 2
    np.testing.assert_allclose(pred, expected, atol=1e-6)


def test_converged(opt_result):
    """Optimizer must report convergence."""
    _, data = opt_result
    assert data["converged"], "Optimizer did not converge"


def test_optimal_x(opt_result):
    """Optimal x should be 0.5 (active lower bound y=0.25 via KRG(x)=x^2)."""
    xvec, _ = opt_result
    x_opt = xvec["src.x"][0]
    assert abs(x_opt - 0.5) < 1e-4, f"x* = {x_opt:.6f}, expected 0.5"


def test_optimal_y(opt_result):
    """Optimal y should be 0.25 (active lower bound)."""
    xvec, _ = opt_result
    y_opt = xvec["src.y"][0]
    assert abs(y_opt - 0.25) < 1e-4, f"y* = {y_opt:.6f}, expected 0.25"


def test_surrogate_constraint_satisfied(opt_result, trained_krg):
    """Surrogate residual KRG(x*) - y* must be near zero at the solution."""
    xvec, _ = opt_result
    x_opt = xvec["src.x"][0]
    y_opt = xvec["src.y"][0]
    pred = trained_krg.predict_values(np.array([[x_opt]]))[0, 0]
    residual = abs(pred - y_opt)
    assert residual < 1e-6, f"Surrogate residual = {residual:.2e}, expected < 1e-6"
