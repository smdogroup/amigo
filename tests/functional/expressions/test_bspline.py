"""
B-splines
---------

- Compiled checks of the B-spline, its basis functions, and the inverse of a coordinate spline against scipy
- The gradient and the Hessian against finite differences
- The Jacobians of the outputs and of the gradient with respect to the spline tables against finite differences
"""

import contextlib
import sys

import numpy as np
import pytest
from scipy.interpolate import BSpline

import amigo as am

COEF = np.array([0.3, -1.2, 0.8, 2.5, -0.4, 1.1, 0.6, -0.9, 1.7, 0.2])

TABLE = np.array(
    [
        [0.5, -0.3, 1.0],
        [1.5, 0.7, -0.4],
        [-0.2, 2.0, 0.3],
        [0.9, -1.1, 0.8],
        [0.1, 0.4, -0.6],
        [1.2, -0.8, 0.5],
        [-0.5, 1.3, 0.9],
        [0.7, 0.2, -1.4],
    ]
)

CX = np.array([0.0, 0.2, 0.5, 0.9, 1.4, 2.0, 2.4, 2.9, 3.3, 3.6])

CX2 = np.array(
    [[-1.0, 0.1], [-0.4, 0.2], [0.3, 0.35], [0.9, 0.5], [1.2, 0.8], [2.0, 0.9]]
)

TENSOR = np.array(
    [
        [0.4, -0.2, 1.1, 0.3, -0.7],
        [1.0, 0.6, -0.5, 0.8, 0.2],
        [-0.3, 1.4, 0.9, -1.0, 0.5],
        [0.7, -0.9, 0.2, 1.2, -0.4],
        [0.1, 0.8, -0.6, 0.4, 1.3],
    ]
)

PIECES = [
    "spline",
    "spline_column",
    "basis",
    "inverse",
    "inverse_column",
    "clamped",
    "tensor",
]


class Splines(am.Component):
    """Every spline piece as an output, with their sum as the objective"""

    def __init__(self):
        super().__init__()

        self.add_data("coef", shape=len(COEF))
        self.add_data("table", shape=TABLE.shape)
        self.add_data("cx", shape=len(CX))
        self.add_data("cx2", shape=CX2.shape)
        self.add_data("grid", shape=TENSOR.shape)

        self.add_input("x", shape=3)
        self.add_input("s")
        self.add_input("y")

        self.add_objective("obj")

        for name in PIECES:
            self.add_output(name)

        return

    def pieces(self):
        """Every spline piece by name"""
        x = self.inputs["x"]
        s = self.inputs["s"]
        y = self.inputs["y"]

        # A tensor product of a basis in y and the columns of a spline in s
        tensor = 0.0

        for j in range(TENSOR.shape[1]):
            column = am.bspline(self.data["grid"], s, 4, column=j)
            tensor = tensor + am.bspline_basis(y, j, TENSOR.shape[1], 4) * column

        return {
            "spline": am.bspline(self.data["coef"], s, 4) * y,
            "spline_column": am.bspline(self.data["table"], s * y, 3, column=1),
            "basis": am.bspline_basis(s, 3, 10, 4) * y + am.bspline_basis(y, 0, 6, 3),
            "inverse": am.bspline_inverse(self.data["cx"], 0.2 + 3.0 * y, 4) * s,
            "inverse_column": am.bspline_inverse(self.data["cx2"], x[0], 3, column=1),
            "clamped": am.bspline_inverse(self.data["cx"], x[1] - 5.0, 4)
            + am.bspline_inverse(self.data["cx"], x[2] + 50.0, 4),
            "tensor": tensor,
        }

    def compute(self):
        pieces = self.pieces()

        total = pieces[PIECES[0]]

        for name in PIECES[1:]:
            total = total + pieces[name]

        self.objective["obj"] = total

        return

    def compute_output(self):
        pieces = self.pieces()

        for name in PIECES:
            self.outputs[name] = pieces[name]

        return


def knots(k, m):
    """Open uniform knots of m control points of order k on the unit interval"""
    inner = np.linspace(0.0, 1.0, m - k + 2)[1:-1]

    return np.concatenate([np.zeros(k), inner, np.ones(k)])


def spline(coef, k, t):
    """The scipy spline of order k through the coefficients at t"""
    return float(BSpline(knots(k, len(coef)), coef, k - 1)(t))


def basis(index, m, k, t):
    """One basis function of m control points of order k at t"""
    return spline(np.eye(m)[index], k, t)


def inverse(coef, k, x):
    """Root of the increasing coordinate spline by bisection, held at the ends outside its range"""
    if x <= coef[0]:
        return 0.0

    if x >= coef[-1]:
        return 1.0

    lo = 0.0
    hi = 1.0

    for step in range(80):
        mid = 0.5 * (lo + hi)

        if spline(coef, k, mid) < x:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def reference(x, s, y):
    """Every spline piece by name from scipy"""
    tensor = 0

    for j in range(5):
        tensor = tensor + basis(j, 5, 4, y) * spline(TENSOR[:, j], 4, s)

    return {
        "spline": spline(COEF, 4, s) * y,
        "spline_column": spline(TABLE[:, 1], 3, s * y),
        "basis": basis(3, 10, 4, s) * y + basis(0, 6, 3, y),
        "inverse": inverse(CX, 4, 0.2 + 3.0 * y) * s,
        "inverse_column": inverse(CX2[:, 1], 3, x[0]),
        "clamped": inverse(CX, 4, x[1] - 5.0) + inverse(CX, 4, x[2] + 50.0),
        "tensor": tensor,
    }


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """The model built in a temporary directory with its spline tables set"""
    tmp_path = tmp_path_factory.mktemp("bspline")

    model = am.Model("bspline_test")
    model.add_component("ops", 1, Splines())

    # Build in a temporary directory so the repository CMakeLists.txt stays untouched
    with contextlib.chdir(tmp_path):
        model.build_module()

    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))

    model.initialize()

    # The spline tables
    data = model.get_data_vector()

    data["ops.coef"] = COEF
    data["ops.table"] = TABLE
    data["ops.cx"] = CX
    data["ops.cx2"] = CX2
    data["ops.grid"] = TENSOR

    data.get_vector().copy_host_to_device()

    return model


def point(model, x, s, y):
    """Design vector at x, s, and y"""
    vec = model.create_vector()

    vec["ops.x"] = x
    vec["ops.s"] = s
    vec["ops.y"] = y

    return vec


def gradient(model, x, s, y):
    """Gradient of the objective with respect to x, s, and y"""
    g = model.create_vector()
    model.eval_gradient(point(model, x, s, y), g)

    return np.concatenate([g["ops.x"].ravel(), g["ops.s"].ravel(), g["ops.y"].ravel()])


def state_indices(model):
    """Global indices of x, s, and y in order"""
    indices = []

    for name in ["ops.x", "ops.s", "ops.y"]:
        indices.append(np.asarray(model.get_indices(name)).ravel())

    return np.concatenate(indices)


def hessian(model, x, s, y):
    """Hessian of the objective with respect to x, s, and y"""
    mat = model.create_matrix()
    model.eval_hessian(point(model, x, s, y), mat)

    idx = state_indices(model)

    return am.tocsr(mat).toarray()[np.ix_(idx, idx)]


def total(v):
    """Sum of every reference piece at the point v of x, s, and y"""
    pieces = reference(v[:3], v[3], v[4])

    value = 0

    for name in PIECES:
        value = value + pieces[name]

    return value


def table_indices(model):
    """Data indices of every spline table in order"""
    indices = []

    for name in ["ops.coef", "ops.table", "ops.cx", "ops.cx2", "ops.grid"]:
        indices.append(np.asarray(model.get_indices(name)).ravel())

    return np.concatenate(indices)


def output_indices(model):
    """Indices of every output in order"""
    indices = []

    for name in PIECES:
        indices.append(np.asarray(model.get_indices(f"ops.{name}")).ravel())

    return np.concatenate(indices)


def outputs(model, x, s, y):
    """Every output at x, s, and y"""
    out = model.create_output_vector()
    model.compute_output(point(model, x, s, y), out)

    values = []

    for name in PIECES:
        values.append(np.ravel(out[f"ops.{name}"]))

    return np.concatenate(values)


def tables(model):
    """Entries of every spline table"""
    data = model.get_data_vector().get_vector()

    return data.get_array()[table_indices(model)].copy()


def set_tables(model, values):
    """Set the entries of every spline table"""
    data = model.get_data_vector().get_vector()

    data.get_array()[table_indices(model)] = values
    data.copy_host_to_device()


class TableDifferences:
    """Central differences of a function of the model along every table entry"""

    def __init__(self, model, h=1e-6):
        self.model = model
        self.h = h

    def __call__(self, function):
        base = tables(self.model)

        columns = []

        for j in range(len(base)):
            step = np.zeros(len(base))
            step[j] = self.h

            set_tables(self.model, base + step)
            up = function()

            set_tables(self.model, base - step)
            down = function()

            columns.append((up - down) / (2 * self.h))

        # The tables return to their values for the tests that follow
        set_tables(self.model, base)

        return np.column_stack(columns)


class OutputsAt:
    """Every output at a fixed point"""

    def __init__(self, model, v):
        self.model = model
        self.v = v

    def __call__(self):
        return outputs(self.model, self.v[:3], self.v[3], self.v[4])


class GradientAt:
    """Gradient of the objective at a fixed point"""

    def __init__(self, model, v):
        self.model = model
        self.v = v

    def __call__(self):
        return gradient(self.model, self.v[:3], self.v[3], self.v[4])


@pytest.mark.parametrize(
    "x, s, y",
    [
        # Interior spans of every spline
        ([0.3, -0.4, 1.2], 0.37, 0.62),
        # The first and last spans and an argument on an interior knot
        ([0.85, 0.2, -0.3], 0.02, 0.5),
    ],
)
def test_values_match_scipy(model, x, s, y):
    out = model.create_output_vector()
    model.compute_output(point(model, x, s, y), out)

    expected = reference(np.array(x), s, y)

    for name in PIECES:
        assert abs(out[f"ops.{name}"] - expected[name]) < 1e-12, name


def test_gradient_matches_differences(model):
    v = np.array([0.3, -0.4, 1.2, 0.37, 0.62])
    h = 1e-6

    # Central differences of the reference total along every coordinate
    fd = np.zeros(5)

    for j in range(5):
        e = np.eye(5)[j]
        fd[j] = (total(v + h * e) - total(v - h * e)) / (2 * h)

    assert np.allclose(gradient(model, v[:3], v[3], v[4]), fd, rtol=1e-7, atol=1e-8)


def test_hessian_matches_differences_of_the_gradient(model):
    v = np.array([0.3, -0.4, 1.2, 0.37, 0.62])
    h = 1e-6

    H = hessian(model, v[:3], v[3], v[4])

    # Central differences of the compiled gradient along every coordinate
    for j in range(5):
        e = np.eye(5)[j]

        up = v + h * e
        down = v - h * e

        plus = gradient(model, up[:3], up[3], up[4])
        minus = gradient(model, down[:3], down[3], down[4])
        column = (plus - minus) / (2 * h)

        assert np.allclose(H[:, j], column.ravel(), rtol=1e-6, atol=1e-6), j


def test_clamped_inverse_has_zero_derivatives(model):
    g = gradient(model, np.array([0.3, -0.4, 1.2]), 0.37, 0.62)
    H = hessian(model, np.array([0.3, -0.4, 1.2]), 0.37, 0.62)

    assert g[1] == 0.0 and g[2] == 0.0
    assert np.all(H[1, :] == 0.0) and np.all(H[2, :] == 0.0)


def test_output_jacobian_wrt_data_matches_differences(model):
    v = np.array([0.3, -0.4, 1.2, 0.37, 0.62])

    # The Jacobian of the outputs with respect to the spline tables
    problem = model.get_problem()
    jac = problem.create_output_jacobian_wrt_data()
    problem.output_jacobian_wrt_data(point(model, v[:3], v[3], v[4]).get_vector(), jac)

    rows = output_indices(model)
    J = am.tocsr(jac).toarray()[np.ix_(rows, table_indices(model))]

    differences = TableDifferences(model)
    fd = differences(OutputsAt(model, v))

    assert np.allclose(J, fd, rtol=1e-6, atol=1e-8)


def test_gradient_jacobian_wrt_data_matches_differences(model):
    v = np.array([0.3, -0.4, 1.2, 0.37, 0.62])

    # The Jacobian of the gradient with respect to the spline tables
    problem = model.get_problem()
    jac = problem.create_gradient_jacobian_wrt_data()
    problem.gradient_jacobian_wrt_data(
        point(model, v[:3], v[3], v[4]).get_vector(), jac
    )

    rows = state_indices(model)
    J = am.tocsr(jac).toarray()[np.ix_(rows, table_indices(model))]

    differences = TableDifferences(model)
    fd = differences(GradientAt(model, v))

    assert np.allclose(J, fd, rtol=1e-6, atol=1e-7)
