import amigo as am
import numpy as np


def dot_product(x, y, n=1):
    val = x[0] * y[0]
    for i in range(1, n):
        val = val + x[i] * y[i]
    return val


def mat_vec(A, x, m=1, n=1):
    return [A[0][0] * x[0] + A[0][1] * x[1], A[1][0] * x[0] + A[1][1] * x[1]]


def mat_vec_transpose(A, x, m=1, n=1):
    return [A[0][0] * x[0] + A[1][0] * x[1], A[0][1] * x[0] + A[1][1] * x[1]]


class Basis:
    def __init__(self, names, nnodes=1, kind="input"):
        if isinstance(names, (list, tuple)):
            self.names = names
        elif isinstance(names, str):
            self.names = [names]
        self.nnodes = nnodes
        self.kind = kind

    def add_declarations(self, comp):
        """Add the declarations to the component"""

        if self.kind == "input":
            for name in self.names:
                comp.add_input(name, shape=(self.nnodes,))
        elif self.kind == "data":
            for name in self.names:
                comp.add_data(name, shape=(self.nnodes,))


class H1Basis2D(Basis):
    def __init__(self, names, nnodes=1, kind="input"):
        super().__init__(names, nnodes=nnodes, kind=kind)

    def transform(self, detJ, Jinv, orig):
        soln = {}
        for name in orig:
            value = orig[name]["value"]
            grad = orig[name]["grad"]
            soln[name] = {
                "value": value,
                "grad": mat_vec_transpose(Jinv, grad, n=2, m=2),
            }

        return soln

    def compute_transform(self, geo):
        if "x" not in geo or "y" not in geo:
            raise ValueError("Coordinates not defined")

        x_xi, x_eta = geo["x"]["grad"]
        y_xi, y_eta = geo["y"]["grad"]

        detJ = x_xi * y_eta - x_eta * y_xi
        Jinv = [[y_eta / detJ, -x_eta / detJ], [-y_xi / detJ, x_xi / detJ]]

        return detJ, Jinv


class LinearH1TriangleBasis(H1Basis2D):
    def __init__(self, names, kind="input"):
        super().__init__(names, nnodes=3, kind=kind)

    def eval(self, comp, pt):
        xi = pt[0]
        eta = pt[1]

        N = [1.0 - xi - eta, xi, eta]
        Nxi = [-1.0, 1.0, 0.0]
        Neta = [-1.0, 0.0, 1.0]

        soln = {}
        for name in self.names:
            if self.kind == "input":
                u = comp.inputs[name]
            elif self.kind == "data":
                u = comp.data[name]

            soln[name] = {
                "value": dot_product(u, N, n=self.nnodes),
                "grad": [
                    dot_product(u, Nxi, n=self.nnodes),
                    dot_product(u, Neta, n=self.nnodes),
                ],
            }

        return soln


class LinearH1QuadBasis(H1Basis2D):
    def __init__(self, names, kind="input"):
        super().__init__(names, nnodes=4, kind=kind)

    def eval(self, comp, pt):
        xi = pt[0]
        eta = pt[1]

        N = [
            0.25 * (1.0 - xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 + eta),
            0.25 * (1.0 - xi) * (1.0 + eta),
        ]
        Nxi = [
            -0.25 * (1.0 - eta),
            0.25 * (1.0 - eta),
            0.25 * (1.0 + eta),
            -0.25 * (1.0 + eta),
        ]
        Neta = [
            -0.25 * (1.0 - xi),
            -0.25 * (1.0 + xi),
            0.25 * (1.0 + xi),
            0.25 * (1.0 - xi),
        ]

        soln = {}
        for name in self.names:
            if self.kind == "input":
                u = comp.inputs[name]
            elif self.kind == "data":
                u = comp.data[name]

            soln[name] = {
                "value": dot_product(u, N, n=self.nnodes),
                "grad": [
                    dot_product(u, Nxi, n=self.nnodes),
                    dot_product(u, Neta, n=self.nnodes),
                ],
            }

        return soln


class Quadrature:
    def get_args(self):
        return []


class TriangleQuadrature(Quadrature):
    def __init__(self):
        self.weights = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
        self.points = [[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]]
        self.args = [{"n": 0}, {"n": 1}, {"n": 2}]

    def get_args(self):
        return self.args

    def get_point(self, n=0):
        return self.weights[n], self.points[n]


class QuadQuadrature(Quadrature):
    def __init__(self):
        self.weights = [1.0, 1.0, 1.0, 1.0]

        p = -1.0 / np.sqrt(3.0)
        self.points = [[-p, -p], [p, -p], [-p, p], [p, p]]
        self.args = [{"n": 0}, {"n": 1}, {"n": 2}, {"n": 3}]

    def get_args(self):
        return self.args

    def get_point(self, n=0):
        return self.weights[n], self.points[n]


class SolutionSpace:
    def __init__(self, mapping):
        self._allowed_spaces = ["H1", "L2", "H(div)", "H(curl)"]

        self.names = {}
        for name in mapping:
            if not mapping[name] in self._allowed_spaces:
                raise ValueError(f"{mapping[name]} not an allowed solution space")
            if not mapping[name] in self.names:
                self.names[mapping[name]] = [name]
            else:
                self.names[mapping[name]].append(name)

    def get_spaces(self):
        return list(self.names.keys())

    def get_names(self, space):
        return self.names[space]


class BasisCollection:
    def __init__(self):
        self.basis = []

    def add_basis(self, basis):
        self.basis.append(basis)
