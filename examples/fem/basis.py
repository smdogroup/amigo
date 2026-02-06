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


def eval_monomials(p, xi, eta, exps):
    """out[k] = xi^i * eta^j for each (i,j)."""
    xi_pows = np.ones(p + 1)
    eta_pows = np.ones(p + 1)

    for a in range(1, p + 1):
        xi_pows[a] = xi_pows[a - 1] * xi
    for b in range(1, p + 1):
        eta_pows[b] = eta_pows[b - 1] * eta

    out = np.empty(len(exps), dtype=float)
    for k, (i, j) in enumerate(exps):
        out[k] = xi_pows[i] * eta_pows[j]

    return out


def eval_monomial_grad(p, xi, eta, exps):
    """grads[k] = [i * xi^{i-1} * eta^{j}, j * xi^{i} * eta^{j-1}]"""
    xi_pows = np.ones(p + 1)
    eta_pows = np.ones(p + 1)

    for a in range(1, p + 1):
        xi_pows[a] = xi_pows[a - 1] * xi
    for b in range(1, p + 1):
        eta_pows[b] = eta_pows[b - 1] * eta

    grad = np.zeros((len(exps), 2), dtype=float)
    for k, (i, j) in enumerate(exps):
        if i > 0:
            grad[k, 0] = i * xi_pows[i - 1] * eta_pows[j]
        if j > 0:
            grad[k, 1] = j * xi_pows[i] * eta_pows[j - 1]

    return grad


def build_vandermonde(n, p, pts, exps):
    # Build Vandermonde: V[a,k] = m_k(node_a)
    V = np.zeros((n, n), dtype=float)
    for a, (xi, eta) in enumerate(pts):
        V[a, :] = eval_monomials(p, xi, eta, exps)

    # Compute C = V^{-1}
    I = np.eye(n, dtype=float)
    return np.linalg.solve(V, I)


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


class LagrangeBasis2D(Basis):
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


class TriangleLagrangeBasis(LagrangeBasis2D):
    def __init__(self, p, names, kind="input"):
        if p < 0:
            raise ValueError(f"Degree {p} must be >= 0")

        self.p = p
        nnodes = (p + 1) * (p + 2) // 2
        super().__init__(names, nnodes=nnodes, kind=kind)

        self.pts = self._get_tri_nodes(self.p)
        self.exps = self._get_monomial_exponents(self.p)
        self.C = build_vandermonde(self.nnodes, self.p, self.pts, self.exps)

        return

    def _get_tri_nodes(self, p):
        """Get the node locations"""
        pts = []

        if p == 0:
            return [[1 / 3, 1 / 3]]
        else:
            # Set the vertices
            pts = [[0, 0], [1, 0], [0, 1]]

            # At points on the edges
            edges = [[0, 1], [1, 2], [2, 0]]
            for a, b in edges:
                for i in range(1, p):
                    t = i / p
                    xi = (1 - t) * pts[a][0] + t * pts[b][0]
                    eta = (1 - t) * pts[a][1] + t * pts[b][1]
                    pts.append([xi, eta])

            # Add the remaining points in the interior
            for i in range(1, p):
                for j in range(1, p - i):
                    k = p - i - j
                    xi = j / p
                    eta = k / p
                    pts.append((xi, eta))

        return np.array(pts, dtype=float)

    def _get_monomial_exponents(self, p):
        exps = []
        for i in range(p + 1):
            for j in range(p + 1 - i):
                exps.append((i, j))
        return exps

    def eval(self, comp, pt):
        xi = pt[0]
        eta = pt[1]

        # Evaluate the monomials
        m = eval_monomials(self.p, xi, eta, self.exps)
        N = m @ self.C

        # Evaluate the derivatives of the monomials
        mgrad = eval_monomial_grad(self.p, xi, eta, self.exps)
        Nxi = mgrad[:, 0] @ self.C
        Neta = mgrad[:, 1] @ self.C

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


class QuadLagrangeBasis(LagrangeBasis2D):
    def __init__(self, p, names, kind="input"):
        if p < 0:
            raise ValueError(f"Degree {p} must be >= 0")

        self.p = p
        nnodes = (p + 1) * (p + 1)
        super().__init__(names, nnodes=nnodes, kind=kind)

        self.pts = self._get_quad_nodes(self.p)
        self.exps = self._get_monomial_exponents(self.p)
        self.C = build_vandermonde(self.nnodes, self.p, self.pts, self.exps)

        return

    def _get_quad_nodes(self, p):
        """Get the node locations"""
        pts = []

        if p == 0:
            return [[0, 0]]
        else:
            # Set the vertices
            pts = [[-1, -1], [1, -1], [1, 1], [-1, 1]]

            # At points on the edges
            edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
            for a, b in edges:
                for i in range(1, p):
                    t = i / p
                    xi = (1 - t) * pts[a][0] + t * pts[b][0]
                    eta = (1 - t) * pts[a][1] + t * pts[b][1]
                    pts.append([xi, eta])

            # Add the remaining points in the interior
            for j in range(1, p):
                for i in range(1, p):
                    xi = -1 + 2 * i / p
                    eta = -1 + 2 * j / p
                    pts.append((xi, eta))

        return np.array(pts, dtype=float)

    def _get_monomial_exponents(self, p):
        exps = []
        for j in range(p + 1):
            for i in range(p + 1):
                exps.append((i, j))
        return exps

    def eval(self, comp, pt):
        xi = pt[0]
        eta = pt[1]

        # Evaluate the monomials
        m = eval_monomials(self.p, xi, eta, self.exps)
        N = m @ self.C

        # Evaluate the derivatives of the monomials
        mgrad = eval_monomial_grad(self.p, xi, eta, self.exps)
        Nxi = mgrad[:, 0] @ self.C
        Neta = mgrad[:, 1] @ self.C

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
    def __init__(self, order):
        self.weights = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
        self.points = [[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]]
        self.args = [{"n": 0}, {"n": 1}, {"n": 2}]

        if order == 1:
            # 1-point (exact for degree 1)
            self.xi = np.array([1 / 3])
            self.eta = np.array([1 / 3])
            self.weights = np.array([1.0])

        elif order == 2:
            # 3-point (exact for degree 2)
            self.xi = np.array([1 / 6, 2 / 3, 1 / 6])
            self.eta = np.array([1 / 6, 1 / 6, 2 / 3])
            self.weights = np.array([1 / 3, 1 / 3, 1 / 3])

        elif order == 3:
            # 4-point (exact for degree 3)
            self.xi = np.array([1 / 3, 0.2, 0.2, 0.6])
            self.eta = np.array([1 / 3, 0.2, 0.6, 0.2])
            self.weights = np.array([-27 / 48, 25 / 48, 25 / 48, 25 / 48])

        elif order == 4:
            a = 0.445948490915965
            b = 0.108103018168070
            c = 0.091576213509771
            d = 0.816847572980459

            w1 = 0.223381589678011
            w2 = 0.109951743655322

            self.xi = np.array([a, a, b, c, c, d])
            self.eta = np.array([a, b, a, c, d, c])
            self.weights = np.array([w1, w1, w1, w2, w2, w2])
        else:
            raise NotImplementedError

        self.args = []
        for n in range(len(self.weights)):
            self.args.append({"n": n})

        return

    def get_args(self):
        return self.args

    def get_point(self, n=0):
        return self.weights[n], [self.xi[n], self.eta[n]]


class QuadQuadrature(Quadrature):
    def __init__(self, npts):
        self.points, self.weights = np.polynomial.legendre.leggauss(npts)
        self.args = []

        for m in range(npts):
            for n in range(npts):
                self.args.append({"n": n, "m": m})

    def get_args(self):
        return self.args

    def get_point(self, n=0, m=0):
        wt = self.weights[n] * self.weights[m]
        pt = [self.points[n], self.points[m]]
        return wt, pt


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
