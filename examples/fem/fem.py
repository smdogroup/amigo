import amigo as am


def dot_product(x, y, n=1):
    val = x[0] * y[0]
    for i in range(1, n):
        val = val + x[i] * y[i]
    return val


def mat_product(A, x, m=1, n=1):
    return [A[0][0] * x[0] + A[0][1] * x[1], A[1][0] * x[0] + A[1][1] * x[1]]


# class Mesh:
#     pass  # Load in the INP file and store triangles and quads

# class Problem:
#     def __init__(self, mesh, solution_space, data_space):
#         self.mesh = mesh
#         self.solution_space = solution_space
#         self.data_space = data_space

#         return


class Basis:
    def __init__(self, space):
        self.space = space
        return

    def add_declarations(self, comp):
        pass


class LinearH1TriangleBasis(Basis):
    def __init__(self, names, kind="input"):
        if isinstance(names, (list, tuple)):
            self.names = names
        elif isinstance(names, str):
            self.names = [names]
        self.kind = kind
        self.nnodes = 3

    def add_declarations(self, comp):
        """Add the declarations to the component"""

        if self.kind == "input":
            for name in self.names:
                comp.add_input(name, shape=(self.nnodes,))
        elif self.kind == "data":
            for name in self.names:
                comp.add_data(name, shape=(self.nnodes,))

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

    def transform(self, detJ, Jinv, orig):
        soln = {}
        for name in orig:
            value = orig[name]["value"]
            grad = orig[name]["grad"]
            soln[name] = {"value": value, "grad": mat_product(Jinv, grad, n=2, m=2)}

        return soln

    def compute_transform(self, geo):
        if "x" not in geo or "y" not in geo:
            raise ValueError("Coordinates not defined")

        x_xi, x_eta = geo["x"]["grad"]
        y_xi, y_eta = geo["y"]["grad"]

        detJ = x_xi * y_eta - x_eta * y_xi
        Jinv = [[y_eta / detJ, -x_eta / detJ], [-y_xi / detJ, x_xi / detJ]]

        return detJ, Jinv


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


class FiniteElement(am.Component):
    def __init__(self, name, soln_basis, data_basis, geo_basis, quadrature, weakform):
        super().__init__(name=name)

        self.soln_basis = soln_basis
        self.data_basis = data_basis
        self.geo_basis = geo_basis
        self.quadrature = quadrature
        self.weakform = weakform

        # Add the declarations for each basis used in the element
        self.soln_basis.add_declarations(self)
        self.data_basis.add_declarations(self)
        self.geo_basis.add_declarations(self)

        # Set the arguments to the compute function for each quadrature point
        self.set_args(self.quadrature.get_args())

        self.add_objective("obj")

        return

    def compute(self, **args):

        quad_weight, quad_point = self.quadrature.get_point(**args)

        # Evaluate the solution fields/data fields
        soln_xi = self.soln_basis.eval(self, quad_point)
        data_xi = self.data_basis.eval(self, quad_point)
        geo = self.geo_basis.eval(self, quad_point)

        # Perform the mapping from computational to physical coordinates
        detJ, Jinv = self.geo_basis.compute_transform(geo)
        soln_phys = self.soln_basis.transform(detJ, Jinv, soln_xi)
        data_phys = self.data_basis.transform(detJ, Jinv, data_xi)

        # Add the contributions directly to the Lagrangian
        self.objective["obj"] = (
            quad_weight * detJ * self.weakform(soln_phys, data=data_phys, geo=geo)
        )

        return


def weakform(soln, data=None, geo=None):
    u = soln["u"]
    uvalue = u["value"]
    ugrad = u["grad"]

    rho = data["rho"]["value"]

    return 0.5 * (uvalue**2 + rho * dot_product(ugrad, ugrad, n=2))


# Set it up so that the input is
geo_basis = LinearH1TriangleBasis(["x", "y"], kind="data")
data_basis = LinearH1TriangleBasis("rho", kind="data")
soln_basis = LinearH1TriangleBasis("u", kind="input")

quadrature = TriangleQuadrature()

name = "TriElement"
elem = FiniteElement(name, soln_basis, data_basis, geo_basis, quadrature, weakform)


model = am.Model("test")
model.add_component("tri", 5, elem)

model.build_module()
model.initialize()
