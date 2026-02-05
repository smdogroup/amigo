import amigo as am
import numpy as np
import re


def dot_product(x, y, n=1):
    val = x[0] * y[0]
    for i in range(1, n):
        val = val + x[i] * y[i]
    return val


def mat_product(A, x, m=1, n=1):
    return [A[0][0] * x[0] + A[0][1] * x[1], A[1][0] * x[0] + A[1][1] * x[1]]


class InpParser:
    def __init__(self):
        self.elements = {}

    def _read_file(self, filename):
        with open(filename, "r", errors="ignore") as fp:
            return [line.rstrip("\n") for line in fp]

    def _split_csv_line(self, s):
        # ABAQUS lines are simple CSV-like, no quotes typically
        return [p.strip() for p in s.split(",") if p.strip()]

    def _find_kw(self, header, key):
        m = re.search(rf"\b{re.escape(key)}\s*=\s*([^,\s]+)", header, flags=re.I)
        return m.group(1) if m else None

    def parse_inp(self, filename):
        # Read the entire file in
        lines = self._read_file(filename)

        self.X = {}
        self.elem_conn = {}

        elem_type = None

        index = 0
        section = None
        while index < len(lines):
            raw = lines[index].strip()
            index += 1

            if not raw or raw.startswith("**"):
                continue

            if raw.startswith("*"):
                header = raw.upper()

                if header.startswith("*NODE"):
                    section = "NODE"
                elif header.startswith("*ELEMENT"):
                    section = "ELEMENT"
                    elem_type = self._find_kw(header, "TYPE")
                    elset = self._find_kw(header, "ELSET")
                    if elem_type not in self.elem_conn:
                        self.elem_conn[elset] = {}
                    if elset not in self.elem_conn[elset]:
                        self.elem_conn[elset][elem_type] = {}
                continue

            if section == "NODE":
                parts = self._split_csv_line(raw)
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                self.X[nid - 1] = (x, y, z)

            elif section == "ELEMENT":
                parts = self._split_csv_line(raw)
                eid = int(parts[0])
                conn = [(int(p) - 1) for p in parts[1:]]
                self.elem_conn[elset][elem_type][eid - 1] = conn

    def get_nodes(self):
        return np.array([self.X[k] for k in sorted(self.X.keys())])

    def get_names(self):
        names = {}
        for elset in self.elem_conn:
            names[elset] = []
            for elem_type in self.elem_conn[elset]:
                names[elset].append(elem_type)

        return names

    def get_conn(self, elset, elem_type):
        conn = self.elem_conn[elset][elem_type]
        return np.array([conn[k] for k in sorted(conn.keys())], dtype=int)


class Mesh(InpParser):
    def __init__(self, filename):
        super().__init__()
        self.parse_inp(filename)

    def load_data(self, surface_names, line_names, element_type, line_type):
        # Get the node locations
        X = self.get_nodes()

        # Get element connectivity for each surface name
        conn = {}
        for name in surface_names:
            conn[name] = self.get_conn(name, element_type)

        # Get the node tags for each line/curve in the mesh
        lines = {}
        for name in line_names:
            lines[name] = self.get_conn(name, line_type)

        return X, conn, lines


class NodeSource(am.Component):
    def __init__(self):
        super().__init__()

        # Mesh coordinates
        self.add_data("x")
        self.add_data("y")

        # States
        self.add_input("u")


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


mesh = Mesh("plate.inp")
X, conn, lines = mesh.load_data(
    surface_names=["SURFACE1"],
    line_names=["LINE1", "LINE2", "LINE3", "LINE4"],
    element_type="CPS3",
    line_type="T3D2",
)
nnodes = X.shape[0]
nelems = conn["SURFACE1"].shape[0]


# Set it up so that the input is
geo_basis = LinearH1TriangleBasis(["x", "y"], kind="data")
data_basis = LinearH1TriangleBasis("rho", kind="data")
soln_basis = LinearH1TriangleBasis("u", kind="input")

quadrature = TriangleQuadrature()
node_src = NodeSource()

name = "TriElement"
elem = FiniteElement(name, soln_basis, data_basis, geo_basis, quadrature, weakform)


# Amigo model
model = am.Model("test")
model.add_component("tri", nelems, elem)

# Add source components to the amigo model
model.add_component("src", nnodes, node_src)

# Linke source mesh to the finite element problem
model.link("tri.x", "src.x", tgt_indices=conn["SURFACE1"])
model.link("tri.y", "src.y", tgt_indices=conn["SURFACE1"])

model.build_module()
model.initialize()
