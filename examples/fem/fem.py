import amigo as am
import numpy as np
import re
import basis


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

    def get_domains(self):
        names = {}
        for elset in self.elem_conn:
            names[elset] = []
            for elem_type in self.elem_conn[elset]:
                names[elset].append(elem_type)

        return names

    def get_conn(self, elset, elem_type):
        conn = self.elem_conn[elset][elem_type]
        return np.array([conn[k] for k in sorted(conn.keys())], dtype=int)


class Mesh:
    def __init__(self, filename):
        self.parser = InpParser()
        self.parser.parse_inp(filename)

        self.X = self.parser.get_nodes()

    def get_num_nodes(self):
        return self.X.shape[0]

    def get_domains(self):
        domains = self.parser.get_domains()

        volumes = {}
        for name in domains:
            if "CPS3" in domains[name] or "CPS4" in domains[name]:
                volumes[name] = domains[name]

        return volumes

    def get_conn(self, name, etype):
        return self.parser.get_conn(name, etype)


class NodeSource(am.Component):
    def __init__(self, input_names=[], geo_names=[], data_names=[], con_names=[]):
        super().__init__()

        # Geo and data added as data to the component
        for name in data_names:
            self.add_data(name)
        for name in geo_names:
            self.add_data(name)

        # Add inputs and constraints
        for name in input_names:
            self.add_input(name)
        for name in con_names:
            self.add_constraint(name)

        return


class Problem:
    def __init__(self, mesh, soln_space, weakform, data_space=[], ndim=2):

        self.mesh = mesh
        self.ndim = ndim  # Dimension of the problem

        self.soln_space = soln_space
        self.data_space = data_space

        if self.ndim == 2:
            self.geo_space = basis.SolutionSpace({"x": "H1", "y": "H1"})
        elif self.ndim == 3:
            self.geo_space = basis.SolutionSpace({"x": "H1", "y": "H1", "z": "H1"})

        self.weakform = weakform

        return

    # def add_weak_form(self, name, weakform):
    #     if name in self.domains:
    #         self.weakform[name] = weakform
    #     else:
    #         raise KeyError(f"{name} not in domains")
    #     return

    def create_model(self, module_name: str):
        """Create and link the Amigo model"""

        model = am.Model(module_name)

        # Get the degrees of freedom associated with H1
        input_names = self.soln_space.get_names("H1")
        data_names = self.data_space.get_names("H1")
        geo_names = self.geo_space.get_names("H1")

        # Get the number of nodes from the mesh
        nnodes = self.mesh.get_num_nodes()

        # Create a node source object for all nodes in the mesh
        src = NodeSource(
            input_names=input_names, geo_names=geo_names, data_names=data_names
        )
        model.add_component("src", nnodes, src)

        # Build the elements for all domains
        domains = self.mesh.get_domains()
        for domain in domains:
            for etype in domains[domain]:

                # Build a finite-element for each weak form
                elem_name = f"Element{etype}_{domain}"

                soln_basis = self.get_basis(etype, "H1", input_names, kind="input")
                data_basis = self.get_basis(etype, "H1", data_names, kind="data")
                geo_basis = self.get_basis(etype, "H1", geo_names, kind="data")

                quadrature = self.get_quadrature(etype)

                # Create the element object
                elem = FiniteElement(
                    elem_name,
                    soln_basis,
                    data_basis,
                    geo_basis,
                    quadrature,
                    self.weakform,
                )

                # Get the connectivity
                conn = self.mesh.get_conn(domain, etype)

                # Add the component
                nelems = conn.shape[0]
                model.add_component(elem_name, nelems, elem)

                # Link this component
                for name in input_names + data_names + geo_names:
                    model.link(f"src.{name}", f"{elem_name}.{name}", src_indices=conn)

        return model

    def get_basis(self, etype, space, names=[], kind="input"):
        if etype == "CPS3":
            if space == "H1":
                return basis.LinearH1TriangleBasis(names, kind=kind)
        elif etype == "CPS4":
            if space == "H1":
                return basis.LinearH1QuadBasis(names, kind=kind)

        raise NotImplementedError(
            f"Basis for element {etype} with space {space} not implemented"
        )

    def get_quadrature(self, etype):
        if etype == "CPS3":
            return basis.TriangleQuadrature()
        elif etype == "CPS4":
            return basis.QuadQuadrature()

        raise NotImplementedError(f"Quadrature for element {etype} not implemented")


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

    x = geo["x"]["value"]
    y = geo["y"]["value"]
    rho = data["rho"]["value"]

    f = am.sin(x) ** 2 * am.cos(y) ** 2

    return 0.5 * (uvalue**2 + basis.dot_product(ugrad, ugrad, n=2) - 2.0 * uvalue * f)


soln_space = basis.SolutionSpace({"u": "H1"})
data_space = basis.SolutionSpace({"rho": "H1"})

mesh = Mesh("magnet.inp")
problem = Problem(mesh, soln_space, weakform, data_space=data_space, ndim=2)
model = problem.create_model("test")

# mesh = Mesh("plate.inp")
# X, conn, lines = mesh.load_data(
#     surface_names=["SURFACE1"],
#     line_names=["LINE1", "LINE2", "LINE3", "LINE4"],
#     element_type="CPS3",
#     line_type="T3D2",
# )
# nnodes = X.shape[0]
# nelems = conn["SURFACE1"].shape[0]


# # Set it up so that the input is
# geo_basis = LinearH1TriangleBasis(["x", "y"], kind="data")
# data_basis = LinearH1TriangleBasis("rho", kind="data")
# soln_basis = LinearH1TriangleBasis("u", kind="input")

# quadrature = TriangleQuadrature()
# node_src = NodeSource()

# name = "TriElement"
# elem = FiniteElement(name, soln_basis, data_basis, geo_basis, quadrature, weakform)


# # Amigo model
# model = am.Model("test")
# model.add_component("tri", nelems, elem)

# # Add source components to the amigo model
# model.add_component("src", nnodes, node_src)

# # Linke source mesh to the finite element problem
# model.link("tri.x", "src.x", tgt_indices=conn["SURFACE1"])
# model.link("tri.y", "src.y", tgt_indices=conn["SURFACE1"])

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def plot_solution(
    xyz_nodeCoords, conn, z, title="fig", fname="contour.jpg", flag=False, ax=None
):
    """
    Create a contour plot of the solution.
    Inputs:
        xyz_nodeCoords = [x,y,z] node positions
        z = solution field vector

    Parameters
    ----------
    xyz_nodeCoords : 2d np array
        [x,y,z]
    z : 1d array
        solution vector to plot at each xyz position
    title : str, optional
        figure title, by default "fig"
    fname : str, optional
        name of figure make sure to include extension (.jpg), by default "contour.jpg"
    flag : bool, optional
        save figure, by default False
    """
    min_level = min(z)
    max_level = max(z)
    levels = np.linspace(min_level, max_level, 30)
    x = xyz_nodeCoords[:, 0]
    y = xyz_nodeCoords[:, 1]

    # create a Delaunay triangulation
    # tri = mtri.Triangulation(x, y)
    if conn.shape[1] == 3:
        tri = mtri.Triangulation(x, y, conn)
    elif conn.shape[1] == 4:
        conn2 = np.vstack([conn[:, [0, 1, 2]], conn[:, [0, 2, 3]]])
        tri = mtri.Triangulation(x, y, conn2)

    # Define colormap
    cmap = "coolwarm"

    # Plot solution
    fig = None
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    cntr = ax.tricontourf(tri, z, levels=levels, cmap=cmap)
    ax.tricontour(
        tri, z, levels=levels, colors="k", linewidths=0.3, alpha=0.5
    )  # optional contour lines

    # Overlay mesh
    ax.triplot(tri, color="0.7", lw=0.3, alpha=0.8)  # lighter grey

    if fig is not None:
        norm = mpl.colors.Normalize(vmin=min_level, vmax=max_level)
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, location="right"
        )
        cbar.set_ticks([min(z), (min(z) + max(z)) / 2.0, max(z)])

        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()

    # if flag:
    #     plt.savefig(fname, dpi=800, edgecolor="none")

    return ax


model.build_module()
model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)

problem = model.get_problem()

# Set the problem data
data = model.get_data_vector()
data["src.x"] = mesh.X[:, 0]
data["src.y"] = mesh.X[:, 1]

x = problem.create_vector()
mat = problem.create_matrix()
rhs = model.create_vector()
problem.gradient(1.0, x, rhs.get_vector())
problem.hessian(1.0, x, mat)

chol = am.SparseCholesky(mat)
flag = chol.factor()
print("flag = ", flag)
chol.solve(rhs.get_vector())

u = rhs["src.u"]
conn = mesh.get_conn("SURFACE2", "CPS3")
ax = plot_solution(mesh.X, conn, u)
conn = mesh.get_conn("SURFACE1", "CPS4")
plot_solution(mesh.X, conn, u, ax=ax)

plt.show()
