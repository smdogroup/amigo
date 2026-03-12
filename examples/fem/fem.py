import amigo as am
import numpy as np
import re
import basis
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from connectivity import InpParser
from matplotlib.collections import PolyCollection


class DofSource(am.Component):
    def __init__(self, input_names=[], data_names=[], con_names=[]):
        super().__init__()

        # Geo and data added as data to the component
        for name in data_names:
            self.add_data(name)

        # Add inputs and constraints
        for name in input_names:
            self.add_input(name)
        for name in con_names:
            self.add_constraint(name)

        return


class SymmBCSource(am.Component):
    def __init__(self, input_name=[], scale=[1.0, 1.0]):
        super().__init__()

        self.input_name = input_name
        self.scale = scale

        for name in self.input_name:
            self.add_input(f"{name}0", value=1.0)
            self.add_input(f"{name}1", value=1.0)
        self.add_input("lam", value=1.0)
        self.add_objective("obj")
        return

    def compute(self):
        scale_node_0 = self.scale[0]
        scale_node_1 = self.scale[1]
        for name in self.input_name:
            self.objective["obj"] = (
                scale_node_0 * self.inputs[f"{name}0"]
                + scale_node_1 * self.inputs[f"{name}1"]
            ) * self.inputs["lam"]
        return


class SymmetryDegreesOfFreedom:
    def __init__(self, mesh, bc={}):
        self.mesh = mesh
        self.bc = bc
        return

    def add_bc_source(self, model):
        names = self.bc.keys()
        for name in names:
            # Loop through each bc_type name
            bc_src = SymmBCSource(
                input_name=self.bc[name]["input"],
                scale=self.bc[name]["scale"],
            )

            line_tag = self.bc[name]["target"][0]
            nnodes = self.mesh.get_num_nodes_on_bc(line_tag, "T3D2")

            # Update the number of components based on whether to include start and end
            if self.bc[name]["start"] == False:
                nnodes -= 1
            if self.bc[name]["end"] == False:
                nnodes -= 1

            model.add_component(
                f"src_{name}",
                nnodes,
                bc_src,
            )
        return

    def link_bc_dof(self, model):
        names = self.bc.keys()
        for name in names:
            # Loop through each bc_type name
            input_name = self.bc[name]["input"][0]  # Extract "u"

            for i, line_tag in enumerate(self.bc[name]["target"]):
                conn = self.mesh.get_bc_nodes(line_tag, "T3D2")

                # Slice the nodes based on start and end requirement
                if self.bc[name]["start"] == False and self.bc[name]["end"] == True:
                    conn = conn[1:]
                elif self.bc[name]["start"] == False and self.bc[name]["end"] == False:
                    conn = conn[1:-1]
                elif self.bc[name]["start"] == True and self.bc[name]["end"] == False:
                    conn = conn[0:-1]

                if self.bc[name]["flip"][i] == True:
                    conn = np.flip(conn)

                model.link(
                    f"src_soln.{input_name}",
                    f"src_{name}.{input_name}{i}",
                    src_indices=conn,
                )
        return


class DirichletBCSource(am.Component):
    def __init__(self, input_name=[]):
        super().__init__()

        self.input_name = input_name

        for name in self.input_name:
            self.add_input(f"{name}0", value=1.0)
        self.add_input("lam", value=1.0)
        self.add_objective("obj")
        return

    def compute(self):
        for name in self.input_name:
            self.objective["obj"] = self.inputs[f"{name}0"] * self.inputs["lam"]
        return


class DirichletDegreesOfFreedom:
    def __init__(self, mesh, bc={}):
        self.mesh = mesh
        self.bc = bc
        return

    def add_bc_source(self, model):
        names = self.bc.keys()
        for name in names:
            # Loop through each bc_type name
            input_name = self.bc[name]["input"]
            target_name = self.bc[name]["target"]
            bc_src = DirichletBCSource(input_name=input_name)
            nnodes = self.mesh.get_num_nodes_on_bc(target_name, "T3D2")

            # Update the number of components based on whether to include start and end
            if self.bc[name]["start"] == False:
                nnodes -= 1
            if self.bc[name]["end"] == False:
                nnodes -= 1

            model.add_component(
                f"src_{name}",
                nnodes,
                bc_src,
            )
        return

    def link_bc_dof(self, model):
        names = self.bc.keys()
        for name in names:
            # Loop through each bc target
            target = self.bc[name]["target"]
            input_name = self.bc[name]["input"][0]  # Extract "u"
            conn = self.mesh.get_bc_nodes(target, "T3D2")

            # Slice the nodes based on start and end requirement
            if self.bc[name]["start"] == False and self.bc[name]["end"] == True:
                conn = conn[1:]
            elif self.bc[name]["start"] == False and self.bc[name]["end"] == False:
                conn = conn[1:-1]
            elif self.bc[name]["start"] == True and self.bc[name]["end"] == False:
                conn = conn[0:-1]

            model.link(
                f"src_soln.{input_name}",
                f"src_{name}.{input_name}0",
                src_indices=conn,
            )
        return


class DegreesOfFreedom:
    def __init__(self, mesh, space, kind="input", name="src"):
        """
        Allocate the degrees of freedom on the mesh
        """

        self.mesh = mesh
        self.space = space
        self.kind = kind
        self.name = name

        return

    def add_source(self, model):
        # Get the names of things associated with H1

        for sp in ["H1", "const"]:
            names = self.space.get_names(sp)

            if len(names) == 0:
                continue

            # Create amigo source component with input names and geo names
            input_names = []
            data_names = []
            if self.kind == "input":
                input_names = names
            elif self.kind == "data":
                data_names = names

            dof_src = DofSource(input_names=input_names, data_names=data_names)

            # Add global mesh source component
            if sp == "H1":
                nnodes = self.mesh.get_num_nodes()
                model.add_component(f"src_{self.name}", nnodes, dof_src)

            elif sp == "const":
                nsurfaces = self.mesh.get_num_surfaces()
                model.add_component(f"src_{self.name}", nsurfaces, dof_src)

    def link_dof(self, model, domain, etype, elem_name):
        for sp in ["H1", "const"]:
            names = self.space.get_names(sp)

            if len(names) == 0:
                continue

            if sp == "H1":
                conn = self.mesh.get_conn(domain, etype)
                for name in names:
                    model.link(
                        f"src_{self.name}.{name}",
                        f"{elem_name}.{name}",
                        src_indices=conn,
                    )

            elif sp == "const":
                gmsh_surf_index = int(domain.replace("SURFACE", ""))
                surf_index = gmsh_surf_index - 1

                for name in names:
                    model.link(
                        f"src_{self.name}.{name}[{surf_index}]",
                        f"{elem_name}.{name}[:]",
                    )

        return

    def get_basis(self, etype):
        basis_list = []

        for sp in ["H1", "const"]:
            names = self.space.get_names(sp)

            if len(names) == 0:
                continue

            basis_list.append(self._get_basis(etype, sp, names, self.kind))

        return basis.BasisCollection(basis_list)

    def _get_basis(self, etype, space, names=[], kind="input"):
        if etype == "CPS3":
            if space == "H1":
                return basis.TriangleLagrangeBasis(1, names, kind=kind)
            elif space == "const":
                return basis.ConstantBasis(names=names, kind=kind)
        elif etype == "CPS4":
            if space == "H1":
                return basis.QuadLagrangeBasis(1, names, kind=kind)
            elif space == "const":
                return basis.ConstantBasis(names=names, kind=kind)
        elif etype == "CPS6":
            if space == "H1":
                return basis.TriangleLagrangeBasis(2, names, kind=kind)
        elif etype == "M3D9":
            if space == "H1":
                return basis.QuadLagrangeBasis(2, names, kind=kind)

        raise NotImplementedError(
            f"Basis for element {etype} with space {space} not implemented"
        )

    def get_quadrature(self, etype):
        if etype == "CPS3":
            return basis.TriangleQuadrature(2)
        elif etype == "CPS4":
            return basis.QuadQuadrature(2)
        elif etype == "CPS6":
            return basis.TriangleQuadrature(4)
        elif etype == "M3D9":
            return basis.QuadQuadrature(3)

        raise NotImplementedError(f"Quadrature for element {etype} not implemented")


class Mesh:
    def __init__(self, filename):
        self.parser = InpParser()
        self.parser.parse_inp(filename)

        self.X = self.parser.get_nodes()

    def get_num_nodes(self):
        return self.X.shape[0]

    def get_domains(self):
        domains = self.parser.get_domains()

        element_types = ["CPS3", "CPS4", "CPS6", "M3D9"]

        volumes = {}
        for name in domains:
            for etype in element_types:
                if etype in domains[name]:
                    volumes[name] = domains[name]
                    break

        return volumes

    def get_conn(self, name, etype):
        return self.parser.get_conn(name, etype)

    def get_num_elements(self, name, etype):
        return self.parser.get_conn(name, etype).shape[0]

    def get_num_surfaces(self):
        return self.parser.get_num_surfaces()

    def get_bc_nodes(self, name, etype, flip=False):
        conn = self.parser.get_edge_node(name, etype)
        if flip == True:
            np.flip(conn)
        return conn

    def get_num_nodes_on_bc(self, name, etype):
        return self.parser.get_edge_node(name, etype).shape[0]

    def plot(
        self,
        u,
        ax=None,
        nlevels=30,
        cmap="coolwarm",
        title=None,
        x_offset=0.0,
        y_offset=0.0,
        min_level=None,
        max_level=None,
    ):
        if min_level == None or max_level == None:
            min_level = np.min(u)
            max_level = np.max(u)

        levels = np.linspace(min_level, max_level, nlevels)

        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

        volumes = self.get_domains()
        x = self.X[:, 0] + x_offset
        y = self.X[:, 1] + y_offset

        for name in volumes:
            for etype in volumes[name]:
                # Get the connectivity
                conn = self.convert_conn(etype, self.get_conn(name, etype))
                tri = mtri.Triangulation(x, y, conn)

                # Set the contour plot
                ax.tricontourf(tri, u, levels=levels, cmap=cmap)
                ax.tricontour(
                    tri, u, levels=levels, colors="k", linewidths=0.3, alpha=0.5
                )

                # Overlay the mesh skeleton
                gmsh_conn = self.get_conn(name, etype)
                X2d = self.X[:, 0:2]
                polygons = [X2d[row] for row in gmsh_conn]
                mesh = PolyCollection(
                    polygons,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.4,
                )
                # ax.add_collection(mesh)

                if title is not None:
                    ax.set_title(title)

                ax.set_aspect("equal")
        return ax

    def convert_conn(self, etype, conn):
        if etype == "CPS3":
            return conn
        elif etype == "CPS4":
            c = [[0, 1, 2], [0, 2, 3]]
        elif etype == "CPS6":
            # 2
            # |  .
            # 5     4
            # |        .
            # 0 --- 3 --- 1
            c = [[0, 3, 5], [3, 4, 5], [3, 1, 4], [5, 4, 2]]
        elif etype == "M3D9":
            # 3 --- 6 --- 2
            # |           |
            # 7     8     5
            # |           |
            # 0 --- 4 --- 1
            c = [
                [0, 4, 7],
                [4, 8, 7],
                [4, 1, 8],
                [1, 5, 8],
                [7, 8, 3],
                [8, 6, 3],
                [8, 5, 6],
                [5, 2, 6],
            ]

        cs = []
        for c0 in c:
            cs.append(conn[:, c0])

        return np.vstack(cs)


# Needs to take in bcs here
class Problem:
    # soln_space = object
    def __init__(
        self,
        mesh,
        soln_space,
        weakform_map={},
        data_space=[],
        geo_space=[],
        dirichlet_bc_map={},
        sym_bc_map={},
        ndim=2,
    ):
        self.mesh = mesh
        self.ndim = ndim  # Dimension of the problem

        self.soln_space = soln_space
        self.data_space = data_space
        self.geo_space = geo_space

        self.weakform_map = weakform_map
        self.dirichlet_bc_map = dirichlet_bc_map
        self.sym_bc_map = sym_bc_map

        # Initialize Dof's
        # Take in the soln space -> removes "H1" input
        self.soln_dof = DegreesOfFreedom(
            self.mesh,
            self.soln_space,
            kind="input",
            name="soln",
        )
        self.geo_dof = DegreesOfFreedom(
            self.mesh,
            self.geo_space,
            kind="data",
            name="geo",
        )
        self.data_dof = DegreesOfFreedom(
            self.mesh,
            self.data_space,
            kind="data",
            name="data",
        )
        self.dirichlet_bc_dof = DirichletDegreesOfFreedom(
            self.mesh,
            self.dirichlet_bc_map,
        )
        self.sym_bc_dof = SymmetryDegreesOfFreedom(
            self.mesh,
            self.sym_bc_map,
        )

        return

    def create_model(self, module_name: str):
        """Create and link the Amigo model"""
        model = am.Model(module_name)

        # print("\nsoln_dof source called")
        self.soln_dof.add_source(model)

        # print("\ndata_dof source called")
        self.data_dof.add_source(model)

        # print("\ngeo_dof source called")
        self.geo_dof.add_source(model)

        # Build the elements for all domains
        domains = self.mesh.get_domains()
        wf_name = self.weakform_map["name"]
        for domain in domains:
            for etype in domains[domain]:
                # Each element type has a dictionary of solution basis's
                soln_basis = {}

                # Build a finite-element for each weak form
                elem_name = f"{wf_name}_Element{etype}_{domain}"

                soln_basis = self.soln_dof.get_basis(etype)
                data_basis = self.data_dof.get_basis(etype)
                geo_basis = self.geo_dof.get_basis(etype)

                # Create the quadrature instance
                quadrature = self.soln_dof.get_quadrature(etype)

                # Create the element object
                elem = FiniteElement(
                    elem_name,
                    soln_basis,
                    data_basis,
                    geo_basis,
                    quadrature,
                    self.weakform_map[domain],
                )

                # Add the element/component
                nelems = self.mesh.get_num_elements(domain, etype)
                model.add_component(elem_name, nelems, elem)

                # Link all the element dof to the component
                # print("\nsoln_dof link called")
                self.soln_dof.link_dof(model, domain, etype, elem_name)

                # print("\ndata_dof link called")
                self.data_dof.link_dof(model, domain, etype, elem_name)

                # print("\ngeo_dof link called")
                self.geo_dof.link_dof(model, domain, etype, elem_name)

        # Add BC components and links
        self.dirichlet_bc_dof.add_bc_source(model)
        self.dirichlet_bc_dof.link_bc_dof(model)

        # Add symmetric bcs
        self.sym_bc_dof.add_bc_source(model)
        self.sym_bc_dof.link_bc_dof(model)

        return model


class FiniteElement(am.Component):
    def __init__(
        self,
        name,
        soln_basis,
        data_basis,
        geo_basis,
        quadrature,
        weakform,
    ):
        super().__init__(name=name)

        self.soln_basis = soln_basis
        self.data_basis = data_basis
        self.geo_basis = geo_basis
        self.quadrature = quadrature
        self.weakform = weakform

        # From BasisCollection
        self.soln_basis.add_declarations(self)
        self.geo_basis.add_declarations(self)
        self.data_basis.add_declarations(self)

        # Set the arguments to the compute function for each quadrature point
        self.set_args(self.quadrature.get_args())

        # Add the objective to minimize
        self.add_objective("obj")

        return

    def compute(self, **args):

        quad_weight, quad_point = self.quadrature.get_point(**args)

        # Evaluate the solution fields/data fields (u)
        soln_xi = self.soln_basis.eval(self, quad_point)
        data_xi = self.data_basis.eval(self, quad_point)
        geo = self.geo_basis.eval(self, quad_point)

        # Perform the mapping from computational to physical coordinates (u)
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
    v = soln["v"]

    uvalue = u["value"]
    ugrad = u["grad"]

    vvalue = v["value"]
    vgrad = v["grad"]

    x = geo["x"]["value"]
    y = geo["y"]["value"]

    # f = am.sin(x) ** 2 * am.cos(y) ** 2
    # comp1 = 0.5 * (uvalue**2 + basis.dot_product(ugrad, ugrad, n=2) - 2.0 * uvalue * f)

    alpha = 1.0
    pi = 3.14159265358979

    f1 = 2 * pi**2 * am.sin(pi * x) * am.sin(pi * y) + alpha * am.cos(
        pi * x
    ) * am.cos(pi * y)
    f2 = 2 * pi**2 * am.cos(pi * x) * am.cos(pi * y) + alpha * am.sin(
        pi * x
    ) * am.sin(pi * y)

    comp1 = (
        basis.dot_product(ugrad, ugrad, n=2)
        + alpha * uvalue * uvalue
        + f1 * uvalue
        + basis.dot_product(vgrad, vgrad, n=2)
        + alpha * vvalue * vvalue
        + f2 * vvalue
    )
    return comp1


def weakform_air(soln, data=None, geo=None):
    u = soln["u"]
    uvalue = u["value"]
    ugrad = u["grad"]

    x = geo["x"]["value"]
    y = geo["y"]["value"]

    wf = 0.5 * (basis.dot_product(ugrad, ugrad, n=2))
    return wf


def weakform_NS_Magnet(soln, data=None, geo=None):
    u = soln["u"]
    uvalue = u["value"]
    ugrad = u["grad"]

    x = geo["x"]["value"]
    y = geo["y"]["value"]

    M = [0.0, 1.0]
    f = basis.curl_2d(ugrad, M, n=2)
    wf = 0.5 * (basis.dot_product(ugrad, ugrad, n=2) - f)
    return wf


def weakform_SN_Magnet(soln, data=None, geo=None):
    u = soln["u"]
    uvalue = u["value"]
    ugrad = u["grad"]

    x = geo["x"]["value"]
    y = geo["y"]["value"]

    M = [0.0, -1.0]
    f = basis.curl_2d(ugrad, M, n=2)
    wf = 0.5 * (basis.dot_product(ugrad, ugrad, n=2) - f)
    return wf


def weakform_coils(soln, data=None, geo=None):
    u = soln["u"]
    uvalue = u["value"]
    ugrad = u["grad"]

    Jz = data["Jz"]["value"]
    f = Jz * uvalue

    wf = 0.5 * (basis.dot_product(ugrad, ugrad, n=2) - f)
    return wf
