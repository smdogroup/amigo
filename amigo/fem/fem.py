import amigo as am
import numpy as np
from . import basis
from .connectivity import InpParser

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection


class DofSource(am.Component):
    def __init__(self, input_names=[], data_names=[], con_names=[], output_names=[]):
        super().__init__()

        # Geo and data added as data to the component
        for name in data_names:
            self.add_data(name)

        # Add inputs and constraints
        for name in input_names:
            self.add_input(name)
        for name in con_names:
            self.add_constraint(name)
        for name in output_names:
            self.add_output(name)

        return


class SymmBCSource(am.Component):
    def __init__(self, input_name=[], scale=[]):
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
    def __init__(self, bc_name, mesh, bc={}):
        self.bc_name = bc_name
        self.mesh = mesh
        self.bc = bc
        return

    def _get_bc_nodes(self, targets, start, end):
        all_nodes = []
        for target in targets:
            nodes = self.mesh.get_bc_nodes(target, "T3D2")
            all_nodes.extend(nodes)

        unique = list(dict.fromkeys(all_nodes))  # preserve order

        if not start:
            unique = unique[1:]
        if not end:
            unique = unique[:-1]

        return unique

    def _reorder_nodes(self, nodes_left, nodes_right):
        nodes_left = np.array(nodes_left)
        nodes_right = np.array(nodes_right)

        y_left = self.mesh.X[nodes_left, 1]
        y_right = self.mesh.X[nodes_right, 1]

        idx_left = np.argsort(y_left)
        idx_right = np.argsort(y_right)

        return nodes_left[idx_left], nodes_right[idx_right]

    def add_and_link_source(self, model):
        targets = self.bc["target"]
        start = self.bc["start"]
        end = self.bc["end"]
        input_names = self.bc["input"]
        scale = self.bc["scale"]

        left_target_lines = targets[0]
        right_target_lines = targets[1]
        nodes_left = self._get_bc_nodes(left_target_lines, start, end)
        nodes_right = self._get_bc_nodes(right_target_lines, start, end)

        if len(nodes_left) != len(nodes_right):
            raise Exception(f"nnodes left != nnodes right")

        # Reorder the nodes to match
        nodes_a, nodes_b = self._reorder_nodes(nodes_left, nodes_right)

        bc_src = SymmBCSource(input_names, scale=scale)

        if len(nodes_left) > 0:
            for name in input_names:
                model.add_component(
                    f"src_{self.bc_name}",
                    len(nodes_left),
                    bc_src,
                )

                model.link(
                    f"src_soln.{name}",
                    f"src_{self.bc_name}.{name}0",
                    src_indices=nodes_a,
                )
                model.link(
                    f"src_soln.{name}",
                    f"src_{self.bc_name}.{name}1",
                    src_indices=nodes_b,
                )
        return


class DirichletBCSource(am.Component):
    def __init__(self, name, input_names=[]):
        super().__init__(name=name)

        self.input_names = input_names

        for name in self.input_names:
            self.add_input(f"{name}")
            self.add_input(f"lam_{name}")

        self.add_objective("obj")
        return

    def compute(self):
        obj = 0.0
        for name in self.input_names:
            obj += self.inputs[f"{name}"] * self.inputs[f"lam_{name}"]
        self.objective["obj"] = obj
        return


class DirichletDegreesOfFreedom:
    def __init__(self, bc_name, mesh, bc={}):
        self.bc_name = bc_name
        self.mesh = mesh
        self.bc = bc
        return

    def _get_bc_nodes(self, targets, start, end):
        all_nodes = []
        for target in targets:
            nodes = self.mesh.get_bc_nodes(target, "T3D2")

            if not start:
                nodes = nodes[1:]
            if not end:
                nodes = nodes[:-1]

            all_nodes.append(nodes)

        return np.unique(all_nodes)

    def add_and_link_source(self, model):
        targets = self.bc["target"]
        start = self.bc["start"]
        end = self.bc["end"]
        nodes = self._get_bc_nodes(targets, start, end)

        input_names = self.bc["input"]
        bc_src = DirichletBCSource(self.bc_name, input_names=input_names)

        if len(nodes) > 0:
            model.add_component(
                f"src_{self.bc_name}",
                len(nodes),
                bc_src,
            )

            for name in input_names:
                model.link(
                    f"src_soln.{name}",
                    f"src_{self.bc_name}.{name}",
                    src_indices=nodes,
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
        elif etype == "T3D2":
            if space == "H1":
                return basis.LagrangeBasis1D(1, names, kind=kind)

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
        elif etype == "T3D2":
            return basis.LineQuadrature(2)

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

        element_types = ["CPS3", "CPS4", "CPS6", "M3D9", "T3D2"]

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

    def get_bc_nodes(self, name, etype):
        conn = self.parser.get_edge_node(name, etype)
        return conn

    def get_num_nodes_on_bc(self, name, etype):
        return self.parser.get_edge_node(name, etype).shape[0]

    def plot(
        self,
        u,
        fig=None,
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
                if etype == "T3D2":
                    """do not attempt to plot when line element found"""
                    continue
                # Get the connectivity
                conn = self.convert_conn(etype, self.get_conn(name, etype))
                tri = mtri.Triangulation(x, y, conn)

                # Set the contour plot
                cntr = ax.tricontourf(tri, u, levels=levels, cmap=cmap)
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
                # fig.colorbar(cntr, ax=ax)
        return ax

    def plot_3d(
        self,
        w,
        fig=None,
        ax=None,
        scale=1.0,
        cmap="coolwarm",
        title=None,
        x_offset=0.0,
        y_offset=0.0,
        alpha=0.85,
        show_edges=True,
        min_level=None,
        max_level=None,
    ):
        """
        Plot the 2D mesh lifted into 3D by the out-of-plane displacement field w.

        Parameters
        ----------
        w         : (nnodes,) array — perpendicular displacement at each node
        fig/ax    : existing Figure / Axes3D to draw into (created if None)
        scale     : multiply w before plotting (useful for visual exaggeration)
        cmap      : matplotlib colormap name
        title     : axes title string
        x_offset  : shift all x coordinates before plotting
        y_offset  : shift all y coordinates before plotting
        alpha     : surface transparency
        show_edges: overlay mesh wire-frame on the surface
        min_level / max_level : clamp the colormap range
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        if fig is None or ax is None:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection="3d")

        x = self.X[:, 0] + x_offset
        y = self.X[:, 1] + y_offset
        z = np.asarray(w) * scale

        # Colormap normalisation
        vmin = np.min(z) if min_level is None else min_level * scale
        vmax = np.max(z) if max_level is None else max_level * scale
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

        volumes = self.get_domains()

        for name in volumes:
            for etype in volumes[name]:
                if etype == "T3D2":
                    continue

                # Triangulate so we can colour per-triangle face
                tri_conn = self.convert_conn(etype, self.get_conn(name, etype))

                verts = []
                face_colors = []
                for tri in tri_conn:
                    pts = np.column_stack([x[tri], y[tri], z[tri]])  # (3, 3)
                    verts.append(pts)
                    # colour by mean displacement of the triangle
                    face_colors.append(scalar_map.to_rgba(np.mean(z[tri])))

                poly = Poly3DCollection(
                    verts,
                    facecolors=face_colors,
                    edgecolors="k" if show_edges else "none",
                    linewidths=0.3 if show_edges else 0.0,
                    alpha=alpha,
                )
                ax.add_collection3d(poly)

        # Axis limits derived from data
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_zlim(vmin, vmax)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("w")

        if title is not None:
            ax.set_title(title)

        fig.colorbar(scalar_map, ax=ax, shrink=0.5, label="w")

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

    def plot_tri_mesh_region(self, name, etype, ax, color, label):
        X2d = self.X[:, 0:2]  # Reduce to 2d (x,y coords only)
        gmsh_conn = self.get_conn(name, etype)
        polygons = [X2d[row] for row in gmsh_conn]

        coll = PolyCollection(
            polygons,
            facecolors=color,
            edgecolors="k",
            linewidths=0.01,
            label=label,
            antialiaseds=False,
        )
        ax.add_collection(coll)
        return


# Needs to take in bcs here
class Problem:
    # soln_space = object
    def __init__(
        self,
        mesh,
        soln_space: basis.SolutionSpace,
        data_space: basis.SolutionSpace,
        geo_space: basis.SolutionSpace,
        weakform_map={},
        output_map={},
        bc_map={},
    ):
        self.mesh = mesh
        self.soln_space = soln_space
        self.data_space = data_space
        self.geo_space = geo_space

        self.weakform_map = weakform_map
        self.bc_map = bc_map
        self.output_map = output_map

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

        self.dirichlet_dof = []
        self.symm_dof = []

        for name in bc_map:
            bc = bc_map[name]
            if bc["type"] == "dirichlet":
                self.dirichlet_dof.append(
                    DirichletDegreesOfFreedom(name, self.mesh, bc)
                )
            elif bc["type"] == "symmetry":
                self.symm_dof.append(SymmetryDegreesOfFreedom(name, self.mesh, bc))
            else:
                raise Exception(f"{bc["type"]} not recognized")

        return

    def create_model(self, module_name: str):
        """Create and link the Amigo model"""
        model = am.Model(module_name)

        self.soln_dof.add_source(model)
        self.data_dof.add_source(model)
        self.geo_dof.add_source(model)

        # Get the domain names from the mesh
        domains = self.mesh.get_domains()

        # Figure out which elements need to be created to go with this weak form
        element_objs = {}
        for weakform_name in self.weakform_map:
            targets = self.weakform_map[weakform_name]["target"]
            weakform = self.weakform_map[weakform_name]["weakform"]

            # Figure out the element types that we need
            etypes = []
            for target in targets:
                for etype in domains[target]:
                    if not etype in etypes:
                        etypes.append(etype)

            # Loop over the element types for this weakform
            for etype in etypes:
                # Set the element name
                elem_name = f"Element{weakform_name}{etype}"

                # Get the basis objects for the element type
                soln_basis = self.soln_dof.get_basis(etype)
                data_basis = self.data_dof.get_basis(etype)
                geo_basis = self.geo_dof.get_basis(etype)

                # quadrature = self.soln_dof.get_quadrature(etype)
                # # Create the quadrature instance
                if weakform_name == "shear_potential":
                    quadrature = basis.ReducedQuadQuadrature()
                else:
                    quadrature = self.soln_dof.get_quadrature(etype)

                # Create the element object
                obj = FiniteElement(
                    elem_name, soln_basis, data_basis, geo_basis, quadrature, weakform
                )

                # Set this into the element dictionary
                element_objs[(weakform_name, etype)] = obj

        # Add the element component objects
        for weakform_name in self.weakform_map:
            targets = self.weakform_map[weakform_name]["target"]

            for target in targets:
                for etype in domains[target]:
                    elem = element_objs[(weakform_name, etype)]
                    comp_name = f"Element{weakform_name}{etype}{target}"

                    # Add the element/component
                    nelems = self.mesh.get_num_elements(target, etype)
                    model.add_component(comp_name, nelems, elem)

                    # Link all the element dof to the component
                    self.soln_dof.link_dof(model, target, etype, comp_name)
                    self.data_dof.link_dof(model, target, etype, comp_name)
                    self.geo_dof.link_dof(model, target, etype, comp_name)

        # Add BC components and links
        for dof in self.dirichlet_dof:
            dof.add_and_link_source(model)

        for dof in self.symm_dof:
            dof.add_and_link_source(model)

        # Make a list of all of the outputs
        all_outputs = []
        for out_name in self.output_map:
            for name in self.output_map[out_name]["names"]:
                if not (name in all_outputs):
                    all_outputs.append(name)

        # Add the outputs component
        model.add_component("outputs", 1, DofSource(output_names=all_outputs))

        # Create the output objects
        output_objs = {}
        for out_name in self.output_map:
            targets = self.output_map[out_name]["target"]
            output_names = self.output_map[out_name]["names"]
            output_func = self.output_map[out_name]["function"]

            # Figure out the element types we need
            etypes = []
            for target in targets:
                for etype in domains[target]:
                    if not etype in etypes:
                        etypes.append(etype)

            # Loop over the element types for generating the output function
            for etype in etypes:
                elem_name = f"ElementOutput{out_name}{etype}"

                # Get the basis objects for the element type
                soln_basis = self.soln_dof.get_basis(etype)
                data_basis = self.data_dof.get_basis(etype)
                geo_basis = self.geo_dof.get_basis(etype)

                # Create the quadrature instance
                quadrature = self.soln_dof.get_quadrature(etype)

                # Create the output object
                obj = FiniteElementOutput(
                    elem_name,
                    soln_basis,
                    data_basis,
                    geo_basis,
                    quadrature,
                    output_names,
                    output_func,
                )

                # Set this into the output dictionary
                output_objs[(out_name, etype)] = obj

        for out_name in self.output_map:
            targets = self.output_map[out_name]["target"]

            for target in targets:
                for etype in domains[target]:
                    obj = output_objs[(out_name, etype)]
                    comp_name = f"ElementOutput{out_name}{etype}{target}"

                    # Add the element/component
                    nelems = self.mesh.get_num_elements(target, etype)
                    model.add_component(comp_name, nelems, obj)

                    # Link all the element dof to the component
                    self.soln_dof.link_dof(model, target, etype, comp_name)
                    self.data_dof.link_dof(model, target, etype, comp_name)
                    self.geo_dof.link_dof(model, target, etype, comp_name)

                    # Link the outputs
                    for name in output_names:
                        model.link(f"{comp_name}.{name}", f"outputs.{name}[0]")

        # Link the output to the finite element class
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


class FiniteElementOutput(am.Component):
    def __init__(
        self,
        name,
        soln_basis,
        data_basis,
        geo_basis,
        quadrature,
        output_names,
        output_function,
    ):
        super().__init__(name=name)

        self.soln_basis = soln_basis
        self.data_basis = data_basis
        self.geo_basis = geo_basis
        self.quadrature = quadrature
        self.output_names = output_names
        self.output_function = output_function

        # From BasisCollection
        self.soln_basis.add_declarations(self)
        self.geo_basis.add_declarations(self)
        self.data_basis.add_declarations(self)

        # add the output declarations, assuming scalar functions
        for name in self.output_names:
            self.add_output(name)

        # Set the arguments to the compute function for each quadrature point
        self.set_args(self.quadrature.get_args())

        return

    def compute_output(self, **args):
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
        outputs = self.output_function(soln_phys, data=data_phys, geo=geo)

        for name in self.output_names:
            if name in outputs:
                self.outputs[name] = quad_weight * detJ * outputs[name]
            else:
                self.outputs[name] = 0.0
        return
