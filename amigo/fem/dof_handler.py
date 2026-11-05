from dataclasses import dataclass
import amigo as am
import numpy as np
from .fem_space import Space, Conformity, FunctionSpace, SolutionSpace
from .cell_types import DofLayout
from .mesh import Mesh


@dataclass
class DirichletBCsForDof:
    name: str
    dofs: np.ndarray
    values: np.ndarray


class DofHandler:
    """
    Construct canonical finite-element DOF numbering for each distinct
    FunctionSpace in a SolutionSpace.

    Field names are NOT represented here. For example, if u, v, and w all
    belong to the same H1(p=2, GLOBAL) FunctionSpace, they use the same
    canonical DOF connectivity.

    The DegreesOfFreedom class is responsible for applying that canonical
    numbering separately to u, v, w, etc.
    """

    def __init__(self, mesh: Mesh, solution_space: SolutionSpace):
        self.mesh = mesh
        self.solution_space = solution_space

        # Number of DOFs associated with each FunctionSpace
        #  (FunctionSpace) -> int
        self._num_dof = {}

        # Element connectivity for each function space
        #   (FunctionSpace, domain, CellType) -> ndarray
        # Each array has shape (num_elements, num_element_dof)
        self._dof_conn = {}

        # Save the dof signs for H(div) and H(curl) spaces
        #   (FunctionSpace, domain, CellType) -> ndarray [+/- 1] only
        self._dof_signs = {}

        # Build the DOF numbering.
        self._build()

    def get_mesh(self):
        return self.mesh

    def get_solution_space(self):
        return self.solution_space

    def get_num_dof(self, input: FunctionSpace | str) -> int:
        """
        Return the total number of DOFs for a FunctionSpace.
        """
        if isinstance(input, str):
            func_space = self.solution_space.get_space(input)
            return self._num_dof[func_space]
        elif isinstance(input, FunctionSpace):
            return self._num_dof[input]
        else:
            raise TypeError("Input must be FunctionSpace or str")

    def get_dof_conn(self, space: FunctionSpace, block_id: int) -> np.ndarray:
        """
        Return element-local -> global DOF connectivity.

        Returns
        -------
        conn : ndarray
            Shape (num_elements, num_element_dof).
        """
        return self._dof_conn[(space, block_id)]

    def get_dof_signs(self, space: FunctionSpace, block_id: int) -> np.ndarray:
        """
        Return element-local -> global DOF array of sign transformations.

        Returns
        -------
        signs : ndarray of +/- 1s
            Shape (num_elements, num_element_dof).
        """
        if space.func_space == Space.HDIV or space.func_space == Space.HCURL:
            return self._dof_signs[(space, block_id)]
        else:
            raise NotImplementedError

    def get_dof_in_domain(self, name: str, domain: str):
        """
        Return all the DOFs for the given variable name in the specified domain
        """

        # Get the function space associated with the variable name
        space = self.solution_space.get_space(name)
        block_ids = self.mesh.get_block_ids(domain)

        all_dof = []
        for block_id in block_ids:
            all_dof.extend(self._dof_conn[(space, block_id)])

        return np.array(all_dof, dtype=np.int64)

    def _build(self):
        """
        Build the DOF numbering independently for each FunctionSpace.
        """

        for func_space in self.solution_space.get_spaces():
            self._build_space(func_space)

    def _build_space(self, func_space: FunctionSpace):
        """
        Build one canonical global numbering for a FunctionSpace.
        """

        # These dictionaries identify DOFs associated with mesh entities.
        #
        # key -> global DOF number
        vertex_dof = {}
        edge_dof = {}
        face_dof = {}
        cell_dof = {}

        next_dof = 0

        # Build the layouts
        elem_layouts = {}
        for cell_type in self.mesh.get_cell_types():
            elem_layouts[cell_type] = DofLayout.make_layout(func_space, cell_type)

        # For each cell type in each domain
        for block_id in range(self.mesh.get_num_blocks()):
            # Extract the domain name
            nelem = self.mesh.get_num_elements(block_id)
            domain = self.mesh.get_domain_name(block_id)
            cell_type = self.mesh.get_cell_type(block_id)
            layout = elem_layouts[cell_type]

            # Get the vertex connectivity
            vertex_conn = self.mesh.get_vertex_conn(block_id)

            # Edge/face topology may not exist for every reference cell.
            edge_conn = None
            edge_orientation = None
            if len(layout.edge_dofs) > 0:
                edge_conn, edge_orientation = self.mesh.get_edge_conn(block_id)

            face_conn = None
            face_orientation = None
            if len(layout.face_dofs) > 0:
                face_conn, face_orientation = self.mesh.get_face_conn(block_id)

            # Create the connectivity
            conn = np.empty((nelem, layout.ndof), dtype=np.int64)

            # Create the signs array
            signs = None
            if (
                func_space.func_space == Space.HDIV
                or func_space.func_space == Space.HCURL
            ):
                signs = np.ones((nelem, layout.ndof), dtype=float)

            for elem in range(nelem):
                # Vertex DOFs
                for vertex_index, local_dof in enumerate(layout.vertex_dofs):
                    entity_id = int(vertex_conn[elem, vertex_index])

                    key = self.make_entity_key(
                        func_space=func_space,
                        domain=domain,
                        entity_id=entity_id,
                        entity_dof=0,
                    )

                    if key not in vertex_dof:
                        vertex_dof[key] = next_dof
                        next_dof += 1

                    conn[elem, vertex_index] = vertex_dof[key]

                # Edge DOFs
                for edge_index, local_dofs in enumerate(layout.edge_dofs):
                    entity_id = int(edge_conn[elem, edge_index])

                    for entity_dof, local_dof in enumerate(local_dofs):
                        # Flip the edge orientation
                        edge_entity_dof = entity_dof
                        edge_sign = 1.0
                        if edge_orientation[elem, edge_index] < 0:
                            edge_entity_dof = len(local_dofs) - 1 - entity_dof
                            edge_sign = -1.0

                        key = self.make_entity_key(
                            func_space=func_space,
                            domain=domain,
                            entity_id=entity_id,
                            entity_dof=edge_entity_dof,
                        )

                        if key not in edge_dof:
                            edge_dof[key] = next_dof
                            next_dof += 1

                        conn[elem, local_dof] = edge_dof[key]
                        if signs is not None:
                            signs[elem, local_dof] = edge_sign

                # Face DOFs
                for local_face, local_dofs in enumerate(layout.face_dofs):
                    entity_id = int(face_conn[elem, local_face])

                    for entity_dof, local_dof in enumerate(local_dofs):
                        key = self.make_entity_key(
                            func_space=func_space,
                            domain=domain,
                            entity_id=entity_id,
                            entity_dof=entity_dof,
                        )

                        if key not in face_dof:
                            face_dof[key] = next_dof
                            next_dof += 1

                        conn[elem, local_dof] = face_dof[key]

                # Cell-interior DOFs
                for entity_dof, local_dof in enumerate(layout.cell_dofs):
                    # Interior DOFs belong to the element itself.
                    #
                    # The domain/cell_type/elem tuple uniquely
                    # identifies the element within the mesh chunk.
                    key = (domain, cell_type, elem, entity_dof)

                    if key not in cell_dof:
                        cell_dof[key] = next_dof
                        next_dof += 1

                    conn[elem, local_dof] = cell_dof[key]

            # Store element connectivity.
            chunk_key = (func_space, block_id)
            self._dof_conn[chunk_key] = conn
            self._dof_signs[chunk_key] = signs

        self._num_dof[func_space] = next_dof

        return

    @staticmethod
    def make_entity_key(
        func_space: FunctionSpace,
        domain: str,
        entity_id: int,
        entity_dof: int,
    ):
        """
        Construct the key determining whether two element DOFs represent
        the same global DOF.

        GLOBAL
            DOFs are shared wherever the underlying mesh entity is shared.

        COMPONENT
            DOFs are shared only inside the same domain/component.
        """

        conformity = func_space.conformity
        if conformity == Conformity.GLOBAL:
            return (entity_id, entity_dof)
        elif conformity == Conformity.COMPONENT:
            return (domain, entity_id, entity_dof)

        raise ValueError(f"Unsupported conformity {conformity}")

    def get_dof_to_node_map(self, input: FunctionSpace | str):
        """
        Get the mapping between the ordering of the nodes parsed from the mesh
        and the dof handler ordering.

        x_dof = x_node[mapping]

        Given the dof, the mapping gives the corresponding node

        mapping[dof] = node
        """
        if isinstance(input, str):
            func_space = self.solution_space.get_space(input)
        elif isinstance(input, FunctionSpace):
            func_space = input
        else:
            raise TypeError(f"Expected FunctionSpace or str, got {type(input)}")

        ndof = self._num_dof[func_space]
        mapping = np.empty(ndof, dtype=np.int64)

        for block_id in range(self.mesh.get_num_blocks()):
            # Extract the local connectivity
            chunk_key = (func_space, block_id)
            dof_conn = self._dof_conn[chunk_key]

            # Extract the underlying source node dofs from the mesh input
            block = self.mesh.blocks[block_id]
            block_conn = block.connectivity
            source_dofs = block.element_type.get_entity_dofs()
            if dof_conn.shape[1] > len(source_dofs):
                raise ValueError(
                    f"Function space has {dof_conn.shape[1]} local DOFs, but mesh "
                    f"element type {block.source_type} provides only "
                    f"{len(source_dofs)} source nodes."
                )

            for i, source_dof in enumerate(source_dofs[: dof_conn.shape[1]]):
                mapping[dof_conn[:, i]] = block_conn[:, source_dof]

        return mapping

    def get_dirichlet_bcs(self):
        """Get the dofs and values for the Dirichlet Boundary conditions"""

        # Boundary conditions on the degrees of freedom
        dof_bcs = {}

        # Get the boundary conditions on the underlying nodes (if any)
        bcs = self.mesh.get_dirichlet_bcs()

        if len(bcs) > 0:
            # Build a map of index to name for the solution
            bc_map = {}
            for i, (name, space) in enumerate(self.solution_space.fields):
                bc_map[name] = i

            # Find and apply the mapping for each space
            for space in self.solution_space.get_spaces():
                mapping = self.get_dof_to_node_map(space)
                inverse = np.empty_like(mapping)
                inverse[mapping] = np.arange(len(mapping))

                for name in self.solution_space.get_names(space):
                    index = bc_map[name]

                    dofs = inverse[bcs[index].nodes]
                    dof_bcs[name] = DirichletBCsForDof(
                        name=name, dofs=dofs, values=bcs[index].values
                    )

        return dof_bcs


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


class DegreesOfFreedom:
    def __init__(
        self,
        mesh: Mesh,
        solution_space: SolutionSpace,
        kind: str | list[str] = "input",
        prefix=None,
    ):
        """
        Allocate the degrees of freedom on the mesh
        """

        self.mesh = mesh
        self.solution_space = solution_space
        if isinstance(kind, str):
            self.kind = [kind]
        else:
            self.kind = kind
        self.prefix = prefix

        # Create the DOF handler
        self.dof_handler = DofHandler(mesh, solution_space)

        return

    def get_dof_handler(self):
        return self.dof_handler

    def add_source(self, model: am.Model):
        for space in self.solution_space.get_spaces():
            comp_name = self.solution_space.get_component_name(space)
            names = self.solution_space.get_names(space)

            if len(names) == 0:
                continue

            if (
                space.func_space == Space.CONST
                and space.conformity == Conformity.COMPONENT
            ):
                # Create a sub-model for each data set
                sub_model = am.Model()
                for data_name in names:

                    domain_names = self.mesh.get_domain_names()
                    input_names, data_names = [], []
                    con_names, output_names = [], []
                    if "input" in self.kind:
                        input_names = domain_names
                    if "data" in self.kind:
                        data_names = domain_names
                    if "multiplier" in self.kind:
                        con_names = [f"res_{name}" for name in domain_names]
                    if "output" in self.kind:
                        output_names = domain_names

                    dof_src = DofSource(
                        input_names=input_names,
                        con_names=con_names,
                        data_names=data_names,
                        output_names=output_names,
                    )
                    sub_model.add_component(data_name, 1, dof_src)

                sub_model_name = comp_name
                model.add_model(sub_model_name, sub_model)
            else:
                input_names, data_names = [], []
                con_names, output_names = [], []
                if "input" in self.kind:
                    input_names = names
                if "data" in self.kind:
                    data_names = names
                if "multiplier" in self.kind:
                    con_names = [f"res_{name}" for name in names]
                if "output" in self.kind:
                    output_names = names

                # Create the source component
                dof_src = DofSource(
                    input_names=input_names,
                    con_names=con_names,
                    data_names=data_names,
                    output_names=output_names,
                )

                # Get the number of degrees of freedom associated with the
                # associated space
                if space.func_space == Space.CONST:
                    ndof = 1
                else:
                    ndof = self.dof_handler.get_num_dof(space)

                # Add the dof from the source mesh
                model.add_component(comp_name, ndof, dof_src)

        return

    def link_dof(self, model: am.Model, block_id: int, elem_name: str):
        domain = self.mesh.get_domain_name(block_id)

        for space in self.solution_space.get_spaces():
            comp_name = self.solution_space.get_component_name(space)
            names = self.solution_space.get_names(space)
            if len(names) == 0:
                continue

            if (
                space.func_space == Space.CONST
                and space.conformity == Conformity.COMPONENT
            ):
                for name in names:
                    model.link(
                        f"{comp_name}.{name}.{domain}",
                        f"{elem_name}.{name}[:]",
                    )

            else:
                names_list = []
                if "input" in self.kind:
                    names_list.append(names)
                if "data" in self.kind:
                    names_list.append(names)
                if "multiplier" in self.kind:
                    names_list.append([f"res_{name}" for name in names])

                if space.func_space == Space.CONST:
                    # Fake the connectivity for the constant case
                    conn = [0]
                else:
                    # Get the connectivity for the function space
                    conn = self.dof_handler.get_dof_conn(space, block_id)

                for var_names in names_list:
                    # Link the degrees of freedom
                    if space.func_space == Space.HDIV:
                        for name in var_names:
                            model.link(
                                f"{comp_name}.{name}",
                                f"{elem_name}.{name}[:, 0, :]",
                                src_indices=conn,
                            )
                        for name in var_names:
                            model.link(
                                f"{comp_name}.{name}",
                                f"{elem_name}.{name}[:, 1, :]",
                                src_indices=conn,
                            )
                    else:
                        for name in var_names:
                            model.link(
                                f"{comp_name}.{name}",
                                f"{elem_name}.{name}",
                                src_indices=conn,
                            )
                if "output" in self.kind:
                    if space.func_space == Space.H1:
                        for name in names:
                            for dim in range(conn.shape[1]):
                                model.link(
                                    f"{comp_name}.{name}",
                                    f"{elem_name}.{name}{dim}",
                                    src_indices=conn[:, dim],
                                )

        return

    def get_coordinates(self):
        """
        Get the coordinates from the underlying mesh in the DOF handler order.
        """

        geo_space = self.solution_space.get_spaces()[0]
        dof_to_node = self.dof_handler.get_dof_to_node_map(geo_space)

        if np.any(dof_to_node < 0):
            raise ValueError(
                "Some geometry DOFs are not associated with a mesh vertex; "
                "setting nodal coordinates requires a degree-1 H1 geometry space."
            )

        return self.mesh.X[dof_to_node]
