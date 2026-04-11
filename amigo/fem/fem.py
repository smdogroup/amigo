import amigo as am
import numpy as np
from . import basis
from .connectivity import InpParser
from .element import FiniteElement, FiniteElementOutput
from .plot_utils import plot
from pathlib import Path


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


class ScaledBC(am.Component):
    def __init__(self, name, input_name=[], scale=[1.0, 1.0]):
        super().__init__(name)

        if len(scale) != 2:
            raise ValueError("scale must be of length 2")

        self.input_name = input_name

        self.add_constant(f"scale_left", value=scale[0])
        self.add_constant(f"scale_right", value=scale[1])

        for name in self.input_name:
            self.add_input(f"{name}_left", value=1.0)
            self.add_input(f"{name}_right", value=1.0)
            self.add_constraint(f"res_{name}")

        return

    def compute(self):
        scale_left = self.constants["scale_left"]
        scale_right = self.constants["scale_right"]

        for name in self.input_name:
            self.constraints[f"res_{name}"] = (
                scale_left * self.inputs[f"{name}_left"]
                + scale_right * self.inputs[f"{name}_right"]
            )
        return


class BoundaryConditions:
    def __init__(self, bc_name, mesh, bc={}, integrand_formulation="potential"):
        if not (
            bc["type"] == "dirichlet"
            or bc["type"] == "continuity"
            or bc["type"] == "scaled"
        ):
            typ = bc["type"]
            raise ValueError(f"Unrecognized boundary condition type {typ}")
        self.bc_name = bc_name
        self.mesh = mesh
        self.bc = bc
        self.integrand_formulation = integrand_formulation
        return

    def _get_bc_nodes(self, targets, start=True, end=True):
        all_nodes = []
        for target in targets:
            nodes = self.mesh.get_nodes_in_domain(target)
            all_nodes.extend(nodes)

        unique = list(dict.fromkeys(all_nodes))

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

    def _get_matched_nodes(self, targets, start=True, end=True):
        left_target_lines = targets[0]
        right_target_lines = targets[1]
        nodes_left = self._get_bc_nodes(left_target_lines, start, end)
        nodes_right = self._get_bc_nodes(right_target_lines, start, end)

        if len(nodes_left) != len(nodes_right):
            raise Exception(f"nnodes left != nnodes right")

        # Reorder the nodes to match
        return self._reorder_nodes(nodes_left, nodes_right)

    def add_bcs(self, model):
        """Add the boundary conditions to the model"""

        if self.bc["type"] == "dirichlet":
            nodes = self._get_bc_nodes(self.bc["target"])

            input_names = self.bc["input"]
            for name in input_names:
                model.add_fixed(f"soln.{name}", nodes)

                if self.integrand_formulation == "weak":
                    model.add_fixed(f"multiplier.res_{name}", nodes)

        else:
            targets = self.bc["target"]
            start = self.bc.get("start", True)
            end = self.bc.get("end", True)

            nodes_left, nodes_right = self._get_matched_nodes(
                targets, start=start, end=end
            )

            input_names = self.bc["input"]
            if self.bc["type"] == "continuity":
                for name in input_names:
                    model.link(
                        f"soln.{name}",
                        f"soln.{name}",
                        src_indices=nodes_left,
                        tgt_indices=nodes_right,
                    )

            elif self.bc["type"] == "scaled":
                scale = self.bc["scale"]
                class_name = f"ScaledBC_{self.bc_name}"
                bc_src = ScaledBC(class_name, input_names, scale=scale)

                if len(nodes_left) > 0:
                    model.add_component(
                        f"{self.bc_name}",
                        len(nodes_left),
                        bc_src,
                    )

                    for name in input_names:
                        model.link(
                            f"soln.{name}",
                            f"{self.bc_name}.{name}_left",
                            src_indices=nodes_left,
                        )
                        model.link(
                            f"soln.{name}",
                            f"{self.bc_name}.{name}_right",
                            src_indices=nodes_right,
                        )


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

    def _add_h1_source(self, model):
        names = self.space.get_names("H1")
        if len(names) == 0:
            return

        input_names, data_names, con_names = [], [], []
        if self.kind == "input":
            input_names = names
        elif self.kind == "data":
            data_names = names
        elif self.kind == "multiplier":
            con_names = [f"res_{name}" for name in names]

        # Create the source component
        dof_src = DofSource(
            input_names=input_names, con_names=con_names, data_names=data_names
        )

        # Add global mesh source component
        nnodes = self.mesh.get_num_nodes()
        model.add_component(self.name, nnodes, dof_src)

        return

    def _link_h1(self, model, domain, etype, elem_name):
        names = self.space.get_names("H1")
        if len(names) == 0:
            return

        if self.kind == "multiplier":
            con_names = [f"res_{name}" for name in names]
            names = con_names

        conn = self.mesh.get_conn(domain, etype)
        for name in names:
            model.link(
                f"{self.name}.{name}",
                f"{elem_name}.{name}",
                src_indices=conn,
            )

        return name

    def _link_const(self, model, domain, elem_name):
        for name in self.space.get_names("const"):
            model.link(
                f"{self.name}.{name}.{domain}",
                f"{elem_name}.{name}[:]",
            )

    def _add_const_source(self, model):
        names = self.space.get_names("const")
        if len(names) == 0:
            return

        # Create a sub-model for each data set
        sub_model = am.Model()
        for data_name in names:

            domains = self.mesh.get_domains()

            domain_names = [name for name in domains]
            input_names, data_names, con_names = [], [], []
            if self.kind == "input":
                input_names = domain_names
            elif self.kind == "data":
                data_names = domain_names
            elif self.kind == "multiplier":
                con_names = [f"res_{name}" for name in domain_names]

            dof_src = DofSource(
                input_names=input_names, con_names=con_names, data_names=data_names
            )
            sub_model.add_component(data_name, 1, dof_src)

        model.add_model(self.name, sub_model)

        return

    def add_source(self, model):
        self._add_h1_source(model)
        self._add_const_source(model)
        return

    def link_dof(self, model, domain, etype, elem_name):
        self._link_h1(model, domain, etype, elem_name)
        self._link_const(model, domain, elem_name)
        return

    def get_basis(self, etype):
        return self.mesh.get_basis(self.space, etype, kind=self.kind)

    def get_quadrature(self, etype):
        return self.mesh.get_quadrature(etype)


class Mesh:
    def __init__(self, filename: str):
        ext = Path(filename).suffix
        if ext == ".inp" or ext == ".INP":
            self.parser = InpParser()
            self.parser.parse_inp(filename)
        else:
            raise ValueError(f"Unrecognized file extension {ext}")

        self.X = self.parser.get_nodes()

    def get_num_nodes(self):
        return self.X.shape[0]

    def get_domains(self):
        """
        Get a dictionary of the element types for each domain, indexed by the name
        of each domain
        """
        return self.parser.get_domains()

    def get_conn(self, name, etype):
        """
        Get the connectivity for the given domain name with the given element type
        """
        return self.parser.get_conn(name, etype)

    def get_basis(self, space, etype, kind):
        """
        Get an instance of a Basis class that is matched to the solution space, element
        type and kind of quantity
        """
        return self.parser.get_basis(space, etype, kind=kind)

    def get_quadrature(self, etype):
        """
        Get an instance of Quadrature that is matched to the element type
        """
        return self.parser.get_quadrature(etype)

    def get_nodes_in_domain(self, name):
        """
        Get nodes in the specified domain for all element types (if etype == None),
        or a specific element type.
        """
        return self.parser.get_nodes_in_domain(name)

    def get_num_elements(self, name, etype):
        return self.parser.get_conn(name, etype).shape[0]

    def plot(self, u, **kwargs):
        """Plot the finite element solution on the mesh"""
        plot(self, u, **kwargs)


class Problem:
    def __init__(
        self,
        mesh,
        soln_space: basis.SolutionSpace,
        data_space: basis.SolutionSpace,
        geo_space: basis.SolutionSpace,
        integrand_map={},
        integrand_formulation="potential",
        output_map={},
        bc_map={},
        element_objs={},
        output_objs={},
    ):
        self.mesh = mesh
        self.soln_space = soln_space
        self.data_space = data_space
        self.geo_space = geo_space

        self.integrand_map = integrand_map
        self.integrand_formulation = integrand_formulation
        self.bc_map = bc_map
        self.output_map = output_map

        # Set the element objects and output objects
        self.element_objs = element_objs
        self.output_objs = output_objs

        # Allocate constraints for the weak formulation
        self.test_dof = None
        if self.integrand_formulation == "weak":
            self.test_dof = DegreesOfFreedom(
                self.mesh,
                self.soln_space,
                kind="multiplier",
                name="multiplier",
            )

        # Initialize Dofs
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

        # Build the boundary conditions
        self.boundary_conditions = []

        for name in bc_map:
            bc = bc_map[name]
            self.boundary_conditions.append(
                BoundaryConditions(name, self.mesh, bc, self.integrand_formulation)
            )

        return

    def _create_element_objs(self):
        # Get the domain names from the mesh
        domains = self.mesh.get_domains()

        for integrand_name in self.integrand_map:
            targets = self.integrand_map[integrand_name]["target"]
            integrand = self.integrand_map[integrand_name]["integrand"]

            # Figure out the element types that we need
            etypes = []
            for target in targets:
                for etype in domains[target]:
                    if not etype in etypes:
                        etypes.append(etype)

            # Loop over the element types for this integrand
            for etype in etypes:
                if (integrand_name, etype) in self.element_objs:
                    continue

                # Set the element name
                elem_name = f"Element{integrand_name}{etype}"

                # Get the basis objects for the element type
                test_basis = None
                if self.test_dof is not None:
                    test_basis = self.test_dof.get_basis(etype)
                soln_basis = self.soln_dof.get_basis(etype)
                data_basis = self.data_dof.get_basis(etype)
                geo_basis = self.geo_dof.get_basis(etype)

                # Create the quadrature instance
                quadrature = self.soln_dof.get_quadrature(etype)

                # Create the element object
                obj = FiniteElement(
                    elem_name,
                    soln_basis,
                    data_basis,
                    geo_basis,
                    quadrature,
                    integrand,
                    test_basis=test_basis,
                )

                # Set this into the element dictionary
                self.element_objs[(integrand_name, etype)] = obj

        return

    def _create_output_objs(self):
        # Get the domain names from the mesh
        domains = self.mesh.get_domains()

        # Create the output objects
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
                if (out_name, etype) in self.element_objs:
                    continue

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
                self.output_objs[(out_name, etype)] = obj

        return

    def create_model(self, module_name: str):
        """Create and link the Amigo model"""
        model = am.Model(module_name)

        if self.test_dof is not None:
            self.test_dof.add_source(model)
        self.soln_dof.add_source(model)
        self.data_dof.add_source(model)
        self.geo_dof.add_source(model)

        # Get the domain names from the mesh
        domains = self.mesh.get_domains()

        # Figure out which elements need to be created
        self._create_element_objs()

        # Add the element component objects
        for integrand_name in self.integrand_map:
            targets = self.integrand_map[integrand_name]["target"]

            for target in targets:
                for etype in domains[target]:
                    elem = self.element_objs[(integrand_name, etype)]
                    comp_name = f"Element{integrand_name}{etype}{target}"

                    # Add the element/component
                    nelems = self.mesh.get_num_elements(target, etype)
                    model.add_component(comp_name, nelems, elem)

                    # Link all the element dof to the component
                    self.soln_dof.link_dof(model, target, etype, comp_name)
                    self.data_dof.link_dof(model, target, etype, comp_name)
                    self.geo_dof.link_dof(model, target, etype, comp_name)

                    # Link the constraints (if using the weak formulation)
                    if self.test_dof is not None:
                        self.test_dof.link_dof(model, target, etype, comp_name)

        # Add BC components and links
        for bc in self.boundary_conditions:
            bc.add_bcs(model)

        # Make a list of all of the outputs
        all_outputs = []
        for out_name in self.output_map:
            for name in self.output_map[out_name]["names"]:
                if not (name in all_outputs):
                    all_outputs.append(name)

        # Add the outputs component
        model.add_component("outputs", 1, DofSource(output_names=all_outputs))

        self._create_output_objs()

        for out_name in self.output_map:
            targets = self.output_map[out_name]["target"]
            output_names = self.output_map[out_name]["names"]

            for target in targets:
                for etype in domains[target]:
                    obj = self.output_objs[(out_name, etype)]
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

        # Set the node locations directly
        for k, name in enumerate(self.geo_space.get_names("H1")):
            model.set_data(f"geo.{name}", self.mesh.X[:, k])

        # Link the output to the finite element class
        return model
