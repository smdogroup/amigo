import amigo as am
from abc import ABC, abstractmethod


class FiniteElement(am.Component):
    def __init__(
        self,
        name,
        soln_basis,
        data_basis,
        geo_basis,
        quadrature,
        potential,
    ):
        super().__init__(name=name)

        self.soln_basis = soln_basis
        self.data_basis = data_basis
        self.geo_basis = geo_basis
        self.quadrature = quadrature
        self.potential = potential

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
            quad_weight * detJ * self.potential(soln_phys, data=data_phys, geo=geo)
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


class MITCTyingStrain(ABC):
    @abstractmethod
    def get_tying_points(self):
        """Return all the tying points within the element"""
        pass

    @abstractmethod
    def eval_tying_strain(self, index, geo, soln):
        """Evaluate the covariant component of the strain at the tying point"""
        return 0.0

    @abstractmethod
    def interp_and_transform(self, pt, Jinv, tying_strains):
        """Interpolate and transform the tying strain"""
        return {}


class MITCElement(FiniteElement):
    def __init__(
        self, name, soln_basis, data_basis, geo_basis, quadrature, mitc, potential
    ):
        if not isinstance(mitc, MITCTyingStrain):
            raise ValueError("MITCElement requires instance of MITCTyingStrain")

        super().__init__(name, soln_basis, data_basis, geo_basis, quadrature, potential)

        self.mitc = mitc

    def compute(self, **args):
        # Compute the tensorial strain components at tying points
        tensorial_strains = []
        for index, pt in enumerate(self.mitc.get_tying_points()):
            soln_xi = self.soln_basis.eval(self, pt)
            geo = self.geo_basis.eval(self, pt)

            # Append the tying strain
            tensorial_strains.append(self.mitc.eval_tying_strain(index, geo, soln_xi))

        # Now get the quadrature point
        quad_weight, quad_point = self.quadrature.get_point(**args)

        # Evaluate the solution fields/data fields (u)
        soln_xi = self.soln_basis.eval(self, quad_point)
        data_xi = self.data_basis.eval(self, quad_point)
        geo = self.geo_basis.eval(self, quad_point)

        # Perform the mapping from computational to physical coordinates (u)
        detJ, Jinv = self.geo_basis.compute_transform(geo)
        soln_phys = self.soln_basis.transform(detJ, Jinv, soln_xi)
        data_phys = self.data_basis.transform(detJ, Jinv, data_xi)

        # Add to the physics
        soln_phys.update(
            self.mitc.interp_and_transform(quad_point, Jinv, tensorial_strains)
        )

        # Add the contributions directly to the Lagrangian
        self.objective["obj"] = (
            quad_weight * detJ * self.potential(soln_phys, data=data_phys, geo=geo)
        )


class MITCElementOutput(am.Component):
    def __init__(
        self,
        name,
        soln_basis,
        data_basis,
        geo_basis,
        quadrature,
        mitc,
        output_names,
        output_function,
    ):
        if not isinstance(mitc, MITCTyingStrain):
            raise ValueError("MITCElement requires instance of MITCTyingStrain")

        super().__init__(
            name,
            soln_basis,
            data_basis,
            geo_basis,
            quadrature,
            output_names,
            output_function,
        )
        self.mitc = mitc

        return

    def compute_output(self, **args):
        # Compute the tensorial strain components at tying points
        tensorial_strains = []
        for index, pt in enumerate(self.mitc.get_tying_points()):
            soln_xi = self.soln_basis.eval(self, pt)
            geo = self.geo_basis.eval(self, pt)

            # Append the tying strain
            tensorial_strains.append(self.mitc.eval_tying_strain(index, geo, soln_xi))

        # Now get the quadrature point
        quad_weight, quad_point = self.quadrature.get_point(**args)

        # Evaluate the solution fields/data fields (u)
        soln_xi = self.soln_basis.eval(self, quad_point)
        data_xi = self.data_basis.eval(self, quad_point)
        geo = self.geo_basis.eval(self, quad_point)

        # Perform the mapping from computational to physical coordinates (u)
        detJ, Jinv = self.geo_basis.compute_transform(geo)
        soln_phys = self.soln_basis.transform(detJ, Jinv, soln_xi)
        data_phys = self.data_basis.transform(detJ, Jinv, data_xi)

        # Add to the physics
        soln_phys.update(
            self.mitc.interp_and_transform(quad_point, Jinv, tensorial_strains)
        )

        # Add the contributions directly to the Lagrangian
        outputs = self.output_function(soln_phys, data=data_phys, geo=geo)

        for name in self.output_names:
            if name in outputs:
                self.outputs[name] = quad_weight * detJ * outputs[name]
            else:
                self.outputs[name] = 0.0
        return
