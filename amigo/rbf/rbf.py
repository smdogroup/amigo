import amigo as am
import numpy as np
from typing import List


class RBFSource(am.Component):
    def __init__(self, input_names, output_name):
        """Create a source model"""
        super().__init__()

        for name in input_names:
            self.add_input(name)
        self.add_input(output_name)


class RBFKernel(am.Component):
    def __init__(self, input_names, weight_name, theta_names, base_names, con_name):
        super().__init__()

        # Inputs are variables within the optimization problem
        self.input_names = input_names

        # Fixed values within the optimization procedure
        self.theta_names = theta_names
        self.base_names = base_names

        # Weight for the basis function
        self.weight_name = weight_name

        # What is the output function
        self.con_name = con_name

        for input, theta, base in zip(
            self.input_names, self.theta_names, self.base_names
        ):
            self.add_input(input)
            self.add_data(theta)
            self.add_data(base)

        self.add_data(self.weight_name)
        self.add_constraint(self.con_name)

        return

    def compute(self):

        r2 = 0.0
        for input, theta, base in zip(
            self.input_names, self.theta_names, self.base_names
        ):
            x = self.inputs[input]
            t = self.data[theta]
            x0 = self.data[base]
            r2 += (t * (x - x0)) ** 2

        weight = self.data[self.weight_name]
        self.constraints[self.con_name] = weight * am.exp(-r2)

        return


class RBFConstraint(am.Component):
    def __init__(self, result_name, con_name):
        super().__init__()

        self.con_name = con_name
        self.result_name = result_name
        self.add_input(self.result_name)
        self.add_constraint(self.con_name)

    def compute(self):
        self.constraints[self.con_name] = -self.inputs[self.result_name]
        return


class RBF:
    """
    Perform RBF interpolation.

    for i in range(num_points):
        output_name[i] = weights[j] * exp(- theta[k] * (input[i, k] - base[j, k])**2)
    """

    def __init__(
        self, num_points, output_name: str, input_names: List[str], xt, yt, theta
    ):
        num_basis = len(yt)
        if xt.shape[0] != num_basis:
            raise ValueError("Inconsistent training input and output sizes")
        if len(input_names) != xt.shape[1]:
            raise ValueError("Inconsistent input_names and training data sizes")

        self.xt = xt
        self.yt = yt
        self.theta = theta

        # Construct the surrogate model
        self.weights = self._build_model(self.xt, self.yt, self.theta)

        # Number of interpolation point (i in range(num_points))
        self.num_points = num_points

        # Number of RBF basis functions (j in range(num_basis))
        self.num_basis = num_basis

        self.output_name = output_name
        self.input_names = input_names

        # One theta value for each input name
        self.theta_names = []
        self.base_names = []
        for name in self.input_names:
            self.theta_names.append(f"{name}_theta")
            self.base_names.append(f"{name}_base")

        # Set the name of the weights
        self.weight_name = "weight"

        # Set the name of the constraint
        self.con_name = "constraint"

        return

    def _build_model(self, xt, yt, theta):
        nt = len(yt)

        # Form the matrix Phi
        Phi = np.zeros((nt, nt))

        # Evaluate the function
        for i in range(nt):
            for j in range(nt):
                d = theta * (xt[i] - xt[j])
                r2 = np.dot(d, d)
                Phi[i, j] = np.exp(-r2)

        return np.linalg.solve(Phi, yt)

    def create_model(self, module_name):
        model = am.Model(module_name)

        # Add the source
        src = RBFSource(self.input_names, self.output_name)
        model.add_component("src", self.num_points, src)

        # Create the instance of the kernel component
        kernel = RBFKernel(
            self.input_names,
            self.weight_name,
            self.theta_names,
            self.base_names,
            self.con_name,
        )
        model.add_component("kernel", self.num_points * self.num_basis, kernel)

        # Create the instance of the constraint component
        kernel = RBFConstraint(self.output_name, self.con_name)
        model.add_component("constraint", self.num_points, kernel)

        # Link the constraints
        idx = np.repeat(np.arange(self.num_points), self.num_basis)
        model.link(
            f"constraint.{self.con_name}", f"kernel.{self.con_name}", src_indices=idx
        )

        # Link the input variables
        for name in self.input_names:
            model.link(f"src.{name}", f"kernel.{name}", src_indices=idx)

        # Link the base points
        idx = np.tile(np.arange(self.num_basis), self.num_points - 1)
        for name in self.base_names:
            model.link(
                f"kernel.{name}[:{self.num_basis}]",
                f"kernel.{name}[{self.num_basis}:]",
                src_indices=idx,
            )

        # Link the weights
        model.link(
            f"kernel.{self.weight_name}[:{self.num_basis}]",
            f"kernel.{self.weight_name}[{self.num_basis}:]",
            src_indices=idx,
        )

        # Link the theta values
        for name in self.theta_names:
            model.link(f"kernel.{name}[0]", f"kernel.{name}[1:]")

        # Link the outputs
        model.link(f"src.{self.output_name}", f"constraint.{self.output_name}")

        return model

    def set_data(self, model, sub_name=None):

        data = model.get_data_vector()

        def _get_name(name):
            if sub_name is None:
                return name
            else:
                return f"{sub_name}.name"

        # Set the values of theta, the weights and the base locations
        for i, name in enumerate(self.theta_names):
            data[_get_name(f"kernel.{name}[0]")] = self.theta[i]

        data[_get_name(f"kernel.{self.weight_name}[:{self.num_basis}]")] = self.weights

        for i, name in enumerate(self.base_names):
            data[_get_name(f"kernel.{name}[:{self.num_basis}]")] = self.xt[:, i]
