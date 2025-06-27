import amigo as am
import argparse
from icecream import ic


class Paraboloid(am.Component):
    def __init__(self):
        super().__init__()

        # Add inputs
        self.add_input("x", value=0.0)
        self.add_input("y", value=0.0)

        # Add constants
        self.add_constant("a", value=3.0)
        self.add_constant("b", value=4.0)

        # Add objective, which is the function value
        self.add_objective("f_xy")

        # Add a constraint for the values
        self.add_constraint("con", value=0.0, lower=0.0, upper=10.0)

        return

    def compute(self):
        # Extract inputs and constants
        x = self.inputs["x"]
        y = self.inputs["y"]

        a = self.constants["a"]
        b = self.constants["b"]

        # Compute and set the objective
        obj = (x - a) ** 2 + x * y + (y + b) ** 2 - a
        self.objective["f_xy"] = obj

        # Compute and set the constraint
        con = x + y
        self.constraints["con"] = con

        return


if __name__ == "__main__":
    # Add the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        dest="build",
        action="store_true",
        default=False,
        help="Enable building",
    )
    args = parser.parse_args()

    # Build the amigo model
    model = am.Model("paraboloid")
    model.add_component("paraboloid", size=1, comp_obj=Paraboloid())

    if args.build:
        model.build_module()

    # Initialize the model
    model.initialize()

    # Get the design variable values
    x = model.create_vector()
    x[:] = 0.0

    # Set the initial conditions
    x["paraboloid.x"] = 0.5
    x["paraboloid.y"] = 0.5

    # Perform the optimization
    opt = am.Optimizer(model, x=x)
    opt.optimize()

    x_star = x["paraboloid.x[0]"]
    y_star = x["paraboloid.y[0]"]

    # Compare the final solution to the expected solution
    x_exp = 7.0
    y_exp = -7.0

    x_err = (x_star - x_exp) / x_exp
    y_err = (y_star - y_exp) / y_exp

    print(
        f"\n %15s %20s %20s %20s" % ("Design Var.", "Expected", "Actual", "Rel. Error")
    )
    print(f"%15s %20.10f %20.10f %20.8e" % ("x", x_exp, x_star, x_err))
    print(f"%15s %20.10f %20.10f %20.8e" % ("y", y_exp, y_star, y_err))

    # QUESTION: is there a way to extract the constraints after optimization like with design vars?
    # f_xy_star = x["paraboloid.f_xy"]
