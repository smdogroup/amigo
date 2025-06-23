import amigo as am
import numpy as np
import argparse


class Rosenbrock(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x1", value=-1.0, lower=-2.0, upper=2.0)
        self.add_input("x2", value=-1.0, lower=-2.0, upper=2.0)
        self.add_objective("obj")
        self.add_output("con", value=0.0, lower=-float("inf"), upper=0.0)

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        self.objective["obj"] = (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2
        self.outputs["con"] = x1**2 + x2**2 - 1.0


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

model = am.Model("rosenbrock")
model.add_component("rosenbrock", 1, Rosenbrock())

if args.build:
    model.build_module()

model.initialize()

opt = am.Optimizer(model)
opt.optimize()
