import amigo as am
import argparse


class Disp1(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x")
        self.add_input("z1")
        self.add_input("z2")
        self.add_input("y1")
        self.add_input("y2")

        self.add_constraint("c1")

    def compute(self):
        x = self.inputs["x"]
        z1 = self.inputs["z1"]
        z2 = self.inputs["z2"]
        y1 = self.inputs["y1"]
        y2 = self.inputs["y2"]

        self.constraints["c1"] = z1**2 + z2 + x - 0.2 * y2 - y1


class Disp2(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("z1")
        self.add_input("z2")
        self.add_input("y1")
        self.add_input("y2")

        self.add_constraint("c2")

    def compute(self):
        z1 = self.inputs["z1"]
        z2 = self.inputs["z2"]
        y1 = self.inputs["y1"]
        y2 = self.inputs["y2"]

        self.constraints["c2"] = am.sqrt(y1) + z1 + z2 - y2


class Objective(am.Component):
    def __init__(self):
        super().__init__()
        self.add_objective("obj")

        self.add_input("x")
        self.add_input("z2")
        self.add_input("y1")
        self.add_input("y2")

    def compute(self):
        x = self.inputs["x"]
        z2 = self.inputs["z2"]
        y1 = self.inputs["y1"]
        y2 = self.inputs["y2"]

        self.objective["obj"] = x**2 + z2 + y1 + am.exp(-y2)


class Con1(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("y1")
        self.add_constraint("g1")

    def compute(self):
        self.constraints["g1"] = 3.16 - self.inputs["y1"]


class Con2(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("y2")
        self.add_constraint("g2")

    def compute(self):
        self.constraints["g2"] = self.inputs["y2"] - 24.0


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
parser.add_argument(
    "--link",
    choices=["all", "component", "explicit"],
    default="all",
    help="Set the type of linking procedure",
)
args = parser.parse_args()

model = am.Model("sellar")
model.add_component("disp1", 1, Disp1())
model.add_component("disp2", 1, Disp2())
model.add_component("obj", 1, Objective())
model.add_component("con1", 1, Con1())
model.add_component("con2", 1, Con2())

if args.link == "all":
    # Link the variables together with the same names
    model.link_by_name()
elif args.link == "component":
    # Links by component
    model.link_by_name("disp1", "disp2")
    model.link_by_name("disp1", "obj")
    model.link_by_name("disp1", "con1")
    model.link_by_name("disp1", "con2")
elif args.link == "explicit":
    # Explicit links
    model.link("disp1.z1", "disp2.z1")
    model.link("disp1.z2", "disp2.z2")
    model.link("disp1.y1", "disp2.y1")
    model.link("disp1.y2", "disp2.y2")

    model.link("disp1.x", "obj.x")
    model.link("disp1.z2", "obj.z2")
    model.link("disp1.y1", "obj.y1")
    model.link("disp1.y2", "obj.y2")

    model.link("disp1.y1", "con1.y1")
    model.link("disp1.y2", "con2.y2")

if args.build:
    model.build_module()

model.initialize()

# Set the starting point and the bounds
x = model.create_vector()
lower = model.create_vector()
upper = model.create_vector()

x["disp1.y1"] = 1.0
x["disp1.y2"] = 1.0

x["disp1.z1"] = 1.0
x["disp1.z2"] = 1.0

lower["disp1.x"] = -float("inf")
lower["disp1.y1"] = -float("inf")
lower["disp1.y1"] = -float("inf")
lower["disp1.z1"] = -float("inf")
lower["disp1.z1"] = -float("inf")

upper["disp1.x"] = float("inf")
upper["disp1.y1"] = float("inf")
upper["disp1.y2"] = float("inf")
upper["disp1.z1"] = float("inf")
upper["disp1.z2"] = float("inf")


opt = am.Optimizer(model, x, lower=lower, upper=upper)
data = opt.optimize(
    {
        "initial_barrier_param": 0.1,
        "max_line_search_iterations": 1,
        "max_iterations": 500,
        "init_affine_step_multipliers": False,
    }
)
