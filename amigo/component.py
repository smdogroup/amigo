from expressions import *

_cpp_type_map = {int: "int", float: "double", complex: "std::complex<double>"}


def _generate_cpp_input_defs(
    inputs,
    mode="eval",
    template_name="T__",
    input_name="input__",
    grad_name="boutput__",
    prod_name="pinput__",
    hprod_name="houtput__",
):
    lines = []

    if mode == "eval":
        for index, name in enumerate(inputs):
            node = inputs[name]
            shape = node.shape

            decl = None
            if shape is None:
                decl = f"{template_name}& {name}"
            elif len(shape) == 1:
                decl = f"A2D::Vec<{template_name}, {shape[0]}>& {name}"
            elif len(shape) == 2:
                decl = f"A2D::Mat<{template_name}, {shape[0]}, {shape[1]}>& {name}"
            lines.append(f"{decl} = A2D::get<{index}>({input_name});")
    elif mode == "rev":
        for index, name in enumerate(inputs):
            node = inputs[name]
            shape = node.shape

            decl = None
            if shape is None:
                decl = f"A2D::ADObj<{template_name}&> {name}"
            elif len(shape) == 1:
                decl = f"A2D::ADObj<A2D::Vec<{template_name}, {shape[0]}>&> {name}"
            elif len(shape) == 2:
                decl = f"A2D::ADObj<A2D::Mat<{template_name}, {shape[0]}, {shape[1]}>&> {name}"
            lines.append(
                f"{decl}(A2D::get<{index}>({input_name}), A2D::get<{index}>({grad_name}));"
            )
    elif mode == "hprod":
        for index, name in enumerate(inputs):
            node = inputs[name]
            shape = node.shape

            decl = None
            if shape is None:
                decl = f"A2D::A2DObj<{template_name}&> {name}"
            elif len(shape) == 1:
                decl = f"A2D::A2DObj<A2D::Vec<{template_name}, {shape[0]}>&> {name}"
            elif len(shape) == 2:
                decl = f"A2D::A2DObj<A2D::Mat<{template_name}, {shape[0]}, {shape[1]}>&> {name}"
            lines.append(
                f"{decl}(A2D::get<{index}>({input_name}), A2D::get<{index}>({grad_name}), "
                + f"A2D::get<{index}>({prod_name}), A2D::get<{index}>({hprod_name}),);"
            )

    return lines


def _generate_cpp_var_defs(inputs, mode="eval", template_name="T__"):
    lines = []

    if mode == "eval":
        for index, name in enumerate(inputs):
            node = inputs[name]
            shape = node.shape

            decl = None
            if shape is None:
                decl = f"{template_name} {name};"
            elif len(shape) == 1:
                decl = f"A2D::Vec<{template_name}, {shape[0]}> {name};"
            elif len(shape) == 2:
                decl = f"A2D::Mat<{template_name}, {shape[0]}, {shape[1]}> {name};"
            lines.append(decl)
    elif mode == "rev":
        for index, name in enumerate(inputs):
            node = inputs[name]
            shape = node.shape

            decl = None
            if shape is None:
                decl = f"A2D::ADObj<{template_name}> {name};"
            elif len(shape) == 1:
                decl = f"A2D::ADObj<A2D::Vec<{template_name}, {shape[0]}>> {name};"
            elif len(shape) == 2:
                decl = f"A2D::ADObj<A2D::Mat<{template_name}, {shape[0]}, {shape[1]}>> {name};"
            lines.append(decl)
    elif mode == "hprod":
        for index, name in enumerate(inputs):
            node = inputs[name]
            shape = node.shape

            decl = None
            if shape is None:
                decl = f"A2D::A2DObj<{template_name}> {name};"
            elif len(shape) == 1:
                decl = f"A2D::A2DObj<A2D::Vec<{template_name}, {shape[0]}>> {name};"
            elif len(shape) == 2:
                decl = f"A2D::A2DObj<A2D::Mat<{template_name}, {shape[0]}, {shape[1]}>> {name};"
            lines.append(decl)

    return lines


class InputSet:
    """
    The set of input values that are
    """

    def __init__(self):
        self.inputs = {}

    def add(self, name, shape=None, type=float):
        node = VarNode(name, shape=shape, type=type)
        self.inputs[name] = node
        return

    def __iter__(self):
        return iter(self.inputs)

    def __getitem__(self, name):
        if name not in self.inputs:
            raise KeyError(f"{name} not in declared inputs")
        return Expr(self.inputs[name])

    def generate_cpp_defs(
        self,
        mode="eval",
        template_name="T__",
        input_name="input__",
        grad_name="boutput__",
        prod_name="pinput__",
        hprod_name="houtput__",
    ):
        return _generate_cpp_input_defs(
            self.inputs,
            mode=mode,
            template_name=template_name,
            input_name=input_name,
            grad_name=grad_name,
            prod_name=prod_name,
            hprod_name=hprod_name,
        )


class ConstantSet:
    def __init__(self):
        self.inputs = {}

    def add(self, name, value=0.0, type=float):
        node = ConstNode(value, type=type)
        self.inputs[name] = node
        return

    def __iter__(self):
        return iter(self.inputs)

    def __getitem__(self, name):
        if name not in self.inputs:
            raise KeyError(f"{name} not in declared constants")
        return Expr(self.inputs[name])

    def generate_cpp_defs(self):
        lines = []
        for name in self.inputs:
            node = self.inputs[name]
            lines.append(
                f"static constexpr {_cpp_type_map[node.type]} {name} = {node.value};"
            )
        return lines


class VarSet:
    def __init__(self):
        self.vars = {}
        self.expr = {}

    def add(self, name, shape=None, type=float):
        node = VarNode(name, shape=shape, type=type)
        self.vars[name] = node
        self.expr[name] = None
        return

    def __iter__(self):
        return iter(self.vars)

    def __getitem__(self, name):
        return Expr(self.vars[name])

    def __setitem__(self, name, expr):
        if name not in self.vars:
            raise KeyError(f"{name} not in declared variables")
        expr.node.name = name
        self.expr[name] = expr

    def generate_cpp_defs(self, mode="eval", template_name="T__"):
        return _generate_cpp_var_defs(self.vars, mode=mode, template_name=template_name)

    def generate_cpp(self, mode="eval"):
        lines = []

        if mode == "eval":
            for item in self.outputs:
                if self.outputs[item].shape is None:
                    rhs = self.expr[item].generate_cpp()
                    lines.append(f"{self.outputs[item].name} = {rhs}")
                elif len(self.outputs[item].shape) == 1:
                    for i in range(self.outputs[item].shape[0]):
                        rhs = self.expr[item].expr[i].generate_cpp()
                        lines.append(f"{self.outputs[item].name}[{i}] = {rhs};")
                elif len(self.outputs[item].shape) == 2:
                    for i in range(self.outputs[item].shape[0]):
                        for j in range(self.outputs[item].shape[1]):
                            rhs = self.outputs[item].expr[i][j].generate_cpp()
                            lines.append(
                                f"{self.outputs[item].name}({i}, {j}) = {rhs};"
                            )

        return lines

class OutputSet:
    class OutputExpr:
        def __init__(self, name, type=float, shape=None):
            self.name = name
            self.shape = shape
            self.type = type

            if shape is not None and (shape != 1 or shape != (1)):
                if len(shape) == 1:
                    self.expr = [None for _ in range(shape[0])]
                elif len(shape) == 2:
                    self.expr = [
                        [None for _ in range(shape[1])] for _ in range(shape[0])
                    ]
            else:
                self.expr = None

    def __init__(self):
        self.outputs = {}

    def add(self, name, type=float, shape=None):
        self.outputs[name] = self.OutputExpr(name, shape=shape, type=type)
        return

    def __iter__(self):
        return iter(self.outputs)

    def __getitem__(self, name):
        return self.outputs[name]

    def __setitem__(self, name, expr):
        if name not in self.outputs:
            raise KeyError(f"{name} not in declared outputs")

        if isinstance(expr, Expr):
            if self.outputs[name].shape is None:
                self.outputs[name].expr = expr
        else:
            shape = self.outputs[name].shape
            if len(shape) == 1:
                for i in range(shape[0]):
                    self.outputs[name].expr[i] = expr[i]
            elif len(shape) == 2:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        self.outputs[name].expr[i][j] = expr[i][j]

    def evaluate(self, name, env):
        return self.outputs[name].node.evaluate(env)

    def generate_cpp(self, mode="eval"):
        lines = []

        if mode == "eval":
            for item in self.outputs:
                if self.outputs[item].shape is None:
                    rhs = self.outputs[item].expr.generate_cpp()
                    lines.append(f"{self.outputs[item].name} = {rhs}")
                elif len(self.outputs[item].shape) == 1:
                    for i in range(self.outputs[item].shape[0]):
                        rhs = self.outputs[item].expr[i].generate_cpp()
                        lines.append(f"{self.outputs[item].name}[{i}] = {rhs};")
                elif len(self.outputs[item].shape) == 2:
                    for i in range(self.outputs[item].shape[0]):
                        for j in range(self.outputs[item].shape[1]):
                            rhs = self.outputs[item].expr[i][j].generate_cpp()
                            lines.append(
                                f"{self.outputs[item].name}({i}, {j}) = {rhs};"
                            )

        return lines


class Component:
    def __init__(self):
        self.name = self.__class__.__name__
        self.constants = ConstantSet()
        self.data = InputSet()
        self.inputs = InputSet()
        self.vars = VarSet()
        self.outputs = OutputSet()

    def add_constant(self, name, value=1.0, type=float):
        self.constants.add(name, value=value, type=type)

    def add_input(self, name, type=float, shape=None):
        self.inputs.add(name, type=type, shape=shape)

    def add_var(self, name, type=float, shape=None):
        self.vars.add(name, type=type, shape=shape)

    def add_output(self, name, type=float, shape=None):
        self.outputs.add(name, type=type, shape=shape)

    def generate_cpp(self):
        """
        Generate the code for a c++ implementation
        """

        # Perform the computation to get the outputs as a function of the inputs
        self.compute()

        lines = []
        lines.extend(self.constants.generate_cpp_defs())

        for mode in ["eval", "rev", "hprod"]:
            lines.extend(self.inputs.generate_cpp_defs(mode=mode))
            lines.extend(self.vars.generate_cpp_defs(mode=mode))

            lines.extend(self.outputs.generate_cpp(mode=mode))

        for line in lines:
            print(line)

        return


class CartComponent(Component):
    def __init__(self, g=9.81, L=0.5, m1=0.5, m2=0.3):
        super().__init__()

        self.add_constant("g", value=g)
        self.add_constant("L", value=L)
        self.add_constant("m1", value=m1)
        self.add_constant("m2", value=m2)

        self.add_input("x")
        self.add_input("q", shape=(4,))
        self.add_input("qdot", shape=(4,))

        self.add_var("cost")
        self.add_var("sint")

        self.add_output("res", shape=(4,))

        return

    def compute(self):
        g = self.constants["g"]
        L = self.constants["L"]
        m1 = self.constants["m1"]
        m2 = self.constants["m2"]

        x = self.inputs["x"]
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        # Compute the declared variable values
        self.vars["sint"] = sin(q[1])
        self.vars["cost"] = cos(q[1])

        # Extract a reference to the expressions for convenience
        sint = self.vars["sint"]
        cost = self.vars["cost"]

        res = 4 * [None]
        res[0] = q[2] - qdot[0]
        res[1] = q[3] - qdot[1]
        res[2] = (m1 + m2 * (1.0 - cost * cost)) * qdot[2] - (
            L * m2 * sint * q[3] * q[3] * x + m2 * g * cost * sint
        )
        res[3] = L * (m1 + m2 * (1.0 - cost * cost)) * qdot[3] + (
            L * m2 * cost * sint * q[3] * q[3] + x * cost + (m1 + m2) * g * sint
        )

        self.outputs["res"] = res

        return


cart = CartComponent()
cart.generate_cpp()

