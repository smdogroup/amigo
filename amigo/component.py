from .expressions import *

_cpp_type_map = {int: "int", float: "double", complex: "std::complex<double>"}


def _normalize_shape(shape):
    if shape is None:
        return None
    if isinstance(shape, int):
        shape = (shape,)
    elif not isinstance(shape, tuple):
        raise TypeError("Expecting None, int or tuple")
    if not (len(shape) == 1 or len(shape) == 2):
        raise ValueError("Amigo only accepts shapes with at most two dimensions")
    return shape


def _get_shape_from_list(obj):
    if not isinstance(obj, (list, tuple)):
        return ()

    length = len(obj)

    # Empty container: return current length
    if length == 0 or not isinstance(obj[0], (list, tuple)):
        return (length,)

    # Recurse into first element and check consistency
    sub_shape = _get_shape_from_list(obj[0])
    for sub in obj:
        if _get_shape_from_list(sub) != sub_shape:
            raise ValueError("Inconsistent list-of-list shapes")

    return (length,) + sub_shape


def _generate_cpp_input_decl(
    inputs,
    offset=0,
    mode="eval",
    alt_names=None,
    template_name="T__",
    input_name="input__",
    grad_name="boutput__",
    prod_name="pinput__",
    hprod_name="houtput__",
):
    lines = []

    for index, name in enumerate(inputs):
        node = inputs[name]
        shape = _normalize_shape(node.shape)
        var_name = name if alt_names is None else alt_names[name]

        if mode == "eval":
            decl = None
            if shape is None:
                decl = f"{template_name}& {var_name}"
            else:
                if len(shape) == 1:
                    decl = f"A2D::Vec<{template_name}, {shape[0]}>& {var_name}"
                elif len(shape) == 2:
                    decl = (
                        f"A2D::Mat<{template_name}, {shape[0]}, {shape[1]}>& {var_name}"
                    )
            lines.append(f"{decl} = A2D::get<{index + offset}>({input_name})")
        elif mode == "rev":
            decl = None
            if shape is None:
                decl = f"A2D::ADObj<{template_name}&> {var_name}"
            else:
                if len(shape) == 1:
                    decl = (
                        f"A2D::ADObj<A2D::Vec<{template_name}, {shape[0]}>&> {var_name}"
                    )
                elif len(shape) == 2:
                    decl = f"A2D::ADObj<A2D::Mat<{template_name}, {shape[0]}, {shape[1]}>&> {var_name}"
            lines.append(
                f"{decl}(A2D::get<{index + offset}>({input_name}), A2D::get<{index + offset}>({grad_name}))"
            )
        elif mode == "hprod":
            decl = None
            if shape is None:
                decl = f"A2D::A2DObj<{template_name}&> {var_name}"
            else:
                if len(shape) == 1:
                    decl = f"A2D::A2DObj<A2D::Vec<{template_name}, {shape[0]}>&> {var_name}"
                elif len(shape) == 2:
                    decl = f"A2D::A2DObj<A2D::Mat<{template_name}, {shape[0]}, {shape[1]}>&> {var_name}"
            lines.append(
                f"{decl}(A2D::get<{index + offset}>({input_name}), A2D::get<{index + offset}>({grad_name}), "
                + f"A2D::get<{index + offset}>({prod_name}), A2D::get<{index + offset}>({hprod_name}))"
            )

    return lines


def _generate_cpp_var_decl(
    inputs, active=None, mode="eval", alt_names=None, template_name="T__"
):
    lines = []

    for index, name in enumerate(inputs):
        node = inputs[name]
        shape = _normalize_shape(node.shape)
        var_name = name if alt_names is None else alt_names[name]

        passive = False
        if active is not None:
            if var_name in active and not active[var_name]:
                passive = True

        if mode == "eval" or passive:
            decl = None
            if shape is None:
                decl = f"{template_name} {var_name}"
            else:
                if len(shape) == 1:
                    decl = f"A2D::Vec<{template_name}, {shape[0]}> {var_name}"
                elif len(shape) == 2:
                    decl = (
                        f"A2D::Mat<{template_name}, {shape[0]}, {shape[1]}> {var_name}"
                    )
            lines.append(decl)
        elif mode == "rev":
            decl = None
            if shape is None:
                decl = f"A2D::ADObj<{template_name}> {var_name}"
            else:
                if len(shape) == 1:
                    decl = (
                        f"A2D::ADObj<A2D::Vec<{template_name}, {shape[0]}>> {var_name}"
                    )
                elif len(shape) == 2:
                    decl = f"A2D::ADObj<A2D::Mat<{template_name}, {shape[0]}, {shape[1]}>> {var_name}"
            lines.append(decl)
        elif mode == "hprod":
            decl = None
            if shape is None:
                decl = f"A2D::A2DObj<{template_name}> {var_name}"
            else:
                if len(shape) == 1:
                    decl = (
                        f"A2D::A2DObj<A2D::Vec<{template_name}, {shape[0]}>> {var_name}"
                    )
                elif len(shape) == 2:
                    decl = f"A2D::A2DObj<A2D::Mat<{template_name}, {shape[0]}, {shape[1]}>> {var_name}"
            lines.append(decl)

    return lines


def _generate_cpp_types(inputs, template_name="T__"):
    lines = []

    for index, name in enumerate(inputs):
        node = inputs[name]
        shape = _normalize_shape(node.shape)

        decl = None
        if shape is None:
            decl = f"{template_name}"
        else:
            if len(shape) == 1:
                decl = f"A2D::Vec<{template_name}, {shape[0]}>"
            elif len(shape) == 2:
                decl = f"A2D::Mat<{template_name}, {shape[0]}, {shape[1]}>"
        lines.append(decl)

    return lines


class InputSet:
    """
    The set of input values that are
    """

    def __init__(self):
        self.inputs = {}
        self.labels = {}

    def add(self, name, shape=None, type=float, label="input"):
        node = VarNode(name, shape=shape, type=type, active=True)
        self.inputs[name] = node
        self.labels[name] = label
        return

    def get_num_inputs(self):
        return len(self.inputs)

    def __len__(self):
        return len(self.inputs)

    def __iter__(self):
        return iter(self.inputs)

    def __getitem__(self, name):
        if name not in self.inputs:
            raise KeyError(f"{name} not in declared inputs")
        return Expr(self.inputs[name])

    def get_info(self, name):
        return self.inputs[name].shape, self.inputs[name].type, self.labels[name]

    def generate_cpp_types(self, template_name="T__"):
        return _generate_cpp_types(self.inputs, template_name=template_name)

    def generate_cpp_input_decl(
        self,
        offset=0,
        mode="eval",
        template_name="T__",
        input_name="input__",
        grad_name="boutput__",
        prod_name="pinput__",
        hprod_name="houtput__",
    ):
        return _generate_cpp_input_decl(
            self.inputs,
            offset=offset,
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
        self.labels = {}

    def add(self, name, value, type=float, label="const"):
        node = ConstNode(name=name, value=value, type=type)
        self.inputs[name] = node
        self.labels[name] = label
        return

    def __iter__(self):
        return iter(self.inputs)

    def __getitem__(self, name):
        if name not in self.inputs:
            raise KeyError(f"{name} not in declared constants")
        return Expr(self.inputs[name])

    def get_info(self, name):
        return self.inputs[name].shape, self.inputs[name].type, self.labels[name]

    def generate_cpp_const_decl(self):
        lines = []
        for name in self.inputs:
            node = self.inputs[name]
            lines.append(
                f"static constexpr {_cpp_type_map[node.type]} {name} = {node.value}"
            )
        return lines


class VarSet:
    class VarExpr:
        def __init__(self, name, type=float, shape=None, active=True):
            self.name = name
            self.shape = _normalize_shape(shape)
            self.type = type
            self.var = VarNode(name, shape=shape, type=type, active=active)
            self.active = active

            if self.shape is None:
                self.expr = None
            else:
                if len(self.shape) == 1:
                    self.expr = [None for _ in range(self.shape[0])]
                else:
                    self.expr = [
                        [None for _ in range(self.shape[1])]
                        for _ in range(self.shape[0])
                    ]

    def __init__(self):
        self.vars = {}

    def __iter__(self):
        return iter(self.vars)

    def __getitem__(self, name):
        return Expr(self.vars[name].var)

    def __setitem__(self, name, expr):
        if isinstance(expr, Expr):
            shape = None
            self.vars[name] = self.VarExpr(name, shape=shape, active=expr.active)
            self.vars[name].expr = expr
            expr.node.name = name
        else:
            shape = _get_shape_from_list(expr)
            self.vars[name] = self.VarExpr(name, shape=shape, active=True)

            # Check whether we should reset the active flag
            active = False
            if len(shape) == 1:
                for i in range(shape[0]):
                    self.vars[name].expr[i] = expr[i]
                    active = active or expr[i].active
            elif len(shape) == 2:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        self.vars[name].expr[i][j] = expr[i][j]
                        active = active or expr[i][j].active

            self.vars[name].active = active
            self.vars[name].var.active = active

        return

    def generate_cpp_types(self, template_name="T__"):
        return _generate_cpp_types(self.vars, template_name=template_name)

    def generate_cpp_decl(self, mode="eval", template_name="T__"):
        active = {}
        for name in self.vars:
            active[name] = self.vars[name].active

        return _generate_cpp_var_decl(
            self.vars, active, mode=mode, template_name=template_name
        )

    def generate_passive_cpp(self):
        lines = []
        for name in self.vars:
            if not self.vars[name].active:
                shape = _normalize_shape(self.vars[name].shape)
                if shape is None:
                    rhs = self.vars[name].expr.generate_cpp()
                    lines.append(f"{name} = {rhs}")
                elif len(shape) == 1:
                    for i in range(shape[0]):
                        rhs = self.vars[name].expr[i].generate_cpp()
                        lines.append(f"{name}[{i}] = {rhs}")
                elif len(shape) == 2:
                    for j in range(shape[1]):
                        for i in range(shape[0]):
                            rhs = self.vars[name].expr[i][j].generate_cpp()
                            lines.append(f"{name}({i},{j}) = {rhs}")

        return lines

    def generate_active_cpp(self, mode="eval"):
        lines = []

        if mode == "eval":
            for name in self.vars:
                if self.vars[name].active:
                    shape = _normalize_shape(self.vars[name].shape)
                    if shape is None:
                        rhs = self.vars[name].expr.generate_cpp()
                        lines.append(f"{name} = {rhs}")
                    elif len(shape) == 1:
                        for i in range(shape[0]):
                            rhs = self.vars[name].expr[i].generate_cpp()
                            lines.append(f"{name}[{i}] = {rhs}")
                    elif len(shape) == 2:
                        for j in range(shape[1]):
                            for i in range(shape[0]):
                                rhs = self.vars[name].expr[i][j].generate_cpp()
                                lines.append(f"{name}({i},{j}) = {rhs}")
        else:
            for name in self.vars:
                if self.vars[name].active:
                    shape = _normalize_shape(self.vars[name].shape)
                    if shape is None:
                        rhs = self.vars[name].expr.generate_cpp()
                        lines.append(f"A2D::Eval({rhs}, {name})")
                    elif len(shape) == 1:
                        for i in range(shape[0]):
                            rhs = self.vars[name].expr[i].generate_cpp()
                            lines.append(f"A2D::Eval({rhs}, {name}[{i}])")
                    elif len(shape) == 2:
                        for j in range(shape[1]):
                            for i in range(shape[0]):
                                rhs = self.vars[name].expr[i][j].generate_cpp()
                                lines.append(f"A2D::Eval({rhs}, {name}({i},{j}))")
        return lines


class DataSet:
    def __init__(self):
        self.data = {}
        self.labels = {}

    def add(self, name, shape=None, type=float, label=None):
        self.data[name] = VarNode(name, shape=shape, type=type, active=False)
        self.labels[name] = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, name):
        return Expr(self.data[name])

    def __iter__(self):
        return iter(self.data)

    def generate_cpp_types(self, template_name="T__"):
        return _generate_cpp_types(self.data, template_name=template_name)

    def generate_cpp_input_decl(self, template_name="T__", data_name="data___"):
        lines = _generate_cpp_input_decl(
            self.data, mode="eval", template_name=template_name, input_name=data_name
        )
        return lines

    def get_info(self, name):
        return self.data[name].shape, self.data[name].type, self.labels[name]


class OutputSet:
    class OutputExpr:
        def __init__(self, name, type=float, shape=None):
            self.name = name
            self.shape = _normalize_shape(shape)
            self.type = type

            if self.shape is None:
                self.expr = None
            else:
                if len(self.shape) == 1:
                    self.expr = [None for _ in range(self.shape[0])]
                else:
                    self.expr = [
                        [None for _ in range(self.shape[1])]
                        for _ in range(self.shape[0])
                    ]

    def __init__(self, lagrangian_name="lagrangian__"):
        self.outputs = {}
        self.labels = {}
        self.lagrangian_name = lagrangian_name

    def add(self, name, type=float, shape=None, label="output"):
        self.outputs[name] = self.OutputExpr(name, shape=shape, type=type)
        self.labels[name] = label
        return

    def __len__(self):
        return len(self.outputs)

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

        return

    def get_info(self, name):
        return self.outputs[name].shape, self.outputs[name].type, self.labels[name]

    def evaluate(self, name, env):
        return self.outputs[name].node.evaluate(env)

    def _get_multiplier_names(self):
        mult_names = {}
        for name in self:
            mult_names[name] = "lam_" + name + "__"

        return mult_names

    def generate_cpp_types(self, template_name="T__"):
        return _generate_cpp_types(self.outputs, template_name=template_name)

    def generate_cpp_input_decl(
        self,
        offset=0,
        mode="eval",
        template_name="T__",
        input_name="input__",
        grad_name="boutput__",
        prod_name="pinput__",
        hprod_name="houtput__",
    ):
        mult_names = self._get_multiplier_names()
        lines = _generate_cpp_input_decl(
            self.outputs,
            alt_names=mult_names,
            offset=offset,
            mode=mode,
            template_name=template_name,
            input_name=input_name,
            grad_name=grad_name,
            prod_name=prod_name,
            hprod_name=hprod_name,
        )
        return lines

    def generate_cpp_decl(self, mode="eval", template_name="T__"):
        lines = _generate_cpp_var_decl(
            self.outputs, mode=mode, template_name=template_name
        )
        if mode == "eval":
            lines.append(f"{template_name} {self.lagrangian_name}")
        elif mode == "rev":
            lines.append(f"A2D::ADObj<{template_name}> {self.lagrangian_name}")
        else:
            lines.append(f"A2D::A2DObj<{template_name}> {self.lagrangian_name}")
        return lines

    def _lagrangian_evaluation(self, obj_expr=None):
        expr_list = []

        mult_names = self._get_multiplier_names()
        for item in self.outputs:
            shape = self.outputs[item].shape
            name = self.outputs[item].name

            if shape is None:
                expr_list.append(f"{name} * {mult_names[name]}")
            elif len(shape) == 1:
                for i in range(shape[0]):
                    expr_list.append(f"{name}[{i}] * {mult_names[name]}[{i}]")
            elif len(shape) == 2:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        expr_list.append(
                            f"{name}({i}, {j}) * {mult_names[name]}({i}, {j})"
                        )

        if obj_expr is None:
            expr = ""
        else:
            expr = f"{obj_expr}"
        if len(expr_list) > 0:
            expr = expr_list[0]
            for i in range(1, len(expr_list)):
                expr += " + " + expr_list[i]

        return expr

    def generate_cpp(self, mode="eval", obj_expr=None):
        lines = []

        if mode == "eval":
            for item in self.outputs:
                shape = self.outputs[item].shape
                name = self.outputs[item].name
                if shape is None:
                    rhs = self.outputs[item].expr.generate_cpp()
                    lines.append(f"{name} = {rhs}")
                elif len(shape) == 1:
                    for i in range(shape[0]):
                        rhs = self.outputs[item].expr[i].generate_cpp()
                        lines.append(f"{name}[{i}] = {rhs}")
                elif len(shape) == 2:
                    for i in range(shape[0]):
                        for j in range(shape[1]):
                            rhs = self.outputs[item].expr[i][j].generate_cpp()
                            lines.append(f"{name}({i}, {j}) = {rhs}")

            rhs = self._lagrangian_evaluation(obj_expr=obj_expr)
            lines.append(f"{self.lagrangian_name} = {rhs}")
        else:
            for item in self.outputs:
                shape = self.outputs[item].shape
                name = self.outputs[item].name

                if shape is None:
                    rhs = self.outputs[item].expr.generate_cpp()
                    lines.append(f"A2D::Eval({rhs}, {name})")
                elif len(shape) == 1:
                    for i in range(shape[0]):
                        rhs = self.outputs[item].expr[i].generate_cpp()
                        lines.append(f"A2D::Eval({rhs}, {name}[{i}])")
                elif len(shape) == 2:
                    for i in range(shape[0]):
                        for j in range(shape[1]):
                            rhs = self.outputs[item].expr[i][j].generate_cpp()
                            lines.append(f"A2D::Eval({rhs}, {name}({i}, {j}))")

            rhs = self._lagrangian_evaluation(obj_expr=obj_expr)
            lines.append(f"A2D::Eval({rhs}, {self.lagrangian_name})")

        return lines


class ObjectiveSet:
    def __init__(self):
        self.expr = {}

    def add(self, name, type=float, shape=None):
        if shape != None:
            raise ValueError("Objective must be a scalar")
        if len(self.expr) == 1:
            raise ValueError("Cannot add more than one objective")
        self.expr[name] = None
        return

    def __len__(self):
        return len(self.expr)

    def __setitem__(self, name, expr):
        if name not in self.expr:
            raise KeyError(f"{name} not the declared objective")
        self.expr[name] = expr
        return

    def generate_cpp(self):
        for name in self.expr:
            rhs = self.expr[name].generate_cpp()
            return rhs
        return None


class Component:
    def __init__(self):
        self.name = self.__class__.__name__
        self.constants = ConstantSet()
        self.data = InputSet()
        self.inputs = InputSet()
        self.vars = VarSet()
        self.outputs = OutputSet()
        self.objective = ObjectiveSet()
        self.data = DataSet()
        self.empty = False

    def add_constant(self, name, value, type=float, label="const"):
        self.constants.add(name, value, type=type, label=label)
        return

    def add_input(self, name, type=float, shape=None, label="input"):
        self.inputs.add(name, type=type, shape=shape, label=label)
        return

    def add_output(self, name, type=float, shape=None, label="output"):
        self.outputs.add(name, type=type, shape=shape, label=label)
        return

    def add_objective(self, name, type=float):
        self.objective.add(name, type=type)
        return

    def add_data(self, name, type=float, shape=None):
        self.data.add(name, type=type, shape=shape)
        return

    def is_empty(self):
        if (len(self.objective) == 0 and len(self.outputs) == 0) or self.empty:
            return True
        return False

    def get_var_shapes(self):
        var_shapes = {}
        for name in self.inputs:
            shape, _, _ = self.inputs.get_info(name)
            var_shapes[name] = shape

        for name in self.outputs:
            shape, _, _ = self.outputs.get_info(name)
            var_shapes[name] = shape

        return var_shapes

    def get_data_shapes(self):
        data_shapes = {}
        for name in self.data:
            shape, _, _ = self.data.get_info(name)
            data_shapes[name] = shape

        return data_shapes

    def _get_using_statement(self, name="input", template_name="T__"):
        # Generate the using statement
        if name == "input":
            using = f"using Input = A2D::VarTuple<{template_name}"

            input = self.inputs.generate_cpp_types(template_name=template_name)
            output = self.outputs.generate_cpp_types(template_name=template_name)

            for val in input:
                using += f", {val}"
            for val in output:
                using += f", {val}"
            using += ">"

        elif name == "data":
            using = f"using Data = A2D::VarTuple<{template_name}"

            data = self.data.generate_cpp_types(template_name=template_name)

            for val in data:
                using += f", {val}"
            using += ">"

        return using

    def generate_cpp(
        self,
        template_name="T__",
        data_name="data__",
        input_name="input__",
        grad_name="boutput__",
        prod_name="pinput__",
        hprod_name="houtput__",
        stack_name="stack__",
    ):
        """
        Generate the code for a c++ implementation
        """

        # Perform the computation to get the outputs as a function of the inputs
        self.compute()

        # Create the class structure
        cpp = "\n" + f"template<typename {template_name}> \n"
        cpp += f"class {self.name} " + "{\n"
        cpp += " public:\n"

        # Add the const declarations
        const_decl = self.constants.generate_cpp_const_decl()
        for line in const_decl:
            cpp += "  " + line + ";\n"

        # Add the input statement
        using = self._get_using_statement(name="input", template_name=template_name)
        cpp += "  " + using + ";\n"
        cpp += "  " + "static constexpr int ncomp = Input::ncomp" + ";\n"

        if len(self.data) > 0:
            using = self._get_using_statement(name="data", template_name=template_name)
            cpp += "  " + using + ";\n"
            cpp += "  " + "static constexpr int ndata = Data::ncomp" + ";\n"
        else:
            cpp += (
                "  "
                + f"using Data = typename A2D::VarTuple<{template_name}, {template_name}>;\n"
            )
            cpp += "  " + "static constexpr int ndata = 0;\n"

        # Add the contributions for each of the functions
        for mode in ["eval", "rev", "hprod"]:
            if mode == "eval":
                cpp += (
                    "  "
                    + f"static {template_name} lagrange(Data& {data_name}, Input& {input_name})"
                    + " {\n"
                )
            elif mode == "rev":
                cpp += (
                    "  "
                    f"static void gradient(Data& {data_name}, Input& {input_name}, Input& {grad_name})"
                    + " {\n"
                )
            elif mode == "hprod":
                cpp += (
                    "  "
                    f"static void hessian(Data& {data_name}, Input& {input_name}, Input& {prod_name}, "
                    f"Input& {grad_name}, Input& {hprod_name})" + " {\n"
                )

            data_decl = self.data.generate_cpp_input_decl(
                template_name=template_name, data_name=data_name
            )
            for line in data_decl:
                cpp += "    " + line + ";\n"

            in_decl = self.inputs.generate_cpp_input_decl(
                mode=mode,
                template_name=template_name,
                input_name=input_name,
                grad_name=grad_name,
                prod_name=prod_name,
                hprod_name=hprod_name,
            )
            for line in in_decl:
                cpp += "    " + line + ";\n"

            offset = self.inputs.get_num_inputs()
            out_decl = self.outputs.generate_cpp_input_decl(
                mode=mode,
                offset=offset,
                template_name=template_name,
                input_name=input_name,
                grad_name=grad_name,
                prod_name=prod_name,
                hprod_name=hprod_name,
            )
            for line in out_decl:
                cpp += "    " + line + ";\n"

            var_decl = self.vars.generate_cpp_decl(
                mode=mode, template_name=template_name
            )
            for line in var_decl:
                cpp += "    " + line + ";\n"

            out_decl = self.outputs.generate_cpp_decl(
                mode=mode, template_name=template_name
            )
            for line in out_decl:
                cpp += "    " + line + ";\n"

            lines = self.vars.generate_passive_cpp()
            for line in lines:
                cpp += "    " + f"{line};\n"

            obj_expr = self.objective.generate_cpp()
            body = self.vars.generate_active_cpp(mode=mode)
            body.extend(self.outputs.generate_cpp(mode=mode, obj_expr=obj_expr))

            if mode == "eval":
                for line in body:
                    cpp += "    " + line + ";\n"
                cpp += "    " + f"return {self.outputs.lagrangian_name};\n"
            else:
                cpp += "    " + f"auto {stack_name} = A2D::MakeStack(\n"
                for index, line in enumerate(body):
                    cpp += "      " + line
                    if index == len(body) - 1:
                        cpp += ");\n"
                    else:
                        cpp += ",\n"

                cpp += "    " + f"{self.outputs.lagrangian_name}.bvalue() = 1.0;\n"
                cpp += "    " + f"{stack_name}.reverse();\n"
                if mode == "hprod":
                    cpp += "    " + f"{stack_name}.hforward();\n"
                    cpp += "    " + f"{stack_name}.hreverse();\n"

            cpp += "  }\n"

        cpp += "};\n"

        return cpp

    def generate_pybind11(self, mod_ident="mod"):
        cls = f"amigo::ComponentGroup<double, amigo::{self.name}<double>>"

        module_class_name = f'"{self.name}"'

        cpp = f"py::class_<{cls}, amigo::ComponentGroupBase<double>, std::shared_ptr<{cls}>>"
        cpp += f"({mod_ident}, {module_class_name}).def("
        cpp += "py::init<std::shared_ptr<amigo::Vector<int>>, std::shared_ptr<amigo::Vector<int>>>())"

        return cpp
