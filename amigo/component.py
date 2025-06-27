import types
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


class Meta:
    def __init__(self, name, var_type, **kwargs):
        self.name = name
        options = ["input", "constraint", "output", "data", "objective", "constant"]
        if var_type not in options:
            raise ValueError(f"{var_type} not one of {options}")
        self.var_type = var_type
        self.shape = kwargs.pop("shape", (1))
        self.value = kwargs.pop("value", 0.0)
        self.type = kwargs.pop("type", float)
        self.lower = kwargs.pop("lower", float("-inf"))
        self.upper = kwargs.pop("upper", float("inf"))
        self.units = kwargs.pop("units", None)
        self.scale = kwargs.pop("scale", 1.0)
        self.label = kwargs.pop("label", None)

        if self.value is None:
            self.value = 0.0

        if len(kwargs) > 0:
            raise ValueError(f"Unknown options: {kwargs}")

        self._validate()

    def _validate(self):
        if not isinstance(self.value, self.type):
            raise TypeError(f"value must be of type {self.type.__name__}")
        if not isinstance(self.scale, (int, float)):
            raise TypeError("scale must be a number")
        if not isinstance(self.lower, (int, float)):
            raise TypeError("lower must be a number")
        if not isinstance(self.upper, (int, float)):
            raise TypeError("upper must be a number")
        if self.lower > self.upper:
            raise ValueError("lower bound cannot be greater than upper bound")
        if self.var_type == "input" and not self.lower <= self.value <= self.upper:
            raise ValueError("value must be within [lower, upper]")

    def __getitem__(self, name):
        if name == "name":
            return self.name
        elif name == "shape":
            return self.shape
        elif name == "value":
            return self.value
        elif name == "type":
            return self.type
        elif name == "lower":
            return self.lower
        elif name == "upper":
            return self.upper
        elif name == "units":
            return self.units
        elif name == "scale":
            return self.scale
        elif name == "label":
            return self.label

    def __repr__(self):
        return (
            f"Meta(name={self.name!r}, var_type={self.var_type!r}, shape={self.shape},\n"
            f"     value={self.value}, type={self.type.__name__},\n"
            f"     lower={self.lower}, upper={self.upper}, units={self.units!r},\n"
            f"     scale={self.scale}, label={self.label!r})"
        )

    def todict(self):
        return {
            "name": self.name,
            "shape": self.shape,
            "value": self.value,
            "type": str(self.type),
            "lower": self.lower,
            "upper": self.upper,
            "units": self.units,
            "scale": self.scale,
            "label": self.label,
        }


class InputSet:
    """
    The set of input values that are
    """

    def __init__(self):
        self.inputs = {}
        self.meta = {}

    def add(self, name, shape=None, **kwargs):
        self.inputs[name] = VarNode(name, shape=shape, active=True)
        self.meta[name] = Meta(name, "input", shape=shape, **kwargs)
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

    def get_shape(self, name):
        return self.inputs[name].shape

    def get_meta(self, name):
        return self.meta[name]

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
        self.meta = {}

    def add(self, name, value, type=float, **kwargs):
        if "shape" in kwargs and kwargs["shape"] != None:
            raise ValueError("Constants must be scalars")
        self.inputs[name] = ConstNode(name=name, value=value, type=type)
        self.meta[name] = Meta(
            name, "constant", value=value, shape=None, type=type, **kwargs
        )
        return

    def __iter__(self):
        return iter(self.inputs)

    def __getitem__(self, name):
        if name not in self.inputs:
            raise KeyError(f"{name} not in declared constants")
        return Expr(self.inputs[name])

    def get_shape(self, name):
        return self.inputs[name].shape

    def get_meta(self, name):
        return self.meta[name]

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

    def clear(self):
        self.vars = {}

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
        self.meta = {}

    def add(self, name, shape=None, type=float, **kwargs):
        self.data[name] = VarNode(name, shape=shape, type=type, active=False)
        self.meta[name] = Meta(name, "data", shape=shape, type=type, **kwargs)

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

    def get_shape(self, name):
        return self.data[name].shape

    def get_meta(self, name):
        return self.meta[name]


class ConstraintSet:
    class ConstrExpr:
        def __init__(self, name, type=float, shape=None):
            self.name = name
            self.shape = _normalize_shape(shape)
            self.type = type
            self.clear_expr()

        def clear_expr(self):
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
        self.cons = {}
        self.meta = {}
        self.lagrangian_name = lagrangian_name

    def add(self, name, shape=None, type=float, **kwargs):
        self.cons[name] = self.ConstrExpr(name, shape=shape, type=type)
        self.meta[name] = Meta(name, "constraint", shape=shape, type=type, **kwargs)
        return

    def clear(self):
        for name in self.cons:
            self.cons[name].clear_expr()
        return

    def __len__(self):
        return len(self.cons)

    def __iter__(self):
        return iter(self.cons)

    def __getitem__(self, name):
        return self.cons[name]

    def __setitem__(self, name, expr):
        if name not in self.cons:
            raise KeyError(f"{name} not in declared constraints")

        if isinstance(expr, Expr):
            if self.cons[name].shape is None:
                self.cons[name].expr = expr
        else:
            shape = self.cons[name].shape
            if len(shape) == 1:
                for i in range(shape[0]):
                    self.cons[name].expr[i] = expr[i]
            elif len(shape) == 2:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        self.cons[name].expr[i][j] = expr[i][j]

        return

    def get_shape(self, name):
        return self.cons[name].shape

    def get_meta(self, name):
        return self.meta[name]

    def evaluate(self, name, env):
        return self.cons[name].node.evaluate(env)

    def _get_multiplier_names(self):
        mult_names = {}
        for name in self:
            mult_names[name] = "lam_" + name + "__"

        return mult_names

    def generate_cpp_types(self, template_name="T__"):
        return _generate_cpp_types(self.cons, template_name=template_name)

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
            self.cons,
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
            self.cons, mode=mode, template_name=template_name
        )
        if mode == "eval":
            lines.append(f"{template_name} {self.lagrangian_name}")
        elif mode == "rev":
            lines.append(f"A2D::ADObj<{template_name}> {self.lagrangian_name}")
        else:
            lines.append(f"A2D::A2DObj<{template_name}> {self.lagrangian_name}")
        return lines

    def _lagrangian_evaluation(self, res_names, obj_expr=None):
        expr_list = []

        mult_names = self._get_multiplier_names()
        for item in self.cons:
            shape = self.cons[item].shape
            name = self.cons[item].name

            if shape is None:
                res_name = res_names[name]
                expr_list.append(f"({res_name}) * {mult_names[name]}")
            elif len(shape) == 1:
                for i in range(shape[0]):
                    res_name = res_names[name][i]
                    expr_list.append(f"({res_name}) * {mult_names[name]}[{i}]")
            elif len(shape) == 2:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        res_name = res_names[name][i][j]
                        expr_list.append(f"({res_name}) * {mult_names[name]}({i}, {j})")

        if obj_expr is None:
            expr = ""
        else:
            expr = f"{obj_expr}"

        if len(expr_list) > 0:
            if obj_expr is None:
                expr = expr_list[0]
            else:
                expr += " + " + expr_list[0]
            for i in range(1, len(expr_list)):
                expr += " + " + expr_list[i]

        return expr

    def generate_cpp(self, mode="eval", obj_expr=None):
        lines = []

        if mode == "eval":
            make_line = lambda name, rhs: f"{name} = {rhs}"
        else:
            make_line = lambda name, rhs: f"A2D::Eval({rhs}, {name})"

        res_names = {}
        for item in self.cons:
            shape = self.cons[item].shape
            name = self.cons[item].name
            if shape is None:
                rhs = self.cons[item].expr.generate_cpp()
                if self.cons[name].expr.active:
                    res_names[name] = name
                    lines.append(make_line(name, rhs))
                else:
                    res_names[name] = rhs
            elif len(shape) == 1:
                res_names[name] = []
                for i in range(shape[0]):
                    rhs = self.cons[item].expr[i].generate_cpp()
                    if self.cons[name].expr[i].active:
                        res_names[name].append(f"{name}[{i}]")
                        lines.append(make_line(f"{name}[{i}]", rhs))
                    else:
                        res_names[name].append(rhs)
            elif len(shape) == 2:
                res_names[name] = []
                for i in range(shape[0]):
                    res_names[name][i].append([])
                    for j in range(shape[1]):
                        rhs = self.cons[item].expr[i][j].generate_cpp()
                        if self.cons[name].expr[i][j].active:
                            res_names[name][i].append(f"{name}({i}, {j})")
                            lines.append(make_line(f"{name}({i}, {j}", rhs))
                        else:
                            res_names[name].append(rhs)

        rhs = self._lagrangian_evaluation(res_names, obj_expr=obj_expr)
        lines.append(make_line(f"{self.lagrangian_name}", rhs))

        return lines


class ObjectiveSet:
    def __init__(self):
        self.expr = {}
        self.meta = {}

    def add(self, name, shape=None, type=float, **kwargs):
        if shape != None:
            raise ValueError("Objective must be a scalar")
        if len(self.expr) == 1:
            raise ValueError("Cannot add more than one objective")
        if "scale" in kwargs:
            raise ValueError("Objective function cannot be scaled through kwargs")
        self.expr[name] = None
        self.meta[name] = Meta(name, "objective", shape=shape, type=type, **kwargs)
        return

    def __len__(self):
        return len(self.expr)

    def __setitem__(self, name, expr):
        if name not in self.expr:
            raise KeyError(f"{name} not the declared objective")
        self.expr[name] = expr
        return

    def clear(self):
        for name in self.expr:
            self.expr[name] = None

    def generate_cpp(self):
        for name in self.expr:
            rhs = self.expr[name].generate_cpp()
            return rhs
        return None

    def get_meta(self, name):
        return self.meta[name]


class Component:
    def __init__(self):
        # Set the name - this will be the ComponentGroup name in C++
        self.name = self.__class__.__name__

        # Set the input values
        self.constants = ConstantSet()
        self.inputs = InputSet()
        self.vars = VarSet()
        self.constraints = ConstraintSet()
        self.objective = ObjectiveSet()
        self.data = DataSet()

        # Set the compute function arguments
        self.compute_args = [{}]

    def set_compute_args(self, compute_args):
        if not isinstance(compute_args, list) or not all(
            isinstance(item, dict) for item in compute_args
        ):
            raise TypeError(
                "set_compute_args expects a list of dictionaries for keyword arguments"
            )
        if len(self.compute_args) == 0:
            raise ValueError("Length of compute_args must be at least 1")

        self.compute_args = compute_args
        return

    def add_constant(self, name, value, **kwargs):
        """
        Add constant values to the component
        """
        self.constants.add(name, value=value, **kwargs)
        return

    def add_input(
        self, name, shape=None, lower=float("-inf"), upper=float("inf"), **kwargs
    ):
        """
        Add inputs to the component. Note that inputs by default have no lower or upper bounds.
        """
        self.inputs.add(name, shape=shape, lower=lower, upper=upper, **kwargs)
        return

    def add_constraint(self, name, shape=None, lower=0.0, upper=0.0, **kwargs):
        """
        Add constraint to the component. By default, constraints are equality constraints.
        """
        self.constraints.add(name, shape=shape, lower=lower, upper=upper, **kwargs)
        return

    def add_objective(self, name, **kwargs):
        """
        Add a scalar objective function
        """
        self.objective.add(name, **kwargs)
        return

    def add_data(self, name, shape=None, **kwargs):
        """
        Add component data that will be loaded into the problem
        """
        self.data.add(name, shape=shape, **kwargs)
        return

    def compute(self):
        pass

    def analyze(self):
        pass

    def _is_overridden(self, method_name):
        instance_method = getattr(self, method_name)
        base_method = getattr(type(self).__bases__[0], method_name, None)

        # Unbind methods so we compare function objects directly
        if isinstance(instance_method, types.MethodType):
            instance_method = instance_method.__func__
        if isinstance(base_method, types.MethodType):
            base_method = base_method.__func__

        return instance_method is not base_method

    def is_empty(self):
        if not self._is_overridden("compute"):
            return True
        return False

    def clear(self):
        self.constraints.clear()
        self.vars.clear()
        self.objective.clear()
        return

    def get_input_names(self):
        inputs = []
        for name in self.inputs:
            inputs.append(name)
        return inputs

    def get_constraint_names(self):
        cons = []
        for name in self.constraints:
            cons.append(name)
        return cons

    def get_data_names(self):
        data = []
        for name in self.data:
            data.append(name)
        return data

    def get_var_shapes(self):
        var_shapes = {}
        for name in self.inputs:
            shape = self.inputs.get_shape(name)
            var_shapes[name] = shape

        for name in self.constraints:
            shape = self.constraints.get_shape(name)
            var_shapes[name] = shape

        return var_shapes

    def get_data_shapes(self):
        data_shapes = {}
        for name in self.data:
            shape = self.data.get_shape(name)
            data_shapes[name] = shape

        return data_shapes

    def _get_using_statement(self, name="input", template_name="T__"):
        # Generate the using statement
        if name == "input":
            using = f"using Input = A2D::VarTuple<{template_name}"

            input = self.inputs.generate_cpp_types(template_name=template_name)
            cons = self.constraints.generate_cpp_types(template_name=template_name)

            for val in input:
                using += f", {val}"
            for val in cons:
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

        cpp = ""

        for index, args in enumerate(self.compute_args):
            # Re-initialize any variables or other arguments
            self.clear()

            # Perform the computation to get the constraints as a function of the inputs
            if len(args) > 0:
                self.compute(**args)
            else:
                self.compute()

            if len(self.compute_args) == 1:
                class_name = self.name + "__"
            else:
                class_name = self.name + str(index) + "__"

            # Create the class structure
            cpp += "\n" + f"template<typename {template_name}> \n"
            cpp += f"class {class_name} " + "{\n"
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
                using = self._get_using_statement(
                    name="data", template_name=template_name
                )
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
                pre = "AMIGO_HOST_DEVICE static"
                if mode == "eval":
                    cpp += (
                        "  "
                        + f"{pre} {template_name} lagrange(Data& {data_name}, Input& {input_name})"
                        + " {\n"
                    )
                elif mode == "rev":
                    cpp += (
                        "  "
                        f"{pre} void gradient(Data& {data_name}, Input& {input_name}, Input& {grad_name})"
                        + " {\n"
                    )
                elif mode == "hprod":
                    cpp += (
                        "  "
                        f"{pre} void hessian(Data& {data_name}, Input& {input_name}, Input& {prod_name}, "
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
                out_decl = self.constraints.generate_cpp_input_decl(
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

                out_decl = self.constraints.generate_cpp_decl(
                    mode=mode, template_name=template_name
                )
                for line in out_decl:
                    cpp += "    " + line + ";\n"

                lines = self.vars.generate_passive_cpp()
                for line in lines:
                    cpp += "    " + f"{line};\n"

                obj_expr = self.objective.generate_cpp()
                body = self.vars.generate_active_cpp(mode=mode)
                body.extend(self.constraints.generate_cpp(mode=mode, obj_expr=obj_expr))

                if mode == "eval":
                    for line in body:
                        cpp += "    " + line + ";\n"
                    cpp += "    " + f"return {self.constraints.lagrangian_name};\n"
                else:
                    cpp += "    " + f"auto {stack_name} = A2D::MakeStack(\n"
                    for index, line in enumerate(body):
                        cpp += "      " + line
                        if index == len(body) - 1:
                            cpp += ");\n"
                        else:
                            cpp += ",\n"

                    cpp += (
                        "    " + f"{self.constraints.lagrangian_name}.bvalue() = 1.0;\n"
                    )
                    cpp += "    " + f"{stack_name}.reverse();\n"
                    if mode == "hprod":
                        cpp += "    " + f"{stack_name}.hforward();\n"
                        cpp += "    " + f"{stack_name}.hreverse();\n"

                cpp += "  }\n"

            cpp += "};\n"

        return cpp

    def generate_pybind11(self, mod_ident="mod"):
        # Collect all the group members together...
        cls = f"amigo::ComponentGroup<double"
        for index, args in enumerate(self.compute_args):
            if len(self.compute_args) == 1:
                class_name = self.name + "__"
            else:
                class_name = self.name + str(index) + "__"

            cls += f", amigo::{class_name}<double>"
        cls += ">"

        module_class_name = f'"{self.name}"'

        cpp = f"py::class_<{cls}, amigo::ComponentGroupBase<double>, std::shared_ptr<{cls}>>"
        cpp += f"({mod_ident}, {module_class_name}).def("
        cpp += "py::init<std::shared_ptr<amigo::Vector<int>>, std::shared_ptr<amigo::Vector<int>>>())"

        return cpp
