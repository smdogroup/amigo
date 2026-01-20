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


def _generate_cpp_types(inputs, template_name="T__"):
    lines = []

    for name in inputs:
        shape = _normalize_shape(inputs[name].node.shape)
        if shape is None:
            lines.append(f"{template_name}")
        else:
            if len(shape) == 1:
                lines.append(f"A2D::Vec<{template_name}, {shape[0]}>")
            elif len(shape) == 2:
                lines.append(f"A2D::Mat<{template_name}, {shape[0]}, {shape[1]}>")

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
        self.inputs[name] = Expr(VarNode(name, shape=shape, active=True))
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
        return self.inputs[name]

    def get_shape(self, name):
        return self.inputs[name].node.shape

    def get_meta(self, name):
        return self.meta[name]

    def generate_cpp_types(self, template_name="T__"):
        return _generate_cpp_types(self.inputs, template_name=template_name)


class ConstantSet:
    def __init__(self):
        self.consts = {}
        self.meta = {}

    def add(self, name, value, type=float, **kwargs):
        if "shape" in kwargs and kwargs["shape"] != None:
            raise ValueError("Constants must be scalars")
        self.consts[name] = Expr(ConstNode(name=name, value=value, type=type))
        self.meta[name] = Meta(
            name, "constant", value=value, shape=None, type=type, **kwargs
        )
        return

    def __len__(self):
        return len(self.consts)

    def __iter__(self):
        return iter(self.consts)

    def __getitem__(self, name):
        if name not in self.consts:
            raise KeyError(f"{name} not in declared constants")
        return self.consts[name]

    def get_shape(self, name):
        return self.consts[name].shape

    def get_meta(self, name):
        return self.meta[name]

    def generate_cpp_const_decl(self):
        lines = []
        for name in self.consts:
            node = self.consts[name].node
            vtype = _cpp_type_map[node.type]
            lines.append(
                f"inline static constexpr {vtype} {name} = static_cast<{vtype}>({node.value})"
            )
        return lines


class VarSet:
    class VarExpr:
        def __init__(self, name, type=float, shape=None, active=True):
            self.name = name
            self.shape = _normalize_shape(shape)
            self.type = type
            self.var = Expr(VarNode(name, shape=shape, type=type, active=active))
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

    def __len__(self):
        return len(self.vars)

    def __iter__(self):
        return iter(self.vars)

    def __getitem__(self, name):
        return self.vars[name].expr

    def __setitem__(self, name, expr):
        if isinstance(expr, Expr):
            shape = None
            self.vars[name] = self.VarExpr(name, shape=shape)
            self.vars[name].expr = expr
            expr.name = name
        else:
            shape = _get_shape_from_list(expr)
            self.vars[name] = self.VarExpr(name, shape=shape)

            if len(shape) == 1:
                for i in range(shape[0]):
                    self.vars[name].expr[i] = expr[i]
                    expr[i].name = f"{name}[{i}]"
            elif len(shape) == 2:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        self.vars[name].expr[i][j] = expr[i][j]
                        expr[i][j].name = f"{name}({i}, {j})"

        return

    def clear(self):
        self.vars = {}

    def generate_cpp_types(self, template_name="T__"):
        return _generate_cpp_types(self.vars, template_name=template_name)


class DataSet:
    def __init__(self):
        self.data = {}
        self.meta = {}

    def add(self, name, shape=None, type=float, **kwargs):
        self.data[name] = Expr(VarNode(name, shape=shape, type=type, active=False))
        self.meta[name] = Meta(name, "data", shape=shape, type=type, **kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, name):
        return self.data[name]

    def __iter__(self):
        return iter(self.data)

    def generate_cpp_types(self, template_name="T__"):
        return _generate_cpp_types(self.data, template_name=template_name)

    def get_shape(self, name):
        return self.data[name].node.shape

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

    def __init__(self):
        self.cons = {}
        self.meta = {}

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
        return iter(self.cons.keys())

    def __getitem__(self, name):
        return self.cons[name].expr

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

    def get_num_constraints(self):
        return len(self.cons)

    def get_shape(self, name):
        return self.cons[name].shape

    def get_meta(self, name):
        return self.meta[name]

    def evaluate(self, name, env):
        return self.cons[name].node.evaluate(env)

    def get_multiplier_name(self, name):
        if name in self.cons:
            return f"lam_{name}__"
        else:
            raise KeyError(f"{name} not in declared constraints")

    def generate_cpp_types(self, template_name="T__"):
        lines = []
        for name in self.cons:
            shape = _normalize_shape(self.cons[name].shape)
            if shape is None:
                lines.append(f"{template_name}")
            else:
                if len(shape) == 1:
                    lines.append(f"A2D::Vec<{template_name}, {shape[0]}>")
                elif len(shape) == 2:
                    lines.append(f"A2D::Mat<{template_name}, {shape[0]}, {shape[1]}>")

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

    def __iter__(self):
        return iter(self.expr)

    def __setitem__(self, name, expr):
        if name not in self.expr:
            raise KeyError(f"{name} not the declared objective")
        self.expr[name] = expr
        return

    def __getitem__(self, name):
        if name not in self.expr:
            raise KeyError(f"{name} not the declared objective")
        return self.expr[name]

    def clear(self):
        for name in self.expr:
            self.expr[name] = None

    def get_meta(self, name):
        return self.meta[name]


class OutputSet:
    class OutputExpr:
        def __init__(self, name, type=float, shape=None, active=True):
            self.name = name
            self.shape = _normalize_shape(shape)
            self.type = type
            self.var = Expr(VarNode(name, shape=shape, type=type, active=active))
            self.active = active
            self.expr = None

    def __init__(self):
        self.outputs = {}
        self.meta = {}

    def add(self, name, shape=None, type=float, **kwargs):
        if shape != None:
            raise ValueError("Output values must be scalar")
        if "scale" in kwargs:
            raise ValueError("Output values cannot be scaled through kwargs")
        self.outputs[name] = self.OutputExpr(name, type=type, active=True)
        self.meta[name] = Meta(name, "output", shape=shape, type=type, **kwargs)

    def __len__(self):
        return len(self.outputs)

    def __iter__(self):
        return iter(self.outputs)

    def __setitem__(self, name, expr):
        if name not in self.outputs:
            raise KeyError(f"{name} not the declared outputs")
        self.outputs[name].expr = expr
        return

    def __getitem__(self, name):
        if name not in self.outputs:
            raise KeyError(f"{name} not the declared outputs")
        return self.outputs[name].expr

    def get_shape(self, name):
        return self.outputs[name].shape

    def clear(self):
        for name in self.outputs:
            self.outputs[name].expr = None

    def generate_cpp_types(self, template_name="T__"):
        lines = []
        for name in self.outputs:
            shape = _normalize_shape(self.outputs[name].shape)
            if shape is None:
                lines.append(f"{template_name}")
            else:
                if len(shape) == 1:
                    lines.append(f"A2D::Vec<{template_name}, {shape[0]}>")
                elif len(shape) == 2:
                    lines.append(f"A2D::Mat<{template_name}, {shape[0]}, {shape[1]}>")

        return lines

    def get_var(self, name):
        return self.outputs[name].var

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
        self.outputs = OutputSet()
        self.multipliers = {}

        # Set the compute function arguments
        self.args = [{}]

        # Set whether this is a continuation component or not
        self.continuation_component = False

    def set_args(self, args):
        """
        Set arguments for the compute and compute_output functions
        """
        if not isinstance(args, list) or not all(
            isinstance(item, dict) for item in args
        ):
            raise TypeError(
                "set_args expects a list of dictionaries for keyword arguments"
            )
        if len(self.args) == 0:
            raise ValueError("Length of args must be at least 1")

        self.args = args
        return

    def set_continuation_component(self, flag=True):
        """
        Set a flag to designate this as a continuation component
        """
        self.continuation_component = flag
        return

    def is_continuation_component(self):
        return self.continuation_component

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

    def add_output(self, name, **kwargs):
        """
        Add an output quantity of interest computed by compute_output. These must be scalar
        """
        self.outputs.add(name, shape=None, **kwargs)
        return

    def compute(self, **kwargs):
        pass

    def compute_output(self, **kwargs):
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
        if self.is_compute_empty() and self.is_output_empty():
            return True
        return False

    def is_compute_empty(self):
        if not self._is_overridden("compute"):
            return True
        return False

    def is_output_empty(self):
        if not self._is_overridden("compute_output"):
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

    def get_output_names(self):
        outs = []
        for name in self.outputs:
            outs.append(name)
        return outs

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

    def get_output_shapes(self):
        out_shapes = {}
        for name in self.outputs:
            shape = self.outputs.get_shape(name)
            out_shapes[name] = shape
        return out_shapes

    def _initialize_multipliers(self):
        for name in self.constraints:
            mult_name = self.constraints.get_multiplier_name(name)
            if mult_name not in self.inputs:
                shape = self.constraints.get_shape(name)
                self.multipliers[mult_name] = Expr(VarNode(mult_name, shape=shape))

        return

    def _compute_lagrangian(self):
        # Compute the Lagrangian
        lhs = Expr(VarNode("lagrangian__"))
        alpha = Expr(VarNode("alpha__", active=False))
        rhs = None

        for name in self.objective:
            rhs = alpha * self.objective[name]

        for name in self.constraints:
            mult_name = self.constraints.get_multiplier_name(name)
            con = self.constraints[name]
            lam = self.multipliers[mult_name]

            shape = self.constraints.get_shape(name)

            if shape is None:
                if rhs is None:
                    rhs = con * lam
                else:
                    rhs = rhs + con * lam
            elif len(shape) == 1:
                if rhs is None:
                    rhs = con[0] * lam[0]
                else:
                    rhs = rhs + con[0] * lam[0]
                for i in range(1, shape[0]):
                    rhs = rhs + con[i] * lam[i]
            else:
                for i in range(shape[0]):
                    if rhs is None:
                        rhs = con[i, 0] * lam[i, 0]
                    else:
                        rhs = rhs + con[i, 0] * lam[i, 0]
                    for j in range(1, shape[1]):
                        rhs = rhs + con[i, j] * lam[i, j]

        return rhs, lhs

    def _get_using_statement(self, name="input", template_name="R__"):
        # Generate the using statement
        if name == "input":
            using = f"template <typename {template_name}> "
            using += f"using Input = A2D::VarTuple<{template_name}"

            input = self.inputs.generate_cpp_types(template_name=template_name)
            cons = self.constraints.generate_cpp_types(template_name=template_name)

            for val in input:
                using += f", {val}"
            for val in cons:
                using += f", {val}"
            using += ">"
        elif name == "data":
            using = f"template <typename {template_name}> "
            using += f"using Data = A2D::VarTuple<{template_name}"

            data = self.data.generate_cpp_types(template_name=template_name)
            for val in data:
                using += f", {val}"
            using += ">"
        elif name == "output":
            using = f"template <typename {template_name}> "
            using += f"using Output = A2D::VarTuple<{template_name}"

            output = self.outputs.generate_cpp_types(template_name=template_name)
            for val in output:
                using += f", {val}"
            using += ">"

        return using

    def _generate_cpp_class_types(
        self, class_name, template_name="T__", using_template="R__"
    ):
        cpp = ""

        # Create the class structure
        cpp += "\n" + f"template<typename {template_name}> \n"
        cpp += f"class {class_name} " + "{\n"
        cpp += " public:\n"

        # Add the const declarations
        const_decl = self.constants.generate_cpp_const_decl()
        for line in const_decl:
            cpp += "  " + line + ";\n"

        # Add the input statement
        if len(self.inputs) + len(self.constraints) > 0:
            using = self._get_using_statement(
                name="input", template_name=using_template
            )
            cpp += "  " + using + ";\n"
            cpp += (
                "  " + f"static constexpr int ncomp = Input<{template_name}>::ncomp;\n"
            )
        else:
            cpp += (
                "  "
                + f"template <typename {using_template}> using Input = "
                + f"typename A2D::VarTuple<{using_template}, {using_template}>;\n"
            )
            cpp += "  " + "static constexpr int ncomp = 0;\n"

        # Add the data statement
        if len(self.data) > 0:
            using = self._get_using_statement(name="data", template_name=using_template)
            cpp += "  " + using + ";\n"
            cpp += (
                "  " + f"static constexpr int ndata = Data<{template_name}>::ncomp;\n"
            )
        else:
            cpp += (
                "  "
                + f"template <typename {using_template}> using Data = "
                + f"typename A2D::VarTuple<{using_template}, {using_template}>;\n"
            )
            cpp += "  " + "static constexpr int ndata = 0;\n"

        # Is compute actually empty
        truth = "false"
        if self.is_compute_empty():
            truth = "true"
        cpp += "  " + f"static constexpr bool is_compute_empty = {truth};\n"

        # Is this a continuation component or not
        truth = "false"
        if self.is_continuation_component():
            truth = "true"
        cpp += "  " + f"static constexpr bool is_continuation_component = {truth};\n"

        # Is compute_output actually empty?
        truth = "false"
        if self.is_output_empty():
            truth = "true"
        cpp += "  " + f"static constexpr bool is_output_empty = {truth};\n"

        # Add the output statement
        if len(self.outputs) > 0:
            using = self._get_using_statement(
                name="output", template_name=using_template
            )
            cpp += "  " + using + ";\n"
            cpp += (
                "  "
                + f"static constexpr int noutputs = Output<{template_name}>::ncomp;\n"
            )
        else:
            cpp += (
                "  "
                + f"template<typename {using_template}> "
                + f"using Output = typename A2D::VarTuple<{using_template}, {using_template}>;\n"
            )
            cpp += "  " + "static constexpr int noutputs = 0;\n"

        return cpp

    def generate_cpp(
        self,
        template_name="T__",
        using_template="R__",
        data_name="data__",
        input_name="input__",
        grad_name="boutput__",
        prod_name="pinput__",
        hprod_name="houtput__",
        stack_name="stack__",
        output_name="output__",
    ):
        """
        Generate the code for a c++ implementation
        """

        cpp = ""

        # Initialize the multipliers that are added to the inputs
        self._initialize_multipliers()

        # Make lists of the constants
        consts = [self.constants[name] for name in self.constants]
        data = [self.data[name] for name in self.data]
        inputs = [self.inputs[name] for name in self.inputs]
        inputs += [self.multipliers[name] for name in self.multipliers]

        for index, args in enumerate(self.args):
            # Re-initialize any variables or other arguments
            self.clear()

            # Perform the computation to get the constraints as a function of the inputs
            if len(args) > 0:
                self.compute(**args)
            else:
                self.compute()

            if len(self.args) == 1:
                class_name = self.name + "__"
            else:
                class_name = self.name + str(index) + "__"

            cpp += self._generate_cpp_class_types(
                class_name, template_name=template_name, using_template=using_template
            )

            # Get the expression for the Lagrangian
            rhs, lhs = self._compute_lagrangian()

            if rhs is None:
                rhs = []
            if lhs is None:
                lhs = []

            # Get any variables that might have been set
            vars = [self.vars[name] for name in self.vars]

            # Create the expression builder
            builder = ExprBuilder(consts, data, inputs, vars, rhs=rhs, lhs=lhs)

            for mode in ["eval", "grad", "hprod"]:
                cpp += self._generate_compute_cpp(
                    builder,
                    lhs,
                    mode,
                    template_name=using_template,
                    data_name=data_name,
                    input_name=input_name,
                    grad_name=grad_name,
                    prod_name=prod_name,
                    hprod_name=hprod_name,
                    stack_name=stack_name,
                )

            # Clear the variables for the compute_output function
            self.clear()

            # Perform the computation to get the constraints as a function of the inputs
            if len(args) > 0:
                self.compute_output(**args)
            else:
                self.compute_output()

            # Get any variables that might have been set
            vars = [self.vars[name] for name in self.vars]

            outputs, lhs = [], []
            for name in self.outputs:
                outputs.append(self.outputs[name])
                lhs.append(self.outputs.get_var(name))

            # Create the expression builder
            builder = ExprBuilder(consts, data, inputs, vars, rhs=outputs, lhs=lhs)

            cpp += self._generate_output_cpp(
                builder,
                lhs,
                template_name=using_template,
                data_name=data_name,
                input_name=input_name,
                output_name=output_name,
            )

            cpp += "};\n"

        return cpp

    def _generate_compute_cpp(
        self,
        builder,
        lhs,
        mode,
        template_name="T__",
        data_name="data__",
        input_name="input__",
        grad_name="boutput__",
        prod_name="pinput__",
        hprod_name="houtput__",
        stack_name="stack__",
    ):
        cpp = ""

        # Add the contributions for each of the functions
        cpp += "  " + f"template <typename {template_name}>\n"
        pre = "  AMIGO_HOST_DEVICE static"
        if mode == "eval":
            cpp += (
                f"{pre} {template_name} lagrange({template_name} alpha__, "
                + f"Data<{template_name}>& {data_name}, "
                + f"Input<{template_name}>& {input_name}) "
                + "{\n"
            )
        elif mode == "grad":
            cpp += (
                f"{pre} void gradient({template_name} alpha__, "
                + f"Data<{template_name}>& {data_name}, "
                + f"Input<{template_name}>& {input_name}, "
                + f"Input<{template_name}>& {grad_name})"
                + " {\n"
            )
        elif mode == "hprod":
            cpp += (
                f"{pre} void hessian({template_name} alpha__, "
                + f"Data<{template_name}>& {data_name}, "
                + f"Input<{template_name}>& {input_name}, "
                + f"Input<{template_name}>& {prod_name}, "
                + f"Input<{template_name}>& {grad_name}, "
                + f"Input<{template_name}>& {hprod_name})"
                + " {\n"
            )

        is_empty = self.is_compute_empty()

        if not is_empty:
            decl, passive, active = builder.get_cpp_lines(
                mode=mode,
                template_name=template_name,
                data_name=data_name,
                input_name=input_name,
                grad_name=grad_name,
                prod_name=prod_name,
                hprod_name=hprod_name,
            )

            decl += builder.get_declarations(
                [lhs], reference=False, mode=mode, template_name=template_name
            )

            if mode == "eval":
                for line in decl + passive + active:
                    cpp += "    " + line + ";\n"
                cpp += "    " + f"return {lhs.to_cpp()};\n"
            else:
                for line in decl + passive:
                    cpp += "    " + line + ";\n"

                cpp += "    " + f"auto {stack_name} = A2D::MakeStack(\n"
                for index, line in enumerate(active):
                    cpp += "      " + line
                    if index == len(active) - 1:
                        cpp += ");\n"
                    else:
                        cpp += ",\n"

                cpp += "    " + f"{lhs.to_cpp()}.bvalue() = 1.0;\n"
                cpp += "    " + f"{stack_name}.reverse();\n"
                if mode == "hprod":
                    cpp += "    " + f"{stack_name}.hforward();\n"
                    cpp += "    " + f"{stack_name}.hreverse();\n"

        if mode == "eval" and is_empty:
            cpp += "    " + f"return {template_name}(0.0);\n"

        cpp += "  }\n"

        return cpp

    def _generate_output_cpp(
        self,
        builder,
        lhs,
        template_name="R__",
        data_name="data__",
        input_name="input__",
        output_name="output__",
    ):
        cpp = ""

        cpp += "  " + f"template <typename {template_name}>\n"
        pre = "  " + "AMIGO_HOST_DEVICE static"
        cpp += (
            f"{pre} void compute_output(Data<{template_name}>& {data_name}, Input<{template_name}>& {input_name}, "
            + f"Output<{template_name}>& {output_name})"
            + " {\n"
        )

        if not self.is_output_empty():
            decl, passive, active = builder.get_cpp_lines(
                mode="eval", template_name=template_name, data_name=data_name
            )
            decl += builder.get_input_declarations(
                lhs, mode="eval", template_name=template_name, input_name=data_name
            )

            for line in decl + passive + active:
                cpp += "    " + line + ";\n"

        cpp += "  }\n"

        return cpp

    def generate_pybind11(self, mod_ident="mod"):
        # Collect all the group members together...
        group_type = "ComponentGroup"

        cls = f"amigo::{group_type}<double, policy"
        for index, args in enumerate(self.args):
            if len(self.args) == 1:
                class_name = self.name + "__"
            else:
                class_name = self.name + str(index) + "__"

            cls += f", amigo::{class_name}<double>"
        cls += ">"

        module_class_name = f'"{self.name}"'
        cpp = f"py::class_<{cls}, amigo::{group_type}Base<double, policy>, std::shared_ptr<{cls}>>"
        cpp += f"({mod_ident}, {module_class_name}).def("

        vec_cls = "std::shared_ptr<amigo::Vector<int>>"
        cpp += f"py::init<int, {vec_cls}, {vec_cls}, {vec_cls}>())"

        return cpp
