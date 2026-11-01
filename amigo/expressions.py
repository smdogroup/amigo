from enum import IntEnum, auto


class Linearity(IntEnum):
    CONSTANT = 0
    LINEAR = 1
    NONLINEAR = 2


# Convert from/to strings to/from type
_type_to_str = {float: "float", int: "int"}
_str_to_type = {"float": float, "int": int}


def _normalize_shape(shape):
    if shape is None or shape == ():
        return None
    if isinstance(shape, int):
        if shape == 1:
            return None
        shape = (shape,)
    elif isinstance(shape, list):
        return tuple(shape)
    elif not isinstance(shape, tuple):
        raise TypeError("Expecting None, int or tuple")
    if not (len(shape) == 1 or len(shape) == 2):
        raise ValueError("Amigo only accepts shapes with at most two dimensions")
    return shape


def _broadcast_shape(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if a == b:
        return a
    raise ValueError(f"Cannot broadcast shapes {a} and {b}")


class ExprNode:
    def is_zero(self):
        raise NotImplementedError

    def serialize(self):
        raise NotImplementedError

    def to_cpp(self):
        raise NotImplementedError

    def is_active(self):
        raise NotImplementedError

    def compute_cost(self):
        raise NotImplementedError

    def get_linearity(self, active_vars):
        raise NotImplementedError


class ConstNode(ExprNode):
    def __init__(self, name=None, value=None, type=float):
        super().__init__()
        self.name = name
        self.value = value
        self.type = type
        self.shape = None

    def is_zero(self):
        if self.value is not None and (self.value == 0 or self.value == 0.0):
            return True
        return False

    def serialize(self):
        return ("const", self.name, self.value, _type_to_str[self.type])

    def to_cpp(self):
        if self.name is not None:
            return self.name
        return str(self.value)

    def is_active(self):
        return False

    def compute_cost(self):
        return 0

    def get_linearity(self, active_vars):
        return Linearity.CONSTANT


class VarNode(ExprNode):
    def __init__(self, name, shape=None, type=float, active=True):
        super().__init__()
        self.name = name
        self.shape = _normalize_shape(shape)
        self.type = type
        self.active = active

    def is_zero(self):
        return False

    def serialize(self):
        return ("var", self.name, self.shape, _type_to_str[self.type], self.active)

    def to_cpp(self):
        return self.name

    def is_active(self):
        return self.active

    def compute_cost(self):
        return 0

    def get_linearity(self, active_vars):
        if self in active_vars:
            return Linearity.LINEAR
        return Linearity.CONSTANT


class IndexNode(ExprNode):
    def __init__(self, expr, index):
        super().__init__()
        self.expr = expr
        self.index = index

    def is_zero(self):
        return False

    def serialize(self):
        return ("index", self.expr.serialize(), self.index)

    def to_cpp(self):
        arr = self.expr.to_cpp()
        if isinstance(self.index, tuple):
            return f"{arr}({self.index[0]}, {self.index[1]})"
        else:
            return f"{arr}[{self.index}]"

    def is_active(self):
        return self.expr.is_active()

    def compute_cost(self):
        return 0

    def get_linearity(self, active_vars):
        return self.expr.get_linearity(active_vars)


class PassiveNode(ExprNode):
    def __init__(self, expr):
        self.expr = expr

    def is_zero(self):
        return False

    def serialize(self):
        return ("passive", self.expr.serialize())

    def to_cpp(self):
        a = self.expr.to_cpp()
        return f"A2D::get_data({a})"

    def is_active(self):
        return False

    def compute_cost(self):
        return 0

    def get_linearity(self, active_vars):
        return Linearity.CONSTANT


class UnaryNode(ExprNode):
    unary_ops = [
        "-",
        "fabs",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "exp",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "log",
    ]

    def __init__(self, op, expr):
        super().__init__()
        if not op in UnaryNode.unary_ops:
            raise ValueError("Unrecognized unary operation f{}", op)
        self.op = op
        self.expr = expr
        self.shape = expr.shape

    def is_zero(self):
        return self.expr.is_zero()

    def serialize(self):
        return ("unary", self.op, self.expr.serialize())

    def to_cpp(self):
        a = self.expr.to_cpp()
        if self.op == "-":
            return f"-({a})"
        else:
            return f"A2D::{self.op}({a})"

    def is_active(self):
        return self.expr.is_active()

    def compute_cost(self):
        cost = self.expr.compute_cost()

        trig = ["sin", "cos", "tan", "asin", "acos", "atan"]
        exp = ["exp", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh"]
        log = ["log"]

        if self.op == "-" or self.op == "fabs":
            cost += 1
        elif self.op == "sqrt":
            cost += 20
        elif self.op in trig:
            cost += 100
        elif self.op in exp:
            cost += 100
        elif self.op in log:
            cost += 200

        return cost

    def get_linearity(self, active_vars):
        if self.expr.get_linearity(active_vars) == Linearity.CONSTANT:
            return Linearity.CONSTANT
        return Linearity.NONLINEAR


class BinaryNode(ExprNode):
    binary_ops = ["+", "-", "*", "/", "**", "pow", "atan2", "min2", "max2"]

    def __init__(self, op, left, right):
        super().__init__()
        if not op in BinaryNode.binary_ops:
            raise ValueError("Unrecognized unary operation f{}", op)
        self.op = op
        self.left = left
        self.right = right
        self.shape = _broadcast_shape(left.shape, right.shape)

    def is_zero(self):
        return False

    def serialize(self):
        return ("binary", self.op, self.left.serialize(), self.right.serialize())

    def to_cpp(self):
        a = self.left.to_cpp()
        b = self.right.to_cpp()

        if self.op == "**":
            return f"A2D::pow({a}, {b})"
        elif self.op == "atan2":
            return f"A2D::atan2({a}, {b})"
        elif self.op == "min2":
            return f"A2D::min2({a}, {b})"
        elif self.op == "max2":
            return f"A2D::max2({a}, {b})"
        return f"({a} {self.op} {b})"

    def is_active(self):
        return self.left.is_active() or self.right.is_active()

    def compute_cost(self):
        cost = self.left.compute_cost() + self.right.compute_cost()

        if self.op == "**":
            rnode = self.right.node
            if isinstance(rnode, ConstNode) and rnode.type == int:
                if rnode.value >= 0:
                    cost += rnode.value
                elif rnode.value < 0:
                    cost += 10 + rnode.value
            else:
                cost += 100
        elif self.op == "atan2":
            cost += 100
        elif self.op == "/":
            cost += 10
        else:
            cost += 1

        return cost

    def get_linearity(self, active_vars):
        left = self.left.get_linearity(active_vars)
        right = self.right.get_linearity(active_vars)

        if self.op in ("+", "-"):
            return max(left, right)

        elif self.op == "*":
            if left == Linearity.CONSTANT:
                return right
            elif right == Linearity.CONSTANT:
                return left
            else:
                return Linearity.NONLINEAR
        elif self.op == "/":
            if right == Linearity.CONSTANT:
                return left
            else:
                return Linearity.NONLINEAR
        else:
            return Linearity.NONLINEAR


def _to_expr(val):
    if isinstance(val, Expr):
        return val
    elif isinstance(val, ExprNode):
        return Expr(val)
    elif isinstance(val, int):
        return Expr(ConstNode(value=val, type=int))
    elif isinstance(val, float):
        return Expr(ConstNode(value=val, type=float))
    raise TypeError(f"Unsupported value: {val}")


class Expr:
    def __init__(self, node: ExprNode):
        self.node = node

    @property
    def shape(self):
        return getattr(self.node, "shape", None)

    def is_scalar(self):
        return self.shape is None

    def __neg__(self):
        return Expr(UnaryNode("-", self))

    def __add__(self, other):
        expr = _to_expr(other)
        if self.is_zero():
            return expr
        elif expr.is_zero():
            return self
        return Expr(BinaryNode("+", self, expr))

    def __sub__(self, other):
        expr = _to_expr(other)
        if self.is_zero():
            return Expr(UnaryNode("-", expr))
        elif expr.is_zero():
            return self
        return Expr(BinaryNode("-", self, expr))

    def __mul__(self, other):
        expr = _to_expr(other)
        if self.is_zero() or expr.is_zero():
            return Expr(ConstNode(value=0.0))
        return Expr(BinaryNode("*", self, expr))

    def __truediv__(self, other):
        if self.is_zero():
            return Expr(ConstNode(value=0.0))
        return Expr(BinaryNode("/", self, _to_expr(other)))

    def __pow__(self, other):
        if isinstance(other, int):
            if other == 0:
                return Expr(ConstNode(value=1.0))
            elif other > 0:
                tmp = self
                for i in range(other - 1):
                    tmp = tmp * self
                return tmp
            elif other < 0:
                tmp = self
                for i in range(-other - 1):
                    tmp = tmp * self
                return 1.0 / tmp
        return Expr(BinaryNode("**", self, _to_expr(other)))

    def __radd__(self, other):
        expr = _to_expr(other)
        if self.is_zero():
            return expr
        elif expr.is_zero():
            return self
        return Expr(BinaryNode("+", expr, self))

    def __rsub__(self, other):
        expr = _to_expr(other)
        if self.is_zero():
            return expr
        elif expr.is_zero():
            return Expr(UnaryNode("-", self))
        return Expr(BinaryNode("-", expr, self))

    def __rmul__(self, other):
        expr = _to_expr(other)
        if self.is_zero() or expr.is_zero():
            return Expr(ConstNode(value=0.0))
        return Expr(BinaryNode("*", expr, self))

    def __rtruediv__(self, other):
        expr = _to_expr(other)
        if expr.is_zero():
            return Expr(ConstNode(value=0.0))
        return Expr(BinaryNode("/", expr, self))

    def __rpow__(self, other):
        return Expr(BinaryNode("**", _to_expr(other), self))

    def __getitem__(self, idx):
        if isinstance(self.node, VarNode):
            if isinstance(idx, tuple):
                index = tuple(int(i) for i in idx)
                if index[0] < 0 or index[0] >= self.node.shape[0]:
                    raise ValueError(f"First index {index[0]} out of range")
                if index[1] < 0 or index[1] >= self.node.shape[1]:
                    raise ValueError(f"Second index {index[1]} out of range")
            else:
                index = int(idx)
                if index < 0 or index >= self.node.shape[0]:
                    raise ValueError(f"Index {index} out of range")

            return Expr(IndexNode(self, index))
        elif isinstance(self.node, PassiveNode):
            if isinstance(idx, tuple):
                index = tuple(int(i) for i in idx)
            else:
                index = int(idx)
            return Expr(IndexNode(self, index))
        else:
            raise TypeError("You can only index variables, not general expressions")

    def is_zero(self):
        return self.node.is_zero()

    def serialize(self):
        return self.node.serialize()

    @classmethod
    def deserialize(cls, d):
        if d[0] == "const":
            return cls(ConstNode(name=d[1], value=d[2], type=_str_to_type[d[3]]))
        elif d[0] == "var":
            return cls(
                VarNode(name=d[1], shape=d[2], type=_str_to_type[d[3]], active=d[4])
            )
        elif d[0] == "index":
            return cls(IndexNode(cls.deserialize(d[1]), d[2]))
        elif d[0] == "passive":
            return cls(PassiveNode(cls.deserialize(d[1])))
        elif d[0] == "unary":
            return cls(UnaryNode(d[1], cls.deserialize(d[2])))
        elif d[0] == "binary":
            return cls(BinaryNode(d[1], cls.deserialize(d[2]), cls.deserialize(d[3])))
        else:
            return ValueError("Unrecognized operation type in deserialize")

    def to_cpp(self):
        return self.node.to_cpp()

    def set_active_flag(self, flag):
        if isinstance(self.node, VarNode):
            self.node.active = flag
        else:
            raise ValueError("Cannot set active flag for node ", self.node)

    def is_active(self):
        return self.node.is_active()

    def compute_cost(self):
        return self.node.compute_cost()

    def get_linearity(self, active_vars):
        return self.node.get_linearity(active_vars)


class ExprBuilder:
    def __init__(self, consts=[], data=[], inputs=[], vars=[], rhs=[], lhs=[]):
        self.consts = consts
        self.data = data
        self.inputs = inputs
        self.vars = vars
        if isinstance(rhs, list):
            self.rhs = rhs
        elif isinstance(rhs, Expr):
            self.rhs = [rhs]
        else:
            raise TypeError("Unrecognized rhs type")
        if isinstance(lhs, list):
            self.lhs = lhs
        elif isinstance(lhs, Expr):
            self.lhs = [lhs]
        else:
            raise TypeError("Unrecognized lhs type")

        # Initialize the internal data
        self.counter = 0
        self.key_to_id = {}
        self.nodes = {}
        self.expr_to_id = {}

        # Serialize the constants, data, inputs and vars
        self.const_list = self._serialize_list(self.consts)
        self.data_list = self._serialize_list(self.data)
        self.input_list = self._serialize_list(self.inputs)
        self.var_list = self._serialize_list(self.vars)

        # Serialize the expressions
        self.rhs_list = self._serialize_list(self.rhs)

        self.counts = [0] * len(self.nodes)
        self.node_exprs = [None] * len(self.nodes)
        temp_var_exprs = {}

        # Set the expressions for the nodes
        for i, idx in enumerate(self.const_list):
            self.node_exprs[idx] = self.consts[i]
        for i, idx in enumerate(self.data_list):
            self.node_exprs[idx] = self.data[i]
        for i, idx in enumerate(self.input_list):
            self.node_exprs[idx] = self.inputs[i]

        make_temp_name = lambda idx: f"t{idx}__"

        # Create the temporary variables specified by the user
        self.temp_list = []
        for i, idx in enumerate(self.var_list):
            temp_var_exprs[idx] = self.vars[i]
            self.temp_list.append(idx)
            name = make_temp_name(idx)
            self.node_exprs[idx] = Expr(VarNode(name, type=float))

        # Count up the references to each node, locating temporaries
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            if node["type"] == "unary":
                self.counts[node["expr"]] += 1
            elif node["type"] == "binary":
                self.counts[node["left"]] += 1
                self.counts[node["right"]] += 1

        # Add the temporaries to the list
        for idx in range(len(self.counts)):
            node = self.nodes[idx]
            if node["type"] == "unary" or node["type"] == "binary":
                if self.counts[idx] > 1:
                    self.temp_list.append(idx)
                    name = make_temp_name(idx)
                    self.node_exprs[idx] = Expr(VarNode(name, type=float))

        # Build the expressions
        for idx in self.temp_list:
            temp_var_exprs[idx] = self._build_expr(idx)

        # Set it up so that the temp expressions are in order
        self.temp_exprs = []
        for idx in sorted(set(self.temp_list)):
            # Set whether the temporaries are active or not
            self.node_exprs[idx].set_active_flag(temp_var_exprs[idx].is_active())
            self.temp_exprs.append([self.node_exprs[idx], temp_var_exprs[idx]])

        # Set the new expressions
        self.new_rhs = []
        for idx in self.rhs_list:
            if idx is None:
                self.new_rhs.append(None)
            else:
                self.new_rhs.append(self._build_expr(idx))

        return

    def _serialize_list(self, expr_list):
        new_list = []
        for expr in expr_list:
            if expr is None:
                new_list.append(None)
            else:
                new_list.append(self._serialize_expr(expr))
        return new_list

    def _serialize_expr(self, e: Expr):
        # Fast path: same Expr object already seen
        eid = id(e)
        if eid in self.expr_to_id:
            return self.expr_to_id[eid]

        node = e.node
        if isinstance(node, ConstNode):
            key = ("const", node.name, node.value, node.type)
            info = {
                "type": "const",
                "name": node.name,
                "value": node.value,
                "vartype": _type_to_str[node.type],
            }
        elif isinstance(node, VarNode):
            key = ("var", node.name, node.shape, node.type, node.active)
            info = {
                "type": "var",
                "name": node.name,
                "shape": node.shape,
                "vartype": _type_to_str[node.type],
                "active": node.active,
            }
        elif isinstance(node, IndexNode):
            expr_id = self._serialize_expr(node.expr)
            key = ("index", expr_id, node.index)
            info = {
                "type": "index",
                "expr": expr_id,
                "index": node.index,
            }
        elif isinstance(node, PassiveNode):
            expr_id = self._serialize_expr(node.expr)
            key = ("passive", expr_id)
            info = {
                "type": "passive",
                "expr": expr_id,
            }
        elif isinstance(node, UnaryNode):
            expr_id = self._serialize_expr(node.expr)
            key = ("unary", node.op, expr_id)
            info = {
                "type": "unary",
                "op": node.op,
                "expr": expr_id,
            }
        elif isinstance(node, BinaryNode):
            left_id = self._serialize_expr(node.left)
            right_id = self._serialize_expr(node.right)
            key = ("binary", node.op, left_id, right_id)
            info = {
                "type": "binary",
                "op": node.op,
                "left": left_id,
                "right": right_id,
            }
        else:
            raise TypeError(type(node))

        # Structural CSE: catches distinct Expr objects with identical structure
        if key in self.key_to_id:
            count = self.key_to_id[key]
            self.expr_to_id[eid] = count
            return count

        count = self.counter
        self.counter += 1

        self.key_to_id[key] = count
        self.expr_to_id[eid] = count
        self.nodes[count] = info

        return count

    def _build_expr(self, idx):
        node = self.nodes[idx]
        if node["type"] == "const":
            expr = Expr(
                ConstNode(
                    name=node["name"],
                    value=node["value"],
                    type=_str_to_type[node["vartype"]],
                )
            )
        elif node["type"] == "var":
            expr = Expr(
                VarNode(
                    name=node["name"],
                    shape=node["shape"],
                    type=_str_to_type[node["vartype"]],
                    active=node["active"],
                )
            )
        elif node["type"] == "index":
            expr = Expr(IndexNode(self._build_expr_index(node["expr"]), node["index"]))
        elif node["type"] == "passive":
            expr = Expr(PassiveNode(self._build_expr_index(node["expr"])))
        elif node["type"] == "unary":
            expr = Expr(UnaryNode(node["op"], self._build_expr_index(node["expr"])))
        elif node["type"] == "binary":
            expr = Expr(
                BinaryNode(
                    node["op"],
                    self._build_expr_index(node["left"]),
                    self._build_expr_index(node["right"]),
                )
            )

        return expr

    def _build_expr_index(self, idx):
        if self.node_exprs[idx] is not None:
            return self.node_exprs[idx]

        self.node_exprs[idx] = self._build_expr(idx)
        return self.node_exprs[idx]

    def _get_expr_type(self, expr, template_name="T__"):
        shape = _normalize_shape(expr.node.shape)
        if shape is None:
            return f"{template_name}"
        else:
            if len(shape) == 1:
                return f"A2D::VecRef<{template_name}, {shape[0]}>"
            elif len(shape) == 2:
                return f"A2D::MatRef<{template_name}, {shape[0]}, {shape[1]}>"

    def _get_ad_type(
        self, expr, scalar_reference=False, mode="eval", template_name="T__"
    ):
        decl = self._get_expr_type(expr, template_name=template_name)
        shape = _normalize_shape(expr.node.shape)

        if mode == "eval" or not expr.is_active():
            if shape is None and scalar_reference:
                return f"{decl}&"
            return decl
        elif mode == "grad":
            if shape is None and scalar_reference:
                return f"A2D::ADObj<{decl}&>"
            return f"A2D::ADObj<{decl}>"
        else:
            if shape is None and scalar_reference:
                return f"A2D::A2DObj<{decl}&>"
            return f"A2D::A2DObj<{decl}>"

    def is_linear(self, active_expr):
        active_vars = []
        for expr in active_expr:
            if isinstance(expr, Expr):
                active_vars.append(expr.node)
            elif isinstance(expr, ExprNode):
                active_vars.append(expr)
            else:
                raise ValueError("Expected list of Expr or ExprNode objects")

        # Check the linearity of the intermediate expressions. If a variable is
        # linear, add it to the active varaibles
        linear = True
        for var, rhs in self.temp_exprs:
            if rhs is not None:
                linearity = rhs.get_linearity(active_vars)
                if linearity == Linearity.LINEAR:
                    active_vars.append(var.node)
                elif linearity == Linearity.NONLINEAR:
                    linear = False
                    break

        # Check that the final expressions are nonlinear, only if the intermediate
        # variables are linear.
        if linear:
            for rhs in self.new_rhs:
                if rhs is not None:
                    linearity = rhs.get_linearity(active_vars)
                    if linearity == Linearity.NONLINEAR:
                        linear = False
                        break

        return linear

    def get_declarations(
        self, inputs, scalar_reference=True, mode="eval", template_name="T__"
    ):
        lines = []

        for expr in inputs:
            decl = self._get_ad_type(
                expr,
                scalar_reference=scalar_reference,
                mode=mode,
                template_name=template_name,
            )
            lines.append(f"{decl} {expr.to_cpp()}")

        return lines

    def get_input_declarations(
        self,
        inputs,
        mode="eval",
        template_name="T__",
        input_name="input__",
        grad_name="boutput__",
        prod_name="pinput__",
        hprod_name="houtput__",
    ):
        lines = []

        for index, expr in enumerate(inputs):
            decl = self._get_ad_type(
                expr, scalar_reference=True, mode=mode, template_name=template_name
            )

            line = f"{decl} {expr.to_cpp()}"
            if mode == "eval" or not expr.is_active():
                line = f"{line} = A2D::get<{index}>({input_name})"
            elif mode == "grad":
                line = f"{line}(A2D::get<{index}>({input_name}), A2D::get<{index}>({grad_name}))"
            else:
                line = (
                    f"{line}(A2D::get<{index}>({input_name}), A2D::get<{index}>({grad_name}), "
                    + f"A2D::get<{index}>({prod_name}), A2D::get<{index}>({hprod_name}))"
                )
            lines.append(line)

        return lines

    def get_cpp_lines(
        self,
        mode="eval",
        template_name="T__",
        data_name="data__",
        input_name="input__",
        grad_name="boutput__",
        prod_name="pinput__",
        hprod_name="houtput__",
    ):
        """Make the expressions"""

        decl = []
        passive = []
        active = []

        # Get lines for the data - note that we use the data name here
        decl = self.get_input_declarations(
            self.data, mode=mode, template_name=template_name, input_name=data_name
        )

        # Get the declaration lines for the input
        decl.extend(
            self.get_input_declarations(
                self.inputs,
                mode=mode,
                template_name=template_name,
                input_name=input_name,
                grad_name=grad_name,
                prod_name=prod_name,
                hprod_name=hprod_name,
            )
        )

        # Get the declaration lines for the temporaries
        temps = [t for t, _ in self.temp_exprs]
        decl.extend(
            self.get_declarations(
                temps,
                scalar_reference=False,
                mode=mode,
                template_name=template_name,
            )
        )

        # Now construct lines for the passive and active terms
        make_passive = lambda name, rhs: f"{name} = {rhs}"
        if mode == "eval":
            make_active = lambda name, rhs: f"{name} = {rhs}"
        else:
            make_active = lambda name, rhs: f"A2D::Eval({rhs}, {name})"

        # Add the passive lines to the code
        for var, rhs in self.temp_exprs:
            if not var.is_active():
                passive.append(make_passive(var.to_cpp(), rhs.to_cpp()))

        # Add the active lines to the code
        for var, rhs in self.temp_exprs:
            if var.is_active():
                active.append(make_active(var.to_cpp(), rhs.to_cpp()))

        # Add the new expressions
        for lhs, rhs in zip(self.lhs, self.new_rhs):
            if rhs is not None:
                active.append(make_active(lhs.to_cpp(), rhs.to_cpp()))
            else:
                active.append(make_active(lhs.to_cpp(), f"{template_name}(0.0)"))

        return decl, passive, active
