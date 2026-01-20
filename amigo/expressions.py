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


class ExprNode:
    def to_key(self):
        raise NotImplementedError

    def to_cpp(self):
        raise NotImplementedError

    def is_active(self):
        raise NotImplementedError

    def compute_cost(self):
        raise NotImplementedError


class ConstNode(ExprNode):
    def __init__(self, name=None, value=None, type=float):
        super().__init__()
        self.name = name
        self.value = value
        self.type = type

    def to_key(self):
        return ("const", self.name, self.value, self.type)

    def to_cpp(self):
        if self.name is not None:
            return self.name
        return str(self.value)

    def is_active(self):
        return False

    def compute_cost(self):
        return 0


class VarNode(ExprNode):
    def __init__(self, name, shape=None, type=float, active=True):
        super().__init__()
        self.name = name
        self.shape = _normalize_shape(shape)
        self.type = type
        self.active = active

    def to_key(self):
        return ("var", self.name, self.shape, self.type, self.active)

    def to_cpp(self):
        return self.name

    def is_active(self):
        return self.active

    def compute_cost(self):
        return 0


class IndexNode(ExprNode):
    def __init__(self, expr, index):
        super().__init__()
        self.expr = expr
        self.index = index

    def to_key(self):
        return ("index", self.expr.to_key(), self.index)

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


class PassiveNode(ExprNode):
    def __init__(self, expr):
        self.expr = expr

    def to_key(self):
        return ("passive", self.expr.to_key())

    def to_cpp(self):
        a = self.expr.to_cpp()
        return f"A2D::get_data({a})"

    def is_active(self):
        return False

    def compute_cost(self):
        return 0


class UnaryNode(ExprNode):
    def __init__(self, op, expr):
        super().__init__()
        self.op = op
        self.expr = expr

    def to_key(self):
        return ("unary", self.op, self.expr.to_key())

    def to_cpp(self):
        a = self.expr.to_cpp()
        if self.op == "-":
            return f"-({a})"
        return f"A2D::{self.op}({a})"

    def is_active(self):
        return self.expr.is_active()

    def compute_cost(self):
        cost = self.expr.compute_cost()

        trig = ["sin", "cos", "tan", "asin", "acos", "atan"]
        exp = ["exp", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh"]
        log = ["log", "log10"]

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


class BinaryNode(ExprNode):
    def __init__(self, op, left, right):
        super().__init__()
        self.op = op
        self.left = left
        self.right = right

    def to_key(self):
        return ("binary", self.op, self.left.to_key(), self.right.to_key())

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
        self.name = None
        self.node = node

    def __neg__(self):
        return Expr(UnaryNode("-", self.node))

    def __add__(self, other):
        return Expr(BinaryNode("+", self, _to_expr(other)))

    def __sub__(self, other):
        return Expr(BinaryNode("-", self, _to_expr(other)))

    def __mul__(self, other):
        return Expr(BinaryNode("*", self, _to_expr(other)))

    def __truediv__(self, other):
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
        return Expr(BinaryNode("+", _to_expr(other), self))

    def __rsub__(self, other):
        return Expr(BinaryNode("-", _to_expr(other), self))

    def __rmul__(self, other):
        return Expr(BinaryNode("*", _to_expr(other), self))

    def __rtruediv__(self, other):
        return Expr(BinaryNode("/", _to_expr(other), self))

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

    def to_key(self):
        return self.node.to_key()

    def to_cpp(self, use_vars=True):
        if use_vars and self.name is not None:
            return self.name
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

        # Convert from/to strings to/from type
        self.type_to_str = {float: "float", int: "int"}
        self.str_to_type = {"float": float, "int": int}

        # Initialize the internal data
        self.counter = 0
        self.key_to_id = {}
        self.nodes = {}

        # Serialize the constants, data, inputs and vars
        self.const_list = list(self._serialize_expr(c) for c in self.consts)
        self.data_list = list(self._serialize_expr(d) for d in self.data)
        self.input_list = list(self._serialize_expr(x) for x in self.inputs)
        self.var_list = list(self._serialize_expr(v) for v in self.vars)

        # Serialize the expressions
        self.rhs_list = list(self._serialize_expr(e) for e in self.rhs)

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
            self.new_rhs.append(self._build_expr(idx))

        return

    def _serialize_expr(self, e: Expr):
        key = e.to_key()
        if key in self.key_to_id:
            return self.key_to_id[key]

        node = e.node
        if isinstance(node, ConstNode):
            info = {
                "type": "const",
                "name": node.name,
                "value": node.value,
                "vartype": self.type_to_str[node.type],
            }
        elif isinstance(node, VarNode):
            info = {
                "type": "var",
                "name": node.name,
                "shape": node.shape,
                "vartype": self.type_to_str[node.type],
                "active": node.active,
            }
        elif isinstance(node, IndexNode):
            info = {
                "type": "index",
                "expr": self._serialize_expr(node.expr),
                "index": node.index,
            }
        elif isinstance(node, PassiveNode):
            info = {
                "type": "passive",
                "expr": self._serialize_expr(node.expr),
            }
        elif isinstance(node, UnaryNode):
            info = {
                "type": "unary",
                "op": node.op,
                "expr": self._serialize_expr(node.expr),
            }
        elif isinstance(node, BinaryNode):
            info = {
                "type": "binary",
                "op": node.op,
                "left": self._serialize_expr(node.left),
                "right": self._serialize_expr(node.right),
            }
        else:
            raise TypeError(type(node))

        count = self.counter
        self.key_to_id[key] = count
        self.nodes[count] = info
        self.counter += 1

        return count

    def serialize(self):
        return {"version": 1, "nodes": self.nodes}

    def _build_expr(self, idx):
        node = self.nodes[idx]
        if node["type"] == "const":
            expr = Expr(
                ConstNode(
                    name=node["name"],
                    value=node["value"],
                    type=self.str_to_type[node["vartype"]],
                )
            )
        elif node["type"] == "var":
            expr = Expr(
                VarNode(
                    name=node["name"],
                    shape=node["shape"],
                    type=self.str_to_type[node["vartype"]],
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
                return f"A2D::Vec<{template_name}, {shape[0]}>"
            elif len(shape) == 2:
                return f"A2D::Mat<{template_name}, {shape[0]}, {shape[1]}>"

    def _get_ad_type(self, expr, reference=True, mode="eval", template_name="T__"):
        decl = self._get_expr_type(expr, template_name=template_name)
        if mode == "eval" or not expr.is_active():
            if reference:
                return f"{decl}&"
            return decl
        elif mode == "grad":
            if reference:
                return f"A2D::ADObj<{decl}&>"
            return f"A2D::ADObj<{decl}>"
        else:
            if reference:
                return f"A2D::A2DObj<{decl}&>"
            return f"A2D::A2DObj<{decl}>"

    def get_declarations(
        self, inputs, reference=True, mode="eval", template_name="T__"
    ):
        lines = []

        for expr in inputs:
            decl = self._get_ad_type(
                expr, reference=reference, mode=mode, template_name=template_name
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
            decl = self._get_ad_type(expr, mode=mode, template_name=template_name)

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
                reference=False,
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
            active.append(make_active(lhs.to_cpp(), rhs.to_cpp()))

        return decl, passive, active
