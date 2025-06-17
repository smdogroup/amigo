import math


class ExprNode:
    def __init__(self):
        self.name = None  # For output variable names
        self.active = False

    def evaluate(self, env):
        raise NotImplementedError

    def generate_cpp(self, index=None):
        raise NotImplementedError


class ConstNode(ExprNode):
    def __init__(self, name=None, value=None, type=float):
        super().__init__()
        self.name = name
        self.value = value
        self.type = type
        self.active = False

    def evaluate(self, env):
        return self.value

    def generate_cpp(self, index=None):
        if self.name is not None:
            return self.name
        return str(self.value)


class VarNode(ExprNode):
    def __init__(self, name, shape=None, type=float, active=True):
        super().__init__()
        self.name = name
        self.shape = shape
        self.type = type
        self.active = active

    def generate_cpp(self, index=None):
        if index is None:
            return self.name
        elif isinstance(index, tuple):
            # For 2D index like A[i][j]
            i_str = index[0].generate_cpp()
            j_str = index[1].generate_cpp()
            return f"{self.name}({i_str}, {j_str})"
        else:
            return f"{self.name}[{index.generate_cpp()}]"


class IndexNode(ExprNode):
    def __init__(self, array_node, index_node):
        super().__init__()
        self.array_node = array_node
        self.index_node = index_node  # can be ExprNode or tuple of ExprNodes
        self.active = self.array_node.active

    def evaluate(self, env):
        array_val = self.array_node.evaluate(env)
        if isinstance(self.index_node, tuple):
            idx = tuple(i.evaluate(env) for i in self.index_node)
            return array_val[idx[0]][idx[1]]
        else:
            idx = self.index_node.evaluate(env)
            return array_val[idx]

    def generate_cpp(self, index=None):
        arr = self.array_node.generate_cpp()
        if isinstance(self.index_node, tuple):
            idxs = tuple(i.generate_cpp(index) for i in self.index_node)
            return f"{arr}({idxs[0]}, {idxs[1]})"
        else:
            idx = self.index_node.generate_cpp(index)
            return f"{arr}[{idx}]"


class OpNode(ExprNode):
    def __init__(self, op, left, right):
        super().__init__()
        self.op = op
        self.left = left
        self.right = right
        if self.left.active or self.right.active:
            self.active = True
        else:
            self.active = False

    def evaluate(self, env):
        lval = self.left.evaluate(env)
        rval = self.right.evaluate(env)
        # Broadcast-aware evaluation
        if hasattr(lval, "__len__") and hasattr(rval, "__len__"):
            return [self._op(a, b) for a, b in zip(lval, rval)]
        elif hasattr(lval, "__len__"):
            return [self._op(a, rval) for a in lval]
        elif hasattr(rval, "__len__"):
            return [self._op(lval, b) for b in rval]
        else:
            return self._op(lval, rval)

    def _op(self, a, b):
        if self.op == "+":
            return a + b
        elif self.op == "-":
            return a - b
        elif self.op == "*":
            return a * b
        elif self.op == "/":
            return a / b
        elif self.op == "**":
            return a**b
        elif self.op == "atan2":
            return math.atan2(a, b)
        return None

    def generate_cpp(self, index=None):
        a = self.left.generate_cpp(index)
        b = self.right.generate_cpp(index)
        if self.op == "**":
            return f"A2D::pow({a}, {b})"
        elif self.op == "atan2":
            return f"A2D::atan2({a}, {b})"
        return f"({a} {self.op} {b})"


class UnaryNode(ExprNode):
    def __init__(self, func_name, func, operand):
        super().__init__()
        self.func_name = func_name
        self.func = func
        self.operand = operand
        self.active = operand.active

    def evaluate(self, env):
        val = self.operand.evaluate(env)
        if hasattr(val, "__len__"):
            return [self.func(v) for v in val]
        return self.func(val)

    def generate_cpp(self, index=None):
        a = self.operand.generate_cpp(index)
        return f"A2D::{self.func_name}({a})"


class UnaryNegNode(ExprNode):
    def __init__(self, operand):
        super().__init__()
        self.operand = operand
        self.active = self.operand.active

    def evaluate(self, env):
        val = self.operand.evaluate(env)
        if hasattr(val, "__len__"):
            return [-v for v in val]
        return -val

    def generate_cpp(self, index=None):
        a = self.operand.generate_cpp(index)
        return f"-({a})"


class Expr:
    def __init__(self, node: ExprNode):
        self.node = node
        self.active = node.active

    def __neg__(self):
        return Expr(UnaryNegNode(self.node))

    def __add__(self, other):
        return Expr(OpNode("+", self.node, self._to_node(other)))

    def __sub__(self, other):
        return Expr(OpNode("-", self.node, self._to_node(other)))

    def __mul__(self, other):
        return Expr(OpNode("*", self.node, self._to_node(other)))

    def __truediv__(self, other):
        return Expr(OpNode("/", self.node, self._to_node(other)))

    def __pow__(self, other):
        return Expr(OpNode("**", self.node, self._to_node(other)))

    def __radd__(self, other):
        return Expr(OpNode("+", self._to_node(other), self.node))

    def __rsub__(self, other):
        return Expr(OpNode("-", self._to_node(other), self.node))

    def __rmul__(self, other):
        return Expr(OpNode("*", self._to_node(other), self.node))

    def __rtruediv__(self, other):
        return Expr(OpNode("/", self._to_node(other), self.node))

    def __rpow__(self, other):
        return Expr(OpNode("**", self._to_node(other), self.node))

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx_node = tuple(self._to_node(i) for i in idx)
        else:
            idx_node = self._to_node(idx)
        return Expr(IndexNode(self.node, idx_node))

    def evaluate(self, env):
        return self.node.evaluate(env)

    def generate_cpp(self, index=None):
        return self.node.generate_cpp(index=index)

    def _to_node(self, val):
        if isinstance(val, Expr):
            return val.node
        if isinstance(val, ExprNode):
            return val
        if isinstance(val, (int, float)):
            return ConstNode(value=val)
        raise TypeError(f"Unsupported value: {val}")
