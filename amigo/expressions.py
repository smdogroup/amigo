import math


class ExprNode:
    def __init__(self):
        self.name = None
        self.active = False

    def generate_cpp(self, index=None):
        raise NotImplementedError


class ConstNode(ExprNode):
    def __init__(self, name=None, value=None, type=float):
        super().__init__()
        self.name = name
        self.value = value
        self.type = type
        self.active = False

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

    def generate_cpp(self, index=None):
        a = self.left.generate_cpp(index)
        b = self.right.generate_cpp(index)
        if self.op == "**":
            # Specialize pow for case with constant integer powers
            rnode = self.right.node
            if isinstance(rnode, ConstNode) and rnode.type == int and rnode.value > 0:
                return " * ".join([f"({a})"] * rnode.value)
            else:
                return f"A2D::pow({a}, {b})"
        elif self.op == "atan2":
            return f"A2D::atan2({a}, {b})"
        elif self.op == "min2":
            return f"A2D::min2({a}, {b})"
        elif self.op == "max2":
            return f"A2D::max2({a}, {b})"
        return f"({a} {self.op} {b})"


class UnaryNode(ExprNode):
    def __init__(self, func_name, func, operand):
        super().__init__()
        self.func_name = func_name
        self.func = func
        self.operand = operand
        self.active = operand.active

    def generate_cpp(self, index=None):
        a = self.operand.generate_cpp(index)
        return f"A2D::{self.func_name}({a})"


class UnaryNegNode(ExprNode):
    def __init__(self, operand):
        super().__init__()
        self.operand = operand
        self.active = self.operand.active

    def generate_cpp(self, index=None):
        a = self.operand.generate_cpp(index)
        return f"-({a})"


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
        self.active = node.active

    def __neg__(self):
        return Expr(UnaryNegNode(self.node))

    def __add__(self, other):
        return Expr(OpNode("+", self, _to_expr(other)))

    def __sub__(self, other):
        return Expr(OpNode("-", self, _to_expr(other)))

    def __mul__(self, other):
        return Expr(OpNode("*", self, _to_expr(other)))

    def __truediv__(self, other):
        return Expr(OpNode("/", self, _to_expr(other)))

    def __pow__(self, other):
        return Expr(OpNode("**", self, _to_expr(other)))

    def __radd__(self, other):
        return Expr(OpNode("+", _to_expr(other), self))

    def __rsub__(self, other):
        return Expr(OpNode("-", _to_expr(other), self))

    def __rmul__(self, other):
        return Expr(OpNode("*", _to_expr(other), self))

    def __rtruediv__(self, other):
        return Expr(OpNode("/", _to_expr(other), self))

    def __rpow__(self, other):
        return Expr(OpNode("**", _to_expr(other), self))

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx_node = tuple(_to_expr(i) for i in idx)
        else:
            idx_node = _to_expr(idx)
        return Expr(IndexNode(self.node, idx_node))

    def generate_cpp(self, index=None, use_vars=True):
        if use_vars and self.name is not None:
            return self.name
        return self.node.generate_cpp(index=index)
