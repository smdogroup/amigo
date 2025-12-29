import math
from .expressions import Expr, UnaryNode, OpNode, ConstNode, PassiveNode


def abs(expr):
    return Expr(UnaryNode("fabs", math.fabs, expr.node))


def fabs(expr):
    return Expr(UnaryNode("fabs", math.fabs, expr.node))


def sqrt(expr):
    return Expr(UnaryNode("sqrt", math.sqrt, expr.node))


def sin(expr):
    return Expr(UnaryNode("sin", math.sin, expr.node))


def sqrt(expr):
    return Expr(UnaryNode("sqrt", math.sqrt, expr.node))


def asin(expr):
    return Expr(UnaryNode("asin", math.asin, expr.node))


def cos(expr):
    return Expr(UnaryNode("cos", math.cos, expr.node))


def acos(expr):
    return Expr(UnaryNode("acos", math.acos, expr.node))


def tan(expr):
    return Expr(UnaryNode("tan", math.tan, expr.node))


def atan(expr):
    return Expr(UnaryNode("atan", math.atan, expr.node))


def sinh(expr):
    return Expr(UnaryNode("sinh", math.sinh, expr.node))


def asinh(expr):
    return Expr(UnaryNode("asinh", math.asinh, expr.node))


def cosh(expr):
    return Expr(UnaryNode("cosh", math.cosh, expr.node))


def acosh(expr):
    return Expr(UnaryNode("acosh", math.acosh, expr.node))


def tanh(expr):
    return Expr(UnaryNode("tanh", math.tanh, expr.node))


def atanh(expr):
    return Expr(UnaryNode("atanh", math.atanh, expr.node))


def exp(expr):
    return Expr(UnaryNode("exp", math.exp, expr.node))


def log(expr):
    return Expr(UnaryNode("log", math.log, expr.node))


def log10(expr):
    return Expr(UnaryNode("log10", math.log10, expr.node))


def atan2(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        raise ValueError("Neither argument is active")
    elif isinstance(a, (int, float)):
        return Expr(OpNode("atan2", ConstNode(value=a), b.node))
    elif isinstance(b, (int, float)):
        return Expr(OpNode("atan2", a.node, ConstNode(value=b)))
    else:
        return Expr(OpNode("atan2", a.node, b.node))


def min2(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        raise ValueError("Neither argument is active")
    elif isinstance(a, (int, float)):
        return Expr(OpNode("min2", ConstNode(value=a), b.node))
    elif isinstance(b, (int, float)):
        return Expr(OpNode("min2", a.node, ConstNode(value=b)))
    else:
        return Expr(OpNode("min2", a.node, b.node))


def max2(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        raise ValueError("Neither argument is active")
    elif isinstance(a, (int, float)):
        return Expr(OpNode("max2", ConstNode(value=a), b.node))
    elif isinstance(b, (int, float)):
        return Expr(OpNode("max2", a.node, ConstNode(value=b)))
    else:
        return Expr(OpNode("max2", a.node, b.node))


def passive(expr):
    """Force an expression to be passive"""
    return Expr(PassiveNode(expr))
