from .expressions import Expr, VarNode, UnaryNode, BinaryNode, ConstNode, PassiveNode


def abs(expr):
    return Expr(UnaryNode("fabs", expr))


def fabs(expr):
    return Expr(UnaryNode("fabs", expr))


def sqrt(expr):
    return Expr(UnaryNode("sqrt", expr))


def sin(expr):
    return Expr(UnaryNode("sin", expr))


def sqrt(expr):
    return Expr(UnaryNode("sqrt", expr))


def asin(expr):
    return Expr(UnaryNode("asin", expr))


def cos(expr):
    return Expr(UnaryNode("cos", expr))


def acos(expr):
    return Expr(UnaryNode("acos", expr))


def tan(expr):
    return Expr(UnaryNode("tan", expr))


def atan(expr):
    return Expr(UnaryNode("atan", expr))


def sinh(expr):
    return Expr(UnaryNode("sinh", expr))


def asinh(expr):
    return Expr(UnaryNode("asinh", expr))


def cosh(expr):
    return Expr(UnaryNode("cosh", expr))


def acosh(expr):
    return Expr(UnaryNode("acosh", expr))


def tanh(expr):
    return Expr(UnaryNode("tanh", expr))


def atanh(expr):
    return Expr(UnaryNode("atanh", expr))


def exp(expr):
    return Expr(UnaryNode("exp", expr))


def log(expr):
    return Expr(UnaryNode("log", expr))


def log10(expr):
    return Expr(UnaryNode("log10", expr))


def atan2(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        raise ValueError("Neither argument is active")
    elif isinstance(a, (int, float)):
        return Expr(BinaryNode("atan2", ConstNode(value=a), b))
    elif isinstance(b, (int, float)):
        return Expr(BinaryNode("atan2", a, ConstNode(value=b)))
    elif isinstance(a, Expr) and isinstance(b, Expr):
        return Expr(BinaryNode("atan2", a, b))
    else:
        raise TypeError("Types not recognized for atan2")


def min2(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        raise ValueError("Neither argument is active")
    elif isinstance(a, (int, float)):
        return Expr(BinaryNode("min2", ConstNode(value=a), b))
    elif isinstance(b, (int, float)):
        return Expr(BinaryNode("min2", a, ConstNode(value=b)))
    elif isinstance(a, Expr) and isinstance(b, Expr):
        return Expr(BinaryNode("min2", a, b))
    else:
        raise TypeError("Types not recognized for min2")


def max2(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        raise ValueError("Neither argument is active")
    elif isinstance(a, (int, float)):
        return Expr(BinaryNode("max2", ConstNode(value=a), b))
    elif isinstance(b, (int, float)):
        return Expr(BinaryNode("max2", a, ConstNode(value=b)))
    elif isinstance(a, Expr) and isinstance(b, Expr):
        return Expr(BinaryNode("max2", a, b))
    else:
        raise TypeError("Types not recognized for max2")


def passive(expr):
    """Force an expression to be passive"""
    if not isinstance(expr.node, VarNode):
        raise TypeError("Passive type must be a variable")
    return Expr(PassiveNode(expr))
