"""
Unit tests for amigo/expressions.py.

Covers the ExprNode subclass hierarchy (ConstNode, VarNode, IndexNode,
PassiveNode, UnaryNode, BinaryNode) and the Expr wrapper class.

Test areas:
  - is_zero(): zero-value detection and propagation through unary nodes
  - is_active(): active/passive flag behaviour for each node type
  - compute_cost(): operation cost increments (sqrt +20, trig/exp +100, log +200,
    negation/fabs +1, division +10, addition/subtraction/multiplication +1)
  - to_cpp(): C++ code generation strings for all node types
  - Expr arithmetic operators: zero-propagation shortcuts in __add__, __sub__,
    __mul__, __truediv__, __pow__, __neg__, and their reflected variants
  - Expr.serialize() / Expr.deserialize(): round-trip fidelity for all node types
  - _normalize_shape(): shape normalisation and validation
"""

import pytest
from amigo.expressions import (
    Expr,
    ConstNode,
    VarNode,
    IndexNode,
    PassiveNode,
    UnaryNode,
    BinaryNode,
)


# ---------------------------------------------------------------------------
# is_zero() — zero-value detection on each node type
# ---------------------------------------------------------------------------


class TestIsZero:
    def test_const_zero_int(self):
        assert ConstNode(value=0).is_zero() is True

    def test_const_zero_float(self):
        assert ConstNode(value=0.0).is_zero() is True

    def test_const_nonzero(self):
        assert ConstNode(value=1).is_zero() is False

    def test_var_is_never_zero(self):
        assert VarNode("x").is_zero() is False

    def test_unary_neg_propagates_zero(self):
        # UnaryNode wraps an Expr, not a bare ExprNode
        zero_expr = Expr(ConstNode(value=0))
        node = UnaryNode("-", zero_expr)
        assert node.is_zero() is True

    def test_unary_neg_nonzero(self):
        nonzero_expr = Expr(VarNode("x"))
        node = UnaryNode("-", nonzero_expr)
        assert node.is_zero() is False


# ---------------------------------------------------------------------------
# is_active() — active/passive flag for each node type
# ---------------------------------------------------------------------------


class TestIsActive:
    def test_const_is_never_active(self):
        assert ConstNode(value=1.0).is_active() is False

    def test_const_named_is_never_active(self):
        assert ConstNode(name="c").is_active() is False

    def test_var_active_true(self):
        assert VarNode("x", active=True).is_active() is True

    def test_var_active_false(self):
        assert VarNode("x", active=False).is_active() is False

    def test_passive_node_is_never_active_even_when_inner_is_active(self):
        active_expr = Expr(VarNode("x", active=True))
        node = PassiveNode(active_expr)
        assert node.is_active() is False

    def test_passive_node_is_never_active_when_inner_is_inactive(self):
        inactive_expr = Expr(VarNode("x", active=False))
        node = PassiveNode(inactive_expr)
        assert node.is_active() is False


# ---------------------------------------------------------------------------
# compute_cost() — operation cost increments for each node type
# ---------------------------------------------------------------------------


class TestComputeCost:
    def test_const_cost_is_zero(self):
        assert ConstNode(value=5.0).compute_cost() == 0

    def test_var_cost_is_zero(self):
        assert VarNode("x").compute_cost() == 0

    def test_unary_sqrt_cost(self):
        base_expr = Expr(VarNode("x"))
        node = UnaryNode("sqrt", base_expr)
        assert node.compute_cost() == base_expr.compute_cost() + 20

    def test_unary_sin_cost(self):
        base_expr = Expr(VarNode("x"))
        node = UnaryNode("sin", base_expr)
        assert node.compute_cost() == base_expr.compute_cost() + 100

    def test_unary_log_cost(self):
        base_expr = Expr(VarNode("x"))
        node = UnaryNode("log", base_expr)
        assert node.compute_cost() == base_expr.compute_cost() + 200

    def test_unary_neg_cost(self):
        base_expr = Expr(VarNode("x"))
        node = UnaryNode("-", base_expr)
        assert node.compute_cost() == base_expr.compute_cost() + 1

    def test_unary_fabs_cost(self):
        base_expr = Expr(VarNode("x"))
        node = UnaryNode("fabs", base_expr)
        assert node.compute_cost() == base_expr.compute_cost() + 1

    def test_binary_div_cost(self):
        a = Expr(VarNode("a"))
        b = Expr(VarNode("b"))
        node = BinaryNode("/", a, b)
        assert node.compute_cost() == a.compute_cost() + b.compute_cost() + 10

    def test_binary_add_cost(self):
        a = Expr(VarNode("a"))
        b = Expr(VarNode("b"))
        node = BinaryNode("+", a, b)
        assert node.compute_cost() == a.compute_cost() + b.compute_cost() + 1

    def test_compute_cost_accumulates_over_nested_tree(self):
        # sqrt(x) costs 20; log(sqrt(x)) costs 20 + 200 = 220
        x = Expr(VarNode("x"))
        sqrt_node = UnaryNode("sqrt", x)
        sqrt_expr = Expr(sqrt_node)
        log_node = UnaryNode("log", sqrt_expr)
        assert log_node.compute_cost() == 220


# ---------------------------------------------------------------------------
# to_cpp() — C++ code generation strings for each node type
# ---------------------------------------------------------------------------


class TestToCpp:
    def test_const_named(self):
        assert ConstNode(name="c").to_cpp() == "c"

    def test_const_value(self):
        assert ConstNode(value=3.0).to_cpp() == "3.0"

    def test_var(self):
        assert VarNode("x").to_cpp() == "x"

    def test_index_1d(self):
        # arr_expr wraps VarNode("arr", shape=(5,))
        arr_expr = Expr(VarNode("arr", shape=(5,)))
        node = IndexNode(arr_expr, 2)
        assert node.to_cpp() == "arr[2]"

    def test_index_2d(self):
        # arr_expr wraps VarNode("arr", shape=(3, 2))
        arr_expr = Expr(VarNode("arr", shape=(3, 2)))
        node = IndexNode(arr_expr, (1, 0))
        assert node.to_cpp() == "arr(1, 0)"

    def test_passive_node(self):
        # inner_expr wraps VarNode("x")
        inner_expr = Expr(VarNode("x"))
        node = PassiveNode(inner_expr)
        assert node.to_cpp() == "A2D::get_data(x)"

    def test_unary_neg(self):
        expr = Expr(VarNode("x"))
        node = UnaryNode("-", expr)
        assert node.to_cpp() == "-(x)"

    def test_binary_add(self):
        a = Expr(VarNode("a"))
        b = Expr(VarNode("b"))
        node = BinaryNode("+", a, b)
        assert node.to_cpp() == "(a + b)"

    def test_binary_pow(self):
        a = Expr(VarNode("a"))
        b = Expr(VarNode("b"))
        node = BinaryNode("**", a, b)
        assert node.to_cpp() == "A2D::pow(a, b)"

    def test_binary_atan2(self):
        a = Expr(VarNode("a"))
        b = Expr(VarNode("b"))
        node = BinaryNode("atan2", a, b)
        assert node.to_cpp() == "A2D::atan2(a, b)"

    def test_binary_min2(self):
        a = Expr(VarNode("a"))
        b = Expr(VarNode("b"))
        node = BinaryNode("min2", a, b)
        assert node.to_cpp() == "A2D::min2(a, b)"

    def test_binary_max2(self):
        a = Expr(VarNode("a"))
        b = Expr(VarNode("b"))
        node = BinaryNode("max2", a, b)
        assert node.to_cpp() == "A2D::max2(a, b)"


# ---------------------------------------------------------------------------
# Expr arithmetic operators — zero-propagation shortcuts and operator overloads
#
# The Expr class short-circuits several operations when one operand is zero:
#   zero + e  →  e          (no BinaryNode created)
#   e + zero  →  e
#   zero * e  →  zero const
#   e * zero  →  zero const
#   e - zero  →  e
#   zero / d  →  zero const
# All other combinations produce the expected BinaryNode or UnaryNode tree.
# ---------------------------------------------------------------------------


class TestExprArithmetic:
    """Zero-propagation shortcuts and operator overloads on the Expr wrapper class."""

    @pytest.fixture(autouse=True)
    def _fixtures(self):
        self.zero = Expr(ConstNode(value=0.0))
        self.e = Expr(VarNode("x"))

    # --- __add__ ---

    def test_zero_plus_e_returns_non_zero_operand(self):
        """zero + e → result is the non-zero operand (no BinaryNode wrapper)."""
        result = self.zero + self.e
        assert result.serialize() == self.e.serialize()

    def test_e_plus_zero_returns_non_zero_operand(self):
        """e + zero → result is the non-zero operand."""
        result = self.e + self.zero
        assert result.serialize() == self.e.serialize()

    def test_nonzero_plus_nonzero_produces_binary_node(self):
        """e + e (both non-zero) → BinaryNode('+', ...)."""
        e2 = Expr(VarNode("y"))
        result = self.e + e2
        assert isinstance(result.node, BinaryNode)
        assert result.node.op == "+"

    # --- __mul__ ---

    def test_zero_times_e_is_zero(self):
        """zero * e → result.is_zero() is True."""
        result = self.zero * self.e
        assert result.is_zero() is True

    def test_e_times_zero_is_zero(self):
        """e * zero → result.is_zero() is True."""
        result = self.e * self.zero
        assert result.is_zero() is True

    # --- __pow__ ---

    def test_e_pow_zero_int_is_one(self):
        """e ** 0 (integer 0) → result.is_zero() is False and to_cpp() == '1.0'."""
        result = self.e**0
        assert result.is_zero() is False
        assert result.to_cpp() == "1.0"

    # --- __neg__ ---

    def test_neg_e_wraps_unary_minus(self):
        """-e → result.node is a UnaryNode with op='-'."""
        result = -self.e
        assert isinstance(result.node, UnaryNode)
        assert result.node.op == "-"

    # --- __sub__ ---

    def test_e_minus_zero_returns_non_zero_operand(self):
        """e - zero → result is the non-zero operand."""
        result = self.e - self.zero
        assert result.serialize() == self.e.serialize()

    def test_zero_minus_e_wraps_unary_minus(self):
        """zero - e → result wraps UnaryNode('-', e)."""
        result = self.zero - self.e
        assert isinstance(result.node, UnaryNode)
        assert result.node.op == "-"
        assert result.node.expr.serialize() == self.e.serialize()

    # --- __truediv__ ---

    def test_zero_divided_by_d_is_zero(self):
        """zero / d → result.is_zero() is True."""
        d = Expr(VarNode("d"))
        result = self.zero / d
        assert result.is_zero() is True

    # --- __radd__ ---

    def test_radd_int_produces_binary_node(self):
        """1 + e → BinaryNode('+', const_1, e)."""
        result = 1 + self.e
        assert isinstance(result.node, BinaryNode)
        assert result.node.op == "+"
        assert isinstance(result.node.left.node, ConstNode)
        assert result.node.left.node.value == 1
        assert result.node.right.serialize() == self.e.serialize()

    # --- __rsub__ ---

    def test_rsub_float_produces_binary_node(self):
        """1.0 - e → BinaryNode('-', const_1.0, e)."""
        result = 1.0 - self.e
        assert isinstance(result.node, BinaryNode)
        assert result.node.op == "-"
        assert isinstance(result.node.left.node, ConstNode)
        assert result.node.left.node.value == 1.0
        assert result.node.right.serialize() == self.e.serialize()

    # --- __rmul__ ---

    def test_rmul_int_produces_binary_node(self):
        """2 * e → BinaryNode('*', const_2, e)."""
        result = 2 * self.e
        assert isinstance(result.node, BinaryNode)
        assert result.node.op == "*"
        assert isinstance(result.node.left.node, ConstNode)
        assert result.node.left.node.value == 2
        assert result.node.right.serialize() == self.e.serialize()

    # --- __rtruediv__ ---

    def test_rtruediv_float_produces_binary_node(self):
        """1.0 / e → BinaryNode('/', const_1.0, e)."""
        result = 1.0 / self.e
        assert isinstance(result.node, BinaryNode)
        assert result.node.op == "/"
        assert isinstance(result.node.left.node, ConstNode)
        assert result.node.left.node.value == 1.0
        assert result.node.right.serialize() == self.e.serialize()

    def test_rtruediv_zero_numerator_is_zero(self):
        """0.0 / e → result.is_zero() is True."""
        result = 0.0 / self.e
        assert result.is_zero() is True


# ---------------------------------------------------------------------------
# Expr.serialize() / Expr.deserialize() — round-trip fidelity
#
# Serialization converts an expression tree to a nested tuple structure.
# Deserialization reconstructs the tree. The round-trip property requires
# that deserialize(serialize(e)).serialize() == serialize(e) for any Expr e.
# ---------------------------------------------------------------------------


class TestExprRoundTrip:
    """Expr serialize/deserialize round-trip: all node types and edge cases."""

    def test_const_node_int_type_preserved(self):
        """ConstNode(value=5, type=int) round-trips with type=int preserved."""
        original = Expr(ConstNode(value=5, type=int))
        deserialized = Expr.deserialize(original.serialize())
        assert isinstance(deserialized.node, ConstNode)
        assert deserialized.node.type is int
        assert deserialized.node.value == 5

    def test_var_node_active_false_preserved(self):
        """VarNode('x', active=False) round-trips with active=False preserved."""
        original = Expr(VarNode("x", active=False))
        deserialized = Expr.deserialize(original.serialize())
        assert isinstance(deserialized.node, VarNode)
        assert deserialized.node.active is False
        assert deserialized.node.name == "x"

    def test_nested_expression_round_trip(self):
        """(a + b) * c round-trips: double serialize equals single serialize."""
        a = Expr(VarNode("a"))
        b = Expr(VarNode("b"))
        c = Expr(VarNode("c"))
        expr = (a + b) * c
        once = expr.serialize()
        twice = Expr.deserialize(once).serialize()
        assert twice == once

    def test_index_node_2d_tuple_preserved(self):
        """IndexNode with 2-D tuple index (1, 0) round-trips with index tuple preserved."""
        arr = Expr(VarNode("arr", shape=(3, 2)))
        original = Expr(IndexNode(arr, (1, 0)))
        deserialized = Expr.deserialize(original.serialize())
        assert isinstance(deserialized.node, IndexNode)
        assert deserialized.node.index == (1, 0)

    def test_unrecognized_op_returns_value_error(self):
        """Expr.deserialize with unknown op returns a ValueError (does not raise)."""
        result = Expr.deserialize(("unknown_op", "foo", "bar"))
        assert isinstance(result, ValueError)


# ---------------------------------------------------------------------------
# _normalize_shape() — shape normalisation and validation
#
# Converts None, int, list, or tuple inputs to a canonical tuple form.
# Raises TypeError for unsupported input types and ValueError for shapes
# with more than two dimensions (amigo only supports 1-D and 2-D arrays).
# ---------------------------------------------------------------------------

from amigo.expressions import _normalize_shape


class TestNormalizeShape:
    """_normalize_shape: valid conversions and error cases."""

    def test_none_returns_none(self):
        assert _normalize_shape(None) is None

    def test_int_returns_1d_tuple(self):
        assert _normalize_shape(3) == (3,)

    def test_list_returns_tuple(self):
        assert _normalize_shape([2, 3]) == (2, 3)

    def test_1d_tuple_unchanged(self):
        assert _normalize_shape((4,)) == (4,)

    def test_3d_tuple_raises_value_error(self):
        with pytest.raises(ValueError):
            _normalize_shape((1, 2, 3))

    def test_float_raises_type_error(self):
        with pytest.raises(TypeError):
            _normalize_shape(3.14)
