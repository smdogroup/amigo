"""
Unit tests for amigo/component.py.

Covers the Meta metadata class and the InputSet, ConstantSet, and DataSet
container classes.

Test areas:
  - Meta construction validation: rejects names with double underscores,
    inverted bounds, out-of-range values, wrong value types, and unknown
    keyword arguments; accepts valid inputs with correct defaults.
  - Meta.serialize() / Meta.deserialize(): round-trip fidelity for all fields.
  - InputSet: add() grows the collection, __getitem__ returns an active VarNode
    Expr, and missing keys raise KeyError.
  - ConstantSet: add() stores a ConstNode with the correct value; non-scalar
    shapes are rejected.
  - DataSet: add() stores an inactive VarNode Expr with the correct shape.
"""

import pytest
from amigo.component import Meta, InputSet, ConstantSet, DataSet
from amigo.expressions import VarNode, ConstNode, Expr


# ---------------------------------------------------------------------------
# Meta — construction validation
#
# Meta stores name, var_type, bounds, value, type, scale, units, and label
# for a component variable. It validates inputs at construction time.
# ---------------------------------------------------------------------------


class TestMetaValidation:
    """Meta construction: validation rules and serialize/deserialize round-trip."""

    # Names containing "__" are reserved for internal amigo identifiers
    def test_name_with_double_underscore_raises(self):
        with pytest.raises(ValueError):
            Meta("bad__name", "input")

    # Bounds must satisfy lower <= upper
    def test_lower_greater_than_upper_raises(self):
        with pytest.raises(ValueError):
            Meta("x", "input", lower=10.0, upper=1.0)

    # For input variables the initial value must lie within [lower, upper]
    def test_input_value_outside_bounds_raises(self):
        with pytest.raises(ValueError):
            Meta("x", "input", value=5.0, lower=0.0, upper=4.0)

    # The value must be an instance of the declared type (default float)
    def test_value_wrong_type_raises(self):
        # default type is float; passing int 1 raises TypeError because isinstance(1, float) is False
        with pytest.raises(TypeError):
            Meta("x", "input", value=1)

    # value=1 with type=int is valid (isinstance(1, int) is True)
    def test_value_correct_int_type_succeeds(self):
        m = Meta("x", "input", value=1, type=int)
        assert m.value == 1
        assert m.type is int

    # Unknown keyword arguments are not silently ignored
    def test_unknown_kwarg_raises(self):
        with pytest.raises(ValueError):
            Meta("x", "input", unknown_kwarg=42)

    # Default bounds are (-inf, +inf) with value=0.0 and type=float
    def test_valid_meta_defaults(self):
        m = Meta("x", "input")
        assert m["lower"] == float("-inf")
        assert m["upper"] == float("inf")
        assert m["value"] == 0.0
        assert m["type"] is float

    # serialize() produces a plain dict; deserialize() reconstructs identical fields
    def test_serialize_deserialize_roundtrip(self):
        m = Meta("y", "constraint", lower=-1.0, upper=1.0, value=0.0)
        data = m.serialize()
        # deserialize pops from the dict, so pass a copy
        m2 = Meta.deserialize(dict(data))
        assert m2.name == m.name
        assert m2.var_type == m.var_type
        assert m2.lower == m.lower
        assert m2.upper == m.upper
        assert m2.value == m.value
        assert m2.type is m.type
        assert m2.units == m.units
        assert m2.scale == m.scale
        assert m2.label == m.label


# ---------------------------------------------------------------------------
# InputSet — named active-variable container
#
# InputSet holds named Expr objects backed by active VarNodes. It is used
# to declare the design variables of a Component.
# ---------------------------------------------------------------------------


class TestInputSet:
    """InputSet: add(), __getitem__, and missing-key error behaviour."""

    def test_add_increases_len(self):
        inputs = InputSet()
        assert len(inputs) == 0
        inputs.add("x")
        assert len(inputs) == 1

    def test_getitem_returns_expr_with_active_varnode(self):
        inputs = InputSet()
        inputs.add("x")
        expr = inputs["x"]
        assert isinstance(expr, Expr)
        assert isinstance(expr.node, VarNode)
        assert expr.node.active is True

    def test_getitem_missing_key_raises_keyerror(self):
        inputs = InputSet()
        with pytest.raises(KeyError):
            _ = inputs["missing"]


# ---------------------------------------------------------------------------
# ConstantSet — named scalar constant container
# DataSet — named passive (non-differentiable) data container
# ---------------------------------------------------------------------------


class TestConstantSet:
    """ConstantSet: stores scalar ConstNode values; rejects non-scalar shapes."""

    def test_add_stores_constnode_with_correct_value(self):
        consts = ConstantSet()
        consts.add("c", 3.14)
        expr = consts["c"]
        assert isinstance(expr, Expr)
        assert isinstance(expr.node, ConstNode)
        assert expr.node.value == 3.14

    def test_add_non_scalar_shape_raises_valueerror(self):
        consts = ConstantSet()
        with pytest.raises(ValueError):
            consts.add("c", 1.0, shape=(3,))


class TestDataSet:
    """DataSet: stores inactive VarNode Exprs with the declared shape."""

    def test_add_stores_varnode_inactive_with_shape(self):
        data = DataSet()
        data.add("d", shape=(3,))
        expr = data["d"]
        assert isinstance(expr, Expr)
        node = expr.node
        assert isinstance(node, VarNode)
        assert node.active is False
        assert node.shape == (3,)
