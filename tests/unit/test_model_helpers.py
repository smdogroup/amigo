"""
Unit tests for pure-Python helpers in amigo/model.py.

Covers GlobalIndexPool and the expression-string parser functions.
The compiled C extension (amigo.amigo) is stubbed out by conftest.py so
these tests run without a C++ build.

Test areas:
  - GlobalIndexPool.allocate(): returns correctly shaped numpy arrays with
    contiguous, non-overlapping integer indices; counter advances after each call.
  - _parse_var_expr(): parses dotted-path strings (with optional subscript
    notation) into a (path_list, indices_tuple) pair.
  - _parse_attribute_path(): converts an AST Attribute node into an ordered
    list of name segments.
"""

import ast
import pytest
import numpy as np

from amigo.model import GlobalIndexPool, _parse_var_expr, _parse_attribute_path


# ---------------------------------------------------------------------------
# GlobalIndexPool — contiguous, non-overlapping index allocation
#
# Each call to allocate(shape) returns a numpy array of the requested shape
# filled with the next available integer indices. The pool's counter advances
# by the total number of elements allocated.
# ---------------------------------------------------------------------------


class TestGlobalIndexPool:
    """GlobalIndexPool: shape, values, non-overlapping ranges, and counter."""

    def test_first_allocate_shape_and_values(self):
        """First allocate((3,)) returns array of shape (3,) with values [0, 1, 2]."""
        pool = GlobalIndexPool()
        result = pool.allocate((3,))
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_successive_calls_non_overlapping(self):
        """Successive allocations return non-overlapping index ranges."""
        pool = GlobalIndexPool()
        first = pool.allocate((3,))
        second = pool.allocate((2,))
        np.testing.assert_array_equal(first, [0, 1, 2])
        np.testing.assert_array_equal(second, [3, 4])

    def test_allocate_2d_shape(self):
        """allocate((2, 3)) returns array of shape (2, 3)."""
        pool = GlobalIndexPool()
        result = pool.allocate((2, 3))
        assert result.shape == (2, 3)

    def test_counter_advances_after_each_allocation(self):
        """Counter advances correctly after each allocation."""
        pool = GlobalIndexPool()
        assert pool.counter == 0
        pool.allocate((3,))
        assert pool.counter == 3
        pool.allocate((2, 3))
        assert pool.counter == 9  # 3 + 6


# ---------------------------------------------------------------------------
# _parse_var_expr / _parse_attribute_path — expression string parsing
#
# _parse_var_expr converts strings like "model.group.var" or
# "model.group.var[:, 0]" into a (path_list, indices_tuple) pair.
# _parse_attribute_path converts an AST Attribute node into an ordered
# list of name segments (e.g. "a.b" → ["a", "b"]).
# ---------------------------------------------------------------------------


class TestParseVarExpr:
    """_parse_var_expr: dotted paths, slice notation, and invalid input handling."""

    def test_simple_attribute_path(self):
        """'model.group.var' returns (['model', 'group', 'var'], None)."""
        path, indices = _parse_var_expr("model.group.var")
        assert path == ["model", "group", "var"]
        assert indices is None

    def test_full_slice(self):
        """'model.group.var[:]' returns path and indices containing slice(None, None, None)."""
        path, indices = _parse_var_expr("model.group.var[:]")
        assert path == ["model", "group", "var"]
        assert indices == (slice(None, None, None),)

    def test_multi_index_slice_and_int(self):
        """'model.group.var[:, 0]' returns indices (slice(None, None, None), 0)."""
        path, indices = _parse_var_expr("model.group.var[:, 0]")
        assert path == ["model", "group", "var"]
        assert indices == (slice(None, None, None), 0)

    def test_invalid_expression_raises_value_error(self):
        """A string that is not a valid attribute or subscript expression raises ValueError."""
        with pytest.raises(ValueError):
            _parse_var_expr("not a valid expression!")


class TestParseAttributePath:
    """_parse_attribute_path: reconstructs ordered name segments from an AST node."""

    def test_two_part_path(self):
        """Parsing AST of 'a.b' and calling _parse_attribute_path returns ['a', 'b']."""
        node = ast.parse("a.b", mode="eval").body
        result = _parse_attribute_path(node)
        assert result == ["a", "b"]
