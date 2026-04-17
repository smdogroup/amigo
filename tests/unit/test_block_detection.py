"""
Unit tests for amigo/block_detection.py.

Covers the internal helpers and the top-level block detection function used
by the BISC convexifier to identify block structure in the KKT sparsity pattern.

Test areas:
  - _compute_degrees(): counts primal-primal edges per variable; multiplier
    (dual) rows always have degree 0.
  - _find_connected_components(): BFS-based connected component detection on
    a primal adjacency structure.
  - detect_bfs_level_blocks(): top-level function that classifies primal
    variables into small blocks (direct eigendecomp), BFS chains (block-
    tridiagonal Schur propagation), and hub variables (high-degree nodes).

All tests use hand-crafted CSR arrays (rowp, cols, mult_ind) so no compiled
extension or solver is required.
"""

import numpy as np
import pytest

from amigo.block_detection import (
    detect_bfs_level_blocks,
    _compute_degrees,
    _find_connected_components,
    _build_primal_adjacency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_linear_chain_csr(n):
    """Build CSR arrays for a symmetric linear chain of n nodes."""
    rowp = [0]
    cols = []
    for i in range(n):
        neighbors = []
        if i > 0:
            neighbors.append(i - 1)
        if i < n - 1:
            neighbors.append(i + 1)
        cols.extend(neighbors)
        rowp.append(rowp[-1] + len(neighbors))
    return np.array(rowp), np.array(cols, dtype=np.int64)


# ---------------------------------------------------------------------------
# _compute_degrees — primal-primal degree per variable
#
# Counts the number of primal (non-multiplier) neighbours for each variable.
# Multiplier rows (dual variables) are excluded from both the count and the
# neighbour set, so their degree is always 0.
# ---------------------------------------------------------------------------


class TestComputeDegrees:
    """_compute_degrees: multiplier rows have degree 0; primal degrees are correct."""

    def test_multiplier_row_has_zero_degree(self):
        """Multiplier (dual) rows must always have degree 0."""
        # 3-node graph: node 0 is a multiplier, nodes 1 and 2 are primal
        # Edges: 1-2 (symmetric)
        rowp = np.array([0, 2, 4, 6])
        cols = np.array([1, 2, 0, 2, 0, 1])
        mult_ind = np.array([True, False, False])

        degree = _compute_degrees(rowp, cols, mult_ind)

        assert degree[0] == 0, "Multiplier node must have degree 0"

    def test_triangle_graph_degrees(self):
        """All-primal triangle: each node has primal-primal degree 2."""
        # Symmetric triangle: 0-1, 0-2, 1-2
        rowp = np.array([0, 2, 4, 6])
        cols = np.array([1, 2, 0, 2, 0, 1])
        mult_ind = np.array([False, False, False])

        degree = _compute_degrees(rowp, cols, mult_ind)

        assert degree[0] == 2
        assert degree[1] == 2
        assert degree[2] == 2


# ---------------------------------------------------------------------------
# _find_connected_components — BFS connected component detection
#
# Operates on the primal adjacency structure produced by _build_primal_adjacency.
# Returns a list of components, each a list of variable indices.
# ---------------------------------------------------------------------------


class TestFindConnectedComponents:
    """_find_connected_components: single component and multi-component graphs."""

    def test_single_component_triangle(self):
        """Fully connected triangle → exactly one component with all 3 nodes."""
        rowp = np.array([0, 2, 4, 6])
        cols = np.array([1, 2, 0, 2, 0, 1])
        mult_ind = np.array([False, False, False])

        primal_indices, primal_set, adj = _build_primal_adjacency(
            rowp, cols, mult_ind, hub_set=set()
        )
        components = _find_connected_components(primal_indices, primal_set, adj)

        assert len(components) == 1
        assert sorted(components[0]) == [0, 1, 2]

    def test_two_disconnected_components(self):
        """Nodes 0-1 connected, nodes 2-3 connected → two components."""
        # 4 nodes: 0-1 connected, 2-3 connected, no cross edges
        rowp = np.array([0, 1, 2, 3, 4])
        cols = np.array([1, 0, 3, 2])
        mult_ind = np.array([False, False, False, False])

        primal_indices, primal_set, adj = _build_primal_adjacency(
            rowp, cols, mult_ind, hub_set=set()
        )
        components = _find_connected_components(primal_indices, primal_set, adj)

        assert len(components) == 2

        # Each component should be one of the two pairs
        component_sets = [frozenset(c) for c in components]
        assert frozenset({0, 1}) in component_sets
        assert frozenset({2, 3}) in component_sets


# ---------------------------------------------------------------------------
# detect_bfs_level_blocks — top-level block classification
#
# Classifies all primal variables into:
#   small_blocks  — components small enough for direct eigendecomposition
#   bfs_chains    — large chain-structured components (block-tridiagonal)
#   hub_indices   — high-degree variables isolated as scalar blocks
#
# The function accepts max_eigendecomp_size and max_block_size thresholds
# that control when a component is treated as a chain vs. small blocks.
# ---------------------------------------------------------------------------


class TestDetectBfsLevelBlocks:
    """detect_bfs_level_blocks: isolated nodes, disconnected components, and chain overflow."""

    def test_single_isolated_primal_variable(self):
        """Single isolated primal node → one small block, no chains."""
        rowp = np.array([0, 0])
        cols = np.array([], dtype=np.int64)
        mult_ind = np.array([False])

        small_blocks, bfs_chains, hub_indices = detect_bfs_level_blocks(
            rowp, cols, mult_ind
        )

        assert len(small_blocks) == 1, "Expected exactly one small block"
        assert len(bfs_chains) == 0, "Expected no BFS chains"
        assert list(small_blocks[0]) == [0]

    def test_two_disconnected_components_become_two_small_blocks(self):
        """Two disconnected 2-node components → two small blocks (size <= 20)."""
        rowp = np.array([0, 1, 2, 3, 4])
        cols = np.array([1, 0, 3, 2])
        mult_ind = np.array([False, False, False, False])

        small_blocks, bfs_chains, hub_indices = detect_bfs_level_blocks(
            rowp, cols, mult_ind
        )

        # Both components have size 2 <= max_eigendecomp_size=20 → small blocks
        assert len(small_blocks) == 2
        assert len(bfs_chains) == 0

        all_nodes = sorted(n for block in small_blocks for n in block.tolist())
        assert all_nodes == [0, 1, 2, 3]

    def test_large_chain_with_max_block_size_zero_becomes_small_blocks(self):
        """Linear chain of 25 nodes with max_block_size=0 → all small blocks, no chains."""
        n = 25
        rowp, cols = make_linear_chain_csr(n)
        mult_ind = np.array([False] * n)

        small_blocks, bfs_chains, hub_indices = detect_bfs_level_blocks(
            rowp,
            cols,
            mult_ind,
            max_block_size=0,  # any level size > 0 triggers conversion
            max_eigendecomp_size=20,
        )

        # With max_block_size=0, every BFS level (size >= 1) exceeds the limit,
        # so the chain is broken into individual small blocks.
        assert len(bfs_chains) == 0, "Expected no BFS chains when max_block_size=0"

        # All 25 nodes should appear in small_blocks
        all_nodes = sorted(n for block in small_blocks for n in block.tolist())
        assert len(all_nodes) == n
        assert all_nodes == list(range(n))
