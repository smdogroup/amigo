import numpy as np
import pytest

from amigo.fem.cell_types import CellType, REFERENCE_CELLS
from amigo.fem.visualization import _pv_cell_type, _p1_shape_functions


@pytest.mark.parametrize(
    "cell_type",
    [
        CellType.TETRAHEDRON,
        CellType.HEXAHEDRON,
        CellType.PRISM,
        CellType.PYRAMID,
    ],
)
def test_3d_linear_shape_functions_are_nodal_and_partition_unity(cell_type):
    vertices = np.asarray(REFERENCE_CELLS[cell_type].vertices)

    values = _p1_shape_functions(cell_type, vertices)

    np.testing.assert_allclose(values, np.eye(len(vertices)))
    np.testing.assert_allclose(values.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    ("cell_type", "dof_count"),
    [
        (CellType.TETRAHEDRON, 4),
        (CellType.TETRAHEDRON, 10),
        (CellType.HEXAHEDRON, 8),
        (CellType.HEXAHEDRON, 20),
        (CellType.PRISM, 6),
        (CellType.PRISM, 15),
        (CellType.PYRAMID, 5),
    ],
)
def test_linear_3d_cells_map_to_pyvista_types(cell_type, dof_count):
    assert _pv_cell_type(cell_type, dof_count) is not None
