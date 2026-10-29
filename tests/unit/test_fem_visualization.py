import numpy as np
import pyvista as pv
import pytest

from amigo.fem.cell_types import CellType, REFERENCE_CELLS
from amigo.fem.visualization import (
    _p1_shape_functions,
    _pv_cell_type,
    select_grid_domains,
)


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


def test_select_grid_domains_filters_cells_and_point_data():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    grid = pv.UnstructuredGrid(
        {pv.CellType.QUAD: np.array([[0, 1, 2, 3], [1, 4, 5, 2]])},
        points,
    )
    grid.cell_data["mesh_domain"] = np.array(["left", "right"])
    grid.point_data["u"] = np.arange(grid.n_points, dtype=float)

    selected = select_grid_domains(grid, ["right"])

    assert selected.n_cells == 1
    np.testing.assert_allclose(selected.points[:, 0], [1.0, 1.0, 2.0, 2.0])
    np.testing.assert_array_equal(selected.point_data["u"], [1.0, 2.0, 4.0, 5.0])
    assert selected.cell_data["mesh_domain"].tolist() == ["right"]
