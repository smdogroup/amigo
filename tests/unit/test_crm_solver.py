import numpy as np
import pyvista as pv
import pytest

from apps.crm_solver import deform_crm_grid


def test_deform_crm_grid_applies_translations_without_mutating_source():
    grid = pv.PolyData(np.zeros((2, 3)))
    displacement_grids = {}
    for field_name, values in {
        "u": [1.0, 2.0],
        "v": [3.0, 4.0],
        "w": [5.0, 6.0],
    }.items():
        field_grid = grid.copy(deep=True)
        field_grid.point_data[field_name] = values
        displacement_grids[field_name] = field_grid

    deformed = deform_crm_grid(grid, displacement_grids, scale=2.0)

    np.testing.assert_allclose(
        deformed.points,
        [[2.0, 6.0, 10.0], [4.0, 8.0, 12.0]],
    )
    np.testing.assert_allclose(grid.points, np.zeros((2, 3)))
    assert deformed is not grid


def test_deform_crm_grid_requires_all_translation_fields():
    grid = pv.PolyData(np.zeros((1, 3)))

    with pytest.raises(ValueError, match="field 'u' is missing"):
        deform_crm_grid(grid, {})


@pytest.mark.parametrize("scale", [-1.0, np.inf, np.nan, True])
def test_deform_crm_grid_rejects_invalid_scale(scale):
    grid = pv.PolyData(np.zeros((1, 3)))

    with pytest.raises(ValueError, match="scale must be a finite"):
        deform_crm_grid(grid, {}, scale=scale)
