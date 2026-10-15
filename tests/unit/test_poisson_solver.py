import pytest

from apps.poisson_solver import (
    _build_bc_map,
    _build_integrand_map,
    inspect_mesh,
    write_rectangle_mesh,
)
from apps.problem_config import (
    config_from_json,
    config_to_json,
    default_config,
    mesh_bytes_from_config,
    potential_flow_config,
    space_mapping_from_json,
    validate_data_values,
)


def _rectangle_mesh_bytes(tmp_path):
    mesh_path = tmp_path / "rectangle.inp"
    write_rectangle_mesh(mesh_path, width=2.0, height=1.0, nx=3, ny=2)
    return mesh_path.read_bytes()


def test_write_rectangle_mesh_creates_expected_domains_and_counts(tmp_path):
    mesh_info = inspect_mesh("rectangle.inp", _rectangle_mesh_bytes(tmp_path))
    assert mesh_info["nodes"] == 12
    assert mesh_info["domains"] == [
        "SURFACE1",
        "LEFT",
        "RIGHT",
        "BOTTOM",
        "TOP",
    ]
    assert [block["elements"] for block in mesh_info["blocks"]] == [6, 2, 2, 3, 3]
    assert mesh_info["dimension"] == 2


def test_problem_preset_round_trips_settings_and_embedded_mesh(tmp_path):
    mesh_bytes = _rectangle_mesh_bytes(tmp_path)
    config = default_config()
    config["mesh"]["filename"] = "rectangle.inp"
    config["data_values"]["data.Jz"] = 3.5

    serialized = config_to_json(config, "rectangle.inp", mesh_bytes)
    restored = config_from_json(serialized)

    assert restored["data_values"]["data.Jz"] == 3.5
    assert restored["mesh"]["filename"] == "rectangle.inp"
    assert mesh_bytes_from_config(restored) == mesh_bytes


def test_space_mapping_supports_multiple_fields_in_one_component():
    mapping = space_mapping_from_json(
        {"soln": {"u,v": "H1"}, "pressure": {"p": "L2"}}, "Solution space"
    )

    assert mapping == {
        "soln": {("u", "v"): "H1"},
        "pressure": {("p",): "L2"},
    }


@pytest.mark.parametrize(
    "mapping",
    [
        {},
        {"bad component": {"u": "H1"}},
        {"soln": {"u": "H1", "v": "H1"}},
        {"soln": {"u,": "H1"}},
    ],
)
def test_invalid_space_mapping_is_rejected(mapping):
    with pytest.raises(ValueError):
        space_mapping_from_json(mapping, "Solution space")


@pytest.mark.parametrize(
    "data_values",
    [
        {"Jz": 1.0},
        {"data.Jz": float("inf")},
        {"data.Jz": [1.0, "bad"]},
        {"data.Jz": True},
    ],
)
def test_invalid_data_values_are_rejected(data_values):
    with pytest.raises(ValueError):
        validate_data_values(data_values)


def test_preset_requires_supported_schema_version():
    with pytest.raises(ValueError, match="schema_version"):
        config_from_json('{"schema_version": 99}')


def test_problem_maps_resolve_named_functions_and_normalize_targets():
    integrand = lambda *args: 0.0
    callback = lambda *args: ([], [])

    integrands = _build_integrand_map(
        {"domain": {"target": "SURFACE1", "integrand": "energy"}},
        {"energy": integrand},
    )
    boundary_conditions = _build_bc_map(
        {
            "periodic": {
                "type": "continuity",
                "target": "LEFT",
                "input": "u",
                "callback": "pair_edges",
            }
        },
        {"pair_edges": callback},
    )

    assert integrands["domain"]["target"] == ["SURFACE1"]
    assert integrands["domain"]["integrand"] is integrand
    assert boundary_conditions["periodic"]["target"] == ["LEFT"]
    assert boundary_conditions["periodic"]["input"] == ["u"]
    assert boundary_conditions["periodic"]["callback"] is callback


def test_problem_map_rejects_undefined_integrand_function():
    with pytest.raises(ValueError, match="undefined function"):
        _build_integrand_map(
            {"domain": {"target": "SURFACE1", "integrand": "missing"}},
            {},
        )


def test_potential_flow_template_matches_example_structure():
    config = potential_flow_config(
        source_strength=2.5,
        source_center=(0.2, 0.4),
        sink_center=(0.8, 0.6),
        source_width=0.08,
    )

    assert config["integrand_formulation"] == "potential"
    assert config["solution_space"] == {"soln": {"phi": "H1"}}
    assert config["data_space"] == {}
    assert config["integrand_map"] == {
        "fluid": {"target": ["SURFACE1"], "integrand": "potential"}
    }
    assert config["bc_map"]["boundary"]["input"] == ["phi"]
    assert "SOURCE_STRENGTH = 2.5" in config["function_source"]
    assert "SOURCE_CENTER = (0.2, 0.4)" in config["function_source"]
    assert "SINK_CENTER = (0.8, 0.6)" in config["function_source"]
    assert "SOURCE_WIDTH = 0.08" in config["function_source"]


def test_potential_flow_template_rejects_invalid_gaussian_width():
    with pytest.raises(ValueError, match="width"):
        potential_flow_config(source_width=0.0)
