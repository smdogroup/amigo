from __future__ import annotations

import base64
import copy
import gzip
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any
import zlib

CONFIG_VERSION = 1
MAX_EMBEDDED_MESH_BYTES = 200 * 1024 * 1024

DEFAULT_FUNCTION_SOURCE = """import numpy as np
import amigo as am
from amigo.fem import dot_product


def potential(soln, data=None, geo=None):
    u = soln["u"].value
    ugrad = soln["u"].grad
    f = data["Jz"].value
    return 0.5 * dot_product(ugrad, ugrad, n=2) - f * u
"""


def default_config() -> dict[str, Any]:
    return {
        "schema_version": CONFIG_VERSION,
        "name": "Poisson potential",
        "integrand_formulation": "potential",
        "function_source": DEFAULT_FUNCTION_SOURCE,
        "integrand_map": {"air": {"target": ["SURFACE1"], "integrand": "potential"}},
        "bc_map": {
            "fixed_edges": {
                "type": "dirichlet",
                "target": ["LEFT", "RIGHT", "BOTTOM", "TOP"],
                "input": ["u"],
            }
        },
        "solution_space": {"soln": {"u": "H1"}},
        "geometry_space": {"geo": {"x,y": "H1"}},
        "data_space": {"data": {"Jz": "const"}},
        "data_values": {"data.Jz": 10.0},
        "mesh": {"filename": None, "encoding": "gzip+base64", "content": None},
    }


def potential_flow_config(
    source_strength: float = 1.0,
    source_center: tuple[float, float] = (0.35, 0.5),
    sink_center: tuple[float, float] = (0.65, 0.5),
    source_width: float = 0.04,
) -> dict[str, Any]:
    values = (source_strength, *source_center, *sink_center, source_width)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Potential-flow settings must be finite numbers.")
    if source_width <= 0:
        raise ValueError("The Gaussian source width must be greater than zero.")

    config = default_config()
    config.update(
        {
            "name": "2D source-sink potential flow",
            "function_source": f"""import math
import amigo as am
from amigo.fem import dot_product


SOURCE_STRENGTH = {source_strength!r}
SOURCE_CENTER = ({source_center[0]!r}, {source_center[1]!r})
SINK_CENTER = ({sink_center[0]!r}, {sink_center[1]!r})
SOURCE_WIDTH = {source_width!r}


def gaussian_source(x, y, center, strength):
    dx = x - center[0]
    dy = y - center[1]
    radius_squared = dx**2 + dy**2
    return (
        strength
        / (2.0 * math.pi * SOURCE_WIDTH**2)
        * am.exp(-radius_squared / (2.0 * SOURCE_WIDTH**2))
    )


def potential(soln, data=None, geo=None):
    phi = soln["phi"].value
    grad_phi = soln["phi"].grad
    x = geo["x"].value
    y = geo["y"].value

    source = gaussian_source(x, y, SOURCE_CENTER, SOURCE_STRENGTH)
    sink = gaussian_source(x, y, SINK_CENTER, -SOURCE_STRENGTH)
    return 0.5 * dot_product(grad_phi, grad_phi, n=2) + (source + sink) * phi
""",
            "integrand_map": {
                "fluid": {"target": ["SURFACE1"], "integrand": "potential"}
            },
            "bc_map": {
                "boundary": {
                    "type": "dirichlet",
                    "target": ["LEFT", "RIGHT", "BOTTOM", "TOP"],
                    "input": ["phi"],
                }
            },
            "solution_space": {"soln": {"phi": "H1"}},
            "data_space": {},
            "data_values": {},
            "mesh": {"filename": None, "encoding": "gzip+base64", "content": None},
        }
    )
    return config


def config_to_json(
    config: dict[str, Any],
    mesh_filename: str | None,
    mesh_bytes: bytes | None,
) -> str:
    result = copy.deepcopy(config)
    result["schema_version"] = CONFIG_VERSION
    result["mesh"] = {
        "filename": mesh_filename,
        "encoding": "gzip+base64",
        "content": (
            base64.b64encode(gzip.compress(mesh_bytes)).decode("ascii")
            if mesh_bytes is not None
            else None
        ),
    }
    return json.dumps(result, indent=2, allow_nan=False) + "\n"


def config_from_json(contents: str | bytes) -> dict[str, Any]:
    try:
        config = json.loads(contents)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid JSON configuration: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a JSON object.")
    if config.get("schema_version") != CONFIG_VERSION:
        raise ValueError(
            "Unsupported or missing schema_version. "
            f"This app supports version {CONFIG_VERSION}."
        )

    required = set(default_config())
    missing = required - config.keys()
    if missing:
        raise ValueError(
            f"Configuration is missing fields: {', '.join(sorted(missing))}."
        )
    if not isinstance(config["mesh"], dict):
        raise ValueError("The mesh configuration must be a JSON object.")
    filename = config["mesh"].get("filename")
    if filename is not None and (
        not isinstance(filename, str)
        or Path(filename).suffix.lower() not in {".inp", ".bdf", ".nas", ".dat"}
    ):
        raise ValueError("The preset mesh filename must use an Amigo mesh extension.")
    return config


def mesh_bytes_from_config(config: dict[str, Any]) -> bytes | None:
    mesh = config.get("mesh", {})
    content = mesh.get("content")
    if content is None:
        return None
    if mesh.get("encoding") != "gzip+base64" or not isinstance(content, str):
        raise ValueError("Unsupported embedded mesh encoding in configuration.")
    try:
        compressed = base64.b64decode(content, validate=True)
        with gzip.GzipFile(fileobj=BytesIO(compressed)) as mesh_file:
            mesh_bytes = mesh_file.read(MAX_EMBEDDED_MESH_BYTES + 1)
        if len(mesh_bytes) > MAX_EMBEDDED_MESH_BYTES:
            raise ValueError(
                "The embedded mesh exceeds the 200 MB supported upload limit."
            )
        return mesh_bytes
    except (ValueError, OSError, EOFError, zlib.error) as exc:
        raise ValueError(f"Could not decode the embedded mesh: {exc}") from exc


def space_mapping_from_json(
    value: Any, label: str
) -> dict[str, dict[tuple[str, ...], str]]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty object of component spaces.")
    mapping: dict[str, dict[tuple[str, ...], str]] = {}
    for component, fields in value.items():
        if not isinstance(component, str) or not component.isidentifier():
            raise ValueError(
                f"{label} component names must be valid Python identifiers."
            )
        if not isinstance(fields, dict) or len(fields) != 1:
            raise ValueError(
                f"{label} component '{component}' must contain exactly one "
                "field-group-to-space entry."
            )
        field_text, function_space = next(iter(fields.items()))
        if not isinstance(field_text, str):
            raise ValueError(f"{label} field names must be strings.")
        names = tuple(name.strip() for name in field_text.split(","))
        if not names or any(not name or not name.isidentifier() for name in names):
            raise ValueError(
                f"{label} fields must be comma-separated Python identifiers."
            )
        if not isinstance(function_space, str):
            raise ValueError(f"{label} function-space names must be strings.")
        mapping[component] = {names: function_space}
    return mapping


def validate_json_object(raw: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def validate_data_values(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Data values must be a JSON object keyed by component.field.")
    for name, field_value in value.items():
        if not isinstance(name, str) or "." not in name:
            raise ValueError(
                "Each data value key must use the component.field form, "
                "for example data.Jz."
            )
        _validate_finite_json_value(field_value, name)
    return value


def _validate_finite_json_value(value: Any, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        raise ValueError(f"Data value '{label}' must contain only finite numbers.")
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"Data value '{label}' must contain only finite numbers.")
        return
    if isinstance(value, list):
        for item in value:
            _validate_finite_json_value(item, label)
        return
    raise ValueError(f"Data value '{label}' must contain only finite numbers.")
