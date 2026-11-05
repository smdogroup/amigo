from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

from apps.problem_config import space_mapping_from_json, validate_data_values

_BUILD_LOCK = threading.Lock()
_CACHE_ROOT = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
_BUILD_DIRECTORY = _CACHE_ROOT / "amigo-problem-builder"
_SUPPORTED_MESH_EXTENSIONS = {".inp", ".bdf", ".nas", ".dat"}


def write_rectangle_mesh(
    path: str | Path, width: float, height: float, nx: int, ny: int
) -> None:
    for label, value in (("width", width), ("height", height)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label.capitalize()} must be a finite positive number.")
    if not isinstance(nx, int) or not isinstance(ny, int) or nx < 2 or ny < 2:
        raise ValueError("The mesh must have at least two elements in each direction.")

    node_id = lambda i, j: j * (nx + 1) + i + 1
    lines = ["*Node"]
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = width * i / nx
            y = height * j / ny
            lines.append(f"{node_id(i, j)}, {x:.16g}, {y:.16g}, 0")

    lines.append("*Element, type=CPE4, ELSET=SURFACE1")
    element_id = 1
    for j in range(ny):
        for i in range(nx):
            n0 = node_id(i, j)
            n1 = node_id(i + 1, j)
            n2 = node_id(i + 1, j + 1)
            n3 = node_id(i, j + 1)
            lines.append(f"{element_id}, {n0}, {n1}, {n2}, {n3}")
            element_id += 1

    boundaries = {
        "LEFT": [(node_id(0, j), node_id(0, j + 1)) for j in range(ny)],
        "RIGHT": [(node_id(nx, j), node_id(nx, j + 1)) for j in range(ny)],
        "BOTTOM": [(node_id(i, 0), node_id(i + 1, 0)) for i in range(nx)],
        "TOP": [(node_id(i, ny), node_id(i + 1, ny)) for i in range(nx)],
    }
    for name, edges in boundaries.items():
        lines.append(f"*Element, type=T3D2, ELSET={name}")
        for n0, n1 in edges:
            lines.append(f"{element_id}, {n0}, {n1}")
            element_id += 1

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def inspect_mesh(mesh_name: str, mesh_bytes: bytes) -> dict[str, Any]:
    from amigo.fem import Mesh
    from amigo.fem.cell_types import REFERENCE_CELLS

    extension = Path(mesh_name).suffix.lower()
    if extension not in _SUPPORTED_MESH_EXTENSIONS:
        raise ValueError(
            "Amigo meshes must use one of these extensions: "
            + ", ".join(sorted(_SUPPORTED_MESH_EXTENSIONS))
        )
    if not mesh_bytes:
        raise ValueError("The selected mesh file is empty.")

    with tempfile.TemporaryDirectory(prefix="amigo-user-mesh-") as directory:
        mesh_path = Path(directory) / f"mesh{extension}"
        mesh_path.write_bytes(mesh_bytes)
        mesh = Mesh(str(mesh_path))
    if mesh.get_num_blocks() == 0:
        raise ValueError("The selected mesh contains no supported elements.")

    return {
        "domains": mesh.get_domain_names(),
        "dimension": max(
            REFERENCE_CELLS[mesh.get_cell_type(block_id)].dimension
            for block_id in range(mesh.get_num_blocks())
        ),
        "geometry_dimension": int(
            np.count_nonzero(np.ptp(mesh.X, axis=0) > 1e-12)
        ),
        "nodes": int(mesh.X.shape[0]),
        "blocks": [
            {
                "domain": mesh.get_domain_name(block_id),
                "cell_type": mesh.get_cell_type(block_id).name,
                "elements": mesh.get_num_elements(block_id),
            }
            for block_id in range(mesh.get_num_blocks())
        ],
    }


def _mesh_path(mesh_name: str, mesh_bytes: bytes, directory: Path) -> Path:
    extension = Path(mesh_name).suffix.lower()
    if extension not in _SUPPORTED_MESH_EXTENSIONS:
        raise ValueError(
            "Amigo meshes must use one of these extensions: "
            + ", ".join(sorted(_SUPPORTED_MESH_EXTENSIONS))
        )
    path = directory / f"mesh{extension}"
    path.write_bytes(mesh_bytes)
    return path


def _build_integrand_map(
    raw_map: dict[str, Any], namespace: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_map, dict) or not raw_map:
        raise ValueError("The integrand map must be a non-empty JSON object.")
    result: dict[str, dict[str, Any]] = {}
    for name, definition in raw_map.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("Integrand names must be valid Python identifiers.")
        if not isinstance(definition, dict):
            raise ValueError(f"Integrand '{name}' must be a JSON object.")
        function_name = definition.get("integrand")
        if not isinstance(function_name, str) or not callable(
            namespace.get(function_name)
        ):
            raise ValueError(
                f"Integrand '{name}' references undefined function "
                f"'{function_name}'. Define it in the function code."
            )
        target = definition.get("target")
        if isinstance(target, str):
            target = [target]
        if (
            not isinstance(target, list)
            or not target
            or any(not isinstance(item, str) or not item for item in target)
        ):
            raise ValueError(
                f"Integrand '{name}' needs a target domain name or list of names."
            )
        result[name] = {
            **definition,
            "target": target,
            "integrand": namespace[function_name],
        }
    return result


def _build_bc_map(
    raw_map: dict[str, Any], namespace: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_map, dict):
        raise ValueError("The boundary-condition map must be a JSON object.")
    result: dict[str, dict[str, Any]] = {}
    for name, definition in raw_map.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(
                "Boundary-condition names must be valid Python identifiers."
            )
        if not isinstance(definition, dict):
            raise ValueError(f"Boundary condition '{name}' must be a JSON object.")
        bc = dict(definition)
        bc_type = bc.get("type")
        if not isinstance(bc_type, str) or bc_type not in {
            "dirichlet",
            "continuity",
            "scaled",
        }:
            raise ValueError(
                f"Boundary condition '{name}' has unsupported type '{bc_type}'."
            )
        for key in ("target", "input"):
            value = bc.get(key)
            if isinstance(value, str):
                value = [value]
            if (
                not isinstance(value, list)
                or not value
                or any(not isinstance(item, str) or not item for item in value)
            ):
                raise ValueError(
                    f"Boundary condition '{name}' needs a non-empty '{key}' list."
                )
            bc[key] = value
        if bc_type != "dirichlet":
            callback_name = bc.get("callback")
            callback = (
                namespace.get(callback_name) if isinstance(callback_name, str) else None
            )
            if not callable(callback):
                raise ValueError(
                    f"Boundary condition '{name}' references undefined callback "
                    f"'{callback_name}'. Define it in the function code."
                )
            bc["callback"] = callback
        result[name] = bc
    return result


def _model_cache_key(config: dict[str, Any], mesh_bytes: bytes) -> str:
    serializable = {
        key: config[key]
        for key in (
            "integrand_formulation",
            "function_source",
            "solution_space",
            "geometry_space",
            "data_space",
        )
    }
    serializable["integrand_map"] = {
        name: {key: value for key, value in definition.items() if key != "integrand"}
        for name, definition in config["integrand_map"].items()
    }
    serializable["bc_map"] = {
        name: {key: value for key, value in definition.items() if key != "callback"}
        for name, definition in config["bc_map"].items()
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(serializable, sort_keys=True, separators=(",", ":")).encode()
    )
    digest.update(mesh_bytes)
    return digest.hexdigest()[:16]


def solve_problem(
    config: dict[str, Any],
    mesh_name: str,
    mesh_bytes: bytes,
    mesh_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import amigo as am
    from amigo.fem import Mesh, Problem, SolutionSpace, build_grid, dot_product

    if not isinstance(config.get("function_source"), str):
        raise ValueError("Function code must be provided as Python source text.")
    formulation = config.get("integrand_formulation")
    if not isinstance(formulation, str) or formulation not in {"potential", "weak"}:
        raise ValueError("Integrand formulation must be 'potential' or 'weak'.")
    if formulation != "potential" and config.get("bc_map", {}) == {}:
        raise ValueError("A weak-formulation problem requires boundary conditions.")

    solution_mapping = space_mapping_from_json(
        config.get("solution_space"), "Solution space"
    )
    geometry_mapping = space_mapping_from_json(
        config.get("geometry_space"), "Geometry space"
    )
    data_definition = config.get("data_space")
    data_mapping = (
        space_mapping_from_json(data_definition, "Data space")
        if data_definition
        else None
    )
    data_values = validate_data_values(config.get("data_values", {}))

    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "np": np,
        "am": am,
        "dot_product": dot_product,
    }
    try:
        exec(
            compile(
                config["function_source"], "<user-defined Amigo functions>", "exec"
            ),
            namespace,
        )
    except (SyntaxError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not compile the function code: {exc}") from exc
    integrand_map = _build_integrand_map(config.get("integrand_map", {}), namespace)
    bc_map = _build_bc_map(config.get("bc_map", {}), namespace)

    with tempfile.TemporaryDirectory(prefix="amigo-problem-mesh-") as workdir:
        mesh_path = _mesh_path(mesh_name, mesh_bytes, Path(workdir))
        mesh = Mesh(str(mesh_path))
        available_domains = set(mesh.get_domain_names())
        if mesh_info is not None:
            available_domains = set(mesh_info["domains"])
        for category, definitions in (
            ("integrand", integrand_map),
            ("boundary condition", bc_map),
        ):
            for name, definition in definitions.items():
                missing_domains = set(definition["target"]) - available_domains
                if missing_domains:
                    raise ValueError(
                        f"{category.capitalize()} '{name}' targets unknown mesh "
                        f"domain(s): {', '.join(sorted(missing_domains))}."
                    )

        soln_space = SolutionSpace(solution_mapping)
        geo_space = SolutionSpace(geometry_mapping)
        data_space = SolutionSpace(data_mapping) if data_mapping else None
        problem = Problem(
            mesh,
            soln_space,
            geo_space,
            data_space,
            integrand_map=integrand_map,
            integrand_formulation=formulation,
            bc_map=bc_map,
        )
        module_name = f"amigo_user_model_{_model_cache_key(config, mesh_bytes)}"
        model = problem.create_model(module_name)

        model_directory = _BUILD_DIRECTORY / module_name
        with _BUILD_LOCK:
            model_directory.mkdir(parents=True, exist_ok=True)
            original_cwd = Path.cwd()
            directory_on_path = str(model_directory)
            added_to_path = directory_on_path not in sys.path
            try:
                os.chdir(model_directory)
                if added_to_path:
                    sys.path.insert(0, directory_on_path)
                model.build_module(
                    source_dir=model_directory,
                    build_dir=model_directory / "_amigo_build",
                )
                model.initialize()
            finally:
                if added_to_path:
                    sys.path.remove(directory_on_path)
                os.chdir(original_cwd)

        data = model.get_data_vector()
        valid_data_names = set(model.get_names()[2])
        for name, value in data_values.items():
            if name not in valid_data_names:
                raise ValueError(
                    f"Data value '{name}' is not a field in the configured data "
                    "space."
                )
            data[name] = value

        matrix = model.create_matrix()
        solution = model.create_vector()
        gradient = model.create_vector()
        model.eval_hessian(solution, matrix)
        model.eval_gradient(solution, gradient)
        if formulation == "potential":
            factor = am.SparseLDL(matrix, am.SolverType.CHOLESKY)
            factor.factor()
            solution[:] = -gradient[:]
            factor.solve(solution.get_vector())
        else:
            from scipy.sparse.linalg import spsolve

            solution[:] = spsolve(am.tocsr(matrix), gradient[:])

        scalar_fields: dict[str, Any] = {}
        gradient_fields: dict[str, Any] = {}
        for field_name, _ in soln_space.fields:
            component_name = soln_space.get_component_name(field_name)
            field_values = np.asarray(solution[f"{component_name}.{field_name}"])
            if field_values.ndim == 1:
                grid = build_grid(problem, field_name, solution)
                scalar_fields[field_name] = grid
                gradient_fields[field_name] = grid.compute_derivative(
                    scalars=field_name,
                    gradient="gradient",
                    vorticity=False,
                )

        if not scalar_fields:
            raise ValueError(
                "The solve completed, but no scalar solution fields are available "
                "for PyVista visualization."
            )

        return {
            "grids": scalar_fields,
            "gradient_grids": gradient_fields,
            "node_count": int(mesh.X.shape[0]),
            "element_count": (
                sum(block["elements"] for block in mesh_info["blocks"])
                if mesh_info
                else int(
                    sum(mesh.get_num_elements(i) for i in range(mesh.get_num_blocks()))
                )
            ),
            "solution_fields": list(scalar_fields),
            "mesh_domains": sorted(available_domains),
        }
