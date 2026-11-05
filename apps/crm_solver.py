from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from apps.poisson_solver import (
    _BUILD_DIRECTORY,
    _BUILD_LOCK,
    _mesh_path,
)
from apps.problem_config import (
    crm_shell_config,
    space_mapping_from_json,
    validate_data_values,
)


def deform_crm_grid(
    grid: Any,
    displacement_grids: dict[str, Any],
    scale: float = 1.0,
) -> Any:
    """Return a copy of a CRM result grid with solved translations applied."""
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError("The deformation scale must be a finite non-negative number.")
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("The deformation scale must be a finite non-negative number.")

    translations = []
    for field_name in ("u", "v", "w"):
        field_grid = displacement_grids.get(field_name)
        if field_grid is None or field_name not in field_grid.point_data:
            raise ValueError(
                f"CRM displacement field '{field_name}' is missing from the result."
            )
        values = np.asarray(field_grid.point_data[field_name], dtype=float)
        if values.shape != (grid.n_points,) or not np.isfinite(values).all():
            raise ValueError(
                f"CRM displacement field '{field_name}' must contain one finite "
                "value per mesh point."
            )
        translations.append(values)

    displaced = grid.copy(deep=True)
    displaced.points += float(scale) * np.column_stack(translations)
    if not np.isfinite(displaced.points).all():
        raise ValueError("The deformed CRM mesh contains non-finite coordinates.")
    return displaced


def _validate_cases(cases: Any) -> dict[str, dict[str, float]]:
    if not isinstance(cases, dict) or not cases:
        raise ValueError("Define at least one named CRM load case.")
    validated: dict[str, dict[str, float]] = {}
    for name, values in cases.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("CRM load-case names must be non-empty strings.")
        validated[name] = validate_data_values(values)
    return validated


def _summarize_spc_constraints(bcs: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, bc in bcs.items():
        values = np.asarray(bc.values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"SPC values for shell field '{name}' must be finite.")
        if np.any(values != 0.0):
            raise ValueError(
                "CRM shell analysis currently supports zero-valued Nastran SPCs "
                f"only; field '{name}' has nonzero enforced values."
            )
        counts[name] = int(np.unique(bc.dofs).size)
    return counts


def solve_crm_shell(
    config: dict[str, Any],
    mesh_name: str,
    mesh_bytes: bytes,
    mesh_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import amigo as am
    from amigo.fem import CellType, Mesh, Problem, SolutionSpace, build_grid
    from amigo.fem.elements import MITCShell

    shell_options = config.get("shell_options")
    if not isinstance(shell_options, dict):
        raise ValueError("CRM shell options must be a JSON object.")
    required_shell_options = {"E", "nu", "ks", "t", "penalty"}
    if set(shell_options) != required_shell_options:
        raise ValueError("CRM shell options must define E, nu, ks, t, and penalty.")
    shell_options = crm_shell_config(
        shell_options["E"],
        shell_options["nu"],
        shell_options["ks"],
        shell_options["t"],
        shell_options["penalty"],
    )["shell_options"]
    try:
        shell = MITCShell(**shell_options)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid CRM shell properties: {exc}") from exc

    solver_options = config.get("solver_options")
    if not isinstance(solver_options, dict):
        raise ValueError("CRM solver options must be a JSON object.")
    backend = solver_options.get("backend", "amigo")
    solver_type = solver_options.get("solver_type", "CHOLESKY")
    ordering = solver_options.get("ordering", "NATURAL")
    if backend not in {"amigo", "scipy"}:
        raise ValueError("CRM solver backend must be 'amigo' or 'scipy'.")
    if backend == "amigo" and solver_type not in {"LDL", "CHOLESKY"}:
        raise ValueError("Amigo supports only LDL and CHOLESKY solver types.")
    if ordering not in {
        "NESTED_DISSECTION",
        "AMD",
        "MULTI_COLOR",
        "NATURAL",
        "DEFAULT",
    }:
        raise ValueError(f"Unsupported Amigo matrix ordering '{ordering}'.")
    for name, default in (
        ("ustab", 0.05),
        ("pivot_tol", 1e-14),
        ("delay_growth", 2.0),
    ):
        value = solver_options.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Solver option '{name}' must be a finite number.")
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Solver option '{name}' must be finite and non-negative.")

    cases = _validate_cases(config.get("load_cases"))
    data_definition = config.get("data_space")
    data_mapping = space_mapping_from_json(data_definition, "Data space")
    if not data_mapping:
        raise ValueError("CRM load cases require at least one data-space field.")

    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "np": np,
        "am": am,
    }
    source = config.get("function_source")
    if not isinstance(source, str):
        raise ValueError("CRM traction function must be Python source text.")
    try:
        exec(compile(source, "<user-defined CRM functions>", "exec"), namespace)
    except (SyntaxError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not compile the CRM function code: {exc}") from exc

    raw_integrand_map = config.get("integrand_map")
    if not isinstance(raw_integrand_map, dict) or not {
        "shell",
        "traction",
    }.issubset(raw_integrand_map):
        raise ValueError(
            "The CRM integrand map must include 'shell' and 'traction' entries."
        )
    integrand_map: dict[str, dict[str, Any]] = {}
    for name in ("shell", "traction"):
        definition = raw_integrand_map[name]
        if not isinstance(definition, dict):
            raise ValueError(f"CRM integrand '{name}' must be a JSON object.")
        target = definition.get("target")
        if isinstance(target, str):
            target = [target]
        if (
            not isinstance(target, list)
            or not target
            or any(not isinstance(item, str) or not item for item in target)
        ):
            raise ValueError(f"CRM integrand '{name}' needs target mesh domains.")
        integrand_map[name] = {**definition, "target": target}
    traction_name = raw_integrand_map["traction"].get("integrand")
    traction = namespace.get(traction_name) if isinstance(traction_name, str) else None
    if not callable(traction):
        raise ValueError(
            "The CRM traction map must reference a function defined in the Python editor."
        )

    with tempfile.TemporaryDirectory(prefix="amigo-crm-mesh-") as workdir:
        mesh = Mesh(str(_mesh_path(mesh_name, mesh_bytes, Path(workdir))))
        domains = set(mesh.get_domain_names())
        if mesh_info is not None:
            domains = set(mesh_info["domains"])
        for name, definition in integrand_map.items():
            target = definition["target"]
            if target == ["*"]:
                definition["target"] = sorted(domains)
            missing = set(definition["target"]) - domains
            if missing:
                raise ValueError(
                    f"CRM integrand '{name}' targets unknown mesh domain(s): "
                    + ", ".join(sorted(missing))
                )

        non_quad_blocks = [
            mesh.get_cell_type(block_id).name
            for block_id in range(mesh.get_num_blocks())
            if mesh.get_cell_type(block_id) is not CellType.QUADRILATERAL
        ]
        if non_quad_blocks:
            raise ValueError(
                "MITCShell requires quadrilateral mesh blocks; found "
                + ", ".join(sorted(set(non_quad_blocks)))
                + "."
            )

        soln_space, geo_space, _, output_space = shell.get_default_spaces()
        data_space = SolutionSpace(data_mapping)
        element_objs = {}
        ctype = CellType.QUADRILATERAL
        element_objs[("shell", ctype)] = shell.create_element(
            name="Shell",
            ctype=ctype,
            soln_space=soln_space,
            data_space=data_space,
        )
        element_objs[("traction", ctype)] = shell.create_element(
            name="Traction",
            ctype=ctype,
            soln_space=soln_space,
            data_space=data_space,
            integrand=traction,
        )
        problem = Problem(
            mesh,
            soln_space,
            geo_space,
            data_space,
            integrand_map=integrand_map,
            integrand_formulation="potential",
            bc_map={},
            element_objs=element_objs,
        )
        constraint_counts = _summarize_spc_constraints(
            problem.soln_dof.get_dof_handler().get_dirichlet_bcs()
        )
        module_digest = hashlib.sha256()
        module_digest.update(
            json.dumps(
                {
                    "mesh": hashlib.sha256(mesh_bytes).hexdigest(),
                    "source": source,
                    "shell_options": shell_options,
                    "data_space": data_definition,
                    "integrand_map": integrand_map,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        module_name = f"amigo_crm_model_{module_digest.hexdigest()[:16]}"
        model = problem.create_model(module_name)
        MITCShell.add_shell_normals(
            problem, model, integrand_map["shell"]["target"], output_space
        )

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

        MITCShell.set_shell_normals(model)
        data_vector = model.get_data_vector()
        valid_data_names = set(model.get_names()[2])
        normal_fields = [
            np.asarray(data_vector[f"normals.{axis}"]) for axis in ("nx", "ny", "nz")
        ]
        if any(not np.isfinite(field).all() for field in normal_fields):
            raise RuntimeError(
                "CRM shell-normal initialization produced non-finite values."
            )
        normal_magnitudes = np.sqrt(sum(field**2 for field in normal_fields))
        if np.any(normal_magnitudes <= 0):
            raise RuntimeError(
                "CRM shell-normal initialization produced zero-length normals."
            )
        common_data = validate_data_values(config.get("data_values", {}))
        case_data_names = set(common_data)
        for case_data in cases.values():
            case_data_names.update(case_data)
        case_results: dict[str, dict[str, Any]] = {}

        for case_name, case_data in cases.items():
            data_values = {**common_data, **case_data}
            unknown_data = set(data_values) - valid_data_names
            if unknown_data:
                raise ValueError(
                    f"CRM load case '{case_name}' references unknown data field(s): "
                    + ", ".join(sorted(unknown_data))
                )
            for field_name in case_data_names:
                data_vector[field_name] = 0.0
            for field_name, value in data_values.items():
                data_vector[field_name] = value

            solution = model.create_vector()
            gradient = model.create_vector()
            matrix = model.create_matrix()
            model.eval_gradient(solution, gradient)
            model.eval_hessian(solution, matrix)
            if not np.isfinite(np.asarray(gradient[:])).all():
                raise RuntimeError(
                    f"CRM gradient contains non-finite values in load case "
                    f"'{case_name}'. Check the mesh geometry and traction function."
                )
            if backend == "amigo":
                factor = am.SparseLDL(
                    matrix,
                    getattr(am.SolverType, solver_type),
                    ustab=float(solver_options.get("ustab", 0.05)),
                    pivot_tol=float(solver_options.get("pivot_tol", 1e-14)),
                    delay_growth=float(solver_options.get("delay_growth", 2.0)),
                    order=getattr(am.OrderingType, ordering),
                )
                factor_status = factor.factor()
                if factor_status != 0:
                    raise RuntimeError(
                        f"Amigo factorization failed for load case '{case_name}' "
                        f"with status {factor_status}."
                    )
                solution[:] = -gradient[:]
                factor.solve(solution.get_vector())
            else:
                from scipy.sparse.linalg import spsolve

                solution[:] = spsolve(am.tocsr(matrix), -gradient[:])

            if not np.isfinite(np.asarray(solution[:])).all():
                raise RuntimeError(
                    f"CRM solve produced non-finite displacements in load case "
                    f"'{case_name}'. Check shell constraints, mesh quality, and "
                    "solver settings."
                )
            grids = {
                name: build_grid(problem, name, solution)
                for name, _ in soln_space.fields
            }
            w_values = np.asarray(solution["soln.w"])
            case_results[case_name] = {
                "grids": grids,
                "max_abs_w": float(np.max(np.abs(w_values), initial=0.0)),
            }

        return {
            "case_results": case_results,
            "solution_fields": ["u", "v", "w", "rx", "ry", "rz"],
            "load_cases": list(cases),
            "spc_constraint_counts": constraint_counts,
            "node_count": int(mesh.X.shape[0]),
            "element_count": int(
                sum(mesh.get_num_elements(i) for i in range(mesh.get_num_blocks()))
            ),
            "mesh_domains": sorted(domains),
        }
