from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import streamlit as st
from stpyvista import stpyvista

from amigo.fem.visualization import select_grid_domains
from apps.poisson_solver import inspect_mesh, solve_problem, write_rectangle_mesh
from apps.crm_solver import deform_crm_grid, solve_crm_shell
from apps.problem_config import (
    config_from_json,
    config_to_json,
    crm_shell_config,
    default_config,
    mesh_bytes_from_config,
    potential_flow_config,
    space_mapping_from_json,
    validate_data_values,
    validate_json_object,
)


@st.cache_data(max_entries=8)
def _inspect_mesh_cached(filename: str, content: bytes) -> dict[str, Any]:
    return inspect_mesh(filename, content)


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2)


def _set_domain_visibility(domains: list[str], visible: bool) -> None:
    for domain in domains:
        st.session_state[f"mesh_domain_visible_{domain}"] = visible


def _mark_mesh_upload_changed() -> None:
    st.session_state["mesh_upload_changed"] = True
    st.session_state["last_result"] = None


def _assign_config_widgets(config: dict[str, Any]) -> None:
    st.session_state["problem_name"] = config["name"]
    st.session_state["problem_formulation"] = config["integrand_formulation"]
    st.session_state["function_source"] = config["function_source"]
    st.session_state["integrand_map_text"] = _json_text(config["integrand_map"])
    st.session_state["bc_map_text"] = _json_text(config["bc_map"])
    st.session_state["solution_space_text"] = _json_text(config["solution_space"])
    st.session_state["geometry_space_text"] = _json_text(config["geometry_space"])
    st.session_state["data_space_text"] = _json_text(config["data_space"])
    st.session_state["data_values_text"] = _json_text(config["data_values"])
    st.session_state["load_cases_text"] = _json_text(
        config.get("load_cases", {"1 kN/m^2": {"load.fz": 1000.0}})
    )
    shell_options = config.get("shell_options", crm_shell_config()["shell_options"])
    solver_options = config.get("solver_options", crm_shell_config()["solver_options"])
    for option, key in (
        ("E", "crm_youngs_modulus"),
        ("nu", "crm_poisson_ratio"),
        ("ks", "crm_shear_correction"),
        ("t", "crm_thickness"),
        ("penalty", "crm_drill_penalty"),
    ):
        st.session_state[key] = shell_options[option]
    for option, key in (
        ("solver_type", "crm_solver_type"),
        ("ordering", "crm_ordering"),
        ("backend", "crm_backend"),
        ("ustab", "crm_ustab"),
        ("pivot_tol", "crm_pivot_tol"),
        ("delay_growth", "crm_delay_growth"),
    ):
        st.session_state[key] = solver_options.get(
            option, crm_shell_config()["solver_options"][option]
        )


def _collect_config() -> dict[str, Any]:
    config = dict(st.session_state["active_config"])
    config.update(
        {
            "schema_version": 1,
            "name": st.session_state["problem_name"].strip(),
            "integrand_formulation": st.session_state["problem_formulation"],
            "function_source": st.session_state["function_source"],
            "integrand_map": validate_json_object(
                st.session_state["integrand_map_text"], "Integrand map"
            ),
            "bc_map": validate_json_object(
                st.session_state["bc_map_text"], "Boundary-condition map"
            ),
            "solution_space": validate_json_object(
                st.session_state["solution_space_text"], "Solution space"
            ),
            "geometry_space": validate_json_object(
                st.session_state["geometry_space_text"], "Geometry space"
            ),
            "data_space": validate_json_object(
                st.session_state["data_space_text"], "Data space"
            ),
            "data_values": validate_data_values(
                validate_json_object(
                    st.session_state["data_values_text"], "Data values"
                )
            ),
            "mesh": config.get(
                "mesh", {"filename": None, "encoding": "gzip+base64", "content": None}
            ),
        }
    )
    if not config["name"]:
        raise ValueError("Enter a name for this problem.")
    is_crm = config.get("problem_type") == "crm_shell"
    if is_crm:
        config["shell_options"] = {
            "E": st.session_state["crm_youngs_modulus"],
            "nu": st.session_state["crm_poisson_ratio"],
            "ks": st.session_state["crm_shear_correction"],
            "t": st.session_state["crm_thickness"],
            "penalty": st.session_state["crm_drill_penalty"],
        }
        config["solver_options"] = {
            "backend": st.session_state["crm_backend"],
            "solver_type": st.session_state["crm_solver_type"],
            "ordering": st.session_state["crm_ordering"],
            "ustab": st.session_state["crm_ustab"],
            "pivot_tol": st.session_state["crm_pivot_tol"],
            "delay_growth": st.session_state["crm_delay_growth"],
        }
        config["load_cases"] = validate_json_object(
            st.session_state["load_cases_text"], "CRM load cases"
        )
        if not isinstance(config["load_cases"], dict) or not config["load_cases"]:
            raise ValueError("Define at least one named CRM load case.")
        for case_name, values in config["load_cases"].items():
            if not isinstance(case_name, str) or not case_name.strip():
                raise ValueError("CRM load-case names must be non-empty strings.")
            validate_data_values(values)
        space_mapping_from_json(config["data_space"], "Data space")
    else:
        config["problem_type"] = "fem"
        space_mapping_from_json(config["solution_space"], "Solution space")
        space_mapping_from_json(config["geometry_space"], "Geometry space")
    if config["data_space"]:
        space_mapping_from_json(config["data_space"], "Data space")
    if config["integrand_formulation"] not in {"potential", "weak"}:
        raise ValueError("Choose either potential or weak integrand formulation.")
    return config


def _build_plotter(
    grid: pv.UnstructuredGrid,
    field_name: str,
    dimension: int,
    show_gradient: bool = False,
    gradient_grid: pv.UnstructuredGrid | None = None,
    comparison_grid: pv.UnstructuredGrid | None = None,
    window_size: tuple[int, int] = (1100, 760),
) -> pv.Plotter:
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    try:
        if comparison_grid is not None:
            plotter.add_mesh(
                comparison_grid,
                color="lightgray",
                opacity=0.35,
                show_edges=True,
                edge_color="black",
                label="Undeformed",
            )
        plotter.add_mesh(
            grid,
            scalars=field_name,
            cmap="coolwarm",
            show_edges=True,
            scalar_bar_args={"title": field_name},
        )
        if dimension == 2:
            plotter.view_xy()
            plotter.camera.parallel_projection = True
        else:
            plotter.view_isometric()
        if show_gradient and gradient_grid is not None:
            vector_name = f"{field_name}_gradient"
            stride = max(1, (gradient_grid.n_points + 249) // 250)
            sampled_points = pv.PolyData(gradient_grid.points[::stride])
            sampled_points.point_data[vector_name] = gradient_grid.point_data[
                "gradient"
            ][::stride]
            glyphs = sampled_points.glyph(
                orient=vector_name,
                scale=False,
                factor=0.04 * max(float(grid.length), 1e-12),
            )
            plotter.add_mesh(glyphs, color="black")
        if comparison_grid is not None:
            plotter.add_legend()
        plotter.add_axes()
        return plotter
    except Exception:
        plotter.close()
        raise


st.set_page_config(
    page_title="Amigo | FEM problem builder",
    page_icon=":material/functions:",
    layout="wide",
)
st.title("Amigo FEM problem builder")
st.write(
    "Upload an Amigo-compatible mesh, define a finite-element problem, "
    "and solve it locally."
)

if "active_config" not in st.session_state:
    st.session_state["active_config"] = default_config()
    _assign_config_widgets(st.session_state["active_config"])
if "mesh_bytes" not in st.session_state:
    st.session_state["mesh_bytes"] = None
    st.session_state["mesh_filename"] = None
    st.session_state["last_result"] = None
if "mesh_upload_changed" not in st.session_state:
    st.session_state["mesh_upload_changed"] = False

with st.sidebar:
    st.header("Mesh and presets")
    template = st.selectbox(
        "Problem template",
        options=[
            "Poisson potential",
            "2D source-sink potential flow",
            "CRM MITC shell",
        ],
        key="problem_template",
    )
    if template == "2D source-sink potential flow":
        st.caption("Source and sink coordinates use the mesh's physical units.")
        st.caption("Use a mesh spacing smaller than the Gaussian source width.")
        source_strength = st.number_input(
            "Source strength", value=1.0, step=0.1, key="flow_strength"
        )
        source_x = st.number_input(
            "Source x", value=0.35, step=0.05, key="flow_source_x"
        )
        source_y = st.number_input(
            "Source y", value=0.5, step=0.05, key="flow_source_y"
        )
        sink_x = st.number_input("Sink x", value=0.65, step=0.05, key="flow_sink_x")
        sink_y = st.number_input("Sink y", value=0.5, step=0.05, key="flow_sink_y")
        source_width = st.number_input(
            "Gaussian source width",
            min_value=0.001,
            value=0.04,
            step=0.01,
            key="flow_width",
        )
    if st.button("Load selected template", width="stretch"):
        if template == "Poisson potential":
            selected_config = default_config()
        elif template == "2D source-sink potential flow":
            selected_config = potential_flow_config(
                source_strength,
                (source_x, source_y),
                (sink_x, sink_y),
                source_width,
            )
        else:
            selected_config = crm_shell_config()
        st.session_state["active_config"] = selected_config
        st.session_state["last_result"] = None
        _assign_config_widgets(selected_config)
        st.rerun()

    mesh_upload = st.file_uploader(
        "Upload mesh (.inp, .bdf, .nas, .dat)",
        type=["inp", "bdf", "nas", "dat"],
        key="mesh_upload",
        on_change=_mark_mesh_upload_changed,
    )
    with st.expander("Create a sample rectangular mesh"):
        sample_width = st.number_input(
            "Sample width", min_value=0.1, value=1.0, step=0.1
        )
        sample_height = st.number_input(
            "Sample height", min_value=0.1, value=1.0, step=0.1
        )
        sample_nx = st.number_input(
            "Elements across width", min_value=2, max_value=100, value=12
        )
        sample_ny = st.number_input(
            "Elements across height", min_value=2, max_value=100, value=12
        )
        if st.button("Create sample mesh", width="stretch"):
            with tempfile.TemporaryDirectory(prefix="amigo-sample-mesh-") as directory:
                sample_path = Path(directory) / "rectangle.inp"
                write_rectangle_mesh(
                    sample_path, sample_width, sample_height, sample_nx, sample_ny
                )
                st.session_state["mesh_bytes"] = sample_path.read_bytes()
                st.session_state["mesh_filename"] = "rectangle.inp"
                st.session_state["last_result"] = None
                st.session_state["mesh_upload_changed"] = False
            st.rerun()
    config_upload = st.file_uploader(
        "Load a problem preset (.json)", type=["json"], key="config_upload"
    )
    if config_upload is not None and st.button(
        "Load preset", key="load_config", width="stretch"
    ):
        try:
            loaded_config = config_from_json(config_upload.getvalue())
            embedded_mesh = mesh_bytes_from_config(loaded_config)
            st.session_state["active_config"] = loaded_config
            st.session_state["mesh_bytes"] = embedded_mesh
            st.session_state["mesh_filename"] = loaded_config["mesh"].get("filename")
            st.session_state["last_result"] = None
            st.session_state["mesh_upload_changed"] = False
            _assign_config_widgets(loaded_config)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    embed_mesh = st.checkbox(
        "Include mesh in downloaded preset",
        value=True,
        key="embed_mesh",
        help="Embeds a compressed copy of the selected mesh so the preset is portable.",
    )
    st.caption(
        "Mesh formats supported by Amigo: Abaqus INP and Nastran BDF/NAS/DAT. "
        "The JSON preset can include the mesh."
    )

current_upload_bytes = mesh_upload.getvalue() if mesh_upload is not None else None
if current_upload_bytes is not None and st.session_state["mesh_upload_changed"]:
    mesh_bytes = current_upload_bytes
    mesh_filename = Path(mesh_upload.name).name
else:
    mesh_bytes = st.session_state.get("mesh_bytes")
    mesh_filename = st.session_state.get("mesh_filename")

mesh_info = None
if mesh_bytes and mesh_filename:
    try:
        mesh_info = _inspect_mesh_cached(mesh_filename, mesh_bytes)
        st.session_state["mesh_bytes"] = mesh_bytes
        st.session_state["mesh_filename"] = mesh_filename
    except Exception as exc:
        st.error(f"Could not read the selected mesh: {exc}")
        st.exception(exc)

if mesh_info:
    with st.expander("Mesh details", expanded=False):
        st.write(
            f"**{mesh_filename}** - {mesh_info['dimension']}D, "
            f"{mesh_info['nodes']:,} nodes"
        )
        st.dataframe(
            mesh_info["blocks"], hide_index=True, alt="Mesh domains and element counts"
        )
        st.write("Available domains:", ", ".join(mesh_info["domains"]))

with st.form("problem_definition"):
    st.subheader("Problem definition")
    is_crm = st.session_state["active_config"].get("problem_type") == "crm_shell"
    problem_name = st.text_input("Problem name", key="problem_name")
    st.selectbox(
        "Integrand formulation",
        options=["potential", "weak"],
        key="problem_formulation",
        disabled=is_crm,
        help="Choose the formulation expected by your integrand function.",
    )
    tab_spaces, tab_maps, tab_functions, tab_crm = st.tabs(
        ["Function spaces", "Integrand and BC maps", "Python functions", "CRM shell"]
    )
    with tab_spaces:
        if is_crm:
            st.info(
                "The CRM template uses MITCShell's six H1 displacement/rotation "
                "fields and its 3D geometry/normal spaces. Configure traction data "
                "in the data-space map below."
            )
        left, right = st.columns(2)
        with left:
            st.text_area(
                "Solution space mapping (JSON)",
                key="solution_space_text",
                height=160,
                disabled=is_crm,
                help='Example: {"soln": {"u": "H1"}}. Group fields with comma-separated names such as "u,v".',
            )
            st.text_area(
                "Geometry space mapping (JSON)",
                key="geometry_space_text",
                height=130,
                disabled=is_crm,
                help='Example: {"geo": {"x,y": "H1"}}.',
            )
        with right:
            st.text_area(
                "Data space mapping (JSON)",
                key="data_space_text",
                height=130,
                help='Example: {"data": {"Jz": "const"}}. Use {} if the model has no data fields.',
            )
            st.text_area(
                "Data values (JSON)",
                key="data_values_text",
                height=130,
                disabled=is_crm,
                help='Set model data by scoped name, for example {"data.Jz": 10.0}.',
            )
    with tab_maps:
        st.text_area(
            "Integrand map (JSON)",
            key="integrand_map_text",
            height=220,
            help='Each entry maps a name to mesh target(s) and a function name from the Python code, e.g. {"air": {"target": ["SURFACE1"], "integrand": "potential"}}.',
        )
        st.text_area(
            "Boundary-condition map (JSON)",
            key="bc_map_text",
            height=220,
            help='Dirichlet example: {"bcs": {"type": "dirichlet", "target": ["LINE1"], "input": ["u"]}}. Continuity/scaled callbacks can be defined in the function code.',
        )
    with tab_functions:
        st.warning(
            "Function code is executed by Python when solving. Only load JSON "
            "presets and code you trust."
        )
        st.text_area(
            "Python function definitions",
            key="function_source",
            height=430,
            help="Define each integrand or boundary-condition callback as a Python function.",
        )
    with tab_crm:
        if not is_crm:
            st.caption("Load the CRM MITC shell template to configure these options.")
        st.markdown("**MITCShell material and thickness**")
        shell_cols = st.columns(5)
        with shell_cols[0]:
            st.number_input(
                "Young's modulus E",
                min_value=1e-12,
                format="%.6e",
                key="crm_youngs_modulus",
                disabled=not is_crm,
            )
        with shell_cols[1]:
            st.number_input(
                "Poisson ratio nu",
                min_value=-0.999999,
                max_value=0.499999,
                format="%.6f",
                key="crm_poisson_ratio",
                disabled=not is_crm,
            )
        with shell_cols[2]:
            st.number_input(
                "Shear correction ks",
                min_value=1e-12,
                format="%.6f",
                key="crm_shear_correction",
                disabled=not is_crm,
            )
        with shell_cols[3]:
            st.number_input(
                "Thickness t",
                min_value=1e-12,
                format="%.6e",
                key="crm_thickness",
                disabled=not is_crm,
            )
        with shell_cols[4]:
            st.number_input(
                "Drill penalty",
                min_value=0.0,
                format="%.6e",
                key="crm_drill_penalty",
                disabled=not is_crm,
            )
        st.markdown("**Linear solver**")
        solver_cols = st.columns(3)
        with solver_cols[0]:
            st.selectbox(
                "Backend",
                options=["amigo", "scipy"],
                key="crm_backend",
                disabled=not is_crm,
            )
            st.selectbox(
                "Amigo factorization",
                options=["CHOLESKY", "LDL"],
                key="crm_solver_type",
                disabled=not is_crm,
            )
        with solver_cols[1]:
            st.selectbox(
                "Ordering",
                options=[
                    "NATURAL",
                    "NESTED_DISSECTION",
                    "AMD",
                    "MULTI_COLOR",
                    "DEFAULT",
                ],
                key="crm_ordering",
                disabled=not is_crm,
            )
            st.number_input(
                "Stabilization (ustab)",
                min_value=0.0,
                format="%.6e",
                key="crm_ustab",
                disabled=not is_crm,
            )
        with solver_cols[2]:
            st.number_input(
                "Pivot tolerance",
                min_value=0.0,
                format="%.6e",
                key="crm_pivot_tol",
                disabled=not is_crm,
            )
            st.number_input(
                "Delay growth",
                min_value=0.0,
                format="%.4f",
                key="crm_delay_growth",
                disabled=not is_crm,
            )
        st.markdown("**Named load cases**")
        st.caption(
            "Each case assigns scoped data fields, for example "
            '`{"baseline": {"load.fz": 1000.0}, "2x": {"load.fz": 2000.0}}`. '
            "The traction function is editable in the Python functions tab."
        )
        st.text_area(
            "Load cases (JSON)",
            key="load_cases_text",
            height=160,
            disabled=not is_crm,
        )
    apply_settings = st.form_submit_button("Apply settings", width="stretch")
    solve = st.form_submit_button("Build and solve", type="primary", width="stretch")

if apply_settings or solve:
    st.session_state["last_result"] = None
    try:
        config = _collect_config()
        if mesh_filename:
            config["mesh"]["filename"] = mesh_filename
        st.session_state["active_config"] = config
        st.session_state["mesh_bytes"] = mesh_bytes
        st.session_state["mesh_filename"] = mesh_filename
        if solve:
            if mesh_bytes is None or mesh_filename is None:
                raise ValueError(
                    "Upload a mesh file or load a preset with an embedded mesh."
                )
            if mesh_info is None:
                raise ValueError("The selected mesh could not be read.")
            with st.spinner("Building the Amigo model and solving..."):
                if config.get("problem_type") == "crm_shell":
                    st.session_state["last_result"] = solve_crm_shell(
                        config, mesh_filename, mesh_bytes, mesh_info
                    )
                else:
                    st.session_state["last_result"] = solve_problem(
                        config, mesh_filename, mesh_bytes, mesh_info
                    )
            st.success("Amigo solved the configured problem.")
    except Exception as exc:
        st.error(f"Could not apply or solve this problem: {exc}")
        if solve:
            st.exception(exc)

config = st.session_state["active_config"]
saved_mesh = mesh_bytes if embed_mesh and mesh_bytes is not None else None
download_name = (
    "".join(
        char if char.isalnum() or char in "-_" else "_" for char in config["name"]
    ).strip("_")
    or "amigo-problem"
)
st.caption("Apply settings before downloading to include the latest editor values.")
st.download_button(
    "Download problem preset (.json)",
    data=config_to_json(config, mesh_filename, saved_mesh),
    file_name=f"{download_name}.json",
    mime="application/json",
    width="stretch",
)

result = st.session_state.get("last_result")
if result is not None:
    st.divider()
    st.header("Solution")
    selected_result = result
    if "case_results" in result:
        selected_case = st.selectbox(
            "Load case",
            options=result["load_cases"],
            key="crm_selected_case",
        )
        selected_result = result["case_results"][selected_case]
        st.metric("Maximum |w|", f'{selected_result["max_abs_w"]:.6g}')
    field_name = st.selectbox(
        "Solution field",
        options=result["solution_fields"],
        key="solution_field",
    )
    mesh_view = "Undeformed"
    deformation_scale = 1.0
    if "case_results" in result:
        mesh_view = st.selectbox(
            "Mesh display",
            options=("Undeformed", "Deformed", "Compare"),
            key="crm_mesh_view",
            help="Deformation uses the solved global translations u, v, and w.",
        )
        if mesh_view != "Undeformed":
            deformation_scale = st.number_input(
                "Deformation scale",
                min_value=0.0,
                value=1.0,
                step=1.0,
                format="%.3g",
                key="crm_deformation_scale",
                help=(
                    "1 shows the solved displacement at physical scale. Increase "
                    "this value to amplify small deformations for viewing."
                ),
            )
    visible_domains = [
        domain
        for domain in result.get("mesh_domains", [])
        if st.session_state.get(f"mesh_domain_visible_{domain}", True)
    ]
    show_gradient = False
    if "case_results" not in result:
        show_gradient = st.checkbox(
            "Show gradient vectors",
            value=field_name == "phi",
            key=f"show_gradient_{field_name}",
        )
    left, right = st.columns([3, 1])
    with left:
        plotter = None
        try:
            if "case_results" in result:
                original_grid = selected_result["grids"][field_name]
                comparison_grid = original_grid if mesh_view == "Compare" else None
                display_grid = (
                    deform_crm_grid(
                        original_grid,
                        selected_result["grids"],
                        deformation_scale,
                    )
                    if mesh_view != "Undeformed"
                    else original_grid
                )
                display_grid = select_grid_domains(display_grid, visible_domains)
                if comparison_grid is not None:
                    comparison_grid = select_grid_domains(
                        comparison_grid, visible_domains
                    )
            else:
                display_grid = result["grids"][field_name]
                comparison_grid = None
                display_grid = select_grid_domains(display_grid, visible_domains)
            if not visible_domains:
                st.info("Make at least one mesh domain visible to display the result.")
            else:
                plotter = _build_plotter(
                    display_grid,
                    field_name,
                    mesh_info.get("geometry_dimension", mesh_info["dimension"]),
                    show_gradient=show_gradient,
                    gradient_grid=(
                        None
                        if "case_results" in result
                        else result["gradient_grids"][field_name]
                    ),
                    comparison_grid=comparison_grid,
                )
                stpyvista(
                    plotter,
                    backend="panel",
                    key=(
                        f"interactive-{field_name}-"
                        f"{st.session_state.get('crm_selected_case', 'default')}-"
                        f"{mesh_view.lower()}-{deformation_scale:g}-"
                        f"{'-'.join(visible_domains)}"
                    ),
                )
                st.caption(
                    "Left-drag to rotate, middle-drag or Shift+left-drag to pan, "
                    "and right-drag or scroll to zoom."
                )
                if "case_results" in result and mesh_view != "Undeformed":
                    st.caption(
                        "Deformed coordinates use the solved global translations "
                        "u, v, and w; rotation fields are shown as colors only."
                    )
        except Exception as exc:
            st.error(f"PyVista could not render field '{field_name}': {exc}")
            st.exception(exc)
        finally:
            if plotter is not None:
                plotter.close()
    with right:
        with st.expander("Mesh details", expanded=True):
            st.metric("Mesh nodes", f'{result["node_count"]:,}')
            st.metric("Mesh elements", f'{result["element_count"]:,}')
            if "spc_constraint_counts" in result:
                counts = result["spc_constraint_counts"]
                st.metric("SPC-constrained DOFs", f"{sum(counts.values()):,}")
                st.caption(
                    "Imported zero-valued BDF SPCs: "
                    + ", ".join(f"{field} {count:,}" for field, count in counts.items())
                )
            st.markdown("**Mesh domains**")
            st.caption("Select a domain to open its visibility setting.")
            show_col, hide_col = st.columns(2)
            show_col.button(
                "Show all",
                key="mesh_show_all_domains",
                on_click=_set_domain_visibility,
                args=(result["mesh_domains"], True),
                width="stretch",
            )
            hide_col.button(
                "Hide all",
                key="mesh_hide_all_domains",
                on_click=_set_domain_visibility,
                args=(result["mesh_domains"], False),
                width="stretch",
            )
            for domain in result["mesh_domains"]:
                visibility_key = f"mesh_domain_visible_{domain}"
                is_visible = st.session_state.setdefault(visibility_key, True)
                with st.popover(
                    f"{'Visible' if is_visible else 'Hidden'} · {domain}",
                    width="stretch",
                ):
                    st.markdown(f"**{domain}**")
                    st.checkbox("Visible in viewer", key=visibility_key)
