from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import streamlit as st

from apps.poisson_solver import inspect_mesh, solve_problem, write_rectangle_mesh
from apps.problem_config import (
    config_from_json,
    config_to_json,
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


def _collect_config() -> dict[str, Any]:
    config = {
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
            validate_json_object(st.session_state["data_values_text"], "Data values")
        ),
        "mesh": {"filename": None, "encoding": "gzip+base64", "content": None},
    }
    if not config["name"]:
        raise ValueError("Enter a name for this problem.")
    space_mapping_from_json(config["solution_space"], "Solution space")
    space_mapping_from_json(config["geometry_space"], "Geometry space")
    if config["data_space"]:
        space_mapping_from_json(config["data_space"], "Data space")
    if config["integrand_formulation"] not in {"potential", "weak"}:
        raise ValueError("Choose either potential or weak integrand formulation.")
    return config


def _render_field(
    grid: pv.UnstructuredGrid,
    field_name: str,
    dimension: int,
    show_gradient: bool = False,
    gradient_grid: pv.UnstructuredGrid | None = None,
):
    plotter = pv.Plotter(off_screen=True, window_size=(1100, 760))
    try:
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
        plotter.add_axes()
        return plotter.screenshot(return_img=True)
    finally:
        plotter.close()


st.set_page_config(
    page_title="Amigo | FEM problem builder",
    page_icon=":material/functions:",
    layout="wide",
)
st.title("Amigo FEM problem builder")
st.write(
    "Upload an Amigo-compatible mesh, define the function spaces and maps, "
    "and solve the resulting finite-element problem."
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
        options=["Poisson potential", "2D source-sink potential flow"],
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
        else:
            selected_config = potential_flow_config(
                source_strength,
                (source_x, source_y),
                (sink_x, sink_y),
                source_width,
            )
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
    problem_name = st.text_input("Problem name", key="problem_name")
    st.selectbox(
        "Integrand formulation",
        options=["potential", "weak"],
        key="problem_formulation",
        help="Choose the formulation expected by your integrand function.",
    )
    tab_spaces, tab_maps, tab_functions = st.tabs(
        ["Function spaces", "Integrand and BC maps", "Python functions"]
    )
    with tab_spaces:
        left, right = st.columns(2)
        with left:
            st.text_area(
                "Solution space mapping (JSON)",
                key="solution_space_text",
                height=160,
                help='Example: {"soln": {"u": "H1"}}. Group fields with comma-separated names such as "u,v".',
            )
            st.text_area(
                "Geometry space mapping (JSON)",
                key="geometry_space_text",
                height=130,
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
    field_name = st.selectbox(
        "Solution field",
        options=result["solution_fields"],
        key="solution_field",
    )
    show_gradient = st.checkbox(
        "Show gradient vectors",
        value=field_name == "phi",
        key=f"show_gradient_{field_name}",
    )
    left, right = st.columns([3, 1])
    with left:
        try:
            image = _render_field(
                result["grids"][field_name],
                field_name,
                mesh_info["dimension"],
                show_gradient=show_gradient,
                gradient_grid=result["gradient_grids"][field_name],
            )
            st.image(
                image,
                alt=(
                    f"PyVista visualization of {field_name}"
                    + (" and its gradient vectors" if show_gradient else "")
                ),
            )
        except Exception as exc:
            st.error(f"PyVista could not render field '{field_name}': {exc}")
            st.exception(exc)
    with right:
        st.metric("Mesh nodes", f'{result["node_count"]:,}')
        st.metric("Mesh elements", f'{result["element_count"]:,}')
        st.write("Mesh domains:", ", ".join(result["mesh_domains"]))
