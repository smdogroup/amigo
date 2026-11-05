import numpy as np
import pyvista as pv

from .fem import Problem
from .cell_types import CellType, DofLayout, REFERENCE_CELLS

# Map an Amigo CellType plus the number of local solution DOFs to the matching
# PyVista / VTK cell type. The local DOF count distinguishes the polynomial
# degree (e.g. a 3-node vs 6-node triangle).
#
#   (cell_type, ndof_local) -> pv.CellType
_PV_CELL_TYPES = {
    (CellType.SEGMENT, 2): pv.CellType.LINE,
    (CellType.SEGMENT, 3): pv.CellType.QUADRATIC_EDGE,
    (CellType.TRIANGLE, 3): pv.CellType.TRIANGLE,
    (CellType.TRIANGLE, 6): pv.CellType.QUADRATIC_TRIANGLE,
    (CellType.QUADRILATERAL, 4): pv.CellType.QUAD,
    (CellType.QUADRILATERAL, 8): pv.CellType.QUADRATIC_QUAD,
    (CellType.QUADRILATERAL, 9): pv.CellType.BIQUADRATIC_QUAD,
    (CellType.TETRAHEDRON, 4): pv.CellType.TETRA,
    (CellType.TETRAHEDRON, 10): pv.CellType.QUADRATIC_TETRA,
    (CellType.HEXAHEDRON, 8): pv.CellType.HEXAHEDRON,
    (CellType.HEXAHEDRON, 20): pv.CellType.QUADRATIC_HEXAHEDRON,
    (CellType.PRISM, 6): pv.CellType.WEDGE,
    (CellType.PRISM, 15): pv.CellType.QUADRATIC_WEDGE,
    (CellType.PYRAMID, 5): pv.CellType.PYRAMID,
}


def _pv_cell_type(cell_type: CellType, ndof_local: int) -> "pv.CellType":
    """Map an Amigo ``CellType`` and local DOF count to a PyVista cell type."""
    key = (cell_type, ndof_local)
    if key not in _PV_CELL_TYPES:
        raise NotImplementedError(
            "No PyVista cell type is registered for "
            f"{cell_type.name} with {ndof_local} local DOFs."
        )
    return _PV_CELL_TYPES[key]


def _p1_shape_functions(cell_type: CellType, pts: np.ndarray) -> np.ndarray:
    """Evaluate the degree-1 (vertex) H1 shape functions of ``cell_type``.

    Parameters
    ----------
    cell_type : CellType
        The reference cell the solution DOF points live on.
    pts : ndarray, shape (npts, dim)
        Reference-cell coordinates (e.g. the solution DOF locations).

    Returns
    -------
    N : ndarray, shape (npts, nvert)
        Linear shape-function values, one column per reference-cell vertex.
    """
    pts = np.atleast_2d(np.asarray(pts, dtype=float))

    if cell_type == CellType.TRIANGLE:
        xi, eta = pts[:, 0], pts[:, 1]
        return np.stack([1.0 - xi - eta, xi, eta], axis=1)

    if cell_type == CellType.QUADRILATERAL:
        xi, eta = pts[:, 0], pts[:, 1]
        return 0.25 * np.stack(
            [
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta),
            ],
            axis=1,
        )

    if cell_type == CellType.SEGMENT:
        # Vertices at -1, 1; linear basis.
        xi = pts[:, 0]
        return 0.5 * np.stack([1.0 - xi, 1.0 + xi], axis=1)

    if cell_type == CellType.TETRAHEDRON:
        xi, eta, zeta = pts[:, 0], pts[:, 1], pts[:, 2]
        return np.stack([1.0 - xi - eta - zeta, xi, eta, zeta], axis=1)

    if cell_type == CellType.HEXAHEDRON:
        xi, eta, zeta = pts[:, 0], pts[:, 1], pts[:, 2]
        return 0.125 * np.stack(
            [
                (1.0 - xi) * (1.0 - eta) * (1.0 - zeta),
                (1.0 + xi) * (1.0 - eta) * (1.0 - zeta),
                (1.0 + xi) * (1.0 + eta) * (1.0 - zeta),
                (1.0 - xi) * (1.0 + eta) * (1.0 - zeta),
                (1.0 - xi) * (1.0 - eta) * (1.0 + zeta),
                (1.0 + xi) * (1.0 - eta) * (1.0 + zeta),
                (1.0 + xi) * (1.0 + eta) * (1.0 + zeta),
                (1.0 - xi) * (1.0 + eta) * (1.0 + zeta),
            ],
            axis=1,
        )

    if cell_type == CellType.PRISM:
        xi, eta, zeta = pts[:, 0], pts[:, 1], pts[:, 2]
        lower = 0.5 * (1.0 - zeta)
        upper = 0.5 * (1.0 + zeta)
        return np.stack(
            [
                (1.0 - xi - eta) * lower,
                xi * lower,
                eta * lower,
                (1.0 - xi - eta) * upper,
                xi * upper,
                eta * upper,
            ],
            axis=1,
        )

    if cell_type == CellType.PYRAMID:
        xi, eta, zeta = pts[:, 0], pts[:, 1], pts[:, 2]
        return 0.25 * np.stack(
            [
                (1.0 - xi) * (1.0 - eta) * (1.0 - zeta),
                (1.0 + xi) * (1.0 - eta) * (1.0 - zeta),
                (1.0 + xi) * (1.0 + eta) * (1.0 - zeta),
                (1.0 - xi) * (1.0 + eta) * (1.0 - zeta),
                4.0 * zeta,
            ],
            axis=1,
        )

    raise NotImplementedError(
        f"Degree-1 geometry shape functions are not implemented for {cell_type.name}."
    )


def select_grid_domains(
    grid: pv.UnstructuredGrid, domains: list[str]
) -> pv.UnstructuredGrid:
    """Return a grid containing only cells whose mesh-domain labels are selected."""
    if "mesh_domain" not in grid.cell_data:
        raise ValueError("The result grid is missing its mesh-domain labels.")
    labels = np.asarray(grid.cell_data["mesh_domain"])
    if labels.shape != (grid.n_cells,):
        raise ValueError("Mesh-domain labels do not match the result grid cells.")
    return grid.extract_cells(np.isin(labels, domains))


def build_grid(
    problem: Problem,
    name: str,
    x,
    domains: str | list[str] | None = None,
) -> "pv.UnstructuredGrid":
    """Build a ``pyvista.UnstructuredGrid`` for a solution field.

    Parameters
    ----------
    problem : Problem
        The Amigo FEM problem that defines the mesh, geometry space and
        solution space.
    name : str
        Name of the scalar solution field to visualize (e.g. ``"u"``).
    x : Vector
        The Amigo solution vector. The field values are read from
        ``x[f"{dof_prefix}.{name}"]`` using the solution DOF numbering.
    domains : str or list[str]
        The domain name(s) whose blocks are included in the grid
        (e.g. ``"SURFACE1"``).

    Returns
    -------
    grid : pyvista.UnstructuredGrid
        Grid whose points are the solution DOF locations and whose
        ``point_data[name]`` holds the solution field and whose
        ``cell_data["mesh_domain"]`` identifies the source mesh domain of each
        cell.
    """
    # Geometry: degree-1 H1 space + its DOF handler (used for nodal coords).
    geo_space = problem.geo_space.get_spaces()[0]
    geo_handler = problem.geo_dof.get_dof_handler()

    # Try to find what the user requested to plot
    space, handler = None, None
    comp_name = problem.soln_space.get_component_name(name)
    if comp_name is not None:
        space = problem.soln_space.get_space(name)
        handler = problem.soln_dof.get_dof_handler()
    else:
        comp_name = problem.geo_space.get_component_name(name)
        if comp_name is not None:
            space = problem.geo_space.get_space(name)
            handler = problem.geo_dof.get_dof_handler()

    if space is None:
        raise ValueError(f"Could not find field {name}")

    # Physical coordinates of the geometry nodes, in geo-DOF order.
    Xgeo = np.asarray(problem.geo_dof.get_coordinates())
    dim = Xgeo.shape[1]

    # Global solution field values, indexed by solution DOF number.
    u = np.asarray(x[f"{comp_name}.{name}"]).reshape(-1)

    # Allocate the global point array (one point per solution DOF).
    n_soln_dof = handler.get_num_dof(space)
    points = np.zeros((n_soln_dof, 3), dtype=float)

    # PyVista accepts connectivity as {cell_type: (nelem, ndof_local)} arrays,
    # so we group each block's connectivity by its PyVista cell type. This
    # avoids the flat VTK "[npts, p0, p1, ...]" packing and the parallel
    # cell-type array entirely.
    conn_by_type: dict["pv.CellType", list[np.ndarray]] = {}
    domain_by_type: dict["pv.CellType", list[np.ndarray]] = {}

    block_ids = problem.mesh.get_block_ids(domains)

    for block_id in block_ids:
        cell_type = problem.mesh.get_cell_type(block_id)

        # Solution connectivity in canonical reference-cell DOF ordering.
        soln_conn = handler.get_dof_conn(space, block_id)
        ndof_local = soln_conn.shape[1]

        # Map the cell type + local DOF count to the PyVista cell type.
        pv_type = _pv_cell_type(cell_type, ndof_local)

        # Solution DOF reference points on this cell type.
        soln_layout = DofLayout.make_layout(space, cell_type)
        ref_pts = np.asarray(soln_layout.pts, dtype=float)

        # Degree-1 geometry map evaluated at the solution DOF reference points.
        geo_conn = geo_handler.get_dof_conn(geo_space, block_id)
        nvert = geo_conn.shape[1]
        N = _p1_shape_functions(cell_type, ref_pts)  # (ndof_local, nvert)
        if N.shape[1] != nvert:
            raise ValueError(
                f"Vertex count mismatch for {cell_type.name}: geometry "
                f"connectivity has {nvert} vertices but shape functions "
                f"produced {N.shape[1]}."
            )

        # Physical coordinates of every solution DOF, placed by global DOF
        vcoords = Xgeo[geo_conn]  # (nelem, nvert, dim)
        phys = N @ vcoords  # (nelem, ndof_local, dim)
        points[soln_conn, :dim] = phys

        conn_by_type.setdefault(pv_type, []).append(soln_conn)
        domain_by_type.setdefault(pv_type, []).append(
            np.repeat(problem.mesh.get_domain_name(block_id), soln_conn.shape[0])
        )

    if not conn_by_type:
        raise ValueError(f"No blocks found for domain(s) {domains!r}.")

    cells = {pv_type: np.vstack(conns) for pv_type, conns in conn_by_type.items()}

    # Build the PyVista grid and attach the solution field.
    grid = pv.UnstructuredGrid(cells, points)
    grid.cell_data["mesh_domain"] = np.concatenate(
        [np.concatenate(domain_by_type[pv_type]) for pv_type in conn_by_type]
    )
    grid.point_data[name] = u

    return grid
