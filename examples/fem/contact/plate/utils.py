# Author: Jack Turbush
# Description: Utility functions for VTU output in the plate contact example.

from amigo.fem import Mesh
import amigo as am
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib.tri as tri


def plot_3d(
    mesh: Mesh,
    w,
    dx=None,
    dy=None,
    fig=None,
    ax=None,
    scale=1.0,
    cmap="coolwarm",
    title=None,
    x_offset=0.0,
    y_offset=0.0,
    alpha=0.85,
    show_edges=True,
    show_undeformed=True,
    min_level=None,
    max_level=None,
    view="iso",
):
    """
    Plot the 2D mesh lifted into 3D by the out-of-plane displacement field w,
    with optional in-plane deformation overlay (dx, dy).

    Parameters
    ----------
    mesh            : Mesh object
    w               : (nnodes,) array — perpendicular displacement at each node
    dx, dy          : (nnodes,) arrays — in-plane displacements (optional).
                      If provided, the deformed mesh is plotted in blue and the
                      undeformed ghost is plotted in grey dashes.
    fig/ax          : existing Figure / Axes3D to draw into (created if None)
    scale           : multiply w before plotting (useful for visual exaggeration)
    cmap            : matplotlib colormap name
    title           : axes title string
    x_offset        : shift all x coordinates before plotting
    y_offset        : shift all y coordinates before plotting
    alpha           : surface transparency
    show_edges      : overlay mesh wire-frame on the surface
    show_undeformed : when dx/dy are given, also draw the undeformed ghost mesh
    min_level / max_level : clamp the colormap range
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    if fig is None or ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

    # Base coordinates
    x_base = mesh.X[:, 0] + x_offset
    y_base = mesh.X[:, 1] + y_offset
    z = np.asarray(w) * scale

    # Deformed coordinates (fall back to undeformed if dx/dy not supplied)
    has_inplane = dx is not None and dy is not None
    x = x_base + np.asarray(dx) if has_inplane else x_base
    y = y_base + np.asarray(dy) if has_inplane else y_base

    # Colormap normalisation
    vmin = np.min(z) if min_level is None else min_level * scale
    vmax = np.max(z) if max_level is None else max_level * scale
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

    etypes_2d = {"CPS3", "CPS4", "CPS6", "CPS8"}

    domains = mesh.get_domains()

    def convert_conn(etype, conn):
        if etype == "CPS3":
            return conn
        elif etype == "CPS4":
            c = [[0, 1, 2], [0, 2, 3]]
        elif etype == "CPS6":
            # 2
            # |  .
            # 5     4
            # |        .
            # 0 --- 3 --- 1
            c = [[0, 3, 5], [3, 4, 5], [3, 1, 4], [5, 4, 2]]
        elif etype == "M3D9":
            # 3 --- 6 --- 2
            # |           |
            # 7     8     5
            # |           |
            # 0 --- 4 --- 1
            c = [
                [0, 4, 7],
                [4, 8, 7],
                [4, 1, 8],
                [1, 5, 8],
                [7, 8, 3],
                [8, 6, 3],
                [8, 5, 6],
                [5, 2, 6],
            ]

        cs = []
        for c0 in c:
            cs.append(conn[:, c0])

        return np.vstack(cs)

    for name in domains:
        for etype in domains[name]:
            if etype == "T3D2":
                continue

            conn = mesh.get_conn(name, etype)
            tri_conn = convert_conn(etype, conn)

            # ------------------------------------------------------------------
            # 1. Undeformed ghost mesh (grey dashed)
            # ------------------------------------------------------------------
            if has_inplane and show_undeformed and etype in etypes_2d:
                ghost_segs = []
                for tri_nodes in tri_conn:
                    loop = list(tri_nodes) + [tri_nodes[0]]
                    pts = np.column_stack([x_base[loop], y_base[loop], z[loop]])
                    for k in range(len(loop) - 1):
                        ghost_segs.append([pts[k], pts[k + 1]])

                ghost = Line3DCollection(
                    ghost_segs,
                    colors="grey",
                    linewidths=0.8,
                    linestyles="--",
                    alpha=0.5,
                )
                ax.add_collection3d(ghost)

            # ------------------------------------------------------------------
            # 2. Coloured surface
            # ------------------------------------------------------------------
            verts = []
            face_colors = []
            for tri_nodes in tri_conn:
                pts = np.column_stack([x[tri_nodes], y[tri_nodes], z[tri_nodes]])
                verts.append(pts)
                face_colors.append(scalar_map.to_rgba(np.mean(z[tri_nodes])))

            edge_col = "steelblue" if has_inplane else "k"
            poly = Poly3DCollection(
                verts,
                facecolors=face_colors,
                edgecolors=edge_col if show_edges else "none",
                linewidths=0.8 if show_edges else 0.0,
                alpha=alpha,
            )
            ax.add_collection3d(poly)

    # Axis limits
    ax.set_xlim(min(x.min(), x_base.min()), max(x.max(), x_base.max()))
    ax.set_ylim(min(y.min(), y_base.min()), max(y.max(), y_base.max()))
    ax.set_zlim(vmin, vmax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("w")

    if title is not None:
        ax.set_title(title)

    fig.colorbar(scalar_map, ax=ax, shrink=0.5, label="w")
    views = {
        "front": (0, 0),
        "side": (0, 90),
        "top": (90, 0),
        "iso": (30, 45),
    }

    elev, azim = views.get(view, (30, 45))
    ax.view_init(elev=elev, azim=azim)
    return ax
