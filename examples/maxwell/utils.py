import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def plot_matrix(dense_mat):
    plt.figure(figsize=(6, 5))
    plt.imshow(dense_mat, cmap="viridis", interpolation="nearest", aspect="auto")
    plt.colorbar(label="value")
    plt.tight_layout()
    return


def plot_solution(xyz_nodeCoords, z, title="fig", fname="contour.jpg", flag=False):
    """
    Create a contour plot of the solution.
    Inputs:
        xyz_nodeCoords = [x,y,z] node positions
        z = solution field vector

    Parameters
    ----------
    xyz_nodeCoords : 2d np array
        [x,y,z]
    z : 1d array
        solution vector to plot at each xyz position
    title : str, optional
        figure title, by default "fig"
    fname : str, optional
        name of figure make sure to include extension (.jpg), by default "contour.jpg"
    flag : bool, optional
        save figure, by default False
    """
    min_level = min(z)
    max_level = max(z)
    levels = np.linspace(min_level, max_level, 30)
    x = xyz_nodeCoords[:, 0]
    y = xyz_nodeCoords[:, 1]

    # create a Delaunay triangultion
    tri = mtri.Triangulation(x, y)
    # ntri = tri.triangles.shape[0]

    # Refine the data
    refiner = mtri.UniformTriRefiner(tri)
    # tri_refi, z_refi = refiner.refine_field(z, subdiv=3)

    # Defin colormap
    cmap = "coolwarm"

    # Plot solution
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    # plot = ax.tricontour(tri_refi, z_refi, levels=levels, cmap=cmap)
    plot = ax.tricontour(tri, z, levels=levels, cmap=cmap)
    plot = ax.tricontourf(tri, z, levels=levels, cmap=cmap)
    # ax.set_aspect("equal", adjustable="box")
    # ax.set_title(title, fontsize=10)

    norm = mpl.colors.Normalize(vmin=min_level, vmax=max_level)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, location="right"
    )

    cbar.set_ticks([min(z), (min(z) + max(z)) / 2.0, max(z)])
    # cbar.set_ticklabels([mn,md,mx])

    fig.tight_layout()

    if flag == True:
        plt.savefig(fname, dpi=800, edgecolor="none")

    return
