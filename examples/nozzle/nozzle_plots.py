import numpy as np
import matplotlib.pylab as plt
from matplotlib import font_manager
import niceplots


available_fonts = {f.name for f in font_manager.fontManager.ttflist}


def set_axis_font(ax, fontname):
    if not fontname in available_fonts:
        fontname = "DejaVu Sans"

    for text in ax.findobj(match=type(ax.text(0, 0, ""))):
        text.set_fontname(fontname)


def plot_solution(
    rho,
    u,
    p,
    M_target,
    p_target,
    num_cells,
    length,
    filename="nozzle_stacked.png",
    fontname="Helvetica",
):

    dx = length / num_cells
    xloc = np.linspace(0.5 * dx, length - 0.5 * dx, num_cells)

    with plt.style.context(niceplots.get_style()):
        fig, ax = plt.subplots(3, 1, figsize=(10, 6))
        colors = niceplots.get_colors_list()

        labels = ["Density", "Mach", "Pressure"]
        xlabel = "Location"
        xticks = [0, 2, 4, 6, 8, 10]

        for i, label in enumerate(labels):
            ax[i].set_ylabel(label, rotation="horizontal", horizontalalignment="right")

        line_scaler = 1.0

        indices = [0, 1, 1, 2, 2]
        cindices = [0, 1, 0, 1, 0]

        gamma = 1.4
        a = np.sqrt(gamma * p / rho)

        data = [rho, M_target, u / a, p_target, p]
        label = [None, "target", "solution", "target", "solution"]

        for i, (index, c, y) in enumerate(zip(indices, cindices, data)):
            ax[index].plot(
                xloc,
                y,
                clip_on=False,
                lw=3 * line_scaler,
                color=colors[c],
                label=label[i],
            )

        ax[1].legend(loc="upper left", prop={"size": 12})
        ax[2].legend(loc="lower left", prop={"size": 12})

        for axis in ax:
            niceplots.adjust_spines(axis, outward=True)

        for axis in ax[:-1]:
            axis.get_xaxis().set_visible(False)
            axis.spines["bottom"].set_visible(False)
            axis.get_xaxis().set_ticks([])

        fig.align_labels()
        ax[-1].set_xlabel(xlabel)
        ax[-1].set_xticks(xticks)

        for axis in ax:
            set_axis_font(axis, fontname)

        fig.savefig(filename)

    return


def plot_nozzle(
    A,
    dAdx,
    A_target,
    num_cells,
    length,
    filename="nozzle_design_stacked.png",
    fontname="Helvetica",
):
    dx = length / num_cells
    xintr = np.linspace(0.0, length, num_cells + 1)
    xcell = np.linspace(0.5 * dx, length - 0.5 * dx, num_cells)

    with plt.style.context(niceplots.get_style()):
        fig, ax = plt.subplots(2, 1, figsize=(10, 4))
        colors = niceplots.get_colors_list()

        ylabels = [r"Area", r"$dA/dx$"]
        xlabel = "Location"
        xticks = [0, 2, 4, 6, 8, 10]

        for i, label in enumerate(ylabels):
            ax[i].set_ylabel(label, rotation="horizontal", horizontalalignment="right")

        line_scaler = 1.0

        indices = [0, 0, 1]
        cindices = [1, 0, 0]
        xdata = [xcell, xintr, xcell]
        ydata = [A_target, A, dAdx]
        labels = ["target", "solution", None]

        for i, (index, c, x, y) in enumerate(zip(indices, cindices, xdata, ydata)):
            ax[index].plot(
                x,
                y,
                clip_on=False,
                lw=3 * line_scaler,
                color=colors[c],
                label=labels[i],
            )

        for axis in ax:
            niceplots.adjust_spines(axis, outward=True)

        ax[0].legend(loc="lower left", prop={"size": 12})

        for axis in ax[:-1]:
            axis.get_xaxis().set_visible(False)
            axis.spines["bottom"].set_visible(False)
            axis.get_xaxis().set_ticks([])

        fig.align_labels()
        ax[-1].set_xlabel(xlabel)
        ax[-1].set_xticks(xticks)

        for axis in ax:
            set_axis_font(axis, fontname)

        fig.savefig(filename)

    return


def plot_convergence(nrms, filename="nozzle_residual_norm.png", fontname="Helvetica"):
    with plt.style.context(niceplots.get_style()):
        fig, ax = plt.subplots(1, 1)

        ax.semilogy(nrms, marker="o", clip_on=False, lw=2.0)
        ax.set_ylabel("KKT Residual Norm")
        ax.set_xlabel("Iteration")

        niceplots.adjust_spines(ax)

        set_axis_font(ax, fontname)

        fig.savefig(filename)
