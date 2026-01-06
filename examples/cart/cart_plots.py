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
    d,
    theta,
    xctrl,
    final_time=2.0,
    num_time_steps=100,
    fontname="Helvetica",
    filename="cart_stacked.png",
):
    t = np.linspace(0, final_time, num_time_steps + 1)

    with plt.style.context(niceplots.get_style()):
        data = {}
        data["Cart pos."] = d
        data["Pole angle"] = (180 / np.pi) * theta
        data["Control force"] = xctrl

        fig, ax = niceplots.stacked_plots(
            "Time (s)",
            t,
            [data],
            lines_only=True,
            figsize=(10, 6),
            line_scaler=0.5,
        )

        for axis in ax:
            set_axis_font(axis, fontname)

        fig.savefig(filename)

    return


def plot_for_documentation(
    x,
    final_time=2.0,
    num_time_steps=100,
    fontname="Helvetica",
    filename="cart_pole_solution.png",
):
    """
    Create documentation-style plots showing position, angle, and force
    """
    # Extract solution
    time = np.linspace(0, final_time, num_time_steps + 1)
    position = x["cart.q[:, 0]"]  # Cart position
    angle = x["cart.q[:, 1]"]  # Pole angle
    force = x["cart.x[:]"]  # Control force

    # blue color
    blue_color = "#0072BD"

    # Create figure with 3 subplots (2 states + 1 control)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))

    # Plot every Nth knot point (adjust step to show more/fewer points)
    knot_step = (
        5  # Show every 5th knot point (change to 1 for all points, 10 for fewer, etc.)
    )

    # Position plot
    ax1.plot(time, position, color=blue_color, linewidth=2.5)
    ax1.plot(
        time[::knot_step],
        position[::knot_step],
        "o",
        color="black",
        markersize=5,
        markerfacecolor="none",
        markeredgewidth=1,
    )
    ax1.set_ylabel("Position (m)", fontsize=18)
    ax1.grid(True, alpha=0.25, linewidth=0.5)
    ax1.tick_params(labelsize=10)
    ax1.set_title("State", fontsize=18, fontweight="bold", pad=10)

    # Angle plot
    ax2.plot(time, angle, color=blue_color, linewidth=2.5)
    ax2.plot(
        time[::knot_step],
        angle[::knot_step],
        "o",
        color="black",
        markersize=5,
        markerfacecolor="none",
        markeredgewidth=1,
    )
    ax2.set_ylabel("Angle (rad)", fontsize=18)
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    ax2.tick_params(labelsize=10)

    # Force plot (purple color)
    purple_color = "#8242A2"
    ax3.plot(time, force, color=purple_color, linewidth=2.5)
    ax3.plot(
        time[::knot_step],
        force[::knot_step],
        "o",
        color="black",
        markersize=5,
        markerfacecolor="none",
        markeredgewidth=1,
    )
    ax3.set_xlabel("Time (s)", fontsize=18)
    ax3.set_ylabel("Force (N)", fontsize=18)
    ax3.grid(True, alpha=0.25, linewidth=0.5)
    ax3.tick_params(labelsize=10)
    ax3.set_title("Control", fontsize=18, fontweight="bold", pad=10)

    # Set font for all axes
    fontname = "Helvetica"
    for ax in [ax1, ax2, ax3]:
        set_axis_font(ax, fontname)

    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")

    return


def plot_convergence(nrms, fontname="Helvetica", filename="cart_residual_norm.png"):
    with plt.style.context(niceplots.get_style()):
        fig, ax = plt.subplots(1, 1)

        ax.semilogy(nrms, marker="o", clip_on=False, lw=2.0)
        ax.set_ylabel("KKT Residual Norm")
        ax.set_xlabel("Iteration")

        niceplots.adjust_spines(ax)
        set_axis_font(ax, fontname)

        fig.savefig(filename)


def visualize(d, theta, L=0.5, filename="cart_pole_history.png"):
    with plt.style.context(niceplots.get_style()):
        # Create the time-lapse visualization
        fig, ax = plt.subplots(1, figsize=(10, 4.5))
        ax.axis("equal")
        ax.axis("off")

        values = np.linspace(0.1, 0.9, d.shape[0])
        cmap = plt.get_cmap("viridis")

        hx = 0.03
        hy = 0.03
        xpts = []
        ypts = []
        for i in range(0, d.shape[0]):
            color = cmap(values[i])

            x1 = d[i]
            y1 = 0.0
            x2 = d[i] + L * np.sin(theta[i])
            y2 = -L * np.cos(theta[i])

            xpts.append(x2)
            ypts.append(y2)

            if i % 2 == 0:
                ax.plot([x1, x2], [y1, y2], linewidth=2, color=color)
                ax.fill(
                    [x1 - hx, x1 + hx, x1 + hx, x1 - hx, x1 - hx],
                    [y1, y1, y1 + hy, y1 + hy, y1],
                    alpha=0.5,
                    linewidth=2,
                    color=color,
                )

            ax.plot([x2], [y2], color=color, marker="o")

        # Define the bounding box coordinates (x, y, width, height)
        x = 0
        y = -L
        height = 2 * L
        width = 2 + 0.6 * L

        # Create a Rectangle patch
        import matplotlib.patches as patches

        rect = patches.Rectangle(
            (x, y), width, height, linewidth=2, edgecolor="none", facecolor="none"
        )

        # Add the patch to the axes
        ax.add_patch(rect)

        fig.savefig(filename)

    return
