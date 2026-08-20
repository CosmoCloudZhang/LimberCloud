"""Benchmark-figure helpers."""


def plot_panel(rows, axes, x_values, description, show_legend=False):
    """Plot one benchmark panel with logarithmic axes."""

    for label, y_values, color, marker in rows:
        axes.loglog(
            x_values,
            y_values,
            linestyle="-",
            linewidth=2.0,
            markersize=9,
            markeredgewidth=1.0,
            markeredgecolor="black",
            color=color,
            marker=marker,
            label=label,
            rasterized=True,
        )

    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_ylabel(r"$\mathrm{Cumulative\ time\ (s)}$", fontsize=25)
    axes.set_xlabel(r"$\mathrm{Number\ of\ evaluations}$", fontsize=25)
    axes.grid(True, which="both", alpha=0.50)
    if show_legend:
        axes.legend(loc="lower right", fontsize=20)
    axes.text(
        0.02,
        0.98,
        description,
        fontsize=25,
        verticalalignment="top",
        transform=axes.transAxes,
    )
