import argparse
import os
import time

from matplotlib import pyplot

from limbercloud import Configuration, ProjectPaths
from limbercloud.experiments.timing import (
    load_cosmology_timing,
    require_matching_timings,
)
from limbercloud.plotting import plot_panel


def _load_timing(folder, suffix, configuration, *, family):
    """Load the explicitly labelled sampled population and its actual counts."""
    return load_cosmology_timing(folder, configuration.value, suffix, family=family)


def _family_directory(paths, backend, survey, device=None):
    """Resolve the family/survey directory that holds the timing products."""

    return paths.spectrum_results(backend, survey, device)


def main(tag, label, folder):
    """
    Plot benchmark: cumulative time vs number of evaluations.

    The performance figure covers CCL, Numba and JAX only. NUMERIC timing
    products are read by the spectra and error notebooks, not this script.

    Args:
        tag (str): The tag of the configuration
        label (str): The label of the configuration
        folder (str): The base folder of the dataset

    Returns:
        duration (float): The duration of the process
    """
    # Start
    start = time.time()
    configuration = Configuration.parse(label)
    label = configuration.value
    print(f"Tag: {tag}")

    # Runtime paths
    paths = ProjectPaths.from_root(folder)
    ccl_folder = _family_directory(paths, "CCL", tag)
    numba_folder = _family_directory(paths, "NUMBA", tag)
    jax_gpu_folder = _family_directory(paths, "JAX", tag, "GPU")
    jax_cpu_folder = _family_directory(paths, "JAX", tag, "CPU")
    plot_folder = paths.plots / "benchmarks" / tag
    plot_folder.mkdir(parents=True, exist_ok=True)

    # Label
    label_ccl = r"$\mathtt{CCL}$"
    label_jax_gpu = r"$\mathtt{JAX-GPU}$"
    label_jax_cpu = r"$\mathtt{JAX-CPU}$"
    label_numba_cpu = r"$\mathtt{Numba-CPU}$"

    # Color
    color_ccl = "darkblue"
    color_jax_gpu = "darkred"
    color_jax_cpu = "darkmagenta"
    color_numba_cpu = "darkorange"

    # Marker
    marker_ccl = "o"
    marker_jax_cpu = "^"
    marker_jax_gpu = "D"
    marker_numba_cpu = "s"

    # The CCL curve is the total in every panel, as noted below the figure.
    time_ccl = _load_timing(ccl_folder, "", configuration, family="CCL")

    # Load JAX-GPU
    time_jax_gpu = _load_timing(jax_gpu_folder, "", configuration, family="JAX")
    time_jax_gpu_cosmology = _load_timing(
        jax_gpu_folder, "_COSMOLOGY", configuration, family="JAX"
    )
    time_jax_gpu_projection = _load_timing(
        jax_gpu_folder, "_PROJECTION", configuration, family="JAX"
    )
    time_jax_gpu_coefficient = _load_timing(
        jax_gpu_folder, "_COEFFICIENT", configuration, family="JAX"
    )

    # Load JAX-CPU
    time_jax_cpu = _load_timing(jax_cpu_folder, "", configuration, family="JAX")
    time_jax_cpu_cosmology = _load_timing(
        jax_cpu_folder, "_COSMOLOGY", configuration, family="JAX"
    )
    time_jax_cpu_projection = _load_timing(
        jax_cpu_folder, "_PROJECTION", configuration, family="JAX"
    )
    time_jax_cpu_coefficient = _load_timing(
        jax_cpu_folder, "_COEFFICIENT", configuration, family="JAX"
    )

    # Load Numba-CPU
    time_numba_cpu = _load_timing(numba_folder, "", configuration, family="NUMBA")
    time_numba_cpu_cosmology = _load_timing(
        numba_folder, "_COSMOLOGY", configuration, family="NUMBA"
    )
    time_numba_cpu_projection = _load_timing(
        numba_folder, "_PROJECTION", configuration, family="NUMBA"
    )
    time_numba_cpu_coefficient = _load_timing(
        numba_folder, "_COEFFICIENT", configuration, family="NUMBA"
    )

    # Compare actual checkpoints and the cosmology table before plotting.
    count_list = require_matching_timings(
        time_ccl,
        time_jax_gpu,
        time_jax_gpu_cosmology,
        time_jax_gpu_projection,
        time_jax_gpu_coefficient,
        time_jax_cpu,
        time_jax_cpu_cosmology,
        time_jax_cpu_projection,
        time_jax_cpu_coefficient,
        time_numba_cpu,
        time_numba_cpu_cosmology,
        time_numba_cpu_projection,
        time_numba_cpu_coefficient,
    )
    time_ccl = time_ccl.seconds
    time_jax_gpu = time_jax_gpu.seconds
    time_jax_gpu_cosmology = time_jax_gpu_cosmology.seconds
    time_jax_gpu_projection = time_jax_gpu_projection.seconds
    time_jax_gpu_coefficient = time_jax_gpu_coefficient.seconds
    time_jax_cpu = time_jax_cpu.seconds
    time_jax_cpu_cosmology = time_jax_cpu_cosmology.seconds
    time_jax_cpu_projection = time_jax_cpu_projection.seconds
    time_jax_cpu_coefficient = time_jax_cpu_coefficient.seconds
    time_numba_cpu = time_numba_cpu.seconds
    time_numba_cpu_cosmology = time_numba_cpu_cosmology.seconds
    time_numba_cpu_projection = time_numba_cpu_projection.seconds
    time_numba_cpu_coefficient = time_numba_cpu_coefficient.seconds

    # Figure
    texlive_bin = os.environ.get("LIMBERCLOUD_TEXLIVE_BIN")
    if texlive_bin:
        os.environ["PATH"] = texlive_bin + os.pathsep + os.environ.get("PATH", "")
    pyplot.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    pyplot.rcParams["pgf.texsystem"] = "pdflatex"
    pyplot.rcParams["text.usetex"] = True
    pyplot.rcParams["font.size"] = 25

    figure, plot = pyplot.subplots(nrows=4, ncols=1, figsize=(12, 20), sharex=True)

    # Total
    rows_total = [
        (label_ccl, time_ccl, color_ccl, marker_ccl),
        (label_jax_gpu, time_jax_gpu, color_jax_gpu, marker_jax_gpu),
        (label_jax_cpu, time_jax_cpu, color_jax_cpu, marker_jax_cpu),
        (label_numba_cpu, time_numba_cpu, color_numba_cpu, marker_numba_cpu),
    ]
    plot_panel(rows_total, plot[0], count_list, r"$\mathrm{Total}$", show_legend=True)

    # Cosmology stage
    rows_cosmology = [
        (label_ccl, time_ccl, color_ccl, marker_ccl),
        (label_jax_gpu, time_jax_gpu_cosmology, color_jax_gpu, marker_jax_gpu),
        (label_jax_cpu, time_jax_cpu_cosmology, color_jax_cpu, marker_jax_cpu),
        (label_numba_cpu, time_numba_cpu_cosmology, color_numba_cpu, marker_numba_cpu),
    ]
    plot_panel(
        rows_cosmology,
        plot[1],
        count_list,
        r"$\mathrm{Cosmology \, stage}$",
        show_legend=False,
    )

    # Coefficient stage
    rows_coefficient = [
        (label_ccl, time_ccl, color_ccl, marker_ccl),
        (label_jax_gpu, time_jax_gpu_coefficient, color_jax_gpu, marker_jax_gpu),
        (label_jax_cpu, time_jax_cpu_coefficient, color_jax_cpu, marker_jax_cpu),
        (
            label_numba_cpu,
            time_numba_cpu_coefficient,
            color_numba_cpu,
            marker_numba_cpu,
        ),
    ]
    plot_panel(
        rows_coefficient,
        plot[2],
        count_list,
        r"$\mathrm{Coefficient \, stage}$",
        show_legend=False,
    )

    # Projection stage
    rows_projection = [
        (label_ccl, time_ccl, color_ccl, marker_ccl),
        (label_jax_gpu, time_jax_gpu_projection, color_jax_gpu, marker_jax_gpu),
        (label_jax_cpu, time_jax_cpu_projection, color_jax_cpu, marker_jax_cpu),
        (label_numba_cpu, time_numba_cpu_projection, color_numba_cpu, marker_numba_cpu),
    ]
    plot_panel(
        rows_projection,
        plot[3],
        count_list,
        r"$\mathrm{Projection \, stage}$",
        show_legend=False,
    )

    for index in range(3):
        plot[index].set_xlabel("")
        plot[index].tick_params(axis="x", which="both", labelbottom=False)
    figure.subplots_adjust(hspace=0.0)

    figure.text(
        0.5,
        0.005,
        "CCL shows total time in every panel; other methods show the named stage.",
        ha="center",
        fontsize=14,
    )
    figure.savefig(plot_folder / f"benchmark_{label}.pdf", bbox_inches="tight", dpi=512)
    pyplot.close(figure)

    # Duration
    end = time.time()
    duration = (end - start) / 60.0

    # Return
    print(f"Time: {duration:.2f} minutes")
    return duration


if __name__ == "__main__":
    # Input
    parse = argparse.ArgumentParser(description="Benchmark")
    parse.add_argument(
        "--tag", type=str, required=True, help="The tag of the configuration"
    )
    parse.add_argument(
        "--label", type=str, required=True, help="The label of the configuration"
    )
    parse.add_argument(
        "--folder", type=str, required=True, help="The base folder of the dataset"
    )

    # Parse
    ARGS = parse.parse_args()
    OUTPUT = main(
        ARGS.tag,
        ARGS.label,
        ARGS.folder,
    )
