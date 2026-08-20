import argparse
import os
import time

import numpy
from matplotlib import pyplot

from limbercloud import Configuration, ProjectPaths
from limbercloud.plotting import plot_panel


def _load_timing(folder, suffix, configuration, number):
    """Load a canonical timing file."""

    candidate = folder / f"Time_{configuration.value}_{number}{suffix}.txt"
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing timing file: {candidate}")
    return numpy.loadtxt(candidate)


def main(tag, path, label, folder, number):
    '''
    Plot benchmark: cumulative time vs number of evaluations.

    Arguments:
        tag (str): The tag of the configuration
        path (str): The path of the project scripts
        label (str): The label of the configuration
        folder (str): The base folder of the dataset
        number (int): The number of cores for parallel computation

    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    configuration = Configuration.parse(label)
    label = configuration.value
    print(f'Tag: {tag}')

    # Runtime paths
    paths = ProjectPaths.from_root(folder)
    ccl_folder = paths.spectrum_results('CCL', tag)
    numba_folder = paths.spectrum_results('NUMBA', tag)
    jax_gpu_folder = paths.spectrum_results('JAX', tag, 'GPU')
    jax_cpu_folder = paths.spectrum_results('JAX', tag, 'CPU')
    plot_folder = paths.plots / 'benchmarks' / tag
    plot_folder.mkdir(parents=True, exist_ok=True)

    # Count
    count1 = 100
    count2 = 1000
    count_size = 10
    count_list = numpy.linspace(count1, count2, count_size, dtype=numpy.float64)

    # Label
    label_ccl = r'$\mathtt{CCL}$'
    label_jax_gpu = r'$\mathtt{JAX-GPU}$'
    label_jax_cpu = r'$\mathtt{JAX-CPU}$'
    label_numba_cpu = r'$\mathtt{Numba-CPU}$'

    # Color
    color_ccl = 'darkblue'
    color_jax_gpu = 'darkred'
    color_jax_cpu = 'darkmagenta'
    color_numba_cpu = 'darkorange'

    # Marker
    marker_ccl = 'o'
    marker_jax_cpu = '^'
    marker_jax_gpu = 'D'
    marker_numba_cpu = 's'

    # Load CCL
    time_ccl = _load_timing(ccl_folder, '', configuration, number)

    # Load JAX-GPU
    time_jax_gpu = _load_timing(jax_gpu_folder, '', configuration, number)
    time_jax_gpu_cosmology = _load_timing(jax_gpu_folder, '_COSMOLOGY', configuration, number)
    time_jax_gpu_projection = _load_timing(jax_gpu_folder, '_PROJECTION', configuration, number)
    time_jax_gpu_coefficient = _load_timing(jax_gpu_folder, '_COEFFICIENT', configuration, number)

    # Load JAX-CPU
    time_jax_cpu = _load_timing(jax_cpu_folder, '', configuration, number)
    time_jax_cpu_cosmology = _load_timing(jax_cpu_folder, '_COSMOLOGY', configuration, number)
    time_jax_cpu_projection = _load_timing(jax_cpu_folder, '_PROJECTION', configuration, number)
    time_jax_cpu_coefficient = _load_timing(jax_cpu_folder, '_COEFFICIENT', configuration, number)

    # Load Numba-CPU
    time_numba_cpu = _load_timing(numba_folder, '', configuration, number)
    time_numba_cpu_cosmology = _load_timing(numba_folder, '_COSMOLOGY', configuration, number)
    time_numba_cpu_projection = _load_timing(numba_folder, '_PROJECTION', configuration, number)
    time_numba_cpu_coefficient = _load_timing(numba_folder, '_COEFFICIENT', configuration, number)

    # Figure
    texlive_bin = os.environ.get('LIMBERCLOUD_TEXLIVE_BIN')
    if texlive_bin:
        os.environ['PATH'] = texlive_bin + os.pathsep + os.environ.get('PATH', '')
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25

    figure, plot = pyplot.subplots(nrows=4, ncols=1, figsize=(12, 20), sharex=True)

    # Total
    rows_total = [(label_ccl, time_ccl, color_ccl, marker_ccl), (label_jax_gpu, time_jax_gpu, color_jax_gpu, marker_jax_gpu), (label_jax_cpu, time_jax_cpu, color_jax_cpu, marker_jax_cpu), (label_numba_cpu, time_numba_cpu, color_numba_cpu, marker_numba_cpu)]
    plot_panel(rows_total, plot[0], count_list, r'$\mathrm{Total}$', show_legend=True)

    # Cosmology stage
    rows_cosmology = [(label_ccl, time_ccl, color_ccl, marker_ccl), (label_jax_gpu, time_jax_gpu_cosmology, color_jax_gpu, marker_jax_gpu), (label_jax_cpu, time_jax_cpu_cosmology, color_jax_cpu, marker_jax_cpu), (label_numba_cpu, time_numba_cpu_cosmology, color_numba_cpu, marker_numba_cpu)]
    plot_panel(rows_cosmology, plot[1], count_list, r'$\mathrm{Cosmology \, stage}$', show_legend=False)

    # Coefficient stage
    rows_coefficient = [(label_ccl, time_ccl, color_ccl, marker_ccl), (label_jax_gpu, time_jax_gpu_coefficient, color_jax_gpu, marker_jax_gpu), (label_jax_cpu, time_jax_cpu_coefficient, color_jax_cpu, marker_jax_cpu), (label_numba_cpu, time_numba_cpu_coefficient, color_numba_cpu, marker_numba_cpu)]
    plot_panel(rows_coefficient, plot[2], count_list, r'$\mathrm{Coefficient \, stage}$', show_legend=False)

    # Projection stage
    rows_projection = [(label_ccl, time_ccl, color_ccl, marker_ccl), (label_jax_gpu, time_jax_gpu_projection, color_jax_gpu, marker_jax_gpu), (label_jax_cpu, time_jax_cpu_projection, color_jax_cpu, marker_jax_cpu), (label_numba_cpu, time_numba_cpu_projection, color_numba_cpu, marker_numba_cpu)]
    plot_panel(rows_projection, plot[3], count_list, r'$\mathrm{Projection \, stage}$', show_legend=False)

    for index in range(3):
        plot[index].set_xlabel('')
        plot[index].tick_params(axis='x', which='both', labelbottom=False)
    figure.subplots_adjust(hspace=0.0)

    figure.savefig(plot_folder / f'benchmark_{label}_{number}.pdf', bbox_inches='tight', dpi=512)
    pyplot.close(figure)

    # Duration
    end = time.time()
    duration = (end - start) / 60.0

    # Return
    print(f'Time: {duration:.2f} minutes')
    return duration


if __name__ == '__main__':
    # Input
    parse = argparse.ArgumentParser(description='Benchmark')
    parse.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    parse.add_argument('--path', type=str, required=True, help='The path of the project scripts')
    parse.add_argument('--label', type=str, required=True, help='The label of the configuration')
    parse.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    parse.add_argument('--number', type=int, required=True, help='The number of cores for parallel computation')

    # Parse
    TAG = parse.parse_args().tag
    PATH = parse.parse_args().path
    LABEL = parse.parse_args().label
    FOLDER = parse.parse_args().folder
    NUMBER = parse.parse_args().number

    # OUTPUT
    OUTPUT = main(TAG, PATH, LABEL, FOLDER, NUMBER)
