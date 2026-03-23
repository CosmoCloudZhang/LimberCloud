import os
import sys
import time
import numpy
import argparse
from matplotlib import pyplot


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
    print('Tag: {}'.format(tag))
    
    # Path
    sys.path.insert(0, os.path.join(path, 'BENCHMARK'))
    from PLOT import plot_panel
    
    # Folder
    jax_folder = os.path.join(folder, 'JAX')
    plot_folder = os.path.join(folder, 'PLOT')
    python_folder = os.path.join(folder, 'PYTHON')
    os.makedirs(os.path.join(plot_folder, 'BENCHMARK', tag), exist_ok=True)
    
    # Count
    count1 = 100
    count2 = 1000
    count_size = 10
    count_list = numpy.linspace(count1, count2, count_size, dtype=numpy.float64)
    
    # Label
    label_ccl = r'$\mathrm{CCL}$'
    label_jax_gpu = r'$\mathrm{JAX-GPU}$'
    label_jax_cpu = r'$\mathrm{JAX-CPU}$'
    label_numba_cpu = r'$\mathrm{Numba-CPU}$'
    
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
    time_ccl = numpy.loadtxt(os.path.join(python_folder, 'CCL', tag, 'T_{}_{}.txt'.format(label, number)))
    
    # Load JAX-GPU
    time_jax_gpu = numpy.loadtxt(os.path.join(jax_folder, 'GPU', tag, 'T_{}_{}.txt'.format(label, number)))
    time_jax_gpu_cosmology = numpy.loadtxt(os.path.join(jax_folder, 'GPU', tag, 'T_{}_{}_COSMOLOGY.txt'.format(label, number)))
    time_jax_gpu_projection = numpy.loadtxt(os.path.join(jax_folder, 'GPU', tag, 'T_{}_{}_PROJECTION.txt'.format(label, number)))
    time_jax_gpu_coefficient = numpy.loadtxt(os.path.join(jax_folder, 'GPU', tag, 'T_{}_{}_COEFFICIENT.txt'.format(label, number)))
    
    # Load JAX-CPU
    time_jax_cpu = numpy.loadtxt(os.path.join(jax_folder, 'CPU', tag, 'T_{}_{}.txt'.format(label, number)))
    time_jax_cpu_cosmology = numpy.loadtxt(os.path.join(jax_folder, 'CPU', tag, 'T_{}_{}_COSMOLOGY.txt'.format(label, number)))
    time_jax_cpu_projection = numpy.loadtxt(os.path.join(jax_folder, 'CPU', tag, 'T_{}_{}_PROJECTION.txt'.format(label, number)))
    time_jax_cpu_coefficient = numpy.loadtxt(os.path.join(jax_folder, 'CPU', tag, 'T_{}_{}_COEFFICIENT.txt'.format(label, number)))
    
    # Load Numba-CPU
    time_numba_cpu = numpy.loadtxt(os.path.join(python_folder, 'NUMBA', tag, 'T_{}_{}.txt'.format(label, number)))
    time_numba_cpu_cosmology = numpy.loadtxt(os.path.join(python_folder, 'NUMBA', tag, 'T_{}_{}_COSMOLOGY.txt'.format(label, number)))
    time_numba_cpu_projection = numpy.loadtxt(os.path.join(python_folder, 'NUMBA', tag, 'T_{}_{}_PROJECTION.txt'.format(label, number)))
    time_numba_cpu_coefficient = numpy.loadtxt(os.path.join(python_folder, 'NUMBA', tag, 'T_{}_{}_COEFFICIENT.txt'.format(label, number)))
    
    # Figure
    os.environ['PATH'] = ('/pscratch/sd/y/yhzhang/texlive/2025/bin/x86_64-linux:' + os.environ.get('PATH', ''))
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
    
    figure.savefig(os.path.join(plot_folder, 'BENCHMARK', tag, 'BENCHMARK_{}_{}.pdf'.format(label, number)), bbox_inches='tight')
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60.0
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
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
