import os
import time
import numpy
import argparse
from matplotlib import pyplot

# Must match the timing drivers (e.g. PYTHON/CPU, JAX/CPU, PYTHON/CCL).
COUNT_START = 100
COUNT_END = 1000
COUNT_SIZE = 10
EVAL_COUNTS = numpy.linspace(COUNT_START, COUNT_END, COUNT_SIZE, dtype=numpy.float64)

# One style per backend (colour + marker); CCL is the reference in every panel.
STYLE = {
    'ccl': {'color': '#1f77b4', 'marker': 'o'},
    'cpu': {'color': '#ff7f0e', 'marker': 's'},
    'jax_cpu': {'color': '#2ca02c', 'marker': '^'},
    'jax_gpu': {'color': '#d62728', 'marker': 'D'},
}


def _ravel_float(arr):
    return numpy.asarray(arr, dtype=numpy.float64).ravel()


def _y_for_log_scale(arr):
    '''Positive values only; zeros or negatives -> NaN (skipped on log axes).'''
    y = _ravel_float(arr)
    out = y.copy()
    out[out <= 0] = numpy.nan
    return out


def _load_column(path):
    '''Load a single column of floats from a text file.'''
    return numpy.loadtxt(path)


def plot_panel(ax, x, rows, description):
    '''
    rows: list of (legend_label, y_array, style_key) with style_key in STYLE.
    '''
    for label, y, key in rows:
        sty = STYLE[key]
        ax.loglog(
            x,
            _y_for_log_scale(y),
            linestyle='-',
            linewidth=2.0,
            markersize=9,
            markeredgewidth=1.0,
            markeredgecolor='black',
            color=sty['color'],
            marker=sty['marker'],
            label=label,
        )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Number of evaluations')
    ax.set_ylabel(r'Cumulative time (s)')
    ax.grid(True, which='both', alpha=0.35)
    ax.legend(loc='best', fontsize=15)
    ax.text(
        0.02,
        0.98,
        description,
        transform=ax.transAxes,
        fontsize=17,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.35),
    )


def main(tag, _path, label, folder, number):
    '''
    Plot benchmark: cumulative time vs number of evaluations (log--log).

    Timing files are read from ``folder`` (dataset tree). ``_path`` is kept for
    compatibility with job scripts and is not used for I/O here.
    '''
    t0 = time.time()

    dir_python = os.path.join(folder, 'PYTHON')
    dir_jax = os.path.join(folder, 'JAX')
    dir_plot = os.path.join(folder, 'PLOT')
    os.makedirs(os.path.join(dir_plot, 'BENCHMARK', tag), exist_ok=True)

    def p(*parts):
        return os.path.join(*parts)

    # --- CCL (Python): reference total in every panel
    time_ccl = _load_column(p(dir_python, 'CCL', tag, 'T_{}_{}.txt'.format(label, number)))

    # --- Python / CPU (PyCCL), decomposed
    time_cpu = _load_column(p(dir_python, 'CPU', tag, 'T_{}_{}.txt'.format(label, number)))
    time_cpu_cosmo = _load_column(
        p(dir_python, 'CPU', tag, 'T_{}_{}_COSMOLOGY.txt'.format(label, number))
    )
    time_cpu_coef = _load_column(
        p(dir_python, 'CPU', tag, 'T_{}_{}_COEFFICIENT.txt'.format(label, number))
    )
    time_cpu_proj = _load_column(
        p(dir_python, 'CPU', tag, 'T_{}_{}_PROJECTION.txt'.format(label, number))
    )

    # --- JAX / GPU
    time_jgpu = _load_column(p(dir_jax, 'GPU', tag, 'T_{}_{}.txt'.format(label, number)))
    time_jgpu_cosmo = _load_column(
        p(dir_jax, 'GPU', tag, 'T_{}_{}_COSMOLOGY.txt'.format(label, number))
    )
    time_jgpu_coef = _load_column(
        p(dir_jax, 'GPU', tag, 'T_{}_{}_COEFFICIENT.txt'.format(label, number))
    )
    time_jgpu_proj = _load_column(
        p(dir_jax, 'GPU', tag, 'T_{}_{}_PROJECTION.txt'.format(label, number))
    )

    # --- JAX / CPU
    time_jcpu = _load_column(p(dir_jax, 'CPU', tag, 'T_{}_{}.txt'.format(label, number)))
    time_jcpu_cosmo = _load_column(
        p(dir_jax, 'CPU', tag, 'T_{}_{}_COSMOLOGY.txt'.format(label, number))
    )
    time_jcpu_coef = _load_column(
        p(dir_jax, 'CPU', tag, 'T_{}_{}_COEFFICIENT.txt'.format(label, number))
    )
    time_jcpu_proj = _load_column(
        p(dir_jax, 'CPU', tag, 'T_{}_{}_PROJECTION.txt'.format(label, number))
    )

    # Length check (all series should match the evaluation grid)
    for name, arr in [
        ('CCL total', time_ccl),
        ('CPU total', time_cpu),
        ('JAX CPU total', time_jcpu),
        ('JAX GPU total', time_jgpu),
    ]:
        n = _ravel_float(arr).size
        if n != COUNT_SIZE:
            raise ValueError('{}: expected length {}, got {}'.format(name, COUNT_SIZE, n))

    # --- Matplotlib / LaTeX
    os.environ['PATH'] = (
        '/pscratch/sd/y/yhzhang/texlive/2025/bin/x86_64-linux:' + os.environ.get('PATH', '')
    )
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 22

    fig, axes = pyplot.subplots(4, 1, figsize=(12, 22), sharex=True)

    ccl_ref_label = r'Python / CCL (total)'
    # In (b)--(d), ``time_ccl`` is the same full cumulative time as in (a), for scale.
    ccl_ref_note = r' \textit{(CCL curve = full cumulative time from (a).)}'

    # (a) Total cumulative time
    plot_panel(
        axes[0],
        EVAL_COUNTS,
        [
            (ccl_ref_label, time_ccl, 'ccl'),
            (r'Python / CPU (PyCCL)', time_cpu, 'cpu'),
            (r'JAX / CPU', time_jcpu, 'jax_cpu'),
            (r'JAX / GPU', time_jgpu, 'jax_gpu'),
        ],
        r'\textbf{(a) Total:} end-to-end cumulative wall time.',
    )

    # (b) Cosmology
    plot_panel(
        axes[1],
        EVAL_COUNTS,
        [
            (ccl_ref_label, time_ccl, 'ccl'),
            (r'Python / CPU', time_cpu_cosmo, 'cpu'),
            (r'JAX / CPU', time_jcpu_cosmo, 'jax_cpu'),
            (r'JAX / GPU', time_jgpu_cosmo, 'jax_gpu'),
        ],
        r'\textbf{(b) Cosmology:} CAMB transfer + nonlinear $P(k)$ build.' + ccl_ref_note,
    )

    # (c) Coefficient
    plot_panel(
        axes[2],
        EVAL_COUNTS,
        [
            (ccl_ref_label, time_ccl, 'ccl'),
            (r'Python / CPU', time_cpu_coef, 'cpu'),
            (r'JAX / CPU', time_jcpu_coef, 'jax_cpu'),
            (r'JAX / GPU', time_jgpu_coef, 'jax_gpu'),
        ],
        r'\textbf{(c) Coefficient:} Limber / coefficient stage.' + ccl_ref_note,
    )

    # (d) Projection
    plot_panel(
        axes[3],
        EVAL_COUNTS,
        [
            (ccl_ref_label, time_ccl, 'ccl'),
            (r'Python / CPU', time_cpu_proj, 'cpu'),
            (r'JAX / CPU', time_jcpu_proj, 'jax_cpu'),
            (r'JAX / GPU', time_jgpu_proj, 'jax_gpu'),
        ],
        r'\textbf{(d) Projection:} spectra / $C_\ell$ assembly.' + ccl_ref_note,
    )

    fig.suptitle(
        r'Benchmark: {} / {} (cores={})'.format(tag, label, number),
        fontsize=24,
        y=1.01,
    )
    fig.tight_layout()

    out_base = os.path.join(dir_plot, 'BENCHMARK', tag, 'BENCHMARK_{}_{}'.format(label, number))
    fig.savefig(out_base + '.pdf', bbox_inches='tight')
    fig.savefig(out_base + '.png', bbox_inches='tight', dpi=200)
    pyplot.close(fig)

    elapsed_min = (time.time() - t0) / 60.0
    print('Saved: {}.pdf / .png'.format(out_base))
    print('Time: {:.2f} minutes'.format(elapsed_min))
    return elapsed_min


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark plot')
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--number', type=int, required=True)
    args = parser.parse_args()
    main(args.tag, args.path, args.label, args.folder, args.number)
