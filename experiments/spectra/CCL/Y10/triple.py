import argparse
import json
import os
import time
from itertools import product

import numpy
import pyccl
import scipy

from limbercloud import Configuration, ProjectPaths
from limbercloud.experiments import (
    add_evaluation_arguments,
    build_checkpoint_counts,
    resolve_sample_count,
)
from limbercloud.validation.assembly import (
    ccl_magnification_bias,
)
from limbercloud.validation.samples import (
    ccl_cosmology_kwargs,
    sampled_parameter_rows,
)


def main(tag, label, folder, number, sample_count=None, fiducial_only=False, sample_table=None, run_id=None, include_fiducial=False, resume=False):
    """
    Calculate the angular power spectra under the triple configuration

    Args:
        tag (str): The tag of the configuration
        label (str): The label of the configuration
        folder (str): The base folder of the dataset
        number (int): Host CPU allocation label for output filenames
        sample_count (int | None): Non-fiducial sample rows; default 0
        fiducial_only (bool): When true, forces zero sampled rows

    Returns:
        duration (float): The duration of the process
    """
    # Start
    start = time.time()
    label = Configuration.parse(label).value
    if resume:
        raise ValueError(
            "Timing entries do not resume HDF5 checkpoints. Resume validated "
            "sample IDs through the shared artifact writer."
        )

    print(f'Tag: {tag}')

    # Runtime paths
    paths = ProjectPaths.from_root(folder)
    data_folder = str(paths.survey_data(tag))
    result_folder = paths.spectrum_results('CCL', tag, run_id=run_id)
    result_folder.mkdir(parents=True, exist_ok=True)

    # Grid
    z1 = 0.0
    z2 = 3.5
    grid_size = 350
    z_grid = numpy.linspace(z1, z2, grid_size + 1)

    # Lens
    lens = numpy.load(os.path.join(data_folder, 'lsst_lens_bins.npy'), allow_pickle=True).item()
    lens_redshift = lens['redshift_range']
    lens_bin_size = len(lens['bins'])

    lens_psi_grid = numpy.zeros((lens_bin_size, grid_size + 1))
    for bin_index in range(lens_bin_size):
        lens_psi_grid[bin_index, :] = numpy.interp(x=z_grid, xp=lens_redshift, fp=lens['bins'][bin_index])
    lens_psi_grid = lens_psi_grid / scipy.integrate.trapezoid(x=z_grid, y=lens_psi_grid, axis=1)[:, numpy.newaxis]

    # Source
    source = numpy.load(os.path.join(data_folder, 'lsst_source_bins.npy'), allow_pickle=True).item()
    source_redshift = source['redshift_range']
    source_bin_size = len(source['bins'])

    source_psi_grid = numpy.zeros((source_bin_size, grid_size + 1))
    for bin_index in range(source_bin_size):
        source_psi_grid[bin_index, :] = numpy.interp(x=z_grid, xp=source_redshift, fp=source['bins'][bin_index])
    source_psi_grid = source_psi_grid / scipy.integrate.trapezoid(x=z_grid, y=source_psi_grid, axis=1)[:, numpy.newaxis]

    # Galaxy
    with paths.config_file('galaxy_bias').open('r') as file:
        galaxy_info = json.load(file)
    galaxy_bias = numpy.array(galaxy_info[tag])

    # Magnification
    with paths.config_file('magnification_bias').open('r') as file:
        magnification_info = json.load(file)
    magnification_bias = ccl_magnification_bias(magnification_info[tag])

    # Alignment
    with paths.config_file('intrinsic_alignment').open('r') as file:
        alignment_info = json.load(file)
    alignment_bias = numpy.array(alignment_info['A'])

    # Multipole
    ell1 = 20
    ell2 = 2000
    ell_size = 20
    ell_grid = numpy.geomspace(ell1, ell2, ell_size + 1)
    ell_data = numpy.sqrt(ell_grid[1:] * (ell_grid[:-1]))

    # Count (sample_count = non-fiducial rows; default 0 is safe)
    count2 = resolve_sample_count(sample_count, fiducial_only)
    count_list = build_checkpoint_counts(count2)
    count_size = int(count_list.size)
    count_targets = {int(count): index for index, count in enumerate(count_list)}

    parameter_rows = sampled_parameter_rows(
        sample_count=count2,
        sample_table=sample_table,
        include_fiducial=include_fiducial,
        fiducial_only=fiducial_only,
    )


    # Time
    time_list = numpy.zeros(count_size)
    time_cell_list = numpy.zeros(count_size)
    time_cosmology_list = numpy.zeros(count_size)

    # Loop
    cell_duration = 0.0
    cosmology_duration = 0.0
    for index in range(int(count_list.max()) if count_list.size else 0):
        t0 = time.time()
        row = parameter_rows[index]
        cosmology = pyccl.Cosmology(**ccl_cosmology_kwargs(row))

        pyccl.gsl_params['NZ_NORM_SPLINE_INTEGRATION'] = False
        pyccl.gsl_params['LENSING_KERNEL_SPLINE_INTEGRATION'] = False

        pyccl.gsl_params['INTEGRATION_GAUSS_KRONROD_POINTS'] = 100
        pyccl.gsl_params['INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS'] = 100

        t1 = time.time()
        cosmology_duration += (t1 - t0)

        c_ccl_ee = numpy.zeros((source_bin_size, source_bin_size, ell_size))
        for (bin_index1, bin_index2) in product(range(source_bin_size), range(source_bin_size)):
            tracer1 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=[z_grid, source_psi_grid[bin_index1, :]], has_shear=True, ia_bias=[z_grid, alignment_bias], use_A_ia=False, n_samples=grid_size + 1)
            tracer2 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=[z_grid, source_psi_grid[bin_index2, :]], has_shear=True, ia_bias=[z_grid, alignment_bias], use_A_ia=False, n_samples=grid_size + 1)
            c_ccl_ee[bin_index1, bin_index2, :] = pyccl.cells.angular_cl(cosmo=cosmology, tracer1=tracer1, tracer2=tracer2, ell=ell_data, p_of_k_a='delta_matter:delta_matter', l_limber=-1, limber_max_error=0.001, limber_integration_method='spline', p_of_k_a_lin='delta_matter:delta_matter', return_meta=False)

        c_ccl_te = numpy.zeros((lens_bin_size, source_bin_size, ell_size))
        for (bin_index1, bin_index2) in product(range(lens_bin_size), range(source_bin_size)):
            tracer1 = pyccl.tracers.NumberCountsTracer(cosmo=cosmology, dndz=[z_grid, lens_psi_grid[bin_index1, :]], bias=[z_grid, galaxy_bias], mag_bias=[z_grid, magnification_bias[bin_index1] * numpy.ones(grid_size + 1)], has_rsd=False, n_samples=grid_size + 1)
            tracer2 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=[z_grid, source_psi_grid[bin_index2, :]], has_shear=True, ia_bias=[z_grid, alignment_bias], use_A_ia=False, n_samples=grid_size + 1)
            c_ccl_te[bin_index1, bin_index2, :] = pyccl.cells.angular_cl(cosmo=cosmology, tracer1=tracer1, tracer2=tracer2, ell=ell_data, p_of_k_a='delta_matter:delta_matter', l_limber=-1, limber_max_error=0.001, limber_integration_method='spline', p_of_k_a_lin='delta_matter:delta_matter', return_meta=False)

        c_ccl_tt = numpy.zeros((lens_bin_size, lens_bin_size, ell_size))
        for (bin_index1, bin_index2) in product(range(lens_bin_size), range(lens_bin_size)):
            tracer1 = pyccl.tracers.NumberCountsTracer(cosmo=cosmology, dndz=[z_grid, lens_psi_grid[bin_index1, :]], bias=[z_grid, galaxy_bias], mag_bias=[z_grid, magnification_bias[bin_index1] * numpy.ones(grid_size + 1)], has_rsd=False, n_samples=grid_size + 1)
            tracer2 = pyccl.tracers.NumberCountsTracer(cosmo=cosmology, dndz=[z_grid, lens_psi_grid[bin_index2, :]], bias=[z_grid, galaxy_bias], mag_bias=[z_grid, magnification_bias[bin_index2] * numpy.ones(grid_size + 1)], has_rsd=False, n_samples=grid_size + 1)
            c_ccl_tt[bin_index1, bin_index2, :] = pyccl.cells.angular_cl(cosmo=cosmology, tracer1=tracer1, tracer2=tracer2, ell=ell_data, p_of_k_a='delta_matter:delta_matter', l_limber=-1, limber_max_error=0.001, limber_integration_method='spline', p_of_k_a_lin='delta_matter:delta_matter', return_meta=False)

        t2 = time.time()
        cell_duration += (t2 - t1)

        if (index + 1) in count_targets:
            count_index = count_targets[index + 1]

            time_cell_list[count_index] = cell_duration
            time_cosmology_list[count_index] = cosmology_duration
            time_list[count_index] = cell_duration + cosmology_duration

    # Save
    numpy.savetxt(os.path.join(result_folder, f'Time_{label}_{number}.txt'), time_list)
    numpy.savetxt(os.path.join(result_folder, f'Time_{label}_{number}_CELL.txt'), time_cell_list)
    numpy.savetxt(os.path.join(result_folder, f'Time_{label}_{number}_COSMOLOGY.txt'), time_cosmology_list)

    # Duration
    end = time.time()
    duration = (end - start) / 60

    # Return
    print(f'Time: {duration:.2f} minutes')
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Triple')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    PARSE.add_argument('--number', type=int, required=True, help='Host CPU allocation label for output filenames')
    add_evaluation_arguments(PARSE)
    # Parse
    ARGS = PARSE.parse_args()
    OUTPUT = main(
        ARGS.tag,
        ARGS.label,
        ARGS.folder,
        ARGS.number,
        sample_count=ARGS.sample_count,
        fiducial_only=ARGS.fiducial_only,
        sample_table=ARGS.sample_table,
        run_id=ARGS.run_id,
        include_fiducial=ARGS.include_fiducial,
        resume=ARGS.resume,
    )
