import argparse
import json
import logging
import os
import time

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
    active_lensing_amplitude,
    analytical_magnification_response,
    hubble_distance_factor,
)
from limbercloud.validation.samples import (
    ccl_cosmology_kwargs,
    sampled_parameter_rows,
)

logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

# Configure JAX logging before importing the backend.
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)


def main(tag, label, folder, number, sample_count=None, fiducial_only=False, sample_table=None, run_id=None, include_fiducial=False, resume=False):
    """
    Calculate the angular power spectra under the triple configuration

    Args:
        tag (str): The tag of the configuration
        label (str): The label of the configuration
        folder (str): The base folder of the dataset
        number (int): The number of cores for parallel computation

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
    from limbercloud.projection.jax_backend import NN, NS, SN, SS, TENSOR

    paths = ProjectPaths.from_root(folder)
    data_folder = str(paths.survey_data(tag))
    result_folder = paths.spectrum_results('JAX', tag, 'CPU', run_id=run_id)
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
    magnification_bias = analytical_magnification_response(magnification_info[tag])

    # Alignment
    with paths.config_file('intrinsic_alignment').open('r') as file:
        alignment_info = json.load(file)
    alignment_bias = numpy.array(alignment_info['A'])

    # Cosmology
    with paths.config_file('cosmology').open('r') as file:
        cosmology_info = json.load(file)

    # Multipole
    ell1 = 20
    ell2 = 2000
    ell_size = 20
    ell_grid = numpy.geomspace(ell1, ell2, ell_size + 1)

    # Factor
    factor_ss = numpy.array((1 + 3 / (2 * ell_grid + 1)) * (1 + 1 / (2 * ell_grid + 1)) * (1 - 1 / (2 * ell_grid + 1)) * (1 - 3 / (2 * ell_grid + 1)), dtype=numpy.float64)
    factor_si = numpy.array((1 + 3 / (2 * ell_grid + 1)) * (1 + 1 / (2 * ell_grid + 1)) * (1 - 1 / (2 * ell_grid + 1)) * (1 - 3 / (2 * ell_grid + 1)), dtype=numpy.float64)
    factor_is = numpy.array((1 + 3 / (2 * ell_grid + 1)) * (1 + 1 / (2 * ell_grid + 1)) * (1 - 1 / (2 * ell_grid + 1)) * (1 - 3 / (2 * ell_grid + 1)), dtype=numpy.float64)
    factor_ii = numpy.array((1 + 3 / (2 * ell_grid + 1)) * (1 + 1 / (2 * ell_grid + 1)) * (1 - 1 / (2 * ell_grid + 1)) * (1 - 3 / (2 * ell_grid + 1)), dtype=numpy.float64)

    factor_ms = numpy.sqrt((1 + 3 / (2 * ell_grid + 1)) * (1 + 1 / (2 * ell_grid + 1)) * (1 - 1 / (2 * ell_grid + 1)) * (1 - 3 / (2 * ell_grid + 1))) * ell_grid * (ell_grid + 1) / (ell_grid + 1 / 2) ** 2
    factor_mi = numpy.sqrt((1 + 3 / (2 * ell_grid + 1)) * (1 + 1 / (2 * ell_grid + 1)) * (1 - 1 / (2 * ell_grid + 1)) * (1 - 3 / (2 * ell_grid + 1))) * ell_grid * (ell_grid + 1) / (ell_grid + 1 / 2) ** 2
    factor_gs = numpy.sqrt((1 + 3 / (2 * ell_grid + 1)) * (1 + 1 / (2 * ell_grid + 1)) * (1 - 1 / (2 * ell_grid + 1)) * (1 - 3 / (2 * ell_grid + 1)))
    factor_gi = numpy.sqrt((1 + 3 / (2 * ell_grid + 1)) * (1 + 1 / (2 * ell_grid + 1)) * (1 - 1 / (2 * ell_grid + 1)) * (1 - 3 / (2 * ell_grid + 1)))

    factor_mm = ell_grid ** 2 * (ell_grid + 1) ** 2 / (ell_grid + 1 / 2) ** 4
    factor_mg = ell_grid * (ell_grid + 1) / (ell_grid + 1 / 2) ** 2
    factor_gm = ell_grid * (ell_grid + 1) / (ell_grid + 1 / 2) ** 2
    factor_gg = numpy.ones(ell_size + 1)

    # Amplitude
    amplitude = 3 / 2 * cosmology_info['OMEGA_M'] * (cosmology_info['H'] * 100000 / scipy.constants.c) ** 2

    amplitude_ss = amplitude ** 2
    amplitude_si = amplitude * alignment_bias
    amplitude_is = alignment_bias * amplitude
    amplitude_ii = alignment_bias ** 2

    amplitude_ms = amplitude ** 2
    amplitude_mi = amplitude * alignment_bias
    amplitude_gs = galaxy_bias * amplitude
    amplitude_gi = galaxy_bias * alignment_bias

    amplitude_mm = amplitude ** 2
    amplitude_mg = amplitude * galaxy_bias
    amplitude_gm = galaxy_bias * amplitude
    amplitude_gg = galaxy_bias ** 2

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
    time_cosmology_list = numpy.zeros(count_size)
    time_projection_list = numpy.zeros(count_size)
    time_coefficient_list = numpy.zeros(count_size)

    # Loop
    duration_cosmology = 0.0
    duration_projection = 0.0
    duration_coefficient = 0.0
    for index in range(int(count_list.max()) if count_list.size else 0):
        t0 = time.time()
        row = parameter_rows[index]
        cosmology = pyccl.Cosmology(**ccl_cosmology_kwargs(row))

        amplitude = active_lensing_amplitude(float(cosmology["Omega_m"]), float(cosmology["h"]))
        amplitude_ss = amplitude ** 2
        amplitude_si = amplitude * alignment_bias
        amplitude_is = alignment_bias * amplitude
        amplitude_ii = alignment_bias ** 2
        amplitude_ms = amplitude ** 2
        amplitude_mi = amplitude * alignment_bias
        amplitude_gs = galaxy_bias * amplitude
        amplitude_gi = galaxy_bias * alignment_bias
        amplitude_mm = amplitude ** 2
        amplitude_mg = amplitude * galaxy_bias
        amplitude_gm = galaxy_bias * amplitude
        amplitude_gg = galaxy_bias ** 2

        pyccl.gsl_params['NZ_NORM_SPLINE_INTEGRATION'] = False
        pyccl.gsl_params['LENSING_KERNEL_SPLINE_INTEGRATION'] = False

        pyccl.gsl_params['INTEGRATION_GAUSS_KRONROD_POINTS'] = 100
        pyccl.gsl_params['INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS'] = 100

        # Phi
        a_grid = 1 / (1 + z_grid)
        chi_grid = pyccl.background.comoving_radial_distance(cosmo=cosmology, a=a_grid)
        lens_phi_grid = lens_psi_grid * cosmology.h_over_h0(a=a_grid) * hubble_distance_factor(float(cosmology["h"]))
        source_phi_grid = source_psi_grid * cosmology.h_over_h0(a=a_grid) * hubble_distance_factor(float(cosmology["h"]))

        chi_mesh, ell_mesh = numpy.meshgrid(chi_grid, ell_grid)
        scale_grid = numpy.nan_to_num(numpy.divide(ell_mesh + 1/2, chi_mesh, out=numpy.zeros((ell_size + 1, grid_size + 1)) + numpy.inf, where=chi_mesh > 0))

        # Power
        power_grid = numpy.zeros((ell_size + 1, grid_size + 1))
        for grid_index in range(grid_size + 1):
            power_grid[:,grid_index] = pyccl.power.nonlin_matter_power(cosmo=cosmology, k=scale_grid[:,grid_index], a=a_grid[grid_index])

        t1 = time.time()
        duration_cosmology += (t1 - t0)

        # Coefficients EE
        c_ss = SS.coefficient(
            chi_grid=numpy.array(chi_grid, dtype=numpy.float64),
            power_grid=numpy.array(power_grid * amplitude_ss, dtype=numpy.float64),
            redshift_grid=numpy.array(z_grid, dtype=numpy.float64)
        )

        c_si = SN.coefficient(
            chi_grid=numpy.array(chi_grid, dtype=numpy.float64),
            power_grid=numpy.array(power_grid * amplitude_si, dtype=numpy.float64),
            redshift_grid=numpy.array(z_grid, dtype=numpy.float64)
        )

        c_is = NS.coefficient(
            chi_grid=numpy.array(chi_grid, dtype=numpy.float64),
            power_grid=numpy.array(power_grid * amplitude_is, dtype=numpy.float64),
            redshift_grid=numpy.array(z_grid, dtype=numpy.float64)
        )

        c_ii = NN.coefficient(
            chi_grid=numpy.array(chi_grid, dtype=numpy.float64),
            power_grid=numpy.array(power_grid * amplitude_ii, dtype=numpy.float64)
        )

        # Coefficient TE
        c_ms = SS.coefficient(
            chi_grid=numpy.array(chi_grid, dtype=numpy.float64),
            power_grid=numpy.array(power_grid * amplitude_ms, dtype=numpy.float64),
            redshift_grid=numpy.array(z_grid, dtype=numpy.float64)
        )

        c_mi = SN.coefficient(
            chi_grid=numpy.array(chi_grid, dtype=numpy.float64),
            power_grid=numpy.array(power_grid * amplitude_mi, dtype=numpy.float64),
            redshift_grid=numpy.array(z_grid, dtype=numpy.float64)
        )

        c_gs = NS.coefficient(
            chi_grid=numpy.array(chi_grid, dtype=numpy.float64),
            power_grid=numpy.array(power_grid * amplitude_gs, dtype=numpy.float64),
            redshift_grid=numpy.array(z_grid, dtype=numpy.float64)
        )

        c_gi = NN.coefficient(
            chi_grid=numpy.array(chi_grid, dtype=numpy.float64),
            power_grid=numpy.array(power_grid * amplitude_gi, dtype=numpy.float64)
        )

        # Coefficient TT
        c_mm = SS.coefficient(
            chi_grid=numpy.array(chi_grid, dtype=numpy.float64),
            power_grid=numpy.array(power_grid * amplitude_mm, dtype=numpy.float64),
            redshift_grid=numpy.array(z_grid, dtype=numpy.float64)
        )

        c_mg = SN.coefficient(
            chi_grid=numpy.array(chi_grid, dtype=numpy.float64),
            power_grid=numpy.array(power_grid * amplitude_mg, dtype=numpy.float64),
            redshift_grid=numpy.array(z_grid, dtype=numpy.float64)
        )

        c_gm = NS.coefficient(
            chi_grid=numpy.array(chi_grid, dtype=numpy.float64),
            power_grid=numpy.array(power_grid * amplitude_gm, dtype=numpy.float64),
            redshift_grid=numpy.array(z_grid, dtype=numpy.float64)
        )

        c_gg = NN.coefficient(
            chi_grid=numpy.array(chi_grid, dtype=numpy.float64),
            power_grid=numpy.array(power_grid * amplitude_gg, dtype=numpy.float64)
        )

        c_ss.block_until_ready()
        c_si.block_until_ready()
        c_is.block_until_ready()
        c_ii.block_until_ready()

        c_ms.block_until_ready()
        c_mi.block_until_ready()
        c_gs.block_until_ready()
        c_gi.block_until_ready()

        c_mm.block_until_ready()
        c_mg.block_until_ready()
        c_gm.block_until_ready()
        c_gg.block_until_ready()

        t2 = time.time()
        duration_coefficient += (t2 - t1)

        # Cell EE
        cell_data_ss = TENSOR.spectra(
            factor=numpy.array(factor_ss, dtype=numpy.float64),
            phi_a_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
            phi_b_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
            coefficients=c_ss
        )

        cell_data_si = TENSOR.spectra(
            factor=numpy.array(factor_si, dtype=numpy.float64),
            phi_a_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
            phi_b_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
            coefficients=c_si
        )

        cell_data_is = TENSOR.spectra(
            factor=numpy.array(factor_is, dtype=numpy.float64),
            phi_a_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
            phi_b_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
            coefficients=c_is
        )

        cell_data_ii = TENSOR.spectra(
            factor=numpy.array(factor_ii, dtype=numpy.float64),
            phi_a_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
            phi_b_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
            coefficients=c_ii
        )

        # Cell TE
        cell_data_ms = TENSOR.spectra(
            factor=numpy.array(factor_ms, dtype=numpy.float64),
            phi_a_grid=numpy.array(lens_phi_grid * magnification_bias[:, numpy.newaxis], dtype=numpy.float64),
            phi_b_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
            coefficients=c_ms
        )

        cell_data_mi = TENSOR.spectra(
            factor=numpy.array(factor_mi, dtype=numpy.float64),
            phi_a_grid=numpy.array(lens_phi_grid * magnification_bias[:, numpy.newaxis], dtype=numpy.float64),
            phi_b_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
            coefficients=c_mi
        )

        cell_data_gs = TENSOR.spectra(
            factor=numpy.array(factor_gs, dtype=numpy.float64),
            phi_a_grid=numpy.array(lens_phi_grid, dtype=numpy.float64),
            phi_b_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
            coefficients=c_gs
        )

        cell_data_gi = TENSOR.spectra(
            factor=numpy.array(factor_gi, dtype=numpy.float64),
            phi_a_grid=numpy.array(lens_phi_grid, dtype=numpy.float64),
            phi_b_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
            coefficients=c_gi
        )

        # Cell TT
        cell_data_mm = TENSOR.spectra(
            factor=numpy.array(factor_mm, dtype=numpy.float64),
            phi_a_grid=numpy.array(lens_phi_grid * magnification_bias[:, numpy.newaxis], dtype=numpy.float64),
            phi_b_grid=numpy.array(lens_phi_grid * magnification_bias[:, numpy.newaxis], dtype=numpy.float64),
            coefficients=c_mm
        )

        cell_data_mg = TENSOR.spectra(
            factor=numpy.array(factor_mg, dtype=numpy.float64),
            phi_a_grid=numpy.array(lens_phi_grid * magnification_bias[:, numpy.newaxis], dtype=numpy.float64),
            phi_b_grid=numpy.array(lens_phi_grid, dtype=numpy.float64),
            coefficients=c_mg
        )

        cell_data_gm = TENSOR.spectra(
            factor=numpy.array(factor_gm, dtype=numpy.float64),
            phi_a_grid=numpy.array(lens_phi_grid, dtype=numpy.float64),
            phi_b_grid=numpy.array(lens_phi_grid * magnification_bias[:, numpy.newaxis], dtype=numpy.float64),
            coefficients=c_gm
        )

        cell_data_gg = TENSOR.spectra(
            factor=numpy.array(factor_gg, dtype=numpy.float64),
            phi_a_grid=numpy.array(lens_phi_grid, dtype=numpy.float64),
            phi_b_grid=numpy.array(lens_phi_grid, dtype=numpy.float64),
            coefficients=c_gg
        )

        cell_data_ss.block_until_ready()
        cell_data_si.block_until_ready()
        cell_data_is.block_until_ready()
        cell_data_ii.block_until_ready()

        cell_data_ms.block_until_ready()
        cell_data_mi.block_until_ready()
        cell_data_gs.block_until_ready()
        cell_data_gi.block_until_ready()

        cell_data_mm.block_until_ready()
        cell_data_mg.block_until_ready()
        cell_data_gm.block_until_ready()
        cell_data_gg.block_until_ready()

        cell_data_ee = cell_data_ss + cell_data_si + cell_data_is + cell_data_ii
        cell_data_ee.block_until_ready()
        cell_data_te = cell_data_ms + cell_data_mi + cell_data_gs + cell_data_gi
        cell_data_te.block_until_ready()
        cell_data_tt = cell_data_mm + cell_data_mg + cell_data_gm + cell_data_gg
        cell_data_tt.block_until_ready()

        t3 = time.time()
        duration_projection += (t3 - t2)

        if (index + 1) in count_targets:
            count_index = count_targets[index + 1]

            time_cosmology_list[count_index] = duration_cosmology
            time_projection_list[count_index] = duration_projection
            time_coefficient_list[count_index] = duration_coefficient
            time_list[count_index] = duration_projection + duration_cosmology + duration_coefficient

    # Save
    numpy.savetxt(os.path.join(result_folder, f'Time_{label}_{number}.txt'), time_list)
    numpy.savetxt(os.path.join(result_folder, f'Time_{label}_{number}_COSMOLOGY.txt'), time_cosmology_list)
    numpy.savetxt(os.path.join(result_folder, f'Time_{label}_{number}_PROJECTION.txt'), time_projection_list)
    numpy.savetxt(os.path.join(result_folder, f'Time_{label}_{number}_COEFFICIENT.txt'), time_coefficient_list)

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
