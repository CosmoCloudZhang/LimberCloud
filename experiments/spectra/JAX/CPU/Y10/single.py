import argparse
import json
import logging
import os
import time

import numpy
import pyccl
from scipy.constants import c as speed_of_light
from scipy.integrate import trapezoid

from limbercloud import Configuration, ProjectPaths
from limbercloud.experiments import (
    add_evaluation_arguments,
    build_checkpoint_counts,
    prepare_result_directory,
    require_effective_work,
    resolve_sample_count,
)
from limbercloud.experiments.timing import write_timing_products
from limbercloud.validation.assembly import (
    active_lensing_amplitude,
    component_activity,
    hubble_distance_factor,
)
from limbercloud.validation.cosmology import sample_limber_power
from limbercloud.validation.nuisance import (
    load_alignment,
    require_nuisance_compatibility,
)
from limbercloud.validation.samples import (
    ccl_cosmology_kwargs,
    load_cosmology_table,
    select_rows,
)

logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

# Configure JAX logging before importing the backend.
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)


def main(
    tag,
    label,
    folder,
    sample_count=None,
    sample_table=None,
):
    """
    Calculate the angular power spectra under the single configuration

    Args:
        tag (str): The tag of the configuration
        label (str): The label of the configuration
        folder (str): The base folder of the dataset

    Returns:
        duration (float): The duration of the process
    """
    # Start
    start = time.time()
    label = Configuration.parse(label).value
    sampled_count = require_effective_work(
        sample_count=resolve_sample_count(sample_count),
        sample_table=sample_table,
    )

    print(f"Tag: {tag}")

    # Runtime paths
    from limbercloud.projection.jax_backend import NN, NS, SN, SS, TENSOR

    paths = ProjectPaths.from_root(folder)
    data_folder = str(paths.survey_data(tag))
    result_folder = paths.spectrum_results("JAX", tag, "CPU")

    # Grid
    z1 = 0.0
    z2 = 3.5
    grid_size = 350
    z_grid = numpy.linspace(z1, z2, grid_size + 1)

    # Source
    source = numpy.load(
        os.path.join(data_folder, "lsst_source_bins.npy"), allow_pickle=True
    ).item()
    source_redshift = source["redshift_range"]
    source_bin_size = len(source["bins"])

    source_psi_grid = numpy.zeros((source_bin_size, grid_size + 1))
    for bin_index in range(source_bin_size):
        source_psi_grid[bin_index, :] = numpy.interp(
            x=z_grid, xp=source_redshift, fp=source["bins"][bin_index]
        )
    source_psi_grid = (
        source_psi_grid
        / trapezoid(x=z_grid, y=source_psi_grid, axis=1)[:, numpy.newaxis]
    )

    # Alignment
    alignment_bias, alignment_provenance = load_alignment(
        paths.config_file("intrinsic_alignment"), z_grid
    )
    print(f"Alignment: {alignment_provenance.generating_model_fingerprint}")

    # Cosmology
    with paths.config_file("cosmology").open("r") as file:
        cosmology_info = json.load(file)

    # Multipole
    ell1 = 20
    ell2 = 2000
    ell_size = 20
    ell_grid = numpy.geomspace(ell1, ell2, ell_size + 1)

    # Factor
    factor_ss = numpy.array(
        (1 + 3 / (2 * ell_grid + 1))
        * (1 + 1 / (2 * ell_grid + 1))
        * (1 - 1 / (2 * ell_grid + 1))
        * (1 - 3 / (2 * ell_grid + 1)),
        dtype=numpy.float64,
    )
    factor_si = numpy.array(
        (1 + 3 / (2 * ell_grid + 1))
        * (1 + 1 / (2 * ell_grid + 1))
        * (1 - 1 / (2 * ell_grid + 1))
        * (1 - 3 / (2 * ell_grid + 1)),
        dtype=numpy.float64,
    )
    factor_is = numpy.array(
        (1 + 3 / (2 * ell_grid + 1))
        * (1 + 1 / (2 * ell_grid + 1))
        * (1 - 1 / (2 * ell_grid + 1))
        * (1 - 3 / (2 * ell_grid + 1)),
        dtype=numpy.float64,
    )
    factor_ii = numpy.array(
        (1 + 3 / (2 * ell_grid + 1))
        * (1 + 1 / (2 * ell_grid + 1))
        * (1 - 1 / (2 * ell_grid + 1))
        * (1 - 3 / (2 * ell_grid + 1)),
        dtype=numpy.float64,
    )

    # Amplitude
    amplitude = (
        3
        / 2
        * cosmology_info["OMEGA_M"]
        * (cosmology_info["H"] * 100000 / speed_of_light) ** 2
    )
    amplitude_ss = amplitude**2
    amplitude_si = amplitude * alignment_bias
    amplitude_is = alignment_bias * amplitude
    amplitude_ii = alignment_bias**2

    # Count (sample_count = non-fiducial rows; default 0 is safe)
    count2 = sampled_count
    count_list = build_checkpoint_counts(count2)
    count_size = int(count_list.size)
    count_targets = {int(count): index for index, count in enumerate(count_list)}

    table = load_cosmology_table(sample_table)
    require_nuisance_compatibility(table, alignment_provenance)
    parameter_rows = select_rows(table, [0, *range(1, count2 + 1)])
    activity = component_activity(alignment_bias)

    result_folder = prepare_result_directory(result_folder)

    # Time
    time_list = numpy.zeros(count_size)
    time_cosmology_list = numpy.zeros(count_size)
    time_projection_list = numpy.zeros(count_size)
    time_coefficient_list = numpy.zeros(count_size)

    # Loop
    duration_cosmology = 0.0
    duration_projection = 0.0
    duration_coefficient = 0.0
    for index in range(-1, int(count_list.max()) if count_list.size else 0):
        t0 = time.time()
        row = parameter_rows[index + 1]
        cosmology = pyccl.Cosmology(**ccl_cosmology_kwargs(row))

        amplitude = active_lensing_amplitude(
            float(cosmology["Omega_m"]), float(cosmology["h"])
        )
        amplitude_ss = amplitude**2
        amplitude_si = amplitude * alignment_bias
        amplitude_is = alignment_bias * amplitude
        amplitude_ii = alignment_bias**2

        pyccl.gsl_params["NZ_NORM_SPLINE_INTEGRATION"] = False
        pyccl.gsl_params["LENSING_KERNEL_SPLINE_INTEGRATION"] = False

        pyccl.gsl_params["INTEGRATION_GAUSS_KRONROD_POINTS"] = 100
        pyccl.gsl_params["INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS"] = 100

        # Phi
        a_grid = 1 / (1 + z_grid)
        chi_grid = pyccl.background.comoving_radial_distance(cosmo=cosmology, a=a_grid)
        source_phi_grid = (
            source_psi_grid
            * cosmology.h_over_h0(a=a_grid)
            * hubble_distance_factor(float(cosmology["h"]))
        )

        # The observer ordinate is fixed by the cubic reconstruction; only
        # positive-distance nodes query the matter-power provider.
        power_grid = sample_limber_power(
            cosmology,
            chi_grid,
            a_grid,
            ell_grid,
            power_provider=pyccl.power.nonlin_matter_power,
        )

        t1 = time.time()
        duration_cosmology += t1 - t0

        # Coefficients EE
        if activity["SS"]:
            c_ss = SS.coefficient(
                chi_grid=numpy.array(chi_grid, dtype=numpy.float64),
                power_grid=numpy.array(power_grid * amplitude_ss, dtype=numpy.float64),
                redshift_grid=numpy.array(z_grid, dtype=numpy.float64),
            )

        if activity["SI"]:
            c_si = SN.coefficient(
                chi_grid=numpy.array(chi_grid, dtype=numpy.float64),
                power_grid=numpy.array(power_grid * amplitude_si, dtype=numpy.float64),
                redshift_grid=numpy.array(z_grid, dtype=numpy.float64),
            )

        if activity["IS"]:
            c_is = NS.coefficient(
                chi_grid=numpy.array(chi_grid, dtype=numpy.float64),
                power_grid=numpy.array(power_grid * amplitude_is, dtype=numpy.float64),
                redshift_grid=numpy.array(z_grid, dtype=numpy.float64),
            )

        if activity["II"]:
            c_ii = NN.coefficient(
                chi_grid=numpy.array(chi_grid, dtype=numpy.float64),
                power_grid=numpy.array(power_grid * amplitude_ii, dtype=numpy.float64),
            )

        if activity["SS"]:
            c_ss.block_until_ready()
        if activity["SI"]:
            c_si.block_until_ready()
        if activity["IS"]:
            c_is.block_until_ready()
        if activity["II"]:
            c_ii.block_until_ready()

        t2 = time.time()
        duration_coefficient += t2 - t1

        if activity["SS"]:
            cell_data_ss = TENSOR.spectra(
                factor=numpy.array(factor_ss, dtype=numpy.float64),
                phi_a_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
                phi_b_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
                coefficients=c_ss,
            )
        else:
            cell_data_ss = numpy.zeros((source_bin_size, source_bin_size, ell_size + 1))

        if activity["SI"]:
            cell_data_si = TENSOR.spectra(
                factor=numpy.array(factor_si, dtype=numpy.float64),
                phi_a_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
                phi_b_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
                coefficients=c_si,
            )
        else:
            cell_data_si = numpy.zeros((source_bin_size, source_bin_size, ell_size + 1))

        if activity["IS"]:
            cell_data_is = TENSOR.spectra(
                factor=numpy.array(factor_is, dtype=numpy.float64),
                phi_a_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
                phi_b_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
                coefficients=c_is,
            )
        else:
            cell_data_is = numpy.zeros((source_bin_size, source_bin_size, ell_size + 1))

        if activity["II"]:
            cell_data_ii = TENSOR.spectra(
                factor=numpy.array(factor_ii, dtype=numpy.float64),
                phi_a_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
                phi_b_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
                coefficients=c_ii,
            )
        else:
            cell_data_ii = numpy.zeros((source_bin_size, source_bin_size, ell_size + 1))

        if activity["SS"]:
            cell_data_ss.block_until_ready()
        if activity["SI"]:
            cell_data_si.block_until_ready()
        if activity["IS"]:
            cell_data_is.block_until_ready()
        if activity["II"]:
            cell_data_ii.block_until_ready()

        cell_data_ee = cell_data_ss + cell_data_si + cell_data_is + cell_data_ii
        cell_data_ee.block_until_ready()

        t3 = time.time()
        duration_projection += t3 - t2

        if index < 0:
            fiducial_seconds = (
                duration_projection + duration_cosmology + duration_coefficient
            )
            print(f"Fiducial time: {fiducial_seconds:.2f} seconds")
            fiducial_timings = {
                "": fiducial_seconds,
                "COSMOLOGY": duration_cosmology,
                "PROJECTION": duration_projection,
                "COEFFICIENT": duration_coefficient,
            }
            duration_cosmology = 0.0
            duration_projection = 0.0
            duration_coefficient = 0.0
            continue
        if (index + 1) in count_targets:
            count_index = count_targets[index + 1]

            time_cosmology_list[count_index] = duration_cosmology
            time_projection_list[count_index] = duration_projection
            time_coefficient_list[count_index] = duration_coefficient
            time_list[count_index] = (
                duration_projection + duration_cosmology + duration_coefficient
            )

    # Save fiducial and sampled timing populations separately.
    write_timing_products(
        result_folder,
        label,
        family="JAX",
        sample_table_hash=table.content_hash,
        counts=count_list,
        fiducial=fiducial_timings,
        sampled={
            "": time_list,
            "COSMOLOGY": time_cosmology_list,
            "PROJECTION": time_projection_list,
            "COEFFICIENT": time_coefficient_list,
        },
    )

    # Duration
    end = time.time()
    duration = (end - start) / 60

    # Return
    print(f"Time: {duration:.2f} minutes")
    return duration


if __name__ == "__main__":
    # Input
    PARSE = argparse.ArgumentParser(description="Single")
    PARSE.add_argument(
        "--tag", type=str, required=True, help="The tag of the configuration"
    )
    PARSE.add_argument(
        "--label", type=str, required=True, help="The label of the configuration"
    )
    PARSE.add_argument(
        "--folder", type=str, required=True, help="The base folder of the dataset"
    )
    add_evaluation_arguments(PARSE)

    # Parse
    ARGS = PARSE.parse_args()
    OUTPUT = main(
        ARGS.tag,
        ARGS.label,
        ARGS.folder,
        sample_count=ARGS.sample_count,
        sample_table=ARGS.sample_table,
    )
