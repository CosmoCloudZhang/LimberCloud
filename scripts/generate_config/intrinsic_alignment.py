import argparse
import json
import time

import numpy

from limbercloud import ProjectPaths
from limbercloud.validation.contract import (
    A_IA_AMPLITUDE,
    ETA_IA_ADOPTED_VALUE,
    ETA_IA_HISTORICAL_GENERATOR_VALUE,
    ETA_IA_SOURCE,
    ETA_IA_STATUS_ADOPTED,
    ETA_IA_STATUS_DIAGNOSTIC,
    IA_CRITICAL_DENSITY_CONSTANT,
    IA_DENSITY_CONVENTION,
    IA_GROWTH_NORMALIZATION,
    IA_PIVOT_REDSHIFT,
    NUISANCE_COSMOLOGY_POLICY,
)
from limbercloud.validation.cosmology import (
    FIDUCIAL_SOLVER,
    build_effective_cosmology,
    model_fingerprint,
    parameter_hash,
    solver_package_versions,
)


def main(folder, eta_ia=None):
    '''
    Store the fiducial values of intrinsic alignment

    Arguments:
        folder (str): The base folder of the datasets
        eta_ia (float | None): Redshift slope. Omitting it adopts the campaign
            value 0.0. Any other value is written as an explicitly diagnostic
            array and is refused by accepted-campaign readers.

    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()

    # Path
    paths = ProjectPaths.from_root(folder)
    paths.config.mkdir(parents=True, exist_ok=True)

    # Cosmology
    with paths.config_file('cosmology').open('r') as file:
        cosmology_info = json.load(file)

    cosmology = build_effective_cosmology(cosmology_info)

    # Redshift
    z1 = 0.0
    z2 = 3.5
    grid_size = 350
    z_grid = numpy.linspace(z1, z2, grid_size + 1)

    # Alignment law. eta is the adopted SRD slope; A and the pivot are distinct
    # quantities that both happen to equal 0.5 and are not changed with it.
    z_pivot = IA_PIVOT_REDSHIFT
    a_pivot = A_IA_AMPLITUDE
    if eta_ia is None:
        eta_pivot = ETA_IA_ADOPTED_VALUE
        eta_decision = ETA_IA_STATUS_ADOPTED
    else:
        eta_pivot = float(eta_ia)
        eta_decision = (
            ETA_IA_STATUS_ADOPTED
            if eta_pivot == ETA_IA_ADOPTED_VALUE
            else ETA_IA_STATUS_DIAGNOSTIC
        )

    import pyccl

    constant = 5e-14 / numpy.square(cosmology_info['H'])
    growth = pyccl.background.growth_factor(cosmo=cosmology, a=1.0 / (1.0 + z_grid))
    rho_m = pyccl.background.rho_x(cosmo=cosmology, a=1.0, species='matter', is_comoving=True)
    a_grid = - constant * rho_m / growth * a_pivot * numpy.power((1 + z_grid) / (1 + z_pivot), eta_pivot)

    alignment_info = {
        'A': a_grid.tolist(),
        'redshift': z_grid.tolist(),
        'redshift_axis': {'minimum': z1, 'maximum': z2, 'intervals': grid_size, 'spacing': 'linear'},
        'eta_pivot': eta_pivot,
        'eta_decision': eta_decision,
        'eta_source': ETA_IA_SOURCE,
        'eta_historical_generator_value': ETA_IA_HISTORICAL_GENERATOR_VALUE,
        'z_pivot': z_pivot,
        'a_pivot': a_pivot,
        'effective_law': (
            'A(z) = -C1 * rho_m(0) / D(z) * A_IA * ((1+z)/(1+z_pivot))**eta_pivot; '
            'signed NLA with comoving present-day matter density'
        ),
        'density_convention': IA_DENSITY_CONVENTION,
        'growth_normalization': IA_GROWTH_NORMALIZATION,
        'C1': IA_CRITICAL_DENSITY_CONSTANT,
        'nuisance_cosmology_policy': NUISANCE_COSMOLOGY_POLICY,
        'fiducial_input_hash': parameter_hash(cosmology_info),
        'generating_model_fingerprint': model_fingerprint(cosmology_info),
        'solver_fingerprint': FIDUCIAL_SOLVER.fingerprint(),
        'solver_settings': FIDUCIAL_SOLVER.as_dict(),
        'package_versions': solver_package_versions(),
    }

    with paths.config_file('intrinsic_alignment').open('w') as file:
        json.dump(alignment_info, file, indent=4)

    # Duration
    end = time.time()
    duration = (end - start) / 60

    # Return
    print(f'Time: {duration:.2f} minutes')
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Info Alignment')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the datasets')
    PARSE.add_argument('--eta-ia', type=float, default=None, help='Redshift slope. Omit to adopt the campaign value 0.0; any other value is written as a diagnostic array.')

    # Parse
    ARGS = PARSE.parse_args()

    # Output
    OUTPUT = main(ARGS.folder, eta_ia=ARGS.eta_ia)
