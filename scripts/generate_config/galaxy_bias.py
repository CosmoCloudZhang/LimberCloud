import argparse
import json
import time

import numpy

from limbercloud import ProjectPaths
from limbercloud.validation.contract import NUISANCE_COSMOLOGY_POLICY
from limbercloud.validation.cosmology import (
    FIDUCIAL_SOLVER,
    build_effective_cosmology,
    model_fingerprint,
    parameter_hash,
    solver_package_versions,
)


def main(folder):
    '''
    Store the fiducial values of linear galaxy bias

    Arguments:
        folder (str): The base folder of the datasets

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

    # Galaxy
    import pyccl

    galaxy = {}
    tag_list = ['Y1', 'Y10']
    factor = {'Y1': 1.05, 'Y10': 0.95}

    for tag in tag_list:

        growth_factor = pyccl.background.growth_factor(cosmo=cosmology, a=1.0 / (1 + z_grid))
        galaxy[tag] = list(factor[tag] / growth_factor)

    galaxy['_redshift'] = z_grid.tolist()
    galaxy['_redshift_axis'] = {'minimum': z1, 'maximum': z2, 'intervals': grid_size, 'spacing': 'linear'}
    galaxy['_policy'] = NUISANCE_COSMOLOGY_POLICY
    galaxy['_redshift_convention'] = 'factor/D(z) at the fiducial cosmology; not regenerated per sample'
    galaxy['_amplitudes'] = {tag: factor[tag] for tag in tag_list}
    galaxy['_fiducial_input_hash'] = parameter_hash(cosmology_info)
    galaxy['_generating_model_fingerprint'] = model_fingerprint(cosmology_info)
    galaxy['_solver_fingerprint'] = FIDUCIAL_SOLVER.fingerprint()
    galaxy['_solver_settings'] = FIDUCIAL_SOLVER.as_dict()
    galaxy['_package_versions'] = solver_package_versions()

    with paths.config_file('galaxy_bias').open('w') as file:
        json.dump(galaxy, file, indent=4)

    # Duration
    end = time.time()
    duration = (end - start) / 60

    # Return
    print(f'Time: {duration:.2f} minutes')
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Info Galaxy')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the datasets')

    # Parse
    FOLDER = PARSE.parse_args().folder

    # Output
    OUTPUT = main(FOLDER)
