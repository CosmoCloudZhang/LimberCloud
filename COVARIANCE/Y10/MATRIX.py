import os
import json
import time
import numpy
import scipy
import pyccl
import argparse
from astropy import table
from itertools import product


def main(tag, folder):
    '''
    Calculate information for covariance matrix of angular power spectra
    
    Arguments:
        tag (str): The tag of the configuration
        folder (str): The base folder of the dataset
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    print('Tag: {}'.format(tag))
    
    # Path
    data_folder = os.path.join(folder, 'DATA/')
    info_folder = os.path.join(folder, 'INFO/')
    covariance_folder = os.path.join(folder, 'COVARIANCE/')
    os.makedirs(os.path.join(covariance_folder, '{}/'.format(tag)), exist_ok = True)
    
    # Source
    source = numpy.load(os.path.join(data_folder, '{}/lsst_source_bins.npy'.format(tag)), allow_pickle=True).item()
    source_bin_size = len(source['bins'])
    source_redshift = source['redshift_range']
    
    # Define the redshift grid
    grid_size = 350
    z1 = source_redshift.min()
    z2 = source_redshift.max()
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    source_psi_grid = numpy.zeros((source_bin_size, grid_size + 1))
    for m in range(source_bin_size):
        source_psi_grid[m, :] = numpy.interp(x=z_grid, xp=source_redshift, fp=source['bins'][m])
    source_psi_grid = source_psi_grid / scipy.integrate.trapezoid(x=z_grid, y=source_psi_grid, axis=1)[:, numpy.newaxis]
    
    table_source = table.Table()
    table_source['redshift'] = z_grid
    for m in range(source_bin_size):
        table_source['n_{}(z)'.format(m + 1)] = source_psi_grid[m, :]
    table_source.write(os.path.join(covariance_folder, '{}/SOURCE.ascii'.format(tag)), overwrite = True, format = 'ascii')
    
    # Lens
    lens = numpy.load(os.path.join(data_folder, '{}/lsst_lens_bins.npy'.format(tag)), allow_pickle=True).item()
    lens_bin_size = len(lens['bins'])
    lens_redshift = lens['redshift_range']
    
    # Define the redshift grid
    grid_size = 350
    z1 = lens_redshift.min()
    z2 = lens_redshift.max()
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    lens_psi_grid = numpy.zeros((lens_bin_size, grid_size + 1))
    for m in range(lens_bin_size):
        lens_psi_grid[m, :] = numpy.interp(x=z_grid, xp=lens_redshift, fp=lens['bins'][m])
    lens_psi_grid = lens_psi_grid / scipy.integrate.trapezoid(x=z_grid, y=lens_psi_grid, axis=1)[:, numpy.newaxis]
    
    table_lens = table.Table()
    table_lens['redshift'] = z_grid
    for m in range(lens_bin_size):
        table_lens['n_{}(z)'.format(m + 1)] = lens_psi_grid[m, :]
    table_lens.write(os.path.join(covariance_folder, '{}/LENS.ascii'.format(tag)), overwrite = True, format = 'ascii')
    
    # Alignment
    with open(os.path.join(info_folder, 'ALIGNMENT.json'), 'r') as file:
        alignment_info = json.load(file)
    alignment_bias = numpy.array(alignment_info['A'])
    
    table_alignment = table.Table()
    table_alignment['redshift'] = z_grid
    for m in range(source_bin_size):
        table_alignment['A_{}(z)'.format(m + 1)] = alignment_bias
    table_alignment.write(os.path.join(covariance_folder, '{}/ALIGNMENT.ascii'.format(tag)), overwrite = True, format = 'ascii')
    
    # Galaxy
    with open(os.path.join(info_folder, 'GALAXY.json'), 'r') as file:
        galaxy_info = json.load(file)
    galaxy_bias = numpy.array(galaxy_info[tag])
    
    table_galaxy = table.Table()
    table_galaxy['redshift'] = z_grid
    for m in range(lens_bin_size):
        table_galaxy['b_{}(z)'.format(m + 1)] = galaxy_bias
    table_galaxy.write(os.path.join(covariance_folder, '{}/GALAXY.ascii'.format(tag)), overwrite = True, format = 'ascii')
    
    # Magnification
    with open(os.path.join(info_folder, 'MAGNIFICATION.json'), 'r') as file:
        magnification_info = json.load(file)
    magnification_bias = numpy.array(magnification_info[tag])
    
    table_magnification = table.Table()
    table_magnification['redshift'] = z_grid
    for m in range(lens_bin_size):
        table_magnification['m_{}(z)'.format(m + 1)] = magnification_bias[m] * numpy.ones(grid_size + 1)
    table_magnification.write(os.path.join(covariance_folder, '{}/MAGNIFICATION.ascii'.format(tag)), overwrite = True, format = 'ascii')
    
    # Cosmology
    with open(os.path.join(info_folder, 'COSMOLOGY.json'), 'r') as file:
        cosmology_info = json.load(file)
    
    cosmology = pyccl.Cosmology(
        h=cosmology_info['H'],
        w0=cosmology_info['W0'],
        wa=cosmology_info['WA'], 
        n_s=cosmology_info['NS'], 
        A_s=cosmology_info['AS'],
        m_nu=cosmology_info['M_NU'],  
        Neff=cosmology_info['N_EFF'],
        Omega_b=cosmology_info['OMEGA_B'], 
        Omega_k=cosmology_info['OMEGA_K'], 
        Omega_c=cosmology_info['OMEGA_CDM'], 
        mass_split='single', matter_power_spectrum='halofit', transfer_function='boltzmann_camb',
        extra_parameters={'camb': {'kmax': 50, 'lmax': 5000, 'halofit_version': 'mead2020_feedback', 'HMCode_logT_AGN': 7.8}}
    )
    
    pyccl.gsl_params['NZ_NORM_SPLINE_INTEGRATION'] = False
    pyccl.gsl_params['LENSING_KERNEL_SPLINE_INTEGRATION'] = False
    
    pyccl.gsl_params['INTEGRATION_GAUSS_KRONROD_POINTS'] = 100
    pyccl.gsl_params['INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS'] = 100
    
    # Multipole
    ell1 = 20
    ell2 = 2000
    ell_size = 100
    ell_grid = numpy.geomspace(ell1, ell2, ell_size + 1)
    
    # Cell EE
    ell = numpy.zeros((source_bin_size, source_bin_size, ell_size + 1), dtype=numpy.float32)
    index1 = numpy.zeros((source_bin_size, source_bin_size, ell_size + 1), dtype=numpy.int32)
    index2 = numpy.zeros((source_bin_size, source_bin_size, ell_size + 1), dtype=numpy.int32)
    value = numpy.zeros((source_bin_size, source_bin_size, ell_size + 1), dtype=numpy.float32)
    
    for (i, j) in product(range(source_bin_size), range(source_bin_size)):
        tracer1 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=[z_grid, source_psi_grid[i, :]], has_shear=True, ia_bias=[z_grid, alignment_bias], use_A_ia=False, n_samples=grid_size + 1)
        tracer2 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=[z_grid, source_psi_grid[j, :]], has_shear=True, ia_bias=[z_grid, alignment_bias], use_A_ia=False, n_samples=grid_size + 1)
        value[i, j, :] = pyccl.cells.angular_cl(cosmo=cosmology, tracer1=tracer1, tracer2=tracer2, ell=ell_grid, p_of_k_a='delta_matter:delta_matter', l_limber=-1, limber_max_error=0.001, limber_integration_method='spline', p_of_k_a_lin='delta_matter:delta_matter', return_meta=False)
        
        index1[i, j, :] = i + 1
        index2[i, j, :] = j + 1
        ell[i, j, :] = ell_grid
    
    table_cell = table.Table()
    order = numpy.argsort(ell.flatten())
    table_cell['ell'] = ell.flatten()[order]
    table_cell['tomo_i'] = index1.flatten()[order]
    table_cell['tomo_j'] = index2.flatten()[order]
    table_cell['Cell_kappakappa'] = value.flatten()[order]
    table_cell.write(os.path.join(covariance_folder, '{}/Cell_kappakappa.ascii'.format(tag)), overwrite = True, format = 'ascii')
    
    # Cell TE
    ell = numpy.zeros((lens_bin_size, source_bin_size, ell_size + 1), dtype=numpy.float32)
    index1 = numpy.zeros((lens_bin_size, source_bin_size, ell_size + 1), dtype=numpy.int32)
    index2 = numpy.zeros((lens_bin_size, source_bin_size, ell_size + 1), dtype=numpy.int32)
    value = numpy.zeros((lens_bin_size, source_bin_size, ell_size + 1), dtype=numpy.float32)
    
    for (i, j) in product(range(lens_bin_size), range(source_bin_size)):
        tracer1 = pyccl.tracers.NumberCountsTracer(cosmo=cosmology, dndz=[z_grid, lens_psi_grid[i, :]], bias=[z_grid, galaxy_bias], mag_bias=[z_grid, magnification_bias[i] * numpy.ones(grid_size + 1)], has_rsd=False, n_samples=grid_size + 1)
        tracer2 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=[z_grid, source_psi_grid[j, :]], has_shear=True, ia_bias=[z_grid, alignment_bias], use_A_ia=False, n_samples=grid_size + 1)
        value[i, j, :] = pyccl.cells.angular_cl(cosmo=cosmology, tracer1=tracer1, tracer2=tracer2, ell=ell_grid, p_of_k_a='delta_matter:delta_matter', l_limber=-1, limber_max_error=0.001, limber_integration_method='spline', p_of_k_a_lin='delta_matter:delta_matter', return_meta=False)
        
        index1[i, j, :] = i + 1
        index2[i, j, :] = j + 1
        ell[i, j, :] = ell_grid
    
    table_cell = table.Table()
    order = numpy.argsort(ell.flatten())
    table_cell['ell'] = ell.flatten()[order]
    table_cell['tomo_i'] = index1.flatten()[order]
    table_cell['tomo_j'] = index2.flatten()[order]
    table_cell['Cell_gkappa'] = value.flatten()[order]
    table_cell.write(os.path.join(covariance_folder, '{}/Cell_gkappa.ascii'.format(tag)), overwrite = True, format = 'ascii')
    
    # Cell TT
    ell = numpy.zeros((lens_bin_size, lens_bin_size, ell_size + 1), dtype=numpy.float32)
    index1 = numpy.zeros((lens_bin_size, lens_bin_size, ell_size + 1), dtype=numpy.int32)
    index2 = numpy.zeros((lens_bin_size, lens_bin_size, ell_size + 1), dtype=numpy.int32)
    value = numpy.zeros((lens_bin_size, lens_bin_size, ell_size + 1), dtype=numpy.float32)
    
    for (i, j) in product(range(lens_bin_size), range(lens_bin_size)):
        tracer1 = pyccl.tracers.NumberCountsTracer(cosmo=cosmology, dndz=[z_grid, lens_psi_grid[i, :]], bias=[z_grid, galaxy_bias], mag_bias=[z_grid, magnification_bias[i] * numpy.ones(grid_size + 1)], has_rsd=False, n_samples=grid_size + 1)
        tracer2 = pyccl.tracers.NumberCountsTracer(cosmo=cosmology, dndz=[z_grid, lens_psi_grid[j, :]], bias=[z_grid, galaxy_bias], mag_bias=[z_grid, magnification_bias[j] * numpy.ones(grid_size + 1)], has_rsd=False, n_samples=grid_size + 1)
        value[i, j, :] = pyccl.cells.angular_cl(cosmo=cosmology, tracer1=tracer1, tracer2=tracer2, ell=ell_grid, p_of_k_a='delta_matter:delta_matter', l_limber=-1, limber_max_error=0.001, limber_integration_method='spline', p_of_k_a_lin='delta_matter:delta_matter', return_meta=False)
        
        index1[i, j, :] = i + 1
        index2[i, j, :] = j + 1
        ell[i, j, :] = ell_grid
    
    table_cell = table.Table()
    order = numpy.argsort(ell.flatten())
    table_cell['ell'] = ell.flatten()[order]
    table_cell['tomo_i'] = index1.flatten()[order]
    table_cell['tomo_j'] = index2.flatten()[order]
    table_cell['Cell_gg'] = value.flatten()[order]
    table_cell.write(os.path.join(covariance_folder, '{}/Cell_gg.ascii'.format(tag)), overwrite = True, format = 'ascii')
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Covariance')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)