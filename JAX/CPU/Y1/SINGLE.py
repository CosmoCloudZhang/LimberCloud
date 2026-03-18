import os
import sys
import json
import time
import pyccl
import scipy
import numpy
import logging
import argparse
logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

import jax # noqa: E402
jax.config.update("jax_enable_x64", True)

def main(tag, path, label, folder, number):
    '''
    Calculate the angular power spectra under the single configuration
    
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
    sys.path.insert(0, os.path.join(path, 'JAX'))
    from PROJECTION import SS, SN, NS, NN, TENSOR
    
    # Folder
    data_folder = os.path.join(folder, 'DATA/', tag)
    info_folder = os.path.join(folder, 'INFO/')
    
    jax_folder = os.path.join(folder, 'JAX/')
    os.makedirs(os.path.join(jax_folder, 'CPU/'), exist_ok=True)
    os.makedirs(os.path.join(jax_folder, 'CPU/', tag), exist_ok=True)
    
    # Grid
    z1 = 0.0
    z2 = 3.5
    grid_size = 350
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Source
    source = numpy.load(os.path.join(data_folder, 'lsst_source_bins.npy'), allow_pickle=True).item()
    source_redshift = source['redshift_range']
    source_bin_size = len(source['bins'])
    
    source_psi_grid = numpy.zeros((source_bin_size, grid_size + 1))
    for bin_index in range(source_bin_size):
        source_psi_grid[bin_index, :] = numpy.interp(x=z_grid, xp=source_redshift, fp=source['bins'][bin_index])
    source_psi_grid = source_psi_grid / scipy.integrate.trapezoid(x=z_grid, y=source_psi_grid, axis=1)[:, numpy.newaxis]
    
    # Alignment
    with open(os.path.join(info_folder, 'ALIGNMENT.json'), 'r') as file:
        alignment_info = json.load(file)
    alignment_bias = numpy.array(alignment_info['A'])
    
    # Cosmology
    with open(os.path.join(info_folder, 'COSMOLOGY.json'), 'r') as file:
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
    
    # Amplitude
    amplitude = 3 / 2 * cosmology_info['OMEGA_M'] * (cosmology_info['H'] * 100000 / scipy.constants.c) ** 2
    amplitude_ss = amplitude ** 2 
    amplitude_si = amplitude * alignment_bias
    amplitude_is = alignment_bias * amplitude
    amplitude_ii = alignment_bias ** 2
    
    # Count
    count1 = 100
    count2 = 1000
    count_size = 10
    count_step = int((count2 - count1) // (count_size - 1))
    count_list = numpy.linspace(count1, count2, count_size, dtype='int32')
    
    # Time
    time_list = numpy.zeros(count_size)
    time_cosmology_list = numpy.zeros(count_size)
    time_projection_list = numpy.zeros(count_size)
    
    # Loop
    cosmology_duration = 0.0
    projection_duration = 0.0
    for index in range(count_list.max()):
        t0 = time.time()
        cosmology = pyccl.Cosmology(
            h=numpy.random.uniform(cosmology_info['H'] * 0.9, cosmology_info['H'] * 1.1),
            w0=numpy.random.uniform(cosmology_info['W0'] * 0.9, cosmology_info['W0'] * 1.1),
            wa=numpy.random.uniform(cosmology_info['WA'] * 0.9, cosmology_info['WA'] * 1.1), 
            n_s=numpy.random.uniform(cosmology_info['NS'] * 0.9, cosmology_info['NS'] * 1.1), 
            A_s=numpy.random.uniform(cosmology_info['AS'] * 0.9, cosmology_info['AS'] * 1.1),
            m_nu=numpy.random.uniform(cosmology_info['M_NU'] * 0.9, cosmology_info['M_NU'] * 1.1),  
            Neff=numpy.random.uniform(cosmology_info['N_EFF'] * 0.9, cosmology_info['N_EFF'] * 1.1),
            Omega_b=numpy.random.uniform(cosmology_info['OMEGA_B'] * 0.9, cosmology_info['OMEGA_B'] * 1.1), 
            Omega_k=numpy.random.uniform(cosmology_info['OMEGA_K'] * 0.9, cosmology_info['OMEGA_K'] * 1.1), 
            Omega_c=numpy.random.uniform(cosmology_info['OMEGA_CDM'] * 0.9, cosmology_info['OMEGA_CDM'] * 1.1), 
            mass_split='single', matter_power_spectrum='halofit', transfer_function='boltzmann_camb',
            extra_parameters={'camb': {'kmax': 50, 'lmax': 5000, 'halofit_version': 'mead2020_feedback', 'HMCode_logT_AGN': 7.8}}
        )
        
        pyccl.gsl_params['NZ_NORM_SPLINE_INTEGRATION'] = False
        pyccl.gsl_params['LENSING_KERNEL_SPLINE_INTEGRATION'] = False
        
        pyccl.gsl_params['INTEGRATION_GAUSS_KRONROD_POINTS'] = 100
        pyccl.gsl_params['INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS'] = 100
        
        # Phi
        a_grid = 1 / (1 + z_grid)
        chi_grid = pyccl.background.comoving_radial_distance(cosmo=cosmology, a=a_grid)
        source_phi_grid = source_psi_grid * cosmology.h_over_h0(a=a_grid) * cosmology_info['H'] * 100000 / scipy.constants.c
        
        chi_mesh, ell_mesh = numpy.meshgrid(chi_grid, ell_grid)
        scale_grid = numpy.nan_to_num(numpy.divide(ell_mesh + 1/2, chi_mesh, out=numpy.zeros((ell_size + 1, grid_size + 1)) + numpy.inf, where=chi_mesh > 0))
        
        # Power
        power_grid = numpy.zeros((ell_size + 1, grid_size + 1))
        for grid_index in range(grid_size + 1):
            power_grid[:,grid_index] = pyccl.power.nonlin_matter_power(cosmo=cosmology, k=scale_grid[:,grid_index], a=a_grid[grid_index])
        
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
        
        c_ss.block_until_ready()
        c_si.block_until_ready()
        c_is.block_until_ready()
        c_ii.block_until_ready()
        
        t1 = time.time()
        cosmology_duration += (t1 - t0)
        
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
        
        cell_data_ss.block_until_ready()
        cell_data_si.block_until_ready()
        cell_data_is.block_until_ready()
        cell_data_ii.block_until_ready()
        
        t2 = time.time()
        projection_duration += (t2 - t1)
        
        if (index + 1) % count_step == 0:
            count_index = int((index + 1) // count_step) - 1
            
            time_cosmology_list[count_index] = cosmology_duration
            time_projection_list[count_index] = projection_duration
            time_list[count_index] = projection_duration + cosmology_duration
    
    # Save
    numpy.savetxt(os.path.join(jax_folder, 'CPU/', tag, 'T_{}_{}.txt'.format(label, number)), time_list)
    numpy.savetxt(os.path.join(jax_folder, 'CPU/', tag, 'T_{}_{}_COSMOLOGY.txt'.format(label, number)), time_cosmology_list)
    numpy.savetxt(os.path.join(jax_folder, 'CPU/', tag, 'T_{}_{}_PROJECTION.txt'.format(label, number)), time_projection_list)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Single')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--path', type=str, required=True, help='The path of the project scripts')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    PARSE.add_argument('--number', type=int, required=True, help='The number of cores for parallel computation')
    
    # Parse
    TAG = PARSE.parse_args().tag
    PATH = PARSE.parse_args().path
    LABEL = PARSE.parse_args().label
    FOLDER = PARSE.parse_args().folder
    NUMBER = PARSE.parse_args().number
    
    # Output
    OUTPUT = main(TAG, PATH, LABEL, FOLDER, NUMBER)
