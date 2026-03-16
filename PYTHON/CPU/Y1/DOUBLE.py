import os
import sys
import json
import time
import pyccl
import scipy
import numpy
import argparse

def main(tag, path, label, folder):
    '''
    Calculate the angular power spectra under the double configuration
    
    Arguments:
        tag (str): The tag of the configuration
        path (str): The path of the project scripts
        label (str): The label of the configuration
        folder (str): The base folder of the dataset
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    print('Tag: {}'.format(tag))
    
    # Path
    sys.path.insert(0, os.path.join(path, 'PYTHON'))
    from PROJECTION import SS, SN, NS, NN
    
    # Folder
    data_folder = os.path.join(folder, 'DATA/', tag)
    info_folder = os.path.join(folder, 'INFO/')
    
    python_folder = os.path.join(folder, 'PYTHON/')
    os.makedirs(os.path.join(python_folder, 'CPU/'), exist_ok = True)
    os.makedirs(os.path.join(python_folder, 'CPU/', tag), exist_ok = True)
    
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
    with open(os.path.join(info_folder, 'GALAXY.json'), 'r') as file:
        galaxy_info = json.load(file)
    galaxy_bias = numpy.array(galaxy_info[tag])
    
    # Magnification
    with open(os.path.join(info_folder, 'MAGNIFICATION.json'), 'r') as file:
        magnification_info = json.load(file)
    magnification_bias = 5 * numpy.array(magnification_info[tag]) - 2
    
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
    amplitude_ms = amplitude ** 2
    amplitude_mi = amplitude * alignment_bias
    amplitude_gs = galaxy_bias * amplitude
    amplitude_gi = galaxy_bias * alignment_bias
    
    amplitude_mm = amplitude ** 2
    amplitude_mg = amplitude * galaxy_bias
    amplitude_gm = galaxy_bias * amplitude
    amplitude_gg = galaxy_bias ** 2
    
    # Number
    count1 = 10
    count2 = 100
    count_size = 10
    count_list = numpy.linspace(count1, count2, count_size, dtype = 'int32')
    
    # Time
    time_list = numpy.zeros(count_size)
    for index, count in enumerate(count_list):
        print(index, count)
        begin = time.time()
        for _ in range(count):
            cosmology = pyccl.Cosmology(
                h=numpy.random.uniform(cosmology_info['H'] * 0.95, cosmology_info['H'] * 1.05),
                w0=numpy.random.uniform(cosmology_info['W0'] * 0.95, cosmology_info['W0'] * 1.05),
                wa=numpy.random.uniform(cosmology_info['WA'] * 0.95, cosmology_info['WA'] * 1.05), 
                n_s=numpy.random.uniform(cosmology_info['NS'] * 0.95, cosmology_info['NS'] * 1.05), 
                A_s=numpy.random.uniform(cosmology_info['AS'] * 0.95, cosmology_info['AS'] * 1.05),
                m_nu=numpy.random.uniform(cosmology_info['M_NU'] * 0.95, cosmology_info['M_NU'] * 1.05),  
                Neff=numpy.random.uniform(cosmology_info['N_EFF'] * 0.95, cosmology_info['N_EFF'] * 1.05),
                Omega_b=numpy.random.uniform(cosmology_info['OMEGA_B'] * 0.95, cosmology_info['OMEGA_B'] * 1.05), 
                Omega_k=numpy.random.uniform(cosmology_info['OMEGA_K'] * 0.95, cosmology_info['OMEGA_K'] * 1.05), 
                Omega_c=numpy.random.uniform(cosmology_info['OMEGA_CDM'] * 0.95, cosmology_info['OMEGA_CDM'] * 1.05), 
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
            lens_phi_grid = lens_psi_grid * cosmology.h_over_h0(a=a_grid) * cosmology_info['H'] * 100000 / scipy.constants.c
            source_phi_grid = source_psi_grid * cosmology.h_over_h0(a=a_grid) * cosmology_info['H'] * 100000 / scipy.constants.c
            
            chi_mesh, ell_mesh = numpy.meshgrid(chi_grid, ell_grid)
            scale_grid = numpy.nan_to_num(numpy.divide(ell_mesh + 1/2, chi_mesh, out=numpy.zeros((ell_size + 1, grid_size + 1)) + numpy.inf, where=chi_mesh > 0))
            
            # Power
            power_grid = numpy.zeros((ell_size + 1, grid_size + 1))
            for grid_index in range(grid_size + 1):
                power_grid[:,grid_index] = pyccl.power.nonlin_matter_power(cosmo=cosmology, k=scale_grid[:,grid_index], a=a_grid[grid_index])
            
            # Cell TE
            c_data_te = numpy.zeros((source_bin_size, source_bin_size, ell_size + 1))
            
            c_data_te += SS.spectra(
                factor=numpy.array(factor_ms, dtype=numpy.float64), 
                phi_a_grid=numpy.array(lens_phi_grid * magnification_bias[:, numpy.newaxis], dtype=numpy.float64), 
                phi_b_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
                chi_grid=numpy.array(chi_grid, dtype=numpy.float64), 
                power_grid=numpy.array(power_grid * amplitude_ms, dtype=numpy.float64), 
                redshift_grid=numpy.array(z_grid, dtype=numpy.float64)
            )
            
            c_data_te += SN.spectra(
                factor=numpy.array(factor_mi, dtype=numpy.float64), 
                phi_a_grid=numpy.array(lens_phi_grid * magnification_bias[:, numpy.newaxis], dtype=numpy.float64), 
                phi_b_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
                chi_grid=numpy.array(chi_grid, dtype=numpy.float64), 
                power_grid=numpy.array(power_grid * amplitude_mi, dtype=numpy.float64), 
                redshift_grid=numpy.array(z_grid, dtype=numpy.float64)
            )
            
            c_data_te += NS.spectra(
                factor=numpy.array(factor_gs, dtype=numpy.float64), 
                phi_a_grid=numpy.array(lens_phi_grid, dtype=numpy.float64), 
                phi_b_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
                chi_grid=numpy.array(chi_grid, dtype=numpy.float64), 
                power_grid=numpy.array(power_grid * amplitude_gs, dtype=numpy.float64), 
                redshift_grid=numpy.array(z_grid, dtype=numpy.float64)
            )
            
            c_data_te += NN.spectra(
                factor=numpy.array(factor_gi, dtype=numpy.float64), 
                phi_a_grid=numpy.array(lens_phi_grid, dtype=numpy.float64), 
                phi_b_grid=numpy.array(source_phi_grid, dtype=numpy.float64),
                chi_grid=numpy.array(chi_grid, dtype=numpy.float64), 
                power_grid=numpy.array(power_grid * amplitude_gi, dtype=numpy.float64)
            )
            
            # Cell TT
            c_data_tt = numpy.zeros((lens_bin_size, lens_bin_size, ell_size + 1))
            
            c_data_tt += SS.spectra(
                factor=numpy.array(factor_mm, dtype=numpy.float64), 
                phi_a_grid=numpy.array(lens_phi_grid * magnification_bias[:, numpy.newaxis], dtype=numpy.float64), 
                phi_b_grid=numpy.array(lens_phi_grid * magnification_bias[:, numpy.newaxis], dtype=numpy.float64),
                chi_grid=numpy.array(chi_grid, dtype=numpy.float64), 
                power_grid=numpy.array(power_grid * amplitude_mm, dtype=numpy.float64), 
                redshift_grid=numpy.array(z_grid, dtype=numpy.float64)
            )
            
            c_data_tt += SN.spectra(
                factor=numpy.array(factor_mg, dtype=numpy.float64), 
                phi_a_grid=numpy.array(lens_phi_grid * magnification_bias[:, numpy.newaxis], dtype=numpy.float64), 
                phi_b_grid=numpy.array(lens_phi_grid, dtype=numpy.float64),
                chi_grid=numpy.array(chi_grid, dtype=numpy.float64), 
                power_grid=numpy.array(power_grid * amplitude_mg, dtype=numpy.float64), 
                redshift_grid=numpy.array(z_grid, dtype=numpy.float64)
            )
            
            c_data_tt += NS.spectra(
                factor=numpy.array(factor_gm, dtype=numpy.float64), 
                phi_a_grid=numpy.array(lens_phi_grid, dtype=numpy.float64), 
                phi_b_grid=numpy.array(lens_phi_grid * magnification_bias[:, numpy.newaxis], dtype=numpy.float64),
                chi_grid=numpy.array(chi_grid, dtype=numpy.float64), 
                power_grid=numpy.array(power_grid * amplitude_gm, dtype=numpy.float64), 
                redshift_grid=numpy.array(z_grid, dtype=numpy.float64)
            )
            
            c_data_tt += NN.spectra(
                factor=numpy.array(factor_gg, dtype=numpy.float64), 
                phi_a_grid=numpy.array(lens_phi_grid, dtype=numpy.float64), 
                phi_b_grid=numpy.array(lens_phi_grid, dtype=numpy.float64),
                chi_grid=numpy.array(chi_grid, dtype=numpy.float64), 
                power_grid=numpy.array(power_grid * amplitude_gg, dtype=numpy.float64)
            )
        
        stop = time.time()
        time_list[index] = (stop - begin) / 60
    
    # Save
    numpy.savetxt(os.path.join(python_folder, 'CPU/', tag, 'T_{}.txt'.format(label)), time_list)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Double')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--path', type=str, required=True, help='The path of the project scripts')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    
    # Parse
    TAG = PARSE.parse_args().tag
    PATH = PARSE.parse_args().path
    LABEL = PARSE.parse_args().label
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, PATH, LABEL, FOLDER)