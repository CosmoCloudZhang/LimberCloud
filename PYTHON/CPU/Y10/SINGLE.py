import os
import sys
import json
import time
import pyccl
import scipy
import numpy
import argparse
from itertools import product


def main(tag, path, label, folder):
    '''
    Calculate the angular power spectra under the single configuration
    
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
    data_folder = os.path.join(folder, 'DATA/', tag)
    info_folder = os.path.join(folder, 'INFO/')
    
    python_folder = os.path.join(folder, 'PYTHON/')
    os.makedirs(os.path.join(python_folder, 'CCL/'), exist_ok = True)
    os.makedirs(os.path.join(python_folder, 'CCL/', tag), exist_ok = True)
    
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
    ell_data = numpy.sqrt(ell_grid[1:] * (ell_grid[:-1]))
    
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
            
            c_ccl_ee = numpy.zeros((source_bin_size, source_bin_size, ell_size))
            for (bin_index1, bin_index2) in product(range(source_bin_size), range(source_bin_size)):
                
                tracer1 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=[z_grid, source_psi_grid[bin_index1, :]], has_shear=True, ia_bias=[z_grid, alignment_bias], use_A_ia=False, n_samples=grid_size + 1)
                tracer2 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=[z_grid, source_psi_grid[bin_index2, :]], has_shear=True, ia_bias=[z_grid, alignment_bias], use_A_ia=False, n_samples=grid_size + 1)
                
                c_ccl_ee[bin_index1, bin_index2, :] = pyccl.cells.angular_cl(cosmo=cosmology, tracer1=tracer1, tracer2=tracer2, ell=ell_data, p_of_k_a='delta_matter:delta_matter', l_limber=-1, limber_max_error=0.001, limber_integration_method='spline', p_of_k_a_lin='delta_matter:delta_matter', return_meta=False)
        
        stop = time.time()
        time_list[index] = (stop - begin) / 60
    
    # Save
    numpy.savetxt(os.path.join(python_folder, 'CCL/', tag, 'T_{}.txt'.format(label)), time_list)
    
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
    
    # Parse
    TAG = PARSE.parse_args().tag
    PATH = PARSE.parse_args().path
    LABEL = PARSE.parse_args().label
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, PATH, LABEL, FOLDER)