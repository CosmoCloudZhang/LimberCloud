import os
import json
import time
import argparse


def main(folder):
    '''
    Store the fiducial values of magnification bias
    
    Arguments:
        folder (str): The base folder of the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    info_folder = os.path.join(folder, 'INFO/')
    
    # Magnification
    magnification = {
        'Y1': [
            0.6592776202425515,
            0.7753410427009177,
            0.8148095743802405,
            0.8463012774961294,
            0.9939547805759833
        ],
        'Y10': [
            0.6388110321707254,
            0.7190901521723749,
            0.7836701305038943,
            0.7666587718796247,
            0.7524433409426237,
            0.6513359445895796,
            0.7641762190034304,
            0.7950074733615955,
            0.7976443261028644,
            0.6230940665100914
        ]
    }
    
    # Save
    with open(os.path.join(info_folder, 'MAGNIFICATION.json'), 'w') as file:
        json.dump(magnification, file, indent=4)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Info Magnification')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the datasets')
    
    # Parse
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(FOLDER)