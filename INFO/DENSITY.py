import os
import json
import time
import argparse


def main(folder):
    '''
    Store the fiducial values of density configuration
    
    Arguments:
        folder (str): The base folder of the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    info_folder = os.path.join(folder, 'INFO/')
    
    # Density
    density = {
        'Y1': {
            'LENS': [
                0.08720538020133972,
                0.09592237323522568,
                0.13990487158298492,
                0.1811719387769699,
                0.16735374927520752
            ],
            'SOURCE': [
                2.089827060699463,
                2.2984490394592285,
                2.1046831607818604,
                1.900587797164917,
                1.8386759757995605
            ]
        },
        'Y10': {
            'LENS': [
                0.04866538941860199,
                0.03815827891230583,
                0.045559294521808624,
                0.05072904750704765,
                0.06561077386140823,
                0.07414277642965317,
                0.08920416235923767,
                0.09422051906585693,
                0.10330963134765625,
                0.0956401526927948
            ],
            'SOURCE': [
                5.742310047149658,
                5.716140270233154,
                5.32042932510376,
                4.828889846801758,
                4.577329158782959
            ]
        }
    }
    
    # Save
    with open(os.path.join(info_folder, 'DENSITY.json'), 'w') as file:
        json.dump(density, file, indent=4)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Info density')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the datasets')
    
    # Parse
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(FOLDER)