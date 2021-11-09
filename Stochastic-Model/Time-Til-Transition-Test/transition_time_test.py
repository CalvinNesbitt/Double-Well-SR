"""
How long to wait for a transition in the double well?

In this notebook we run the stochastic double well for different noise strengths and see how long we must wait for a transition to occur.

We save these in pickle dictionaries.
"""

# Importing python utilities
from pathlib import Path
import sys
sys.path.append(str(Path.home()/'python_utilities'))
from import_helper import *

# Importing Double Well
add_double_well_dir()
from stochastic_double_well import *
from deterministic_double_well import *
from joblib import Parallel, delayed

import os
import pickle
import time as tm

# Experiment Set Up

# Total integration time = number_of_blocks * block_length
number_of_blocks = int(1.e7) # Setting Experiment Length
block_length = 100. # How often we check for a transitions
dt = 0.1
time = np.arange(0, block_length, dt)

# Experiment Parameters
alpha = 0.1
sigmas = [0.4, 0.3, 0.25, 0.2, 0.18, 0.16, 0.14, 0.12, 0.1, 0.08, 0.06]
ic = cold_point 

# Functions Needed for Experiment

def cold_in_ts(x):
    return np.any(x[:, 0] < 0)

def hot_in_ts(x):
    return np.any(x[:, 0] > 0)

def update_results():
    experiment_results['cpu_times(s)'].append(cpu_time)
    experiment_results['integration_times'].append(integration_time)
    experiment_results['number_of_transitions'] += 1

def save_directory(alpha):
    "Returns different if we're on cluster or not"
    if str(Path.home()) == '/Users/cfn18':
        return f'/Users/cfn18/Documents/PhD-Work/Third-Year/Instanton-Work/Double-Well-SR/Stochastic-Model/Time-Til-Transition-Test/Timing-Results/alpha_{alpha:.2f}'.replace('.', '_')
    else:
        return f'/rds/general/user/cfn18/home/Double-Well-SR/Stochastic-Model/Time-Til-Transition-Test/Timing-Results/alpha_{alpha:.2f}'.replace('.', '_')
        
def save_results():
    save_dir = save_directory(alpha)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + f'/sigma_{sigma:.3f}_results'.replace('.', '_') + '.pickle', 'wb') as handle:
        pickle.dump(experiment_results, handle)
    print(f'\nSaved Results at {save_dir}/sigma_{sigma:.3f}_results.pickle\n')

def experiment_header(p):
    print('\n***RUNNING EXPERIMENT****')
    print()
    print(f'alpha = {alpha}, sigma  = {sigma}')
    print()
    
def load_results(alpha):
    results = []
    pd = save_directory(alpha)
    files = os.listdir(pd)
    files.sort()
    files.reverse()
    for f in files:
        with open(pd + '/' + f, 'rb') as handle:
            results.append(pickle.load(handle))
    return results

# Running the Experiment
if __name__ == "__main__":
    sigma = sigmas[int(sys.argv[1]) - 1]
    p = [alpha, sigma]
    
    # Initialising Results
    experiment_results = {'sigma': sigma, 'cpu_times(s)': [], 'integration_times': [], 'number_of_transitions': 0}
    last_success_block = 0 

    # Looped Search for Transitions
    experiment_header(p)
    for i in range(number_of_blocks):
        start = tm.time()
        ts = double_well_em(ic, time, p)
        ic = ts[-1]

        if hot_in_ts(ts): # Have we found a transition in last block?
            end = tm.time()
            cpu_time = end - start
            integration_time = (i - last_success_block + 1) * time[-1]
            update_results()
            save_results()

            # Check if we have enough samples
            if experiment_results['number_of_transitions'] == 100:
                print(f'Found 100 transitions - will quit.')
                quit()  
           

            # Reset Experiment
            ic = cold_point
            last_success_block = i
