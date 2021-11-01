"""
How long to wait for a transition in the double well?

In this notebook we run the stochastic double well for different noise strengths and see how long we must wait for a transition to occur.

We save these in pickle dictionaries.
"""

import os
import sys
sys.path.append('/rds/general/user/cfn18/home/Double-Well-SR/Stochastic-Model/Remote-Run/')
from utilities import *
import numpy as np
import matplotlib.pyplot as plt
import pickle

def experiment(p, T):
    "Test if we go from hot point to cold basin within a given time"
    hot_ic = np.array([1, 0])
    alpha, sigma = p
    time = np.arange(0, T, 0.1)
    result = euler_maruyama(hot_ic, time, p, timer=False)
    cold_at_any_point = np.any(result[:, 0] < 0)
    return cold_at_any_point

def save_results():
    "Save timing results in dicitionary indexed by sigma"
    save_dir = f'/rds/general/user/cfn18/home/Double-Well-SR/Stochastic-Model/Time-Til-Transition-Test/Timing-Results/alpha_{alpha:.2f}/'.replace('.', '_')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + 'results.pickle', 'wb') as handle:
        pickle.dump(timing_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\nSaved Results at {save_dir}/results.pickle\n')
    
def load_results(alpha):
    "Load timing results for particular alpha"
    save_dir = f'/rds/general/user/cfn18/home/Double-Well-SR/Stochastic-Model/Time-Til-Transition-Test/Timing-Results/alpha_{alpha:.2f}/'.replace('.', '_')
    with open(save_dir + 'results.pickle', 'rb') as handle:
        return pickle.load(handle)

def experiment_header(p):
    print('\n***RUNNING EXPERIMENT****')
    print()
    print(f'alpha = {alpha}, sigma  = {sigma}')
    print()

# Running the experiment
block_size = 1000 # Size of block before we test for transition
num_of_blocks = 1000000
runs_per_sigma = 10

# Results dicitionary
timing_results = {}
alphas = [0.0, 0.25, 0.5, 1.0]
alpha = alphas[0] # For now let's stick to one alpha
sigmas = [0.5, 0.2, 0.175, 0.15, 0.1, 0.05]# different noise strengths we try
sigma = sigmas[int(sys.argv[1]) - 1]

results_list = []
for i in range(runs_per_sigma):
    p = [alpha, sigma]
    experiment_header(p)
    for i in tqdm(range(num_of_blocks)):
        result = experiment(p, block_size)
        if result: # Have we found a transition on this block?
            time_til_transition = block_size * (i+1) 
            results_list.append(time_til_transition)
            print(f'Transition found within {time_til_transition}')
            timing_results[sigma] = results_list
            save_results()
            break
