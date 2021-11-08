##########################################
## TO DO PRE SUBMITTING:
## - Set Parameters
## - Set save_directory
## - Update PBS Name + settings in other shell script
##########################################

##########################################
## Imports
##########################################

# 2D Well Euler Maruyama Scheme Import

import sys
sys.path.append('Instantons/Rotated-2D-Well/Stochastic-Model/Remote-Run/')
from utilities import *

# Standard Imports
import os
import numpy.random as rm

##########################################
## Setting Parameters
##########################################

alphas = [0., 0.25, 0.5, 1.]
epsilons = [10., 1., 0.1, 0.01, 0.001]

# Pair all combos of alpha and epsilons
alpha_eps_pairs = []
for alpha in alphas:
    for eps in epsilons:
        alpha_eps_pairs.append((alpha, eps))

# Use array jobs to decide input
alpha, eps = alpha_eps_pairs[int(sys.argv[1]) - 1]

dt = 0.01
tf = 500
time = np.arange(0, tf, dt)

p = [alpha, eps]

##########################################
## Save Details
##########################################

# Results saved after each block
# Ensemble size = 2 * blocks * block_len
# Half of the ensemble start at the cold point, half at the hot point
blocks = 10
block_len = 1000

# Where We Save Output
parent_dir = f'/rds/general/user/cfn18/ephemeral/Rotated-2D-Well-Stochastic-Model/'
alpha_sub_dir = f'alpha_{alpha}/'.replace('.', '_')
eps_sub_dir = f'eps_{eps}/'.replace('.', '_')
save_directory = parent_dir + alpha_sub_dir + eps_sub_dir

if not os.path.exists(save_directory):
    os.makedirs(save_directory + '/cold-ensemble')
    os.makedirs(save_directory + '/hot-ensemble')

##########################################
## Running and Saving in Blocks
##########################################

print('**STARTING INTEGRATION**\n')

cold_ensemble = []
hot_ensemble = []

bl = int(block_len/2) # half of points start in each basin

for i in range(blocks):

    print(f'Running Block {i}\n')

    for k in range(bl):
        ensemble_member = bl * i + k + 1
        cold_ensemble.append(euler_maruyama(cold_point, time, p))
        hot_ensemble.append(euler_maruyama(hot_point, time, p))

    # Save and clear

    save_ensemble(cold_ensemble, time, p, save_directory + f'cold-ensemble/{i + 1}')
    save_ensemble(hot_ensemble, time, p, save_directory + f'hot-ensemble/{i + 1}')
    cold_ensemble = []
    hot_ensemble = []
