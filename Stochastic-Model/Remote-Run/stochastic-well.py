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

# Importing python utilities
from pathlib import Path
import sys
sys.path.append(str(Path.home()/'python_utilities'))
from import_helper import *

# Importing Double Well
add_double_well_dir()
from stochastic_double_well import *

# Standard Imports
import os
import numpy.random as rm
import time as tm

##########################################
## Setting Parameters
##########################################

alphas = [0., 0.25, 0.5, 1.]
sigmas = [0.22, 0.21, 0.2, 0.19, 0.18]
ic = cold_point

# Pair all combos of alpha and sigma
alpha_sigmas_pairs = []
for alpha in alphas:
    for sigma in sigmas:
        alpha_sigma_pairs.append((alpha, sigma))

# Use array jobs to decide input
alpha, sigma = alpha_sigma_pairs[int(sys.argv[1]) - 1]

# Integration Length
dt = 0.1
T = int(1.e0) # reccomend 10**8 for sigma=0.18
time = np.arange(0, T, dt)

p = [alpha, sigma]

##########################################
## Save Details
##########################################

# How many times should we run each integration 
number_of_integrations = 100

# Where We Save Output
parent_dir = f'/rds/general/user/cfn18/ephemeral/Rotated-2D-Well-Stochastic-Model/'
alpha_sub_dir = f'alpha_{alpha}/'.replace('.', '_')
sigma_sub_dir = f'sigma_{sigma}/'.replace('.', '_')
if ic == cold_point:
    ic_sub_sir = 'cold_ic/'
elif ic == hot_point:
    ic_sub_sir = 'hot_ic/'

save_directory = parent_dir + alpha_sub_dir + sigma_sub_dir + ic_sub_sir

if not os.path.exists(save_directory):
    os.makedirs(save_directory)
    
def save_result():
    # xr setup
    dims = ['x', 'y']
    coords = [time]
    attrs = {'alpha': alpha, 'sigma': sigma}
    
    # Creating xr.Dataset
    data = integration_result
    x_data = xr.DataArray(data[:, 0], coords = {'time': time},
                        dims = ['time'], attrs=attrs, name='x')
    y_data = xr.DataArray(data[:, 1],  coords = {'time': time},
                        dims = ['time'], attrs=attrs, name='y')
    ensemble_ds = xr.merge([x_data, y_data])
    ensemble_ds.attrs = attrs
    
    # Saving Dataset
    save_name = save_directory + f'{i+1}'
    ensemble_ds.to_netcdf(save_name + '.nc')
    print(f'Saved at {save_name}\n')
        
##########################################
## Running and Saving in Blocks
##########################################

print('**STARTING INTEGRATION**\n')

bl = int(block_len/2) # half of points start in each basin

for i in range(number_of_integrations):

    print(f'Running Integration {i} of {number_of_integrations}\n')
    start = tm.time()
    integration_result = euler_maruyama(ic, time, p)
    end = tm.time()
    time_in_hours = (end - start)/60**2
    print(f'Integration {i} took approximately {time_in_hours:.2g} hours.')

    # Save
    save_result()