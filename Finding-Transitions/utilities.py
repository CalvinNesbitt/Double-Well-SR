"""
Functions that are useful for idenifying transitions amongst stochastic
integrations.
"""
import os
import xarray as xr
import numpy as np
import sys
from tqdm import tqdm
import glob

################################################################################
## Functions for finding transitions
################################################################################

def _tag_cold_points(ts, ball_size=0.1):
    return -(np.sqrt((ts.x + 1 )**2 + ts.y **2) < ball_size).values.astype(int)

def _tag_hot_points(ts, ball_size=0.1):
    return (np.sqrt((ts.x - 1 )**2 + ts.y **2) < ball_size).values.astype(int)

def symbolic_ts(ts, ball_size=0.1):
    """
    Identifies all points in a timeseries with -1, 0 or 1.
    -1 means you're close to (-1, 0)
    1 means you're close to (1, 0)
    0 is rest of points.
    """
    cold_points = _tag_cold_points(ts, ball_size)
    hot_points = _tag_hot_points(ts, ball_size)
    return cold_points + hot_points

def cold_to_hot_transitions(ts, ball_size=0.1):
    " Take a single timeseries and return transitions from cold to hot point."
    c2h = [] # List of transitions
    symbolic_path = symbolic_ts(ts, ball_size)

    # Identifying sequences of the form (-1, 0, ..., 0, 1)
    for i in np.where(symbolic_path == -1)[0]:
        next_hot_point = np.argmax(symbolic_path[i:])

        # Is it only 0's between the -1 and 1?
        if (np.sum(symbolic_path[slice(i, i + next_hot_point + 1)]) == 0):
            transition = ts.isel(time=slice(i, i + next_hot_point + 1))
            c2h.append(transition)
    return c2h

def hot_to_cold_transitions(ts, ball_size=0.1):
    " Take a single timeseries and return transitions from hot to cold point."
    h2c = [] # List of transitions
    symbolic_path = symbolic_ts(ts, ball_size)

    # Identifying sequences of the form (1, 0, ..., 0, -1)
    for i in np.where(symbolic_path == 1)[0]:
        next_cold_point = np.argmin(symbolic_path[i:])

        # Is it only 0's between the 1 and -1?
        if (np.sum(symbolic_path[slice(i, i + next_cold_point + 1)]) == 0):
            transition = ts.isel(time=slice(i, i + next_cold_point + 1))
            h2c.append(transition)
    return h2c

def get_transitions(ds, ball_size=0.1):
    "Gets transitions both ways for a given dataset"
    c2h = []
    h2c = []
    c2h += cold_to_hot_transitions(ds, ball_size)
    h2c += hot_to_cold_transitions(ds, ball_size)
    return c2h, h2c

################################################################################
## Functions for loading integration files
################################################################################
def alpha_sigma_pairs(alphas, sigmas):
    "List (a, e) pairs for lists of alphas and epsilons."
    alpha_sigma_pairs = []
    for alpha in alphas:
        for sigma in sigmas:
            alpha_sigma_pairs.append((alpha, sigma))
    return alpha_sigma_pairs

def integration_directory(alpha, sigma):
    return f'/rds/general/user/cfn18/ephemeral/Rotated-2D-Well-Stochastic-Model/alpha_{alpha}/sigma_{sigma}/cold_ic/'.replace('.', '_')

def xr_files(alpha, sigma):
    return glob.glob(integration_directory(alpha, sigma) + '*.nc')

################################################################################
## Functions for saving transition files
################################################################################

def transition_parent_dir(cluster=False):
    "Specify parent directory where you save the transitions"
    if cluster:
        return '/rds/general/user/cfn18/ephemeral/Rotated-2D-Well-Transition-Data/'
    else:
        return '/Users/cfn18/Desktop/Double-Well-SR/Finding-Transitions/Test-Data/Test-Transition-Data/'

def transition_dir(alpha, sigma, cluster=False):
    "For fixed alpha, sigma returns save directory."
    p_dir = transition_parent_dir(cluster)
    alpha_sub_dir = f'alpha_{alpha}/'.replace('.', '_')
    sigma_sub_dir = f'sigma_{sigma}/'.replace('.', '_')
    transition_directory = p_dir + alpha_sub_dir + sigma_sub_dir

    directories = [transition_directory + x for x in ['cold-to-hot/', 'hot-to-cold/']]
    for d in directories:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f'Made directory at {d}')
    return directories

def save_list(ds_list, save_dir):
    "Save ds list in specified directory."
    for i, ds in enumerate(ds_list):
        save_name = f'{i+1}.nc'
        ds.to_netcdf(save_dir + save_name)
        ds.close()
    print(f'Saved list at {save_dir}')
