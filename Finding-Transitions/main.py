"""
This script takes a load of stochastic integrations from the rotated double well
and identifies the transitions in them.
"""
from utilities import *
import sys

################################################################################
## Specify parameters
################################################################################
ball_size = 0.1 # how clost to fixed point for "transitions"
alphas = [0., 0.25, 0.5, 1.]
sigmas = [0.22, 0.21, 0.2, 0.19, 0.18]
as_pairs = alpha_sigma_pairs(alphas, sigmas)

################################################################################
## Specify whether we're running on the cluster or not
################################################################################
cluster = True

################################################################################
## Finding Transitions and saving
################################################################################

# Choose alpha, eps from array jobs
alpha, sigma = as_pairs[int(sys.argv[1]) - 1]

# Initialise list of transitions for fixed alpha, sigma pair
c2h_transitions = []
h2c_transtions = []

# Loop through hot/cold ensemble files and find transitions
integration_files = xr_files(alpha, sigma)
for file in integration_files:
    ds = xr.open_dataset(file)
    c2h, h2c = get_transitions(ds, ball_size)
    c2h_transitions += c2h # add transitions to long list
    h2c_transtions += h2c

# Save Transitions for given alpha, eps
c2h_save_dir, h2c_save_dir = transition_dir(alpha, sigma, cluster)
save_list(c2h_transitions, c2h_save_dir)
save_list(c2h_transitions, h2c_save_dir)
