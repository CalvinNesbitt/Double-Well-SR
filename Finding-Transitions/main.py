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
epsilons = [10., 1., 0.1, 0.01, 0.001]
ae_pairs = alpha_eps_pairs(alphas, epsilons)

################################################################################
## Specify whether we're running on the cluster or not
################################################################################
cluster = True

################################################################################
## Finding Transitions and saving
################################################################################

# Choose alpha, eps from array jobs
alpha, eps = ae_pairs[int(sys.argv[1]) - 1]

# Initialise list of transitions for fixed a, e pair
c2h_transitions = []
h2c_transtions = []

# Loop through hot/cold ensemble files and find transitions
integration_files = xr_files(alpha, eps, cluster)
for file in integration_files:
    ds = xr.open_dataset(file)
    c2h, h2c = get_transitions(ds, ball_size)
    c2h_transitions += c2h # add transitions to long list
    h2c_transtions += h2c

# Save Transitions for given alpha, eps
c2h_save_dir, h2c_save_dir = transition_dir(alpha, eps, cluster)
save_list(c2h_transitions, c2h_save_dir)
save_list(c2h_transitions, h2c_save_dir)
