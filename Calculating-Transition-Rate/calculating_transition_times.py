"""
Script opens transitions and saves transition time data as xr.dataarrays.
"""
cluster=True
import xarray as xr
import sys
import os
if cluster:
    sys.path.append('/rds/general/user/cfn18/home/Double-Well-SR/Finding-Transitions/')
else:
    sys.path.append('/Users/cfn18/Desktop/Double-Well-SR/Finding-Transitions/')
from utilities import transition_dir, alpha_eps_pairs

################################################################################
## Specify Parameters
################################################################################
if __name__ == "__main__":
    cluster=True
    alphas = [0., 0.25, 0.5, 1.]
    epsilons = [10., 1., 0.1, 0.01, 0.001]
    ae_pairs = alpha_eps_pairs(alphas, epsilons)
    alpha, eps = ae_pairs[int(sys.argv[1]) - 1]

c2h=True #looking at cold to hot or reverse?

################################################################################
## Functions for opening transitions
################################################################################

def nc_test(file):
    return file[-3:] == '.nc'

def transition_file_list(alpha, eps, c2h=True, cluster=False):
    c2h_obs_dir, h2c_obs_dir = transition_dir(alpha, eps, cluster=cluster)

    if c2h:
        obs_dir = c2h_obs_dir
    else:
        obs_dir = h2c_obs_dir

    obs_files=[]
    for file in os.listdir(obs_dir):
        if nc_test(file):
            obs_files.append(obs_dir + file)
    return obs_files

################################################################################
## Function for getting transition time
################################################################################

def get_transition_times(file_list):
    transition_times = []
    for file in file_list:
        ds = xr.open_dataset(file)
        transition_times.append((ds.time.max() - ds.time.min()).item())
        ds.close()
    return transition_times

################################################################################
## Function for saving transition times
################################################################################

def parent_tt_dir(cluster=False):
    "parent save directory"
    if cluster:
        return '/rds/general/user/cfn18/home/Double-Well-SR/Calculating-Transition-Rate/'
    else:
        return '/Users/cfn18/Desktop/Double-Well-SR/Calculating-Transition-Rate/'

def ensure_directory_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)
        print(f'Made directory at {d}')

def save_transition_time_data(data, alpha, eps, save_dir, c2h):
    "Save transition time data as a data array."
    da = xr.DataArray(transition_times, name='Transtion_Time', attrs={'alpha':alpha, 'epsilon':eps})

    if c2h:
        save_dir += 'c2h/'
    else:
        save_dir += 'h2c/'

    save_name = f'alpha{alpha}_eps{eps}'.replace('.', '_') + '.nc'

    ensure_directory_exists(save_dir)
    da.to_netcdf(save_dir + save_name)
    print(f'Saved at {save_dir + save_name}')


################################################################################
## Actually Calculating the transition times and then saving results
################################################################################
if __name__ == "__main__":
    transition_files = transition_file_list(alpha, eps, c2h=c2h, cluster=cluster)
    transition_times = get_transition_times(transition_files)
    save_dir = parent_tt_dir(cluster) + 'transition-time-data/'
    save_transition_time_data(transition_times, alpha, eps, save_dir, c2h)
