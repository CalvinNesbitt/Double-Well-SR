# Functions to get relaxation as IC

import xarray as xr
import numpy as np

def get_relaxation(alpha, m2c):
    relaxation_pd = '/Users/cfn18/Documents/PhD-Work/Third-Year/Instanton-Work/Double-Well-SR/Deterministic-Model/Data/'
    
    if m2c:
        relaxation_file = f'alpha{alpha}'.replace('.', '_') + '/cold-relaxation.nc'
    else:
        relaxation_file = f'alpha{alpha}'.replace('.', '_') + '/hot-relaxation.nc'
    
    return xr.open_dataset(relaxation_pd + relaxation_file)

def flip_and_stretch_time(ds, time):
    "Interpolate a relaxation ds on to a reversed time."
    T = time[-1]
    dt = time[1] - time[0] # assuming constant spacing

    # First flip and stretch/shorten ds.time to fit our wanted time
    ds['time'] = np.linspace(T, 0, len(ds.time))

    # Now interpolate on to out desired time
    return ds.interp(time=time)

def ds_to_np(ds):
    X = ds.x.values
    Y = ds.y.values
    return np.column_stack((X, Y))

def get_reversed_relaxation_ic(alpha, c2h, time):
    relaxation = get_relaxation(alpha, not c2h)
    inst_ic  = flip_and_stretch_time(relaxation, time)
    return ds_to_np(inst_ic)