""" 
Functions used for remote run of stochastic double well integration

"""

##########################################
## Imports
##########################################

import xarray as xr
import numpy as np
import numpy.random as rm
from tqdm.notebook import tqdm

##########################################
## Fixed points
##########################################

cold_point = np.array([-1, 0])
hot_point = np.array([1, 0])
saddle_point = np.array([0, 0])

##########################################
## E.M. Scheme Definition
##########################################

R = np.array([[0, -1], [1, 0]]) # 90 degree rotation matrix

def grad_V(x):
    return np.array([x[0]*(x[0]**2 -1), 2 * x[1]])

def drift(x, p):
    alpha, eps = p
    return - (np.eye(2) + alpha * R) @ grad_V(x)

def euler_maruyama(x0, t, p, timer=False):
    alpha, eps = p
    N = len(t)
    x = np.zeros(np.append(N, x0.shape))
    x[0] = x0
    for i in tqdm(range(N-1), disable=not timer):
        dt = t[i+1]-t[i]
        dWt = rm.normal(0, np.sqrt(dt), 2)
        x[i+1] = x[i] + drift(x[i], p) * dt + np.sqrt(eps) * dWt
    return x

##########################################
## Function for saving results
##########################################

def save_ensemble(ensemble, time, p, save_name):
    
    alpha, eps = p

    # xr setup
    dims = ['x', 'y']
    coords = [time]
    attrs = {'alpha': alpha, 'eps': eps}
    
    # Creating xr.Dataset
    data = np.asarray(ensemble)
    x_data = xr.DataArray(data[:, :, 0], coords = {'realisation':np.arange(len(ensemble)) + 1,
                                        'time': time},
                        dims = ['realisation','time'], attrs=attrs, name='x')
    y_data = xr.DataArray(data[:, :, 1], coords = {'realisation':np.arange(len(ensemble)) + 1,
                                        'time': time},
                        dims = ['realisation','time'], attrs=attrs, name='y')
    ensemble_ds = xr.merge([x_data, y_data])
    ensemble_ds.attrs = attrs
    
    # Saving Dataset
    ensemble_ds.to_netcdf(save_name + '.nc')
    print(f'Saved at {save_name}\n')