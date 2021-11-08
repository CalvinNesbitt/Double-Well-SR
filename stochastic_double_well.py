"""
Contains EM scheme for solving integrating stochastic double well.
"""
from deterministic_double_well import warped_well, hot_point, cold_point
import numpy as np
import numpy.random as rm
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

def double_well_em(x0, t, p, timer=False):
    alpha, sigma = p
    N = len(t)
    x = np.zeros(np.append(N, x0.shape))
    x[0] = x0
    for i in tqdm(range(N-1), disable=not timer):
        dt = t[i+1]-t[i]
        dWt = rm.normal(0, np.sqrt(dt), 2)
        x[i+1] = x[i] + warped_well(x[i], alpha) * dt + sigma * dWt
    return x

def double_well_ensemble_simulation(ic_list, time, p, multiprocess=True):
    results = []

    if multiprocess:
        results = Parallel(n_jobs = -2)(delayed(double_well_em)(ic, time, p) for ic in tqdm(ic_list))
    else:
        for ic in tqdm(ic_list):
            results.append(double_well_em(ic, time, p))
    return results

def ic_spread(x, n):
    xy_min = x - 0.1
    xy_max = x + 0.1
    return rm.uniform(low=xy_min, high=xy_max, size=(n,2))

def cold_ic_spread(n):
    return ic_spread(cold_point, n)

def hot_ic_spread(n):
    return ic_spread(hot_point, n)
