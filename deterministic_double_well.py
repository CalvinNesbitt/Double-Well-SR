"""
Functions defining deterministic 2D double well.

##########################################
Contents
##########################################

- Functions defining the double well

- Fixed points of the well

- Deterministic integrator & Trajectory Observer

"""

##########################################
## Imports
##########################################

import xarray as xr
import numpy as np
import numpy.random as rm
import scipy.integrate
from tqdm.notebook import tqdm

##########################################
## Well Definition
##########################################

R = np.array([[0, -1], [1, 0]]) # 90 degree rotation matrix

def V(x):
    return 0.25 * (x[0]**2 - 1)**2 + x[1]**2

def grad_V(x):
    return np.array([x[0]*(x[0]**2 -1), 2 * x[1]])

def warped_well(x, alpha):
    return - (np.eye(2) + alpha * R) @ grad_V(x)

##########################################
## Fixed points
##########################################

cold_point = np.array([-1, 0])
hot_point = np.array([1, 0])
saddle_point = np.array([0, 0])

##########################################
## Deterministic Integrator
##########################################


class FancyWellIntegrator:
    """
    Integrates antisymetric 2d double well.
    """

    def __init__(self, alpha, X_init=None):
        "Alpha controls strength of asymetry"

        self.alpha = alpha
        self.time = 0
        
        if X_init is None:
            self._state = rm.normal(scale=np.sqrt(2), size=2)
        else:
            self._state = X_init

    def _rhs_dt(self, t, state):
        return warped_well(state, self.alpha)

    def integrate(self, how_long):
        """time: how long we integrate for in adimensional time."""

        # Where We are
        t = self.time
        IC = self.state

        # Integration, uses RK45 with adaptive stepping. THIS IS THE HEART.
        solver_return = scipy.integrate.solve_ivp(self._rhs_dt, (t, t + how_long), IC, dense_output = True)

        # Updating variables
        self.set_state(solver_return.y[:,-1])
        self.time = t + how_long
        
    def set_state(self, x):
        """x is [X, T]."""
        self._state =x
        return

    @property
    def state(self):
        """Where we are in phase space."""
        return self._state

    @property
    def parameter_dict(self):
        param = {
        'alpha': self.alpha,
        }
        return param
    
# ------------------------------------------
# TrajectoryObserver
# ------------------------------------------

class TrajectoryObserver():
    """Observes the trajectory of Asymettric Double Well. Dumps to netcdf."""

    def __init__(self, integrator, name='Fancy2Well'):
        """param, integrator: integrator being observed."""

        # Needed knowledge of the integrator
        self._parameters = integrator.parameter_dict

        # Trajectory Observation logs
        self.time_obs = [] # Times we've made observations
        self.x_obs = []
        self.y_obs = []

    def look(self, integrator):
        """Observes trajectory """

        # Note the time
        self.time_obs.append(integrator.time)

        # Making Observations
        self.x_obs.append(integrator.state[0].copy())
        self.y_obs.append(integrator.state[1].copy())
        return

    @property
    def observations(self):
        """cupboard: Directory where to write netcdf."""
        if (len(self.x_obs) == 0):
            print('I have no observations! :(')
            return

        dic = {}
        _time = self.time_obs
        dic['x'] = xr.DataArray(self.x_obs, dims=['time'], name='X',
                                coords = {'time': _time})
        dic['y'] = xr.DataArray(self.y_obs, dims=['time'], name='Y',
                                coords = {'time': _time})
        return xr.Dataset(dic, attrs= self._parameters)

    def wipe(self):
        """Erases observations"""
        self.time_obs = []
        self.x_obs = []
        self.y_obs = []

    def dump(self, cupboard, name=None):
        """ Saves observations to netcdf and wipes.
        cupboard: Directory where to write netcdf.
        name: file name"""

        if (len(self.x_obs) == 0):
            print('I have no observations! :(')
            return

        if name == None:
            name=self.name

        save = cupboard + f'/{name}' + f'{self.dump_count + 1}.nc'
        self.observations.to_netcdf(save)
        print(f'Observations written to {save}. Erasing personal log.\n')
        self.wipe()
        self.dump_count +=1
        
# ------------------------------------------
# make_observations
# ------------------------------------------

def make_observations(runner, looker, obs_num, obs_freq, noprog=True):
    """Makes observations given runner and looker.
    runner, integrator object.
    looker, observer object.
    obs_num, how many observations you want.
    obs_freq, adimensional time between observations"""
    for step in tqdm(np.repeat(obs_freq, obs_num), disable=noprog):
        runner.integrate(obs_freq)
        looker.look(runner)