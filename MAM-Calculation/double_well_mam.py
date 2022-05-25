"""
Contains classes for performing MAM with the double well model.
Also includes example run script
"""

"""
Example run of MAM for the 2D Double Well.
"""

# Standard Dependencies
import sys
import os
import time as tm
import xarray as xr

###################################################
## Importing MAM Code
###################################################
mam_code_location = '/Users/cfn18/Documents/PhD-Work/Third-Year/Instanton-Work/Action-Minimisation-Algorithm/'
sys.path.append(mam_code_location)
from fw_action import *
from mam import *

###################################################
## Double Model Defintion
###################################################

# Drift
double_well_file = '/Users/cfn18/Documents/PhD-Work/Third-Year/Instanton-Work/Double-Well-SR/'
sys.path.append(double_well_file)
from deterministic_double_well import jax_warped_well
drift = jax_warped_well

# Diffusion Definition
import jax.numpy as jnp
def diff_inv(x, s):
    return jnp.eye(2, 2)

###################################################
## Example Run
###################################################

if __name__ == '__main__':

    # Where Results Saved
    sd = '/Users/cfn18/Desktop/Double-Well-Test/'

    # Warping parameter
    alpha = 0.5

    # Time over which instanton is parameterised
    steps = 500
    dt = 0.1
    time = np.arange(0, dt * steps, dt)

    # Initial Instanton
    initial_point = [1, 0]
    final_point = [0, 0]
    inst_ic  = np.linspace(initial_point, final_point, steps)

    # Run Model
    run_additive_mam_double_well(1, inst_ic, time, sd, bounds=None,
                                 block_len=5, number_of_blocks=10, instanton_snapshot_blocks=[0, 1, 2, 3, 4, 5])

###################################################
## Function for Running MAM in the double Well
###################################################

def run_additive_mam_double_well(alpha, inst_ic, time, save_location, bounds=None,
                                 block_len=100, number_of_blocks=10, instanton_snapshot_blocks=[0]):
    """
    Minimises the FW action for the L96-EBM Model with additive noise.

    --------------------
    Arguments
    --------------------
    alpha, float
        Parameter controlling well warpdness

    inst_ic, initial instanton guess
        Should be np.array

    time, np.array
        The time over which the minimum action path is parameterised.

    save_location, string
        Where output is saved.

    block_len, int
        How many iterations between saving minimimasition result thus far.

    number_of_blocks, int
        How many blocks of minimisation to run.
        Total number of minimisation iterations will be: block_len * number_of_blocks.

    """

    ##########################################
    ## Setting Up MAM Objects
    ##########################################
    print('\n*** Setting up jfw object. *** \n')
    # Object for calculating the FW Action and it's jacobian
    p = np.array([alpha, 1])
    fw = JFW(drift, diff_inv) # Freidlin-Wentzell object

    # Minimisation object
    print('\n*** Setting up MAM object. *** \n')
    mamjax = MamJax(fw, inst_ic, time, p) # MAM algorithm object

    if bounds is not None:
        mamjax.bnds=bounds

    # Observer object
    print('\n*** Setting up MAM observer. *** \n')
    observer = doubleWellMAMO(mamjax, save_location)
    # mamjax.run({'maxiter': 0, 'maxfun': 0}) # initialise result object to be observed
    # mamjax.nit = 0
    observer.save_instanton_ic()
    # observer.save_instanton_snapshot()
    # observer.save_av()
    # observer.save_status()

    ##########################################
    ## Running and Saving in Blocks
    ##########################################

    # Runnning MAM
    opt={'maxiter': block_len, 'maxfun': 10 * block_len}

    print('\n*** Starting MAM *** \n')

    for i in range(number_of_blocks):

        print(f'\nRunning block {i+1}/{number_of_blocks}\n')
        mamjax.run(opt)
        observer.save_av() # Status/AV saved after each block
        observer.save_status()
        observer.save_instanton()
        if i+1 in instanton_snapshot_blocks:
            observer.save_instanton_snapshot()

        # Check if you've converged
        if (mamjax.res.success is True):
            print('Success, quitting MAM and saving result.')
            break

        else:
            print('No convergence thus far. Will continue minimisation.\n')
    return

###################################################
## Class for observing MAM in the double Well
###################################################

class doubleWellMAMO:
    """
    Class for observing Double Well MAM minimisation.

    - Currently observes: instanton, action value, status and parameters.
    - Initialised with a MAMJAX object and a save location.

    Methods
    -----------
    save_parameters()
        Runs when initialised, saves parameters in pickled dictionary.

    snaphot()
        To be called after each bout of minimisation.
        Saves action value, instanton and status in save_loc directory.

    Attributes
    -----------
    mj: MamJax
        Object used to rune the MAM algorithm with jax.
        Imagine this would work with a mam object, haven't tried.

    save_loc: string
        Directory where the observations saved.

    """

    def __init__(self, mj, pd):
        """
        Parameters
        ----------
        mj: MamJax
            Object used to run the MAM algorithm with jax.
            This should also work with a mam object, but I haven't tried.

        pd: string
            Directory where the observations saved.

        """
        self.mj = mj
        self.pd = pd
        self.instanton_dir = self.pd + '/Instanton_Snapshots/' # where we save snapshots
        if not os.path.exists(self.instanton_dir):
            os.makedirs(self.instanton_dir)
            print(f'\nMade directory at {self.instanton_dir}\n')

        self.save_parameters()
        self.av_list = []
        self.starting_cpu_time = tm.time()

    @property
    def parameters(self):
        "Parameters used for a run. Access by instanton"

        # Unpack parameters and put in labelled dictionary
        alpha = self.mj.p[0]
        param = {
        'alpha' : alpha,
        'iteration': self.mj.nit
        }
        return param

    def save_parameters(self):
        with open(self.pd +'/parameters.pickle','wb') as file:
            pickle.dump(self.parameters, file)
            print(f'Parameters saved at {self.pd}/parameters.pickle')

    def save_av(self):

        # Save as (nit, av) pairs
        nit = self.mj.nit
        av = self.mj.res.fun
        self.av_list.append((nit, av))

        # Pickle av_list
        with open(self.pd + '/action_timeseries.pickle','wb') as file:
            pickle.dump(self.av_list, file)
            print(f'\nAction values saved at {self.pd}/av.pickle\n')

    def save_status(self):

        # Info we want
        success = self.mj.res.success
        message = self.mj.res.message
        nit = self.mj.nit
        cpu_time = tm.time() - self.starting_cpu_time

        # Write to text file
        f = open(self.pd + "/current_status.txt", "w")
        f.write(f'{nit} iterations completed\nRunning Time(s):{cpu_time}\nSuccess: {success}\n{message}\n')
        print(f'Converged: {success}')
        f.close()

    def save_instanton_snapshot(self):
        "Saves instanton snapshot."

        # Put Instanton in xr form
        dic = {}
        _time = self.mj.time
        _X = self.mj.instanton[:, 0]
        _Y = self.mj.instanton[:, 1]
        dic['X'] = xr.DataArray(_X, dims=['time'], name='X', coords = {'time': _time})
        dic['Y'] = xr.DataArray(_Y, dims=['time'], name='Y', coords = {'time': _time})

        instanton = xr.Dataset(dic, attrs= self.parameters)

        # Save Instanton
        nit = self.mj.nit
        instanton.to_netcdf(self.instanton_dir + f'/iteration_{nit}.nc')
        print('\nInstanton saved at ' + self.instanton_dir + f'/iteration_{nit}.nc\n')
        return

    def save_instanton(self):
        "Saves instanton snapshot."

        # Put Instanton in xr form
        dic = {}
        _time = self.mj.time
        _X = self.mj.instanton[:, 0]
        _Y = self.mj.instanton[:, 1]
        dic['X'] = xr.DataArray(_X, dims=['time'], name='X', coords = {'time': _time})
        dic['Y'] = xr.DataArray(_Y, dims=['time'], name='Y', coords = {'time': _time})

        instanton = xr.Dataset(dic, attrs= self.parameters)

        # Save Instanton
        instanton.to_netcdf(self.pd + f'/Instanton.nc')
        print('\nInstanton saved at ' + self.pd + f'/Instanton.nc')
        return

    def save_instanton_ic(self):
        "Saves instanton ic."

        # Put Instanton in xr form
        dic = {}
        _time = self.mj.time
        _X = self.mj.instanton[:, 0]
        _Y = self.mj.instanton[:, 1]
        dic['X'] = xr.DataArray(_X, dims=['time'], name='X', coords = {'time': _time})
        dic['Y'] = xr.DataArray(_Y, dims=['time'], name='Y', coords = {'time': _time})

        instanton = xr.Dataset(dic, attrs= self.parameters)

        # Save Instanton
        instanton.to_netcdf(self.pd + f'/Instanton_IC.nc')
        print('\nInstanton saved at ' + self.pd + f'/Instanton_IC.nc')
        return
