"""
Functions and classes for plotting paths in the double well system

##########################################
Contents
##########################################

- DoubleWellPath class

- Fixed points of the well
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

##########################################
## Plotting Functions
##########################################
from deterministic_double_well import cold_point, hot_point, saddle_point

def init_2d_fax():
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    return fig, ax

def fancy_well_background_plot(alpha, fax=None):

    if fax is None:
        fig, ax = init_2d_fax()
    else:
        fig, ax = fax

    # Plot Misc
    x_range = (-1.5, 1.5)
    y_range = (-1, 1)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # Gradient Arrows
    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x, y)
    vx = X*(X**2 - 1) - 2 * alpha * Y
    vy = alpha *X*(X**2 - 1) + 2 * Y
    speed = np.sqrt(vx**2 + vy**2)
    ax.streamplot(x, y, -vx, -vy, color='1')
    ax.pcolormesh(X, Y, speed, cmap = 'Purples', shading='auto')
    return fig, ax

    # alpha label
def alpha_label_box(alpha, ax, xpos=0, ypos=0):
    ax.text(xpos, ypos, fr'$\alpha = {alpha:.2f}$', fontsize=15, bbox={'facecolor': '1', 'pad': 10})
    return

def add_arrow(path, fax, start_index=None, end_index=None, head_width=0.05, **kwargs):
    fig, ax = fax
    X = path.X
    Y = path.Y
    if start_index is None:
        mid_point = int(len(X)/2)
        if path.direction == 'c2h':
            start_index = mid_point
            end_index = mid_point + 1
        elif path.direction == 'h2c':
            start_index = mid_point - 1
            end_index = mid_point

    X_start = X[start_index]
    X_next = X[end_index]
    Y_start = Y[start_index]
    Y_next = Y[end_index]
    ax.arrow(X_start, Y_start, X_next - X_start, Y_next - Y_start, head_width=head_width,**kwargs)
    return

def plot_ball(centre, radius, fax=None, **kwargs):
    if fax is None:
        fig, ax = init_2d_fax()
    else:
        fig, ax = fax

    disc = plt.circle(centre, radius, **kwargs)
    ax.add_patch(disc)
    return

def plot_cold_point(radius, fax=None, **kwargs):
    return plot_ball((cold_point[0], cold_point[1]), radius=radius, fax=fax, **kwargs)

def plot_saddle_point(radius, fax=None, **kwargs):
    return plot_ball((saddle_point[0], saddle_point[1]), radius=radius, fax=fax, **kwargs)

def plot_hot_point(radius, fax=None, **kwargs):
    return plot_ball((hot_point[0], hot_point[1]), radius=radius, fax=fax, **kwargs)

##########################################
## DoubleWellPath Definition
##########################################

class DoubleWellPath:
    """
    Path in the Double Well model. Contains a variety of plotting and clipping functionality.
    """

    def __init__(self, file_location):
        self.file_location = file_location
        self.ds = xr.open_dataset(file_location)
        self.alpha = self.ds.attrs['alpha']
        self._determine_direction()

    @property
    def X(self):
        return self.ds.X

    @property
    def Y(self):
        return self.ds.Y

    @property
    def time(self):
        return self.ds.time

    def _determine_direction(self):
        if self.ds.X.values[0] < self.ds.X.values[-1]:
            self.direction = 'c2h'
        else:
            self.direction = 'h2c'
        return

    def clip_path(self, i, j, normalise_time = True):
        "Reduce path to just points those between time indices i and j."
        dt = (self.ds.time[1] - self.ds.time[0]).item() # assuming constant dt
        self.ds = self.ds.isel(time=slice(i, j))
        if normalise_time:
            self.ds = self.ds.assign_coords({'time': dt * np.arange(len(self.ds.time))})
        return

    def ds_as_np(ds):
        X = ds.X.values
        Y = ds.Y.values
        return np.column_stack((X, Y))

    def get_observable_points_as_np(self, obs):
        return obs(self.ds).values.flatten()

    def _2d_plot_projection(self, plot, fax=None, *args, **kwargs):
        if fax is None:
            fig, ax = init_2d_fax()
        else:
            fig, ax = fax
        plot(self.ds, ax=ax, *args, **kwargs)
        return

    def path_plot(self, fax=None, flow_lines=True, arrow=True, **kwargs):
        if fax is None:
            fig, ax = init_2d_fax()
        else:
            fig, ax = fax

        if flow_lines:
            fancy_well_background_plot(alpha=self.alpha, fax=[fig, ax])

        #Plotting Path
        ax.plot(self.X, self.Y, **kwargs)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        if arrow:
            add_arrow(self, fax=[fig, ax], color=kwargs.pop('c'))
        return fig, ax

    def X_timeseries_plot(self, fax=None, *args, **kwargs):
        if fax is None:
            fig, ax = init_2d_fax()
        else:
            fig, ax = fax

        #Plotting Path
        ax.plot(self.time, self.X, **kwargs)
        ax.set_xlabel('Time')
        ax.set_ylabel('X')
        ax.grid()
        return fig, ax

    def Y_timeseries_plot(self, fax=None, *args, **kwargs):
        if fax is None:
            fig, ax = init_2d_fax()
        else:
            fig, ax = fax

        #Plotting Path
        ax.plot(self.time, self.Y, **kwargs)
        ax.set_xlabel('Time')
        ax.set_ylabel('Y')
        ax.grid()
        return fig, ax
