#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 13:26:47 2025

@author: jocelynreahl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib
from matplotlib.colors import LightSource
import matplotlib.animation as animation

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for data with a
    negative min and positive max and you want the middle of the colormap's
    dynamic range to be at zero.

    Parameters
    ----------
    cmap : matplotlib.colormap
        The matplotlib colormap to be altered.
        
    start : int or float, optional
        Offset from lowest point in the colormap's range.
        Should be between 0.0 and `midpoint`.
          
        The default is 0.
          
    midpoint : float, optional
        The new center of the colormap. Should be between 0.0 and 1.0.
        In general, this should be  1 - vmax / (vmax + abs(vmin)).
        For example, if your data range from -15.0 to +5.0 and you want the
        center of the colormap at 0.0, `midpoint` should be set to 
        1 - 5/(5 + 15)) or 0.75.
        
        The default is 0.5.
          
    stop : int or float, optional
        Offset from highest point in the colormap's range. Should be between
        `midpoint` and 1.0.
        
        The default is 1.0.
    
    Returns
    -------
    newcmap : matplotlib.colormap
        Scaled colormap.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)

    return newcmap


def get_shiftedColorMap(var, orig_cmap='coolwarm_r',
                        multiplier=None, return_v=False):
    '''
    Compute custom diverging colormap for input var variable.

    Parameters
    ----------
    var : list, xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset,
          or numpy.ndarray
        Variable to derive shifted colormap from.
        
    orig_cmap : str, optional
        Original colormap for function to reference.
        
        The default is 'coolwarm_r'.
        
    multiplier : float, optional
        Multiplier to apply to vmin and vmax for unit conversions.
        
        The default is None.
        
    return_v : bool, optional
        Whether or not to return the vmin and vmax values.
        
        The default is False.

    Returns
    -------
    shifted_cmap : matplotlib.colormap
        Shifted colormap scaled to var data.
    
    (vmin, vmax) : tuple, returns if return_v == True
        Minimum and maximum colormap values to scale plots against.

    '''
    if type(var) == list:
        vmin = np.min([var[i].min() for i in range(len(var))])
        vmax = np.max([var[i].max() for i in range(len(var))])
    elif type(var) == xr.core.dataarray.DataArray:
        vmin = var.min()
        vmax = var.max()
    elif type(var) == np.ndarray:
        vmin = np.nanmin(var)
        vmax = np.nanmax(var)
    if type(multiplier) == float:
        vmin, vmax = vmin*multiplier, vmax*multiplier
    midpoint = 1-(vmax)/(vmax + np.abs(vmin))
    if type(orig_cmap) == str:
        orig_cmap = getattr(matplotlib.cm, orig_cmap)
    shifted_cmap = shiftedColorMap(orig_cmap, midpoint=midpoint,
                                   name='shifted')
    if return_v == False:
        return shifted_cmap
    if return_v == True:
        return shifted_cmap, (vmin, vmax)


def get_extent(var):
    '''
    Calculates extent of var for matplotlib.pyplot.imshow function.

    Parameters
    ----------
    var : xarray.core.dataarray.DataArray or xarray.core.dataset.Dataset
        Input variable to scale imshow extent against.

    Returns
    -------
    extent : tuple
        Format (ll_E, ur_E, ll_N, ur_N), where:
            - ll_E = lower-left Easting
            - ur_E = upper-right Easting
            - ll_N = lower-left Northing
            - ur_N = upper-right Northing.

    '''
    ll_E, ll_N, ur_E, ur_N = (var.x.min(), var.y.min(),
                              var.x.max(), var.y.max())
    extent = (ll_E, ur_E, ll_N, ur_N)
    return extent


class IGMPlotter():
    def __init__(self, path):
        '''
        Initialize IGMPlotter object.
        
        Parameters
        ----------
        
        path : str
            Path to the output.nc file in your directory
        
        
        Callable Parameters
        -------------------
        
        data_orig : xarray.Dataset
            Original output.nc file, without data cleaning.
            
        data : xarray.Dataet
            Lightly-modified output.nc file where variables 'thk' and onward
            have been masked to the ice area at every timestep.
        
        data_keys : list
            List of variables in output.nc file.
            
        time : np.ndarray
            Years that the model was run for.
        
        '''
        self.data_orig = xr.open_dataset(path)
        self.data = self.data_orig.copy()
        self.data_keys = list(self.data.keys())
        for key in self.data_keys[3:]:
            attrs = self.data[key].attrs
            self.data[key] = xr.where(self.data['thk']==0.0,
                           np.nan,
                           self.data[key])
            self.data[key].attrs = attrs
        tmp_attrs = self.data['thk'].attrs
        self.data['thk'] = xr.where(self.data['thk']==0.0,
                            np.nan,
                            self.data['thk'])
        self.data['thk'].attrs = tmp_attrs
        self.time = self.data.time.values
        
        
    def imshow_timestep(self, variable, year=None,
                        azdeg=315, altdeg=58, ve=1,
                        ax=None, plot_topg=True, zero_centered_cmap=False,
                        cmap=None, **kwargs):
        '''
        Plot a chosen variable at a given year.

        Parameters
        ----------
        variable : str
            Variable to plot.
            Reference 'data' parameter to see possible variables.
            
        year : int or float, optional
            Chosen year.
            If None, the year will default to the last year in the model run.
            
        azdeg : int, optional
            Azimuth in degrees (out of 360) of lightsource for hillshade
            background.
            The default is 315.
            
        altdeg : int, optional
            Altitude in degrees (out of 90) of lightsource for hillshade
            background.
            The default is 58 (latitude of Juneau, Alaska!)
            
        ve : int, optional
            Vertical exaggeration of shading for lightsource.
            Suggest using powers of 10 (e.g. 1e-1, 1e0, 1e1, etc.).
            The default is 1.
            
        ax : matplotlib.axes.Axes, optional
            Axes object to draw plot on. If None, will generate a new Axes
            object.
            The default is None.
            
        plot_topg : bool, optional
            Will plot the hillshade background if True.
            The default is True.
            
        zero_centered_cmap : bool, optional
            If True, will center the colormap to be at 0. This is especially
            useful for plotting surface mass balance.
            The default is False.
            
        cmap : str or matplotlib.colorbar.Colorbar, optional
            Chosen colormap. str must be one of the default Matplotlib colormap
            options.
            The default is None.
            
        **kwargs : Passed to ax.imshow()

        Returns
        -------
        ax : matplotlib.axes.Axes
            Updated Axes object.
            
        cbar : matplotlib.colorbar.Colorbar
            Output colorbar object.

        '''
        extent = get_extent(self.data)
        
        if year is None:
            year = self.time[-1] # defaults to final time
            
        if ax is None:
            fig, ax = plt.subplots()
        
        if plot_topg == True:
            # create hillshade background
            dx, dy = np.diff(self.data.x.values)[0], np.diff(self.data.y.values)[0]
            ls = LightSource(azdeg=azdeg, altdeg=altdeg)
            topg = self.data.isel(time=0).topg.values
            hillshader = ls.hillshade(topg, vert_exag=ve, dx=dx, dy=dy)
            ax.imshow(hillshader, cmap='gray', origin='lower',
                      extent=extent)
            
        if zero_centered_cmap == True:
            if cmap is None:
                cmap = 'coolwarm_r'
            cmap = get_shiftedColorMap(self.data.sel(time=year)[variable],
                                       orig_cmap=cmap)

        im = ax.imshow(self.data.sel(time=year)[variable],
                       origin='lower', extent=extent,
                       cmap=cmap, **kwargs)
        cbar = plt.colorbar(im, label=(
            self.data[variable].attrs['long_name'] +
            ' [' +
            self.data[variable].attrs['units'] +
            ']'
        ),
                            ax=ax)
        #plt.show()
        
        return ax, cbar
        
    
    def make_anim(self, variable, filename='anim.gif',
                  writer='PillowWriter', fps=6, const_cbar=True,
                  zero_centered_cmap=False, cmap=None, **kwargs):
        '''
        Plot a chosen variable at a given year.

        Parameters
        ----------
        variable : str
            Variable to plot.
            Reference 'data' parameter to see possible variables.
            
        filename : str, optional
            Filename path for the output file.
            The default is anim.gif.
            
        writer : str, optional
            Matplotlib.animation writer to generate gif.
            The default is 'PillowWriter'
            
        fps : int, optional
            Frames per second of run.
            The default is 6.
            
        const_cbar : bool, optional
            If True, the colorbar scale will be constant throughout the
            animation.
            The default is True.
            
        zero_centered_cmap : bool, optional
            If True, will center the colormap to be at 0. This is especially
            useful for plotting surface mass balance.
            The default is False.
            
        cmap : str or matplotlib.colorbar.Colorbar, optional
            Chosen colormap. str must be one of the default Matplotlib colormap
            options.
            The default is None.
            
        **kwargs : Passed to IGMPlotter.imshow_timestep method

        '''
        # initialize figure
        fig, ax = plt.subplots()
        
        # initialize writer
        writer = getattr(animation, writer)(fps=fps)
        writer.setup(fig, filename)
        
        if const_cbar == True:
            if zero_centered_cmap == True:
                if cmap is None:
                    cmap = 'coolwarm_r'
                cmap, vrange = get_shiftedColorMap(self.data[variable],
                                                   orig_cmap=cmap,
                                                   return_v=True)
            else:
                vrange = (self.data[variable].max(),
                          self.data[variable].min())
        else:
            vrange = (None, None)
        
        for i in range(len(self.time)):
            # plot timestep
            ax, cbar = self.imshow_timestep(variable, year=self.time[i], ax=ax,
                                      vmin=vrange[0], vmax=vrange[1],
                                      cmap=cmap, **kwargs)
            ax.set_title('%d'%(self.time[i]))
            ax.set_xlabel('Eastings [m]')
            ax.set_ylabel('Northings [m]')

            # grab state of figure
            writer.grab_frame()
            
            # clean axes
            cbar.remove()
            ax.cla()
        plt.close()
        writer.finish()
            
            
            
