#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 10:03:41 2025

@author: jocelynreahl
"""

# Import standard libraries:
import xarray as xr
import numpy as np
from numpy.polynomial import polynomial as P
import pandas as pd
import matplotlib

# Import  niche libraries that you may need to install to your environment:
import rioxarray
import rasterio
from oggm import utils, workflow, graphics
import geopandas as gpd

def prep_xarray_dataset(path, **kwargs):
    '''
    Import original netCDF file into memory, give it a CRS (EPSG:4326), and
    rename the 'lon' and 'lat' coordinates to 'x' and 'y', respectively.

    Parameters
    ----------
    path : str
        File path to .nc file from Ing et al. (2025).
        
        Strings and Path objects are interpreted as a path to a netCDF file or
        an OpenDAP URL and opened with python-netCDF4, unless the filename ends
        with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). Byte-strings or file-like
        objects are opened by scipy.io.netcdf (netCDF3) or h5py (netCDF4/HDF).
        
    **kwargs : ({engine=None, chunks=None, cache=None, decode_cf=None,
                 mask_and_scale=None, decode_times=None, decode_timedelta=None,
                 use_cftime=None, concat_characters=None, decode_coords=None,
                 drop_variables=None, inline_array=False,
                 chunked_array_type=None, from_array_kwargs=None,
                 backend_kwargs=None, **kwargs}, optional)
    
        Additional keyword arguments passed to xarray.open_dataset():
            engine : ({"netcdf4", "scipy", "pydap", "h5netcdf", "zarr", None},
                      installed backend or subclass of
                      xarray.backends.BackendEntrypoint, optional)
                Engine to use when reading files. If not provided, the default
                engine is chosen based on available dependencies, with a
                preference for “netcdf4”. A custom backend class (a subclass of
                BackendEntrypoint) can also be used.
                
            chunks : (int, dict, 'auto' or None, default: None)
                If provided, used to load the data into dask arrays.

                    - chunks="auto" will use dask auto chunking taking into
                      account the engine preferred chunks.

                    - chunks=None skips using dask, which is generally faster
                      for small arrays.

                    - chunks=-1 loads the data with dask using a single chunk
                      for all arrays.

                    - chunks={} loads the data with dask using the engine’s
                      preferred chunk size, generally identical to the format’s
                      chunk size. If not available, a single chunk for all
                      arrays.

                See dask chunking for more details.
            
            cache : (bool, optional)
                If True, cache data loaded from the underlying datastore in
                memory as NumPy arrays when accessed to avoid reading from the
                underlying data- store multiple times. Defaults to True unless
                you specify the chunks argument to use dask, in which case it
                defaults to False. Does not change the behavior of coordinates
                corresponding to dimensions, which always load their data from
                disk into a pandas.Index.
            
            decode_cf : (bool, optional)
                Whether to decode these variables, assuming they were saved
                according to CF conventions.
            
            mask_and_scale : (bool or dict-like, optional)
                If True, replace array values equal to _FillValue with NA and
                scale values according to the formula
                original_values * scale_factor + add_offset,
                where _FillValue, scale_factor and add_offset are taken from
                variable attributes (if they exist). If the _FillValue or
                missing_value attribute contains multiple values a warning will
                be issued and all array values matching one of the multiple 
                alues will be replaced by NA. Pass a mapping, e.g.
                {"my_variable": False}, to toggle this feature per-variable
                individually. This keyword may not be supported by all the
                backends.
            
            decode_times : (bool, CFDatetimeCoder or dict-like, optional)
                If True, decode times encoded in the standard NetCDF datetime
                format into datetime objects. Otherwise, use
                coders.CFDatetimeCoder or leave them encoded as numbers. Pass a
                mapping, e.g. {"my_variable": False}, to toggle this feature
                per-variable individually. This keyword may not be supported by
                all the backends.
            
            decode_timedelta : (bool, CFTimedeltaCoder, or dict-like, optional)
                If True, decode variables and coordinates with time units in
                {“days”, “hours”, “minutes”, “seconds”, “milliseconds”,
                “microseconds”} into timedelta objects. If False, leave them
                encoded as numbers. If None (default), assume the same value of
                decode_times; if decode_times is a coders.CFDatetimeCoder
                instance, this takes the form of a coders.CFTimedeltaCoder
                instance with a matching time_unit. Pass a mapping, e.g.
                {"my_variable": False}, to toggle this feature per-variable
                individually. This keyword may not be supported by all the
                backends.
            
            use_cftime : (bool or dict-like, optional)
                ---------------------------------------------------------------
                Deprecated since version 2025.01.1: Please pass a
                coders.CFDatetimeCoder instance initialized with use_cftime to
                the decode_times kwarg instead.
                ---------------------------------------------------------------
                Only relevant if encoded dates come from a standard calendar
                (e.g. “gregorian”, “proleptic_gregorian”, “standard”, or not
                 specified). If None (default), attempt to decode times to
                np.datetime64[ns] objects; if this is not possible, decode
                times to cftime.datetime objects. If True, always decode times
                to cftime.datetime objects, regardless of whether or not they
                can be represented using np.datetime64[ns] objects. If False,
                always decode times to np.datetime64[ns] objects; if this is
                not possible raise an error. Pass a mapping, e.g.
                {"my_variable": False}, to toggle this feature per-variable
                individually. This keyword may not be supported by all the
                backends.
            
            concat_characters : (bool or dict-like, optional)
                If True, concatenate along the last dimension of character
                arrays to form string arrays. Dimensions will only be
                concatenated over (and removed) if they have no corresponding
                variable and if they are only used as the last dimension of
                character arrays. Pass a mapping, e.g. {"my_variable": False},
                to toggle this feature per-variable individually. This keyword
                may not be supported by all the backends.
            
            decode_coords : (bool or {"coordinates", "all"}, optional)
                Controls which variables are set as coordinate variables:
                    - “coordinates” or True: Set variables referred to in the
                      'coordinates' attribute of the datasets or individual
                      variables as coordinate variables.

                    - “all”: Set variables referred to in 'grid_mapping',
                      'bounds' and other attributes as coordinate variables.

                Only existing variables can be set as coordinates. Missing
                variables will be silently ignored.
            
            drop_variables : (str or iterable of str, optional)
                A variable or list of variables to exclude from being parsed
                from the dataset. This may be useful to drop variables with
                problems or inconsistent values.
            
            inline_array : (bool, default: False)
                How to include the array in the dask task graph. By
                default(inline_array=False) the array is included in a task by
                itself, and each chunk refers to that task by its key. With
                inline_array=True, Dask will instead inline the array directly
                in the values of the task graph. See dask.array.from_array().
            
            chunked_array_type : (str, optional)
                Which chunked array type to coerce this datasets’ arrays to.
                Defaults to ‘dask’ if installed, else whatever is registered
                via the ChunkManagerEnetryPoint system. Experimental API that
                should not be relied upon.
            
            from_array_kwargs : (dict)
                Additional keyword arguments passed on to the
                ChunkManagerEntrypoint.from_array method used to create chunked
                arrays, via whichever chunk manager is specified through the
                chunked_array_type kwarg. For example if dask.array.Array()
                objects are used for chunking, additional kwargs will be passed
                to dask.array.from_array(). Experimental API that should not be
                relied upon.
            
            backend_kwargs : (dict)
                Additional keyword arguments passed on to the engine open
                function, equivalent to **kwargs.
            
            **kwargs : (dict)
                Additional keyword arguments passed on to the engine open
                function. For example:
                    - ‘group’: path to the netCDF4 group in the given file to
                      open given as a str,supported by “netcdf4”, “h5netcdf”,
                      “zarr”.
                    - ‘lock’: resource lock to use when reading data from disk.
                      Only relevant when using dask or another form of
                      parallelism. By default, appropriate locks are chosen to
                      safely read and write files with the currently active
                      dask scheduler. Supported by “netcdf4”, “h5netcdf”,
                      “scipy”.

                See engine open function for kwargs accepted by each specific
                engine.
            
    Returns
    -------
    full_file : xarray.core.dataarray.Dataset
        Variable(s) of choice, reprojected to EPSG:4326 and with
        'lon' and 'lat' coordinates changed to 'x' and 'y'.
    
    Example
    -------
    # import library
    import ing_utils as ing
    
    # define path to .nc file:
    path = 'cosipy_output_CFSR_JIF_1980_2019.nc'
    
    # get entire CFSR run file into xarray dataset:
    full_file = ing.prep_xarray_dataset(path)
    print(full_file)
    
    '''
    print('Opening original .nc file as xarray.Dataset...')
    # Import path as an xarray Dataset
    full_file = xr.open_dataset(path, decode_coords="all", **kwargs)
    print('Writing EPSG:4326 coordinate system to xr.Dataset...')
    # Assign a coordinate system to full_file (original EPSG4326)
    full_file = full_file.rio.write_crs(
        4326, inplace=True
        ).rio.set_spatial_dims(
        x_dim='lon',
        y_dim='lat',
        inplace=True
    ).rio.write_coordinate_system(inplace=True)
    print(
    "Renaming xr.Dataset lon and lat coords to x and y, respectively..."
        )
    full_file = full_file.rename(lon='x', lat='y')
    return full_file


def convert_subvariables_to_utm(full_file, subvar, crs='EPSG:32608'):
    '''
    Return one of the variables from full_file (named var, selected by
    'subvar'), reprojected to a UTM CRS of choice.

    Parameters
    ----------  
    subvar : str or list of str, optional
        File variable to extract and reproject CRS for.
        
        Can select multiple variables. If selecting multiple variables, report
        the variables in a list.
        
        Possible subvar strings:
            Variables constant in time (lat, lon):
                - 'HGT' : Elevation [m] (lat, lon),
                - 'MASK' : Glacier mask [boolean] (lat, lon),
                - 'SLOPE' : Terrain slope [degrees] (lat, lon),
                - 'ASPECT' : Aspect of slope [degrees] (lat, lon),
            
            Time-varying variables (time, lat, lon):
            * Climate input variables:
                - 'T2' : Air temperature at 2 m [K],
                - 'RH2' : Relative humidity at 2 m [%],
                - 'U2' : Wind velocity at 2 m [m s^(-1)],
                - 'PRES' : Atmospheric pressure [hPa],
                - 'G' : Incoming shortwave radiation [W m^(-2)],
                - 'RRR' : Total precipitation [mm],
                - 'SNOWFALL' : Snowfall [m w.e.],
                - 'N' : Cloud fraction [dimensionless],
                - 'LWin' : Incoming longwave radiation [W m^(-2)],
                - 'RAIN' : Liquid precipitation [mm],
                - 'LWout' : Outgoing longwave radiation [W m^(-2)],
                - 'H' : Sensible heat flux [W m^(-2)],
                - 'LE' : Latent heat flux [W m^(-2)],
                - 'B' : Ground heat flux [W m^(-2)],
                - 'QRR' : Rain heat flux [W m^(-2)],
            
            * Model outputs:
                - 'surfMB' : Surface mass balance [m w.e.],
                - 'MB' : Mass balance [m w.e.],
                - 'Q' : Runoff [m w.e.],
                - 'SNOWHEIGHT' : Snow height [m],
                - 'TOTALHEIGHT' : Total domain height [m],
                - 'TS' : Surface temperature [K],
                - 'ALBEDO' : Albedo [dimensionless],
                - 'LAYERS' : Number of layers in snowpack [dimensionless],
                - 'ME' : Available melt energy [W m^(-2)],
                - 'intMB' : Internal mass balance [m w.e.],
                - 'EVAPORATION' : Evaporation [m w.e.],
                - 'SUBLIMATION' : Sublimation [m w.e.],
                - 'CONDENSATION' : Condensation [m w.e.],
                - 'DEPOSITION' : Deposition [m w.e.],
                - 'REFREEZE' : Refreezing [m w.e.],
                - 'subM' : Subsurface melt [m w.e.],
                - 'Z0' : Roughness length [m],
                - 'surfM' : Surface melt [m w.e.]
        
    crs : str, optional
        EPSG crs code to reproject 'subvar' to.
        
        The default is 'EPSG:32608', i.e. UTM8V, which is the UTM system for
        the Juneau Icefield.
        
    Returns
    -------
    var : xarray.core.dataarray.DataArray if single variable,
          xarray.core.dataset.Dataset if multiple variables.
        Variable(s) of choice, reprojected to chosen CRS and with
        'lon' and 'lat' coordinates changed to 'x' and 'y'.
    
    Example
    -------
    # import library
    import ing_utils as ing
    
    # define path to .nc file:
    path = 'cosipy_output_CFSR_JIF_1980_2019.nc'
    
    # get entire CFSR run file into xarray dataset:
    full_file = ing.prep_xarray_dataset(path)
    print(full_file)
    
    # subset runoff variable Q as an xr.DataArray:
    Q = ing.convert_subvariables_to_utm(full_file, 'Q')
    print(Q)
    
    # get multiple variables from CFSR, e.g. ['surfMB, 'Q']:
    # this will take a bit to run:
    smb_Q = ing.convert_subvariables_to_utm(full_file, ['surfMB', 'Q'])
    print(smb_Q)
    
    '''
    if type(subvar) == list:
        print(
            'Reprojecting following variables to CRS ' + crs + ':'
            )
        print(subvar)
    else:
        print(
        'Reprojecting variable ' + subvar + ' to CRS ' + crs + '...'
        )
    # Subset full_file into variable of interest `subvar` and reproject
    # into crs of choice, default UTM8V via EPSG:32608 for JIF:
    var = full_file[subvar].rio.reproject(crs)
    return var


def check_day(day, monthday='10-01'):
    '''
    Checks if the input 'day' datetime variable has the same month and day
    as the 'monthday' string.

    Parameters
    ----------
    day : np.datetime64
        YYYY-MM-DD datetime to check.
        
    monthday : str
        String in 'MM-DD' format as the reference month and day to check
        'day' against.

    Returns
    -------
    daymatch : bool
        If True, the input datetime has a matching month and day to the
        monthday string.

    '''
    # Cast 'day' as type 'datetime64[D]' in case it isn't in that already
    day = day.astype('datetime64[D]')
    
    # split up 'day' datetime into a month string and a day string:
    monthstr = str(day.item().month)
    daystr = str(day.item().day)
    
    # If the number is <10 (i.e. 1 character long in str form),
    # add a zero to the front fo the string
    if len(monthstr) == 1:
        monthstr = '0' + monthstr
    if len(daystr) == 1:
        daystr = '0' + daystr
    
    # Finally! Check if the monthday str equals the day str
    if monthday == monthstr + '-' + daystr:
        daymatch = True
    else:
        daymatch = False
    return daymatch


def adjust_time_bounds(var, monthday='10-01', time_var='time'):
    '''
    Adjusts the input xarray.Dataset object to only have data for
    times starting and ending according to the recurrence date monthday.

    As an example, if you want your data subsetted by the hydrologic year,
    starting on October 1st and ending on September 30th, you'd select your
    monthday as '10-01'.

    Parameters
    ----------
    var : xarray.core.dataarray.DataArray or xarray.core.dataset.Dataset
        Variable to subset by time bounds.

    monthday : str, optional
        Format 'MM-DD'. The month and date that you want to evaluate the array
        over.
        
        The default is '10-01', i.e. October 1st (start of hydrological year).
    
    time_var : str, optional
        Name of the time variable in var you're wanting to adjust.
        
        The default is 'time'.

    Returns
    -------
    var : xarray.core.dataarray.DataArray or xarray.core.dataset.Dataset
        Original variable(s), now with data adjusted to show intervals at
        monthday and with a new axis 'years' that allows for future subsetting
        of the time data by the year.
    
    Examples
    --------
    # import library
    import ing_utils as ing
    
    # define path to .nc file:
    path = 'cosipy_output_CFSR_JIF_1980_2019.nc'
    
    # get entire CFSR run file into xarray dataset:
    full_file = ing.prep_xarray_dataset(path)
    print(full_file)
    
    # subset runoff variable Q as an xr.DataArray:
    Q = ing.convert_subvariables_to_utm(full_file, 'Q')
    print(Q)
    
    # subset Q by the hydrologic year:
    Q_hydro, hydro_start_time, hydro_end_time = (
        ing.adjust_time_bounds(Q, monthday='10-01')
        ) # time_var='time'
    
    # subset Q by the normal year:
    Q_annual, annual_start_time, annual_end_time = (
        ing.adjust_time_bounds(Q, monthday='01-01')
        ) # time_var='time'
    
    print(Q_hydro, hydro_start_time, hydro_end_time)
    print(Q_annual, annual_start_time, annual_end_time)
    
    '''
    print('Adjusting time bounds to start and end on ' + monthday)
    # -------------PART 1: FILTER TIME VALUES TO MONTHDAY SPAN:----------------
    # extract first and last day from current dataset:
    first_day = var[time_var].values.min().astype('datetime64[D]')
    last_day = var[time_var].values.max().astype('datetime64[D]')

    # Identify nearest start and end dates to clip to based off of monthday:
    timespan_first = np.datetime64(
        str(first_day.item().year) + '-' + monthday
        )
    timespan_last = np.datetime64(
        str(last_day.item().year) + '-' + monthday
        )-1
    # Now use the check_day function to see if the first_day and last_day of
    # var match with the monthday str.
    
    # if check_day == True, we want to set the first_day variable as the
    # vstart_time to maximize the amount of data we're using:
    if check_day(first_day) == True:
        vstart_time = first_day 
    # Otherwise, we can compare against timespan_first:
    else:
        # If the selected monthday happens after the original data's first_day
        # but still in the same year, just set the vstart_time equal to the
        # timespan_first value.
        if timespan_first > first_day:
            vstart_time = timespan_first
        # Otherwise, if your selected monthday would have happened before the
        # start of first_day, then we need to skip ahead to the next monthday,
        # which would happen in the following year from first_day:
        else:
            vstart_time = np.datetime64(
                str(first_day.item().year+1) + '-' + monthday)
            
    # Conceptually, the last day that we want to subset the data by should be
    # the day *before* the monthday recurrence interval. So for this check, we
    # need to check whether or not the last_day PLUS 1 day matches with the
    # monthday.

    # # if check_day is true, make the vend_time == last_day
    if check_day(last_day + 1) == True:
        vend_time = last_day
    
    # Otherwise, we can compare against timespan_last:
    else:
        # If the selected monthday happens before the original data's last_day
        # but still in the same year, just set the vend_time equal to the 
        # timespan_last value:
        if timespan_last < last_day:
            vend_time = timespan_last
        # Otherwise, if your selected monthday would have happened after the
        # end of last_day, then we need to choose the monthday from the year
        # before.
        else:
            vend_time = np.datetime64(
                str(last_day.item().year-1) + '-' + monthday)
    
    # Find the idx in var that is equal to the vstart_time and vend_time:
    vstart_idx = np.where(var[time_var].values.astype('datetime64[D]')
                          == vstart_time)[0][0]
    vend_idx = np.where(var[time_var].values.astype('datetime64[D]')
                        == vend_time)[0][0]
    
    # Now adjust the time bounds of var to be the start and end times:
    var = var.isel({time_var : np.arange(vstart_idx, vend_idx+1)})
    return var, vstart_time, vend_time


def create_years_axis(var, vstart_time, vend_time, monthday='10-01',
                      time_var='time'):
    '''
    Adds a new coordinate axis 'years' that allows users to subset data annually.

    Parameters
    ----------
    var : xarray.core.dataarray.DataArray or xarray.core.dataset.Dataset
        Variable to subset by time bounds.
        
    vstart_time, vend_time : np.datetime64
        Start and end time outputs from adjust_time_bounds.

    monthday : str, optional
        Format 'MM-DD'. The month and date that you want to evaluate the array
        over.
        
        The default is '10-01', i.e. October 1st (start of hydrological year).
    
    time_var : str, optional
        Name of the time variable in var you're wanting to adjust.
        
        The default is 'time'.

    Returns
    -------
    var : xarray.core.dataarray.DataArray or xarray.core.dataset.Dataset
        Original variable(s), now with a new axis 'years' that allows for
        future subsetting of the time data by the year.
    
    Examples
    --------
    # import library
    import ing_utils as ing
    
    # define path to .nc file:
    path = 'cosipy_output_CFSR_JIF_1980_2019.nc'
    
    # get entire CFSR run file into xarray dataset:
    full_file = ing.prep_xarray_dataset(path)
    print(full_file)
    
    # subset runoff variable Q as an xr.DataArray:
    Q = ing.convert_subvariables_to_utm(full_file, 'Q')
    print(Q)
    
    monthday='10-01'
    
    # subset Q by the hydrologic year:
    Q_hydro, vstart_time, vend_time = (
        ing.adjust_time_bounds(Q, monthday=monthday)
        ) # time_var='time'
    
    # now create a years axis for Q_hydro
    Q_hydro = ing.create_years_axis(Q_hydro, vstart_time, vend_time,
                                    monthday=monthday)
    print(Q_hydro)

    '''
    print('Creating years axis recurring on ' + monthday)
    # -----------------PART 2: CREATE YEARS COORDINATE AXIS:-------------------
    # Now we're going to create a new coordinate 'years' that has the same
    # dimensions of the time_var, but now all of the days are subsetted into
    # years so they can be summed, averaged, etc. by the user later.
    
    # The 'years' variable is identifying the year interval that a set of days
    # is assigned to. So if monthday = '10-01' (start of hydrologic year),
    # the days 1980-10-01 to 1981-09-30 are in the hydrologic year of 1981.
    
    # First, create an array that adds up the number of days in an average
    # year (not including leap years):
    daysofyr = np.cumsum(
        np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])
        )
    
    # Split up the monthday string into int values:
    month = int(monthday[:2])
    day = int(monthday[3:])
    
    # Using these int values, index the daysofyr value at the month value
    # minus 1 and then add the number of days to get the day of year that
    # the selected monthday string is at:
    monthday_doy = daysofyr[month - 1] + day
    
    # If the monthday_doy is in the first half of the year, make the 'years'
    # array start at the vstart_time's year and end at the vend_time year.
    # Ex if monthday = '01-01', the days 1980-01-01 to 1980-12-31 are assigned
    # to the year 1980.
    if monthday_doy < np.round(365/2):
        years = np.arange(vstart_time.item().year,
                          vend_time.item().year+1)
    
    # If the monthday_doy is in the second half of the year, make the 'years'
    # array start at the vstart_time's year +1 and end at the vend_time year.
    # Ex if monthday = '10-01', the days 1980-10-01 to 1981-09-30 are assigned
    # to the year 1981.
    else:
        years = np.arange(vstart_time.item().year+1,
                          vend_time.item().year+1)
    
    # We're going to initialize our 'new_index' for the 'years' variable:
    new_index = np.array([])
    
    # Then we're going to iterate over the 'years' array to get the size of
    # each year length, get the ID's of those values, and then add to the
    # new_index an array with the same shape as that date span with all the
    # values equal to the years value at i
    for i in range(len(years)):
        if monthday_doy < np.round(365/2):
            date_idx = np.where(
                (var[time_var].values.astype('datetime64[D]')
                 >= np.datetime64(str(years[i]) + '-' + monthday)) &
                (var[time_var].values.astype('datetime64[D]')
                 < np.datetime64(str(years[i]+1) + '-' + monthday)))
        else:
            date_idx = np.where(
                (var[time_var].values.astype('datetime64[D]')
                 >= np.datetime64(str(years[i]-1) + '-' + monthday)) &
                (var[time_var].values.astype('datetime64[D]')
                 < np.datetime64(str(years[i]) + '-' + monthday)))
        datespan = var[time_var].values[date_idx]
        new_index = np.concatenate((new_index,
                                    years[i]*np.ones(datespan.shape)),
                                   axis=0)
    # Cast the final new_index as type 'float32'
    # (this is for IGM compatability)
    new_index = new_index.astype('float32')

    # FINALLY assign a new coordinate axis 'years' that still uses time_var as
    # dimensions, but filling the new axis with the array new_index.
    var = var.assign_coords(years=(time_var, new_index))
    return var


def assign_time(var, monthday='10-01', time_var='time'):
    '''
    Adjusts the input xarray.Dataset object to only have data for
    times recurring at the recurrence date monthday. Also adds a new coordinate
    axis 'years' that allows users to subset data annually.
    
    TL;DR, combines adjust_time_bounds and create_years_axis into one function.

    As an example, if you want your data subsetted by the hydrologic year,
    starting on October 1st and ending on September 30th, you'd select your
    monthday as '10-01'.

    Parameters
    ----------
    var : xarray.core.dataarray.DataArray or xarray.core.dataset.Dataset
        Variable to subset by time bounds.

    monthday : str, optional
        Format 'MM-DD'. The month and date that you want to evaluate the array
        over.
        
        The default is '10-01', i.e. October 1st (start of hydrological year).
    
    time_var : str, optional
        Name of the time variable in var you're wanting to adjust.
        
        The default is 'time'.

    Returns
    -------
    var : xarray.core.dataarray.DataArray or xarray.core.dataset.Dataset
        Original variable(s), now with data adjusted to show intervals at
        monthday and with a new axis 'years' that allows for future subsetting
        of the time data by the year.
    
    Examples
    --------
    # import library
    import ing_utils as ing
    
    # define path to .nc file:
    path = 'cosipy_output_CFSR_JIF_1980_2019.nc'
    
    # get entire CFSR run file into xarray dataset:
    full_file = ing.prep_xarray_dataset(path)
    print(full_file)
    
    # subset runoff variable Q as an xr.DataArray:
    Q = ing.convert_subvariables_to_utm(full_file, 'Q')
    print(Q)
    
    monthday='10-01'
    
    # subset Q by the hydrologic year:
    Q_hydro = ing.assign_time(Q, monthday=monthday)
    print(Q_hydro)
    '''
    var, vstart_time, vend_time = adjust_time_bounds(var, monthday=monthday,
                                                     time_var=time_var)
    var = create_years_axis(var, vstart_time, vend_time, monthday=monthday,
                            time_var=time_var)
    return var


def prepare_analysis_file(path, subvar, crs='EPSG:32608', monthday='10-01',
                          time_var='time', **kwargs):
    '''
    Prepares Ing datast for analysis by combining prep_xarray_dataset,
    convert_subvariables_to_utm, and assign_time functions together.

    Parameters
    ----------
    path : str
        File path to .nc file from Ing et al. (2025).
        
        Strings and Path objects are interpreted as a path to a netCDF file or
        an OpenDAP URL and opened with python-netCDF4, unless the filename ends
        with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). Byte-strings or file-like
        objects are opened by scipy.io.netcdf (netCDF3) or h5py (netCDF4/HDF).
        
    subvar : str or list of str, optional
        File variable to extract and reproject CRS for.
        
        Can select multiple variables. If selecting multiple variables, report
        the variables in a list.
        
        Possible subvar strings:
            Variables constant in time (lat, lon):
                - 'HGT' : Elevation [m] (lat, lon),
                - 'MASK' : Glacier mask [boolean] (lat, lon),
                - 'SLOPE' : Terrain slope [degrees] (lat, lon),
                - 'ASPECT' : Aspect of slope [degrees] (lat, lon),
            
            Time-varying variables (time, lat, lon):
            * Climate input variables:
                - 'T2' : Air temperature at 2 m [K],
                - 'RH2' : Relative humidity at 2 m [%],
                - 'U2' : Wind velocity at 2 m [m s^(-1)],
                - 'PRES' : Atmospheric pressure [hPa],
                - 'G' : Incoming shortwave radiation [W m^(-2)],
                - 'RRR' : Total precipitation [mm],
                - 'SNOWFALL' : Snowfall [m w.e.],
                - 'N' : Cloud fraction [dimensionless],
                - 'LWin' : Incoming longwave radiation [W m^(-2)],
                - 'RAIN' : Liquid precipitation [mm],
                - 'LWout' : Outgoing longwave radiation [W m^(-2)],
                - 'H' : Sensible heat flux [W m^(-2)],
                - 'LE' : Latent heat flux [W m^(-2)],
                - 'B' : Ground heat flux [W m^(-2)],
                - 'QRR' : Rain heat flux [W m^(-2)],
            
            * Model outputs:
                - 'surfMB' : Surface mass balance [m w.e.],
                - 'MB' : Mass balance [m w.e.],
                - 'Q' : Runoff [m w.e.],
                - 'SNOWHEIGHT' : Snow height [m],
                - 'TOTALHEIGHT' : Total domain height [m],
                - 'TS' : Surface temperature [K],
                - 'ALBEDO' : Albedo [dimensionless],
                - 'LAYERS' : Number of layers in snowpack [dimensionless],
                - 'ME' : Available melt energy [W m^(-2)],
                - 'intMB' : Internal mass balance [m w.e.],
                - 'EVAPORATION' : Evaporation [m w.e.],
                - 'SUBLIMATION' : Sublimation [m w.e.],
                - 'CONDENSATION' : Condensation [m w.e.],
                - 'DEPOSITION' : Deposition [m w.e.],
                - 'REFREEZE' : Refreezing [m w.e.],
                - 'subM' : Subsurface melt [m w.e.],
                - 'Z0' : Roughness length [m],
                - 'surfM' : Surface melt [m w.e.]
    crs : str, optional
        EPSG crs code to reproject 'subvar' to.
        
        The default is 'EPSG:32608', i.e. UTM8V, which is the UTM system for
        the Juneau Icefield.
        
    monthday : str, optional
        Format 'MM-DD'. The month and date that you want to evaluate the array
        over.
        
        The default is '10-01', i.e. October 1st (start of hydrological year).
        
    time_var : str, optional
        Name of the time variable in var you're wanting to adjust.
        
        The default is 'time'.
    
    **kwargs : ({engine=None, chunks=None, cache=None, decode_cf=None,
                 mask_and_scale=None, decode_times=None, decode_timedelta=None,
                 use_cftime=None, concat_characters=None, decode_coords=None,
                 drop_variables=None, inline_array=False,
                 chunked_array_type=None, from_array_kwargs=None,
                 backend_kwargs=None, **kwargs}, optional)
    
        Additional keyword arguments passed to xarray.open_dataset():
            engine : ({"netcdf4", "scipy", "pydap", "h5netcdf", "zarr", None},
                      installed backend or subclass of
                      xarray.backends.BackendEntrypoint, optional)
                Engine to use when reading files. If not provided, the default
                engine is chosen based on available dependencies, with a
                preference for “netcdf4”. A custom backend class (a subclass of
                BackendEntrypoint) can also be used.
                
            chunks : (int, dict, 'auto' or None, default: None)
                If provided, used to load the data into dask arrays.

                    - chunks="auto" will use dask auto chunking taking into
                      account the engine preferred chunks.

                    - chunks=None skips using dask, which is generally faster
                      for small arrays.

                    - chunks=-1 loads the data with dask using a single chunk
                      for all arrays.

                    - chunks={} loads the data with dask using the engine’s
                      preferred chunk size, generally identical to the format’s
                      chunk size. If not available, a single chunk for all
                      arrays.

                See dask chunking for more details.
            
            cache : (bool, optional)
                If True, cache data loaded from the underlying datastore in
                memory as NumPy arrays when accessed to avoid reading from the
                underlying data- store multiple times. Defaults to True unless
                you specify the chunks argument to use dask, in which case it
                defaults to False. Does not change the behavior of coordinates
                corresponding to dimensions, which always load their data from
                disk into a pandas.Index.
            
            decode_cf : (bool, optional)
                Whether to decode these variables, assuming they were saved
                according to CF conventions.
            
            mask_and_scale : (bool or dict-like, optional)
                If True, replace array values equal to _FillValue with NA and
                scale values according to the formula
                original_values * scale_factor + add_offset,
                where _FillValue, scale_factor and add_offset are taken from
                variable attributes (if they exist). If the _FillValue or
                missing_value attribute contains multiple values a warning will
                be issued and all array values matching one of the multiple 
                alues will be replaced by NA. Pass a mapping, e.g.
                {"my_variable": False}, to toggle this feature per-variable
                individually. This keyword may not be supported by all the
                backends.
            
            decode_times : (bool, CFDatetimeCoder or dict-like, optional)
                If True, decode times encoded in the standard NetCDF datetime
                format into datetime objects. Otherwise, use
                coders.CFDatetimeCoder or leave them encoded as numbers. Pass a
                mapping, e.g. {"my_variable": False}, to toggle this feature
                per-variable individually. This keyword may not be supported by
                all the backends.
            
            decode_timedelta : (bool, CFTimedeltaCoder, or dict-like, optional)
                If True, decode variables and coordinates with time units in
                {“days”, “hours”, “minutes”, “seconds”, “milliseconds”,
                “microseconds”} into timedelta objects. If False, leave them
                encoded as numbers. If None (default), assume the same value of
                decode_times; if decode_times is a coders.CFDatetimeCoder
                instance, this takes the form of a coders.CFTimedeltaCoder
                instance with a matching time_unit. Pass a mapping, e.g.
                {"my_variable": False}, to toggle this feature per-variable
                individually. This keyword may not be supported by all the
                backends.
            
            use_cftime : (bool or dict-like, optional)
                ---------------------------------------------------------------
                Deprecated since version 2025.01.1: Please pass a
                coders.CFDatetimeCoder instance initialized with use_cftime to
                the decode_times kwarg instead.
                ---------------------------------------------------------------
                Only relevant if encoded dates come from a standard calendar
                (e.g. “gregorian”, “proleptic_gregorian”, “standard”, or not
                 specified). If None (default), attempt to decode times to
                np.datetime64[ns] objects; if this is not possible, decode
                times to cftime.datetime objects. If True, always decode times
                to cftime.datetime objects, regardless of whether or not they
                can be represented using np.datetime64[ns] objects. If False,
                always decode times to np.datetime64[ns] objects; if this is
                not possible raise an error. Pass a mapping, e.g.
                {"my_variable": False}, to toggle this feature per-variable
                individually. This keyword may not be supported by all the
                backends.
            
            concat_characters : (bool or dict-like, optional)
                If True, concatenate along the last dimension of character
                arrays to form string arrays. Dimensions will only be
                concatenated over (and removed) if they have no corresponding
                variable and if they are only used as the last dimension of
                character arrays. Pass a mapping, e.g. {"my_variable": False},
                to toggle this feature per-variable individually. This keyword
                may not be supported by all the backends.
            
            decode_coords : (bool or {"coordinates", "all"}, optional)
                Controls which variables are set as coordinate variables:
                    - “coordinates” or True: Set variables referred to in the
                      'coordinates' attribute of the datasets or individual
                      variables as coordinate variables.

                    - “all”: Set variables referred to in 'grid_mapping',
                      'bounds' and other attributes as coordinate variables.

                Only existing variables can be set as coordinates. Missing
                variables will be silently ignored.
            
            drop_variables : (str or iterable of str, optional)
                A variable or list of variables to exclude from being parsed
                from the dataset. This may be useful to drop variables with
                problems or inconsistent values.
            
            inline_array : (bool, default: False)
                How to include the array in the dask task graph. By
                default(inline_array=False) the array is included in a task by
                itself, and each chunk refers to that task by its key. With
                inline_array=True, Dask will instead inline the array directly
                in the values of the task graph. See dask.array.from_array().
            
            chunked_array_type : (str, optional)
                Which chunked array type to coerce this datasets’ arrays to.
                Defaults to ‘dask’ if installed, else whatever is registered
                via the ChunkManagerEnetryPoint system. Experimental API that
                should not be relied upon.
            
            from_array_kwargs : (dict)
                Additional keyword arguments passed on to the
                ChunkManagerEntrypoint.from_array method used to create chunked
                arrays, via whichever chunk manager is specified through the
                chunked_array_type kwarg. For example if dask.array.Array()
                objects are used for chunking, additional kwargs will be passed
                to dask.array.from_array(). Experimental API that should not be
                relied upon.
            
            backend_kwargs : (dict)
                Additional keyword arguments passed on to the engine open
                function, equivalent to **kwargs.
            
            **kwargs : (dict)
                Additional keyword arguments passed on to the engine open
                function. For example:
                    - ‘group’: path to the netCDF4 group in the given file to
                      open given as a str,supported by “netcdf4”, “h5netcdf”,
                      “zarr”.
                    - ‘lock’: resource lock to use when reading data from disk.
                      Only relevant when using dask or another form of
                      parallelism. By default, appropriate locks are chosen to
                      safely read and write files with the currently active
                      dask scheduler. Supported by “netcdf4”, “h5netcdf”,
                      “scipy”.

                See engine open function for kwargs accepted by each specific
                engine.

    Returns
    -------
    var : xarray.core.dataarray.DataArray or xarray.core.dataset.Dataset
        Original variable(s), now with data adjusted to selected CRS, with
        time intervals recurring at monthday, and with a new axis 'years'
        that allows for future subsetting of the time data by the year.

    Examples
    --------
    # import library
    import ing_utils as ing
    
    # define path to .nc file:
    path = 'cosipy_output_CFSR_JIF_1980_2019.nc'
    
    subvar = ['HGT', 'surfMB', 'SNOWFALL']
    
    # get entire CFSR run file into xarray dataset:
    var = ing.prepare_analysis_file(path, subvar, crs='EPSG:32608',
                                    monthday='10-01', time_var='time')
    '''
    full_file = prep_xarray_dataset(path, **kwargs)
    var = convert_subvariables_to_utm(full_file, subvar, crs=crs)
    var = assign_time(var, monthday=monthday, time_var=time_var)
    print("Analysis file preparation complete.")
    return var


def get_gpd_region(region='01', version='62'):
    '''
    Imports GeoPandas 'gpd' file from Open Global Glacier Model (OGGM) of a
    given RGI region, using default RGI version '62'.

    Parameters
    ----------
    region : str, optional
        RGI region number from ‘01’ to ‘19’.
        The default is '01' for Alaska.

    version : str, optional
        RGI version number, options are ‘62’, ‘70G’, or ‘70C’.
        The default is '62'.

    Returns
    -------
    gpd_region : geopandas.GeoDataFrame
        GeoDataFrame of given RGI region & version containing all glaciers for
        that region.
    
    Example
    -------
    # import library
    import ing_utils as ing
    
    # you'll need to have oggm and geopandas installed to your
    environment/kernel for this to work
    
    # Get the gpd region for RGI region '01' (Alaska) using RGI v. 62
    gpd_region = ing.get_gpd_region()
    
    # Get the gpd region for RGI region '01' (Alaska) using RGI v. 70C
    gpd_regionv7c = ing.get_gpd_region(version='70C')
    '''
    print('Getting RGI (version ' + version + ') region ' + region + '...')
    # Get file for given RGI region and version:
    fr = utils.get_rgi_region_file(region, version=version)
    
    # read the file as a gpd.GeoDataFrame:
    gpd_region = gpd.read_file(fr)
    return gpd_region

def get_range(arr, contour_int):
    cmin = contour_int * np.round(arr[~np.isnan(arr)].min()/contour_int)
    cmax = contour_int * np.round(arr[~np.isnan(arr)].max()/contour_int)
    return cmin, cmax

def get_bins(arr, contour_int):
    cmin, cmax = get_range(arr, contour_int)
    bins = np.arange(cmin, cmax+contour_int, contour_int)
    return bins

def get_hypsometry(dataset, contour_int=50):
    '''
    Calculates the hypsometry for a glacier.

    Parameters
    ----------
    dataset : xarray.core.dataset.Dataset
        Subsetted dataset to extract hypsometry from.
        Must contain variables 'HGT' and 'surfMB' and contain a 'years' axis.

    contour_int : int, optional
        Change in elevation in meters to separate contours by.

        The default is 50 m.

    Returns
    -------
    surfmb_elev : np.ndarray
        Surface mass balance values at each elevation band.
    
    elev : np.ndarray
        Elevation bands.

    areas : np.ndarray
        Areas of each elevation band.
    '''
    if type(dataset) == xr.core.dataset.Dataset:
        surfmb = dataset['surfMB'].groupby('years').sum(skipna=False)
        hgt = np.full(dataset['HGT'].shape, np.nan)
        # surfmb_dataidx is the same for all years (same mask):
        surfmb_dataidx = np.where(~np.isnan(surfmb.isel(years=0).values))
        # replace nan values with height values at points
        # where there are surfmb data:
        hgt[surfmb_dataidx] = dataset['HGT'].values[surfmb_dataidx]

        # Calculate the histogram of pixels in elevation bands defined by
        # contour_int:
        elevbins = get_bins(hgt, contour_int=contour_int)
        counts, elev = np.histogram(hgt[~np.isnan(hgt)],
                                    bins=elevbins,
                                    range=get_range(hgt,
                                                    contour_int=contour_int))
        elev = elev[:-1] # remove the last bin edge (don't need)

        dx, dy = (np.abs(np.diff(dataset.x.values))[0],
                  np.abs(np.diff(dataset.y.values))[0])
        A_pix = dx*dy
        areas = A_pix * counts
        surfmb_elev = []
        for i in range(len(surfmb.years.values)):
            smb_yr = []
            for j in range(len(elevbins)-1):
                idx = np.where((hgt[~np.isnan(hgt)] >= elevbins[j])
                               & (hgt[~np.isnan(hgt)] < elevbins[j+1]))[0]
                smb_percontour = (surfmb.isel(years=i)\
                                  .values[
                                      ~np.isnan(surfmb.isel(years=i).values)
                                      ][idx])
                if smb_percontour.shape == (0,):
                    smb_percontour = np.nan
                else:
                    smb_percontour = smb_percontour.sum()
                smb_yr.append(smb_percontour*A_pix/areas[j])
            smb_yr = np.array(smb_yr)
            surfmb_elev.append(smb_yr)
        return np.array(surfmb_elev), elev, areas
    else:
        raise ValueError('Dataset must be an xarray.core.dataset.Dataset.')
    
    
def calc_r2(y, SSR):
    '''
    Calculate the Coefficient of Determination R2 for a given set of y values
    and the sum of squared residuals (SSR) from the least-squares fit.
    
    Parameters
    ----------
    y : np.ndarray
        Y values used in polynomial fit.
    
    SSR : np.ndarray or float
        Sum of Squared Residuals (SSR) from least-squares fit.
        This is the 0th value in the 1st output from the
        np.polynomial.polynomial.P.polyfit function when `full=True`.
    
    Returns
    -------
    r2 : float or np.ndarray
        Coefficient of Determination R2 for the polynomial fit.
    '''
    SST = ((y - y.mean())**2).sum() # Sum of Square Total
    r2 = 1 - SSR/SST # Calculate Coefficient of Determination
    return r2[0]

def get_polyfit(x, y, deg=1):
    '''
    Calculates the coefficients and R2 for a given x and y.

    Parameters
    ----------
    x : np.ndarray
        X values used in polynomial fit.

    y : np.ndarray
        Y values used in polynomial fit.

    deg : int, optional
        Indicates the degree of the polynomial fit.
        The default is 1.

    Returns
    -------
    coeffs : tuple
        The coefficients of the result.

    r2 : float
        The R2 value of the polynomial fit.
    '''
    if np.any(np.isnan(y)) == True:
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
    coeffs, full = P.polyfit(x, y, deg=deg, full=True)
    r2 = calc_r2(y, full[0])
    return coeffs, r2

def get_smb_simple_param_v2(glacier, txtfilename='smb_simple_param.txt',
                            contour_int=50, year_start=2005., window_size=1,
                            **kwargs):
    '''
    Calculates and returns a simplified table of years, accumulation gradients,
    ablation gradients, equilibrium line altitudes, and max accumulation for
    a selected glacier in the Ing et al. (2025) COSIPY results for the Juneau
    Icefield.
    
    The saved output table is formatted to be used as an input for an IGM run
    using the smb_simple module.

    Parameters
    ----------
    var : xarray.core.dataset.Dataset, optional
        Variable from ing.prep_var() that MUST AT LEAST contain variables
        'HGT', 'surfMB', and 'SNOWFALL'.
        
        Otherwise, the variable will be constructed from the path, if given.
        
        The default is None.
        
    masked_var : xarray.core.dataset.Dataset, optional
        var that has been subsetted to a glacier outline using
        ing.maskprep_var(). glacier MUST AT LEAST contain variables
        'HGT', 'surfMB', and 'SNOWFALL'.
        
        Otherwise, the masked_var variable will be constructed from the path,
        if given.
        
        The default is None.
        
    path : str, optional
        File path to .nc file from Ing et al. (2025).
        
        Strings and Path objects are interpreted as a path to a netCDF file or
        an OpenDAP URL and opened with python-netCDF4, unless the filename ends
        with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). Byte-strings or file-like
        objects are opened by scipy.io.netcdf (netCDF3) or h5py (netCDF4/HDF).
        
        The default is None.
        
    monthday : str, optional
        Format 'MM-DD'. The month and date that you want to evaluate the array
        over.
        
        The default is '10-01', i.e. October 1st (start of hydrological year).
        
    txtfilename : str, optional
        Filename of output saved file.
        
        The default is 'smb_simple_param.txt'.
        
    contour_int : int, optional
        Change in elevation in meters to separate contours by.

        The default is 50 m.
        
    year_start : float32, optional
        The start year (float) that you would like to evaluate over.
        
        The default is 2005..
    
    window_size : int, optional
        The number of years to calculate the averages for. If window_size == 1,
        the output list will be for every year in the model run.
        
        The default is 1.
        
    **kwargs : {'glaciermask': geopandas.DataFrame, 'rgiid': str,
                'gpd_region': geopandas.DataFrame, 'all_touched': bool,
                'time_var': str 'crs': str, 'region' : str, 'version': str},
               optional
        Passed to maskprep_var():
            glaciermask : geopandas.DataFrame, optional
                Glacier of choice based on RGI Id, reprojected to match chosen
                CRS.
                
                If None given, mask_var_by_glaciermask() will use
                get_glaciermask() to generate a glaciermask. If doing this,
                provide an RGI Id str (see next variable description).
                
                The default is None.
                
            rgiid : str, optional
                RGI Id of chosen glacier. Must be provided if glaciermask is
                None.
                
                The default is None.
                
            gpd_region : geopandas.DataFrame, optional
                GeoDataFrame from get_gpd_region() for given RGI region and
                version.
                
                If None given, mask_var_by_glaciermask() will use
                get_glaciermask() to generate a gpd_region.
                
                If you are using an RGI Id from a different RGI version
                (e.g. v5, 7, etc.), or region, use **kwargs to adjust the
                parameters.
                
                The default is None.
            
            all_touched : bool, optional
                If True, pixels overlapping with glaciermask polygon are
                included when var is clipped. If False, only pixels within
                bounds of glaciermask are saved.
            
            time_var : str, optional
                Name of the time variable in var you're wanting to adjust.
                
                The default is 'time'.
                
            crs : str, optional
                CRS to reproject the output glaciermask to.
                The default is 'EPSG:32608'.
            
            region : str
                RGI region number.
                The default is '01' for Alaska.
                
            version : str
                RGI version number.
                The default is '62'.

    Raises
    ------
    ValueError
        If var or glacier do not contain variables 'HGT', 'surfMB', or
        'SNOWFALL' AND if a path is not provided.

    Returns
    -------
    smb_simple_param : pd.DataFrame
        Table output for inline plotting.
        Also simultaneously saves file as a txt file in the current directory.

    '''
    snow = glacier['SNOWFALL'].groupby('years').sum(skipna=False)
    years = snow.years.values
    smb_simple_param = []
    smb_simple_param.append(["time", "gradabl", "gradacc", "ela", "accmax"])
    if window_size == 1:
        #years = years[np.where(years >= year_start)] # initially subset years to shorten record
        surfmb_elev, elev, areas = get_hypsometry(glacier, contour_int=contour_int)
        for i in range(len(years)):
            coeffs_ela, r2_ela = get_polyfit(elev, surfmb_elev[i])
            idx_pos, idx_neg = (np.where(surfmb_elev[i] > 0.0),
                                np.where(surfmb_elev[i] < 0.0))
            if idx_pos[0].shape[0] <= 2:
                coeffs_pos, r2_pos = get_polyfit(elev, surfmb_elev[i])
            else:
                coeffs_pos, r2_pos = get_polyfit(elev[idx_pos],
                                                 surfmb_elev[i][idx_pos])
            if idx_neg[0].shape[0] <= 2:
                coeffs_neg, r2_neg = get_polyfit(elev, surfmb_elev[i])
            else:
                coeffs_neg, r2_neg = get_polyfit(elev[idx_neg],
                                                 surfmb_elev[i][idx_neg])
            ela = int(np.round(-coeffs_ela[-2]/coeffs_ela[-1]))
            gradacc, gradabl = (np.round(coeffs_pos[-1], 3),
                                np.round(coeffs_neg[-1], 3))
            maxacc = np.round(snow.isel(years=i).max().values, 1)
            row_i = [int(years[i]), gradabl, gradacc, ela, maxacc]
            smb_simple_param.append(row_i)
    elif window_size > 1:
        year_windows = years[np.where(np.arange(len(years))%window_size == 0)]
        if len(years)%window_size != 0:
            year_windows = np.append(year_windows, years[-1])
        #if np.isin(year_start, year_windows) == False:
        #    year_windows = year_windows[
        #        np.where(year_windows >= year_start - 2*window_size)
        #        ]
        #else:
        #    year_windows = year_windows[
        #        np.where(year_windows >= year_start - window_size)
        #        ]
        surfmb_elev, elev, areas = get_hypsometry(glacier,
                                                  contour_int=contour_int)
        for i in range(1, len(year_windows)):
            idx = np.where((years >= year_windows[i-1])
                           & (years < year_windows[i]))
            mean_surfmb_elev = np.mean(surfmb_elev[idx], axis=0)
            coeffs_ela, r2_ela = get_polyfit(elev, mean_surfmb_elev, deg=1)
            idx_pos, idx_neg = (np.where(mean_surfmb_elev > 0.0),
                                np.where(mean_surfmb_elev < 0.0))
            if idx_pos[0].shape[0] <= 2:
                coeffs_pos, r2_pos = get_polyfit(elev, mean_surfmb_elev, deg=1)
            else:
                coeffs_pos, r2_pos = get_polyfit(elev[idx_pos],
                                                 mean_surfmb_elev[idx_pos],
                                                 deg=1)
            if idx_neg[0].shape[0] <= 2:
                coeffs_neg, r2_neg = get_polyfit(elev, mean_surfmb_elev, deg=1)
            else:
                coeffs_neg, r2_neg = get_polyfit(elev[idx_neg],
                                                 mean_surfmb_elev[idx_neg],
                                                 deg=1)
            ela = int(np.round(-coeffs_ela[-2]/coeffs_ela[-1]))
            gradacc, gradabl = (np.round(coeffs_pos[-1], 3),
                                np.round(coeffs_neg[-1], 3))
            maxacc = np.round(
                snow.isel(years=idx[0]).max(dim=['x', 'y']).mean().values, 1
                )
            row_i = [int(year_windows[i]), gradabl, gradacc, ela, maxacc]
            smb_simple_param.append(row_i)
    smb_simple_param = pd.DataFrame(smb_simple_param[1:],
                                    columns=smb_simple_param[0])
    smb_simple_param.to_csv(txtfilename, sep=' ', index=False)
    return smb_simple_param

def calc_dailyQ(glacierQ, savetable=False, glacierstr=None):
    '''
    Calculates daily runoff 'Q' in m^3/s that is integrated over the whole
    glacier area for every day in record provided by input glacierQ variable.
    Optionally saves the resulting table as a .csv file.

    Parameters
    ----------
    glacierQ : xarray.core.dataarray.DataArray or xarray.core.dataset.Dataset
        Masked var to glacier RGI that contains at least variable 'Q'.
        
    savetable : bool, optional
        Whether or not to save the output table as a .csv file.
        
        The default is False.
        
    glacierstr : str, optional
        Name of glacier to use in the save file.
        
        The default is None.

    Returns
    -------
    q_daily : numpy.ndarray
        Numpy array containing all daily runoff data Q in m^3/s integrated over
        the whole glacier.

    '''
    A_pix = (np.abs(np.diff(glacierQ.x.values))[0]*
             np.abs(np.diff(glacierQ.y.values))[0])# area of pixel, [m^2]
    q_daily = []
    for i in np.unique(glacierQ.years.values):
        subset_year = glacierQ.isel(
            time=np.where(glacierQ.years.values == i)[0]
            )
        m3s = (subset_year.sum(dim=['x', 'y']) *
               A_pix/(24*60*60)) # value in m^3/s
        q_daily.append(m3s.values)
    q_daily = np.array(q_daily)
    if savetable == True:
        daysinmonth = np.array([0, 31, 28, 31, 30, 31, 30,
                                31, 31, 30, 31, 30, 31])
        years = (glacierQ.groupby('years').sum(skipna=False)\
                 .years.values.astype('int').tolist())
        years = list(map(str, years))
        monthdays = daysinmonth[1:]
        months = np.concatenate([(np.ones(monthdays[i])*i+1).astype('int')
                                 for i in range(len(monthdays))])
        daily_Q_data = np.column_stack(
            [np.arange(1, 365+1, 1).astype('int'),  months, q_daily.T]
            )
        daily_Q_data = pd.DataFrame(daily_Q_data,
                                    columns=['dayofyr', 'month'] + years)
        daily_Q_data.to_csv(glacierstr + '_daily_Q.csv', index=False)
    return q_daily

def calc_summarystats_dailyQ(q_daily=None, glacierQ=None, window_size=7):
    '''
    Calculates summary statistics of daily runoff data from calc_dailyQ (e.g.
    maximum recorded daily Q, minimum recorded daily Q, mean daily Q, and
    rolling mean daily Q determined by window_size).

    Parameters
    ----------
    q_daily : np.ndarray, optional
        Daily runoff statistics for record. If not provided, will generate
        statistics using calc_dailyQ from given glacierQ.
        
        The default is None.
    
    glacierQ : xarray.core.dataarray.DataArray or xarray.core.dataset.Dataset,
               optional
        Masked var to glacier RGI that contains at least variable 'Q'.
        
        The default is None.
        
    window_size : int, optional
        Window size to calculate rolling mean over, in days.
        
        The default is 7.

    Returns
    -------
    dss : pandas.DataFrame
        Summary Pandas DataFrame containing the following columns:
            'maxQ' : Maximum recorded runoff Q per day [m^3/s]
            'minQ' : Minimum recorded runoff Q per day [m^3/s]
            'meanQ' : Average recorded runoff Q per day [m^3/s]
            'rollingQ': n-day rolling mean of runoff Q per day [m^3/s]
    '''
    if q_daily is None:
        q_daily = calc_dailyQ(glacierQ)
    maxQ = q_daily.max(axis=0)
    minQ = q_daily.min(axis=0)
    meanQ = np.mean(q_daily, axis=0)
    rollingQ = pd.Series(meanQ).rolling(window_size,
                                        center=True).mean().to_numpy()
    results = np.column_stack([maxQ, minQ, meanQ, rollingQ])
    dss = pd.DataFrame(results,
                       columns=['maxQ', 'minQ', 'meanQ', 'rollingQ'])
    return dss


def calc_summarystats_monthlyQ(q_daily=None, glacierQ=None, savetable=False,
                               glacierstr=None):
    '''
    Calculates monthly summary statistics of daily runoff data from
    calc_dailyQ (e.g. maximum recorded Q in month, mean maximum recorded Q in
    month, minimum recorded Q in month, mean Q in month, mean minimum recorded
    Q in month, nad minimum recorded Q in month).
    Optionally saves the resulting table as a .csv file.

    Parameters
    ----------
    q_daily : np.ndarray, optional
        Daily runoff statistics for record. If not provided, will generate
        statistics using calc_dailyQ from given glacierQ.
        
        The default is None.
    
    glacierQ : xarray.core.dataarray.DataArray or xarray.core.dataset.Dataset,
               optional
        Masked var to glacier RGI that contains at least variable 'Q'.
        
    savetable : bool, optional
        Whether or not to save the output table as a .csv file.
        
        The default is False.
        
    glacierstr : str, optional
        Name of glacier to use in the save file.
        
        The default is None.

    Returns
    -------
    mss : pandas.DataFrame
        "Daily Summary Statistics". Summary Pandas DataFrame of monthly runoff
        statistics containing columns:
            'month' : month names, abbreviated
            'maxQ' : Maximum recorded Q in month over period [m^3/s]
            'meanmaxQ' : Average Maximum Q in months over period [m^3/s]
            'meanQ' : Average Q in month over period [m^3/s]
            'meanminQ' : Average Minimum Q in months over period [m^3/s]
            'minQ' : Minimum recorded Q in month over period [m^3/s]

    '''
    if q_daily is None:
        q_daily = calc_dailyQ(glacierQ)
    daysinmonth = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    daysofyr = np.cumsum(daysinmonth)
    m_max_Q, m_meanmax_Q = [], []
    m_mean_Q = []
    m_min_Q, m_meanmin_Q = [], []
    for i in range(len(daysofyr)-1):
        m_subset = q_daily[:, daysofyr[i]:daysofyr[i+1]]
        m_max_Q.append(m_subset.max()) # monthly record max Q
        # mean daily maximum in January (or whatever) over period
        m_meanmax_Q.append(m_subset.max(axis=0).mean())
        m_mean_Q.append(m_subset.mean())
        m_meanmin_Q.append(m_subset.min(axis=0).mean())
        m_min_Q.append(m_subset.min())
    m_max_Q, m_meanmax_Q = np.array(m_max_Q), np.array(m_meanmax_Q)
    m_mean_Q = np.array(m_mean_Q)
    m_min_Q, m_meanmin_Q = np.array(m_min_Q), np.array(m_meanmin_Q)
    mss = np.column_stack([m_max_Q, m_meanmax_Q, m_mean_Q, m_meanmin_Q,
                           m_min_Q])
    mss = pd.DataFrame(mss, columns=['maxQ', 'meanmaxQ', 'meanQ',
                                     'meanminQ', 'minQ'])
    mss['month'] = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                    'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    cols = mss.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    mss = mss[cols]
    if savetable == True:
        mss.to_csv(glacierstr + '_monthlyQ.csv', index=False)
    return mss

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


def get_shiftedColorMap(var, orig_cmap=matplotlib.cm.coolwarm_r,
                        multiplier=None, return_v=False):
    '''
    Compute custom diverging colormap for input var variable.

    Parameters
    ----------
    var : list, xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset,
          or numpy.ndarray
        Variable to derive shifted colormap from.
        
    orig_cmap : matplotlib.colormap, optional
        Original colormap for function to reference.
        
        The default is matplotlib.cm.coolwarm_r.
        
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


class AnalysisObject():
    def __init__(self, path, subvar=['HGT', 'surfMB', 'SNOWFALL'],
                 crs='EPSG:32608', monthday='10-01', time_var='time',
                 region='01', version='62', **kwargs):
        self.crs = crs
        self.var = prepare_analysis_file(path, subvar, monthday=monthday,
                                         time_var=time_var, crs=crs, **kwargs)
        self.gpd_region = get_gpd_region(region=region, version=version)
    
    
    def get_glaciermask(self, rgiid):
        '''
        Filters AnalysisObject.gpd_region containing all glaciers within a
        region to a specific glacier, selected by the given RGIId.

        Parameters
        ----------
        rgiid : str
            RGI Id of chosen glacier, corresponding to the RGI version code
            you're using (only applicable to v '62', '70G', or '70C').

        Returns
        -------
        glaciermask : geopandas.GeoDataFrame
            Glacier of choice based on RGIId, reprojected to match chosen CRS.
        
        Examples
        --------
        # import library
        import ing_utils as ing
        
        # Create path
        path = 'cosipy_output_CFSR_JIF_1980_2019.nc'
        
        # Initialize AnalysisObject
        ao = ing.AnalysisObject(path)
        
        # Define rgiid's for Gilkey and Mendenhall glaciers
        gilkey_rgi = 'RGI60-01.00704'
        mendenhall_rgi = 'RGI60-01.00709'
        
        # Get masks as GeoDataFrame objects for Gilkey and Mendenhall:
        gilkey_mask = ao.get_glaciermask(gilkey_rgi)
        mendenhall_mask = ing.get_glaciermask(mendenhall_rgi)
        
        print(gilkey_mask)
        print(mendenhall_mask)
        '''
        print('Deriving glaciermask for rgiid: ' + rgiid)
        # Extract the Glacier based on the input RGIId:
        glaciermask = self.gpd_region[self.gpd_region['RGIId'] == rgiid]
        
        # Adjust glaciermask crs to crs code
        glaciermask = glaciermask.to_crs(self.crs)
        return glaciermask
    
    def mask_var_by_glacier(self, rgiid, all_touched=True):
        '''
        Mask AnalysisObject.var to an RGI outline of a glacier 'glaciermask'.

        Parameters
        ----------            
        rgiid : str, optional
            RGI Id of chosen glacier. Must be provided if glaciermask is None.
            
            The default is None.
        
        all_touched : bool, optional
            If True, pixels overlapping with glaciermask polygon are included when
            var is clipped. If False, only pixels within bounds of glaciermask are
            saved.

        Returns
        -------
        glacier : xarray.core.dataarray.DataArray or xarray.core.dataset.Dataset
            Original variable(s), now with data masked to chosen glacier.
        
        
        Examples
        --------
        # import library
        import ing_utils as ing

        # define path to .nc file:
        path = 'cosipy_output_CFSR_JIF_1980_2019.nc'
        
        # Initialize AnalysisObject
        ao = ing.AnalysisObject(path)
        
        # Define rgiid's for Gilkey and Mendenhall glaciers
        gilkey_rgi = 'RGI60-01.00704'
        mendenhall_rgi = 'RGI60-01.00709'
        
        # Mask AnalysisObjects to RGI Id's!
        gilkey = ao.mask_var_by_glacier(gilkey_rgi)
        mendenhall = ao.mask_var_by_glacier(mendenhall_rgi)
        
        print(gilkey)
        print(mendenhall)
        
        '''
        # Clip var to glaciermask geometry using the glaciermask crs:
        glaciermask = self.get_glaciermask(rgiid)
        print('Subsetting var to glacier outline...')
        glacier = self.var.rio.clip(glaciermask.geometry.values,
                                    glaciermask.crs, all_touched=all_touched)
        return glacier
    


    
    
        
        