import xarray as xr
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import warnings

def seasmean(ds,seas_range):
    ''' Calculate seasonal means of one season

    Note: the output year is always the year of the _first_
    month of the season - so 2013/12 - 2014/2 is returned 
    as the "2013" DJF season. 

    Parameters
    -------------
    ds : :py:class:`xr.Dataset` or :py:class:`xr.DataArray`

    seas_range : `py:class:`list` of :py:class:`int`
        e.g., [3,5] for MAM. A list of the integer indices
        of the months (inclusive) in season. THIS IS 1-,
        NOT 0-INDEXED.

    Returns
    -------------
    ds_out : :py:class:`xr.Dataset` or :py:class:`xr.DataArray`
        Seasonal means of the input data, with the time 
        dimension renamed "year" and showing integer years
        instead of time
    
    '''
    if seas_range[0] < seas_range[1]:
        # If the start season < end season, no wrap around, 
        # so we need months that are least the start
        # month and at most the end month
        tsub = ((ds.time.dt.month>=seas_range[0]) & 
                (ds.time.dt.month<=seas_range[1]))
    else:
        # If the start season > end season, wrap around the
        # new year, so we need months that are least the start
        # month OR at most the end month
        tsub = ((ds.time.dt.month >= seas_range[0]) | 
                (ds.time.dt.month <= seas_range[1]))
        
    # Subset to just this season
    ds_out = ds.isel(time=tsub)
    
    # Resample annually to get seasonal means (using the anchored offset
    # to ensure wrap-around seasons are correctly averaged
    ds_out = (ds_out.resample(time='1YS-'+
                              pd.Timestamp('1900-'+str(seas_range[0]).zfill(2)+'-01').month_name()[0:3].upper()).
          mean(skipna=False))

    # Change time dimension to year int
    ds_out['time'] = ds_out['time'].dt.year
    ds_out = ds_out.rename({'time':'year'})

    # Take out incomplete seasons (from wrap around, both the first
    # and last year's seasons are incomplete means) 
    if seas_range[0] > seas_range[1]:
        ds_out = ds_out.isel(year=slice(1,-1))
    
    return ds_out

def seasmeans(ds,seasons):
    ''' Calculate seasonal means of multiple seasons

    A wrapper for :py:meth:`seasmean`. 

    Parameters
    -------------
    ds : :py:class:`xr.Dataset` or :py:class:`xr.DataArray`

    seasons : :py:class:`dict` 
        Dictionary, with keys as season names and items as 
        lists with length 2 of the start, end months (inclusive)
        of each season. Note that seasonal limits are 1-indexed, 
        i.e., Jan == 1. 

    Returns
    -------------
    ds_out : :py:class:`xr.Dataset` or :py:class:`xr.DataArray`
        Seasonal means of the input data, along a new dimension 
        "season", with the time dimension renamed "year" and 
        showing integer years instead of time

    '''

    if type(ds) == xr.core.dataset.Dataset:
        # Get vars with time dimension
        vars_wtime = [v for v in ds if 'time' in ds[v].sizes]
    
    # Aggregate to season
    ds_out = xr.concat([seasmean(ds[vars_wtime],seas_range) for seas,seas_range in seasons.items()],
                      dim = pd.Index([seas for seas in seasons],name='season'))

    if type(ds) == xr.core.dataset.Dataset:
        # Add back non-time vars
        for var in [v for v in ds if v not in vars_wtime]:
            ds_out[var] = ds[var].copy()

    # Input seasonal information
    ds_out['season_bnds'] = xr.DataArray(np.array([seas_range for seas,seas_range in seasons.items()]),
                                             dims = ('season','bnds'),
                                             coords = {'season':(('season'),[seas for seas in seasons]),
                                                       })
    ds_out['season_bnds'].attrs['DESCRIPTION'] = 'Start, end month (inclusive) of season (1-indexed, i.e., January = 1)'

    return ds_out
    

