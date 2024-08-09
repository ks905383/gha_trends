import xarray as xr
import xagg as xa
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re
import os
import glob
import warnings
import datetime

class NotUniqueFile(Exception):
    """ Exception for when one file needs to be loaded, but the search returned multiple files """
    pass

# Function to convert integer to Roman values
def printRoman(number):
    # from https://www.geeksforgeeks.org/python-program-to-convert-integer-to-roman/
    num = [1, 4, 5, 9, 10, 40, 50, 90,
        100, 400, 500, 900, 1000]
    sym = ["I", "IV", "V", "IX", "X", "XL",
        "L", "XC", "C", "CD", "D", "CM", "M"]
    i = 12

    output = ''
    while number:
        div = number // num[i]
        number %= num[i]
 
        while div:
            output = output + sym[i]
            div -= 1
        i -= 1

    return output

def get_params():
    ''' Get parameters 
    
    Outputs necessary general parameters. 
    
    Parameters:
    ----------------------
    (none)
    
    
    Returns:
    ----------------------
    dir_list : dict()
        a dictionary of directory names for file system 
        managing purposes: 
            - 'raw':   where raw climate files are stored, in 
                        subdirectories by model/product name
            - 'proc':  where processed climate files are stored,
                        in subdirectories by model/product name
            - 'aux':   where aux files (e.g. those that transcend
                        a single data product/model) are stored
    '''

    # Dir_list
    dir_list = pd.read_csv('dir_list.csv')
    dir_list = {d:dir_list.set_index('dir_name').loc[d,'dir_path'] for d in dir_list['dir_name']}


    # Return
    return dir_list

dir_list = get_params()

def get_filepaths(source_dir = 'raw',
                  mod = None,
                  dir_list = dir_list,
                  col_namer = {'(hadley$)|(CMIP[0-9]$)':'forcing_dataset',
                                 'PDO$':'pdo_state',
                                     'AMO$':'amo_state'}):
    ''' Get filepaths of climate data, split up by CMIP filename component
    
    
    Uses modified CMIP5/6 filename standards used by Kevin Schwarzwald's 
    filesystem - in other words, with the additional optional "suffix" 
    between the daterange and the filetype extension. 
    
    Returns
    ------------
    df : pd.DataFrame
        A dataframe containing information for all files in 
        `dir_list[source_dir]/mod/*.nc`, with the full filepath in the
        column `path`, and filename components `varname`, `freq`, 
        `model`, `exp`, `run`, `grid`, `time`, `suffix`, in their own
        columns. `grid` may be Nones if files use CMIP5 conventions, 
        `suffix` may be Nones if no suffixes are found. 
        
        If `exp` has a match for the regex r"-", then additionally
        extra columns for each experiment name component will be 
        created, if possible, using the `col_namer` input.
    
    
    '''
    
    def id_fncomps(comps,col_namer=col_namer):
        # Make sure there are enough components 
        if len(comps)<6:
            # For now - but there has to be a better way to 
            # flag this
            slots = {'varname':None}
        else:
            try:
                # Prepopulate set components
                slots = {s:n for n,s in zip(np.arange(0,5),['varname','freq','model','exp','run'])}

                # Get which slot is the timeframe (fx have "na" as timeframe)
                slots['time'] = np.where([re.search('([0-9]{4,8}'+r'-'+'[0-9]{4,8})|(^na($|'+r'.'+'))',comp) for comp in comps])[0][0]

                # Use the time position to determine whether
                # there's a grid slot (CMIP6) or not (CMIP5)
                if slots['time'] == 5:
                    slots['grid'] = None
                elif slots['time'] == 6:
                    slots['grid'] = 5

                # Use whether the file extension is in the time
                # or one after slot to determine whether there's a 
                # suffix slot
                if np.where([re.search(r'.nc',comp) for comp in comps])[0][0] == slots['time']:
                    slots['suffix'] = None
                elif np.where([re.search(r'.nc',comp) for comp in comps])[0][0] == (slots['time']+1):
                    slots['suffix'] = slots['time']+1

                # Now, assign slots to the components
                slots = {k:re.sub(r'.nc$','',comps[s]) if s is not None else None for k,s in slots.items()}

                # If the experiment slot has multiple sub-experiments,
                # save them seperately using the column namer dict
                exp_comps = re.split(r'-',slots['exp'])
                if len(exp_comps)>1:
                    for exp_comp in exp_comps:
                        if np.any([re.search(k,exp_comp) for k in col_namer]):
                            match_type = [v for k,v in col_namer.items() if re.search(k,exp_comp)]
                            if len(match_type) > 1:
                                warnings.warn('More than one column match found for '+exp_comp+'. Check col_namer, no exp has been split.')
                            else:
                                slots[match_type[0]] = exp_comp
            except:
                # Assuming that if there's an error it's because the
                # file in question isn't in a standard respected form
                # For now - but there has to be a better way to 
                # flag this
                slots = {'varname':None}

        return slots

    #---------- Get list of files ----------
    if mod is None:
        # Get all mods
        mods = [re.split('/',mod)[-1] for mod in glob.glob(dir_list[source_dir]+'*')]
    else:
        mods = [mod]
        
    fns_all = [None]*len(mods)
    for mod,mod_idx in zip(mods,np.arange(0,len(mods))):
        # Get list of subdirectories
        fns = glob.glob(dir_list[source_dir]+mod+'/*.nc')

        # Split up filename by components
        fn_comps = [re.split(r'_',re.split(r'/',fn)[-1]) for fn in fns]
        # Identify components, concatenate with path
        fns_all[mod_idx] = pd.DataFrame([id_fncomps(comps) for comps in fn_comps])
        fns_all[mod_idx] = pd.concat([fns_all[mod_idx],pd.DataFrame([{'path':fn} for fn in fns])],axis=1)

    # Concatentate into single df
    df = pd.concat(fns_all)
    #---------- Return ----------
    return df

def get_cam6_filepaths(dir_search_str='future*',
                       source_dir = 'cam6_runs',
                       col_namer = {'hadley$':'forcing_dataset',
                                   'PDO$':'pdo_state',
                                   'AMO$':'amo_state'},
                       dir_list = dir_list,
                       silent=True,
                       overwrite=False):
    """ Create and save filepaths dataframe
    
    Parameters:
    ------------
    dir_search_str : str, by default 'future*'
        What subdirectories to search for; using "future*" 
        to avoid the test, spinup, etc. directories
        
    source_dir : str, by default 'cam6_runs'
        In which `dir_list[source_dir]` to look for files
    
    """
    output_fn = dir_list['aux']+'filenames_'+source_dir+'.csv'
    
    if overwrite or (not os.path.exists(output_fn)):
        #---------- Get list of run directories ----------
        # Get list of subdirectories
        dns = glob.glob(dir_list[source_dir]+'/'+dir_search_str)

        # Split up subdirectories by SST source, PDO, and AMO positions
        # Actually, do it generally, by splitting by any _*_ field that isn't
        # constant across all directories
        dn_comps = [re.split('_',re.split('/',dn)[-1]) for dn in dns]
        # Subset by identifiers that are not constant across all directories
        identifier_idxs = np.apply_along_axis(lambda x: len(np.unique(x))>1,0,dn_comps)
        dn_comps = np.array(dn_comps)[:,identifier_idxs]

        #---------- Create dataframe with paths ----------
        # Concatenate into dataframe
        df = pd.DataFrame(np.hstack([dn_comps,np.transpose(np.atleast_2d(dns))]))

        # Get default column names
        cols = df.columns.values.astype(object)
        # The last column is the filepath
        cols[-1] = 'dir_path'

        # For each remaining column, try to assign column names through 
        # their contents
        for c_idx in np.arange(0,len(cols)-1):
            names = [v for k,v in col_namer.items() if np.any([re.search(k,t) for t in df[cols[c_idx]]])]
            if len(names) > 1:
                warnings.warn('Multiple possible names found for column '+cols[c_idx]+' (containing: '+
                              print(', '.join(df[cols[c_idx]]))+'); original placeholder name for column kept.')
            else:
                cols[c_idx] = names[0]
        # Set new column names
        df.columns = cols

        #---------- Search each directory for files ----------
        fns_all = [None]*len(df)

        for forc_idx in np.arange(0,len(df)):

            fns_all[forc_idx] = [None]*2
            for domain in ['clm','atm']:
                # Get list of files in domain
                if domain == 'clm':
                    fns = glob.glob(df.loc[forc_idx,:].dir_path+'/'+domain+'/*_*/*.nc')
                elif domain == 'atm':
                    fns = glob.glob(df.loc[forc_idx,:].dir_path+'/'+domain+'/*_*/*/*.nc')

                # Identify variables, runs, v grid type (single or pressure levels)
                var_list = [re.split(r'.nc',re.split(r'/',fn)[-1])[0] for fn in fns]
                if domain == 'clm':
                    run_idx = -2
                    vgrid_list = ['clm']*len(fns)
                elif domain == 'atm':
                    run_idx = -3
                    vgrid_list = [re.split(r'/',fn)[-2] for fn in fns]
                run_list = [re.sub(r'_','',re.split(r'/',fn)[run_idx]) for fn in fns]

                # Concatenate into dataframe
                fns_all[forc_idx][['clm','atm'].index(domain)] = pd.DataFrame(np.transpose(np.array([var_list,run_list,vgrid_list,fns])),columns = ['varname','run','vgrid','path'])

                # Add other columns from the original path
                for col in [c for c in cols if c != 'dir_path']:
                    fns_all[forc_idx][['clm','atm'].index(domain)][col] = df.loc[forc_idx,col]
            # Concatenate 
            fns_all[forc_idx] = pd.concat(fns_all[forc_idx])

        # Concatenate into single dataframe
        fns_all = pd.concat(fns_all).reset_index().drop(columns=['index'])
        
        #---------- Save ----------
        if overwrite and os.path.exists(output_fn):
            os.remove(output_fn)
            if not silent:
                print(output_fn+' removed to allow overwrite!')
        
        fns_all.to_csv(output_fn,index=False)
        if not silent:
            print(output_fn+' saved!')
        
        
    else:
        if not silent:
            print(output_fn+' already exists, loaded!')
        fns_all = pd.read_csv(output_fn)
        
    #---------- Return ----------
    return fns_all


# The next two are from https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7

def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid
    defined by WGS84
    
    Input
    ---------
    lat: vector or latitudes in degrees  
    
    Output
    ----------
    r: vector of radius in meters
    
    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''
    from numpy import deg2rad, sin, cos

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)

    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5)
         / (1 - (e2 * np.cos(lat_gc)**2))**0.5
        )

    return r

def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters
    
    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees
    
    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]
    
    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
    from numpy import meshgrid, deg2rad, gradient, cos
    from xarray import DataArray

    xlon, ylat = meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=0))
    dlon = deg2rad(gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = dy * dx

    xda = DataArray(
        area,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda

def area_mean(ds,assume_rectangular=True):
    """ Calculate area-weighted mean of all variables in a  dataset
    
    Mean over lat / lon, weighted by the relative size of each
    pixel, dependent on latitude. Only weights by latitude, does
    not take into account lat/lon bounds, if present. 
    
    Parameters
    ------------------
    ds : xr.Dataset
    
    Returns
    ------------------
    dsm : xr.Dataset
        The input dataset, `ds`, averaged.
    
    """
    
    if (ds.sizes['lat'] == 1) and (ds.sizes['lon'] == 1):
        # If just one pixel, return that one pixel
        ds = ds.isel(lat=0,lon=0).drop(['lat','lon'])
        
    elif (ds.sizes['lat'] == 1) and (assume_rectangular):
        # If only one lat row, but multiple long rows, 
        # just get the cartesian mean, if assuming rectangular
        # grids. 
        ds = ds.mean(('lat','lon'))
        
    else:
        # Calculate area in each pixel
        weights = area_grid(ds.lat,ds.lon)

        # Remove nans, to make weight sum have the right magnitude
        weights = weights.where(~np.isnan(ds))

        # Calculate mean
        ds = ((ds*weights).sum(('lat','lon'))/weights.sum(('lat','lon')))

    # Return 
    return ds

def subset_to_srat(da,srat_mod = 'CHIRPS',
                   srat_file = None,
                   print_srat_fn = False,
                   drop = False,
                   subset = 'double_peaked',
                   subset_params = {'lat':slice(-3,12.5),'lon':slice(32,55)},
                   regrid_method = 'bilinear'):
    """ Subset dataarray to double-peaked area
    
    Parameters 
    -------------------
    da : xarray.core.dataarray.DataArray
        The DataArray to subset
        
    srat_mod : str, by default 'CHIRPS'
        The data product from which to find the seas_ratio file; 
        if CHIRPS, then the filename is hardcoded, if a different
        model, then the first file to satisfy the search:
            'pr_doyavg_[mod]_*_seasstats*.nc' 
        in that model's [proc] directory is used; if `srat_file` is 
        not None, then this parameter is ignored.
        
    srat_file : str, by default None
        If not None, then the file with this filename is used as
        the source of the `seas_ratio` variable 
        
    print_srat_fn : bool, by default False
        If True, then the filename used for the `seas_ratio` 
        variable is printed. 
        
    subset : str, by default 'double-peaked'
        If `=='double-peaked'`, then data are subset to all locations
        where `seas_ratio<1`. If `=='single-peaked'`, then data are
        subset to all locations where `seas_ratio>1`. 
        
    subset_params : dict, by default {'lat':slice(-3,12.5),'lon':slice(32,55)}
        If not None, then `seas_ratio` variable is subset using 
        this subset dictionary. 
        
    regrid_method : str, by default 'bilinear'
        Which method used to regrid `seas_ratio` to the input `da`
        grid; piped into `xe.Regridder()`
        
    drop : bool, by default False
        If true, then dimension coords with all nans are dropped
        
        
    Returns
    -------------------
    da : xarray.core.dataarray.DataArray
        The input DataArray, but now subset geographically to 
        areas with a double-peaked rainfall climatology, as 
        defined by the `seas_ratio` variable / file used. 
    
    """
    from funcs_load import load_raw
    import xesmf as xe
    
    dir_list = get_params()
        
    # Load seas_ratio from stats file
    if srat_file is not None:
        srat = xr.open_dataset(srat_file).seas_ratio
    else:
        srat, fns_match = load_raw('pr_doyavg_*_seasstats_*HoA.nc',
                                   search_dir=dir_list['proc']+srat_mod+'/',
                                   return_filenames=True)
        srat = srat.seas_ratio.drop('method')
            
    if print_srat_fn:
        print('used '+fns_match[0]+' as source for `seas_ratio` variable.')
            
    # Subset seas_ratio to desired location
    if subset_params is not None:
        srat = srat.sel(**subset_params)
    
    # Get rid of singleton dimensions (e.g., "method")
    srat = srat.squeeze()
    
    # Regrid hdiff to precip grid, if different grids
    if not (np.all([l in srat.lat for l in da.lat.values]) and 
        np.all([l in srat.lon for l in da.lon.values])):
        with warnings.catch_warnings():
            # Ignore the FutureWarning that shows up from inside xesmf 
            # and adds nothing to the conversation
            warnings.simplefilter("ignore") 
            # Regrid 
            rgrd = xe.Regridder(srat,da,method=regrid_method)
            srat = rgrd(srat)

    # Set 0s to nan (artifact of the process) 
    srat = srat.where(srat!=0)
        
    # Subset to double-peaked region
    if subset == 'double_peaked':
        da = da.where(srat<1,drop=drop)
    elif subset == 'single_peaked':
        da = da.where(srat>1,drop=drop)
    else:
        raise KeyError("`subset` must be either 'double-peaked' or 'single-peaked', but was '"+subset+"'")
    
    # Return
    return da


def utility_print(output_fn,formats=['pdf','png']):
    if 'pdf' in formats:
        plt.savefig(output_fn+'.pdf')
        print(output_fn+'.pdf saved!')

    if 'png' in formats:
        plt.savefig(output_fn+'.png',dpi=300)
        print(output_fn+'.png saved!')

    if 'svg' in formats:
        plt.savefig(output_fn+'.svg')
        print(output_fn+'.svg saved!')


def get_varlist(source_dir=None,var=None,varsub='all',
                experiment=None,freq=None,
                empty_warnings=False):
    ''' Get a list of which models have which variables
    
    Searches the filesystem for all models (directory names) and 
    all variables (first part of filenames, before the first 
    underscore), and returns either that information for all 
    models and variables, or an array of models that have 
    files for specified variables. 
    
    NB: if no experiment or frequency is specified, and the
    full dataframe is returned (`var=None`), then the fields
    have True whenever any file with that variable in the filename
    for that model is present (and potentially more than one). 
    In general, the code does not differentiate between multiple
    files for a single model/variable combination. 
    
    Parameters
    ---------------
    source_dir : str; default dir_list['raw']
        a path to the directory with climate data (all 
        subdirectories are assumed to be models, all files in
        these directories are assumed to be climate data files
        in rough CMIP format).
        
    var : str, list; default `None`
        one variable name or a list of variables for which to 
        subset the model list of. If not `None`, then only a list
        of models for which this variable(s) is present is returned
        (instead of the full Dataframe).
        
    varsub : str; default 'all'
        - if 'all', then if `var` has multiple variables, 
          only models that have files for all of the variables 
          are returned
        - if 'any', then if `var` has multiple variables, 
          models that have files for any of the variables are 
          returned
          
    experiment : str; default `None`
        if not None, then only returns models / True if files
        for the given 'experiment' (in CMIP6 parlance, the 
        fourth filename component) are found. If not None, the
        variable is piped into re.search(), allowing for re
        searches for the experiment. 
        
    freq : str; default `None`
        if not None, then only returns models / True if files
        for the given 'frequency' (in CMIP6 parlance, the 
        second filename component) are found. If not None, the
        variable is piped into re.search(), allowing for re
        searches for the frequency. 
        
    empty_warnings : bool; default `False`
        if True, a warning is thrown if no files at all (before 
        subsetting) are found for a model. 
    
    
    Returns
    ---------------
    varindex : pd.DataFrame()
        if `var` is None, then a models x variables pandas
        DataFrame is returned, with `True` if that model has 
        a file with that variable, and `False` otherwise.
        
    mods : list
        if `var` is not None, then a list of model names 
        that have the variables, subject to the subsetting above
    
    
    '''
    if source_dir is None:
        dir_list = get_params()
        source_dir = dir_list['raw']
    
    
    ##### Housekeeping
    # Ensure the var input is a list of strings, and not a string
    if type(var) == str:
        var = [var]
    
    ##### Identify models
    # Figure out in which position of the filename path the model name
    # directory is located (based on how many directory levels there 
    # are in the parent directory)
    modname_idx = len(re.split('/',source_dir)) - 1
    # Get list of all the models (the directory names in source_dir)
    all_mods = [re.split('/',x)[modname_idx] for x in [x[0] for x in os.walk(source_dir)] if re.split('/',x)[modname_idx]!='']
    all_mods = [mod for mod in list(np.unique(all_mods)) if 'ipynb' not in mod]
    
    ##### Identify variables
    # Get list of all variables used and downloaded
    # Make this a pandas dataarray - mod x var
    varlist = []
    for mod in all_mods[:]:
        varlist.append([re.split(r'_',fn)[0] for fn in [x for x in os.walk(source_dir+mod+'/')][0][2]])
    varlist = [item for sublist in varlist for item in sublist]

    varlist = list(np.unique(varlist))

    # Remove "README" and ".nc" files 
    varlist = [var for var in [var for var in varlist if 'READ' not in var] if '.nc' not in var]
    
    ##### Populate dataframe
    # Create empty dataframe to populate with file existence
    varindex = pd.DataFrame(columns=['model',*varlist])

    # Populate the model column
    varindex['model'] = all_mods

    # Actually, just set the models as the index
    varindex = varindex.set_index('model')
    
    # Now populate the dataframe with Trues if that model has that variable as a file
    for mod in all_mods:
        # Get variable name of each file 
        file_varlist = [re.split(r'_',fn)[0] for fn in [x for x in os.walk(source_dir+mod+'/')][0][2]]

        if len(file_varlist) == 0:
            if empty_warnings:
                warnings.warn('No relevant files found for model '+mod)
            varindex.loc[mod] = False
        else:
            # Subset by frequency, or experiment, if desired
            if freq is not None:
                try:
                    freq_bools = [(re.search(freq,re.split(r'_',fn)[1]) != None) for fn in [x for x in os.walk(source_dir+mod+'/')][0][2]]
                except IndexError:
                    freq_bools = [False]*len(file_varlist)
                    if empty_warnings:
                        warnings.warn('Model '+mod+' has files not in CMIP format.')
                    continue
            else:
                freq_bools = [True]*len(file_varlist)

            if experiment is not None:
                try:
                    exp_bools = [(re.search(experiment,re.split(r'_',fn)[3]) != None) for fn in [x for x in os.walk(source_dir+mod+'/')][0][2]]
                except IndexError:
                    exp_bools = [False]*len(file_varlist)
                    if empty_warnings:
                        warnings.warn('Model '+mod+' has files not in CMIP format.')
                    continue
            else:
                exp_bools = [True]*len(file_varlist)

            # Remove from list if it doesn't fit the frequency/experiment subset
            file_varlist = list(np.asarray(file_varlist)[np.asarray(freq_bools) & np.asarray(exp_bools)])

            # Add to dataframe
            varindex.loc[mod] = [var in file_varlist for var in varlist]

    # Fill NaNs with False
    varindex = varindex.fillna(False)

    ##### Return
    if var is None: 
        return varindex
    else:
        if type(var) == str:
            var = [var]
        if varsub == 'all':
            # (1) is to ensure the `all` is across variables/columns, not rows/models
            return list(varindex.index[varindex[var].all(1)].values)
        elif varsub == 'any':
            return list(varindex.index[varindex[var].any(1)].values)
        else:
            raise KeyError(str(varsub) + ' is not a supported variable subsetting method, choose "all" or "any".')

def id_timeframe(r,cond = 'longest',out = 'timestr'):
    ''' Choose between filepaths based on a temporal condition

    Subset output from :py:meth:`get_filepaths()` based on a condition
    on the timelength of a file. 

    Parameters
    -----------------
    r : :py:class:`pd.Dataframe`, output from :py:meth:`get_filepaths`
        Crucially, needs the "time" column

    cond : :py:class:`str`
        One of:
            - 'longest': choose the file with the longest timeframe
            - 'shortest': choose the file with the shortest timeframe
            - 'earliest': choose the file that starts the earliest
            - 'latest': choose the file that ends the latest

    out : :py:class:`str`
        Determines the return, one of: 
            - 'timestr': the timestring of the relevant file
            - 'df': the full dataframe with just that file's row

    Returns
    -----------------
    depends on `out` above

    TODO: allow arbitrary nested conditions (so, if two have the same
    length, choose one using a different condition)
    
    '''
    # Allowable conds
    conds = ['longest','shortest','earliest','latest']
    
    # Get timeframes of each file
    ts = r.time.values

    # Get timeframe in times
    def convert_to_dt(t):
        t_out = []
        for dt in re.split(r'-',t):
            try:
                t_out.append(pd.to_datetime(dt,format='%Y%m%d'))
            except pd.errors.OutOfBoundsDatetime as e1:
                # Dates past 2262 can't fit into pandas, gotta go to 
                # base datetime
                t_out.append(datetime.datetime(int(dt[0:4]),int(dt[4:6]),int(dt[6:8])))
            except ValueError as e2:
                # Some files got saved with incorrect timeframe slots, 
                # i.e., 31 in any month, even in months with fewer days.
                # If the days slot is larger than the # days that should 
                # be in that month, convert to the last day of that month 
                # before turning to datetime format. 
                if int(dt[-2:]) > pd.to_datetime(dt[0:6]+'01',format='%Y%m%d').daysinmonth:
                    t_out.append(pd.to_datetime(dt[0:6]+str(pd.to_datetime(dt[0:6]+'01',format='%Y%m%d').daysinmonth),
                                            format='%Y%m%d'))
                else:
                    raise e2
        return t_out

    tdts = [convert_to_dt(t) for t in ts]

    # Get length of each timeframe
    tlengths = [np.diff(tdt) for tdt in tdts]

    # Get timestring based on condition above
    if cond == 'longest':
        idx = np.argmax(tlengths)
    elif cond == 'shortest':
        idx = np.argmin(tlengths)
    elif cond == 'earliest':
        idx = np.argmin([t[0] for t in tdts])
    elif cond == 'latest':
        idx = np.argmax([t[1] for t in tdts])
    else:
        raise KeyError('cond must be one of '+', '.join(conds)+'.')

    # Return
    if out == 'timestr':
        return ts[idx]
    elif out == 'df':
        return r.iloc[idx,:]
    else:
        raise KeyError('out must be one of "timestr", "df"')
            