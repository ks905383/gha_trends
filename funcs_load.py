import xarray as xr
import xagg as xa
import numpy as np
import pandas as pd
import os
import glob
import re
import warnings

from funcs_support import get_params,get_cam6_filepaths,get_filepaths,id_timeframe
dir_list = get_params()

# TODO: update load_raws to deal with multiple timeframes. I think through 
# a "treat_multiple" dict that's {param:id_params name}, so, e.g. 
# {'time':'longest'} (or {'suffix':'longest'}). In each case, if there are
# multiple files possible for a given idvars combination, then collapse
# that idvar ('time' or 'suffix') based on which file has the 'longest'
# (or 'shortest','earliest', etc.) timeframe. 

def load_cam6(search_params,
              subset_params = {},
              source = 'cam6_raw',
              manually_decode_dates = True,
              silent=False):
    """ Load CAM6 data
    
    Parameters:
    -------------
    search_params : dict
        Of the form, e.g.,: 
            ```
            search_params = {'varname':'T',
                             'vgrid':'PLD',
                             'forcing_dataset':'hadley'}
            ```
        Use the `get_cam6_filepaths()` .csv file's columns
        as dictionary keys.
        
    subset_params : dict, by default {}
        Files are individually subset before concatenation using
        `.sel(**subset_params)`.
        
    source : str, by default 'cam6_raw'
        If 'cam6_raw', then `get_cam6_filepaths()` are used to 
        load and CAM6 raw file parameters are expected. If something 
        else, then that is piped into `get_filepaths(source_dir=source)`
        and CMIP file parameters are expected. 
        
    silent : bool, by default False
        If True, suppresses std out printing
        
    Returns:
    -------------
    ds : `xr.Dataset`
    
    """
    
    #---------- Get filenames -----------
    # Get filenames of all CAM6 files
    if source == 'cam6_raw':
        fns_all = get_cam6_filepaths()
    else:
        fns_all = get_filepaths(source_dir=source,mod='CAM6')
        
        # If more than 8 columns, then there exist split
        # experiment component columns; in this case, 
        # remove the original experiment column
        if len(fns_all.columns)>8:
            fns_all = fns_all.drop(columns='exp')

    # Get subset of files fitting the requirements
    fns = fns_all.loc[np.prod(np.array([fns_all[k] == v for k,v in search_params.items()]),axis=0).astype(bool),:]

    # Get which columns only have one item and therefore don't need to be dims
    fns = fns.drop(columns = fns.columns[fns.apply(lambda x: np.all([v is None for v in x]) 
                                                   or (len(np.unique(x))==1),axis=0)])
    # Set multi-index to all non-path variables
    fns = fns.set_index(list(np.sort([c for c in fns.columns if c != 'path'])))
    # For speed
    fns = fns.sort_index()
    
    #---------- Load data -----------
    # Load individual files and subset
    dss = [(xa.fix_ds(xr.open_dataset(df[1].path,decode_times=False)).
            assign_coords({d:[v] for d,v in zip(fns.index.names,np.atleast_1d(df[0]))}).
            sel(**subset_params)) for df in fns.iterrows()]

    # Get time info, from attributes, assuming the time
    # attribute is of the form '[X] since [T]'. 
    if manually_decode_dates:
        try:
            datefreq = re.split(' ',dss[0].time.units)[0]
            date0 = re.split(' ',dss[0].time.units)[-1]
            dates = [pd.to_datetime(date0)+pd.DateOffset(**{datefreq:np.floor(t).values}) for t in dss[0].time]
        except:
            manually_decode_dates = False
            warnings.warn('Dates were not able to be decoded...')
                      

    # Concatenate into single file
    # This loads the data into memory (even though it shouldn't? I've tried
    # compat='override', coords='minimal', data_vars='minimal',etc...
    if not silent:
        print('combining '+str(len(dss))+' files...')
    dss = xr.combine_by_coords(dss,combine_attrs='drop')

    # Fix time
    if manually_decode_dates:
        dss['time'] = dates
    
    #---------- Return -----------
    return dss



def load_raws(search_params,
              subset_params = {},
              dir_list = dir_list,
              source_dir = 'raw',
              manually_decode_dates = False,
              drop_list = ['lat_bnds','lon_bnds','time_bnds','lat_bounds','lon_bounds','time_bounds','bnds'],
              key_hierarchy = ['model','exp','time'],
              force_return_dict = False,
              force_key = 'model',
              force_load = False,
              attempt_nonrectangular_subsets=False,
              treat_multiple = None,
              new_key_sorting_procedure = True,
              return_nested_dict = True,
              return_errors = False,
              return_fns = False,
              silent=False,
              **open_kwargs):
    """ Load data from multiple models
    
    Workflow:
        1. Get all filepaths in dir_list[source] through
           `get_filepaths()`
        2. Subset filepaths using `search_params`
        3. Put into a `pd.DataFrame`, with a (multi-)index
           based on which filename parameters can't be 
           used to merge or concatenate the resultant files
        4. For each unique index combination, call
           `xr.open_mfdataset()` with an index for all 
           `concat_columns`
        5. Output a dictionary, with keys as the unique 
           index combinations
    
    Parameters:
    -------------
    search_params : dict
        Of the form, e.g.,: 
            ```
            search_params = {'varname':'T',
                             'run':'r1i1p1',
                             'exp':'historical'}
            ```
        Use the `get_cam6_filepaths()` .csv file's columns
        as dictionary keys.
        
    subset_params : dict, by default {}
        Files are individually subset before concatenation using
        `.sel(**subset_params)`.
        
    source_dir : str, by default 'raw'
        Piped into `get_filepaths(source_dir=source_dir)`
        
    drop_list : list, by default ['lon_bounds','lat_bounds','time_bounds']
        Drops variables (with `errors='ignore'`) from loaded datasets
        
    return_nested_dict : bool, by default True
        If True (and only <= 2 levels), then a nested dict is returned
        If False, then a single level dict with tuple keys of the 
        underlying multiindex is returned instead

    treat_multiple : dict, by default None
        A dict of the form {'time':'earliest'} to choose between files
        if multiple are returned for the same combination of non-time
        identifiers. Choices include all from `id_timeframe()`, so
        'earliest', 'latest', 'shortest', 'longest'
        
    ## Diagnostic tools:
    return_errors : bool, by default False
        If True, then returns captured exceptions in the load calls that
        are otherwise silently suppressed
        
    return_fns : bool, by default False
        If True, then returns `dss, fns`, giving the filenames that were
        attempted to be loaded
        
    silent : bool, by default False
        If True, suppresses std out printing (currently not in use) 
        
    Returns:
    -------------
    dss : dict of `xr.Dataset`s
    
    """

    #----------- Setup -----------
    # Define which columns can be easily merged over, 
    # either because of variable names or because they
    # can be dimensions
    merge_columns = ['varname']
    concat_columns = ['run']


    # Function to flatten list of arbitrary depth, from 
    # https://stackoverflow.com/questions/2158395/flatten-an-irregular-arbitrarily-nested-list-of-lists
    from collections.abc import Iterable
    def flatten(xs):
        for x in xs:
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                yield from flatten(x)
            else:
                yield x

    #----------- Find files -----------
    # Get all mods
    mods = [re.split(r'/',mod)[-1] for mod in glob.glob(dir_list[source_dir]+'*')]

    # Get all filepaths for all files
    fns_all = pd.concat([get_filepaths(source_dir=source_dir,mod=mod) for mod in mods])
    
    # Subset by search parameters
    fns = fns_all.loc[np.prod(np.array([fns_all[k] == v for k,v in search_params.items()]),axis=0).astype(bool),:]
    
    # Change NaNs, Nones to '' for easier processing
    fns = fns.fillna('')

    # Choose between time frames, if multiple files for a given combination of 
    # identifiers
    if treat_multiple is not None:
        multiple_collapser = [k for k in treat_multiple][0]
    
        fns = (fns.groupby([col for col in fns.columns if col not in [multiple_collapser,'path']]).
                   apply(id_timeframe,treat_multiple[multiple_collapser],'df',include_groups=False).
                  reset_index())
    
    # Get which columns only have one item and therefore don't need to be dims
    # (unless there's only one row in which case it would just drop the whole thing)
    if len(fns)>1:
        fns = fns.drop(columns = fns.columns[fns.apply(lambda x: np.all([v is None for v in x]) 
                                                           or (len(np.unique(x))==1),axis=0)])

    # Get columns which will be dictionary keys instead of 
    # dimensions in the dataset (since they would affect 
    # the merge - different grids, areas, timeframes,...)
    key_columns = [col for col in fns.columns if col not in ['path',*merge_columns,*concat_columns]]

    if new_key_sorting_procedure: 
        # Sort the key columns, with the desired key_hierarchy up front 
        key_columns = [*[key for key in key_hierarchy if key in key_columns],
                       *[key for key in key_columns if key not in key_hierarchy]]
        
        for key_idx in np.arange(1,len(key_columns)):
            # If the information in all of the columns so far is enough to 
            # uniquely describe all of the key columns, then delete the 
            # remaining key columns 
            # (basically, if you have three columns that are like this: 
            #     mod1    exp1    time1
            #     mod1    exp2    time2
            # then we don't need the time column, because 
            #     mod1exp1
            #     mod1exp2
            # uniquely identifies the rows)
            extra_test_cols = [col for col in concat_columns if col in fns.columns]
            if (len(np.unique(fns[key_columns[0:key_idx]+extra_test_cols].sum(1))) == 
                len(np.unique(fns[key_columns+extra_test_cols].sum(1)))):
                fns = fns.drop(columns = key_columns[key_idx:])
                break
                
        # Set multi-index to all non-path variables
        index_cols = [c for c in fns.columns if c not in ['path',*merge_columns,*concat_columns]]
        
        # Sort the index columns, with the desired key_hierarchy up front 
        index_cols = [*[key for key in key_hierarchy if key in index_cols],
                       *[key for key in index_cols if key not in key_hierarchy]]
        
    else:
        # If some of the key columns don't provide unique information
        # (i.e., if for example a certain experiment only shows up 
        # together with a certain timeframe), then only keep as keys
        # unique values, following the `key_hierarchy` parameter to 
        # decide which key to keep 
        #if len(np.unique([fns[key_columns].drop_duplicates().shape[0],
         #                 *[fns[k].drop_duplicates().shape[0] for k in key_columns]]))==1:
        if len(np.unique([fns[key_columns].drop_duplicates().shape[0],
                          len(np.unique(fns[key_columns].sum(1)))])):
            # Find the highest ranked (earliest-appearing in key_hierarchy) 
            # column of the columns desired 
            try:
                keep_key = key_hierarchy[next(index for index, item in enumerate(key_hierarchy) if item in key_columns)]
                fns = fns.drop(columns=[c for c in key_columns if c != keep_key])
            except:
                if not silent:
                    warnings.warn("Tried to declutter the ouptut dictionary keys using the `key_hierarchy`, "+
                                  "but none of the `key_columns` ("+', '.join(key_columns)+") are in the `key_hierarchy`.")
                pass

        # Set multi-index to all non-path variables
        index_cols = list(np.sort([c for c in fns.columns if c not in ['path',*merge_columns,*concat_columns]]))
    if len(index_cols)>0:
        fns = fns.set_index(index_cols)
    # For speed
    fns = fns.sort_index()

    #----------- Load files -----------
    # Define loading function that can deal with several 
    # edge cases
    def load_mfdataset(rows,drop_list = drop_list,force_load = force_load,
                       subset_params = subset_params,concat_columns = concat_columns,
                       attempt_nonrectangular_subsets = attempt_nonrectangular_subsets,
                       **open_kwargs):
        try: 
            if len(rows) == 1:
                ds = xr.open_dataset(rows['path'].iloc[0]).drop_vars(drop_list,errors='ignore')

            else:
                # Load
                ds = xr.open_mfdataset(flatten(rows[['path']].values.tolist()),
                                       concat_dim = [pd.Index(flatten(rows[[col]].values),name=col)
                                                     for col in concat_columns],
                                       combine='nested',**open_kwargs).drop_vars(drop_list,errors='ignore')

            # Standardize lat/lon order, lon values to -180:180, and dimension names
            try:
                ds = xa.fix_ds(ds)
            except NameError:
                # If no lat/lon found (xagg error) then skip
                pass

            # Subset 
            if (attempt_nonrectangular_subsets and
                ('lon' not in ds.sizes) and
                ('lon' in ds)):
                # If non-rectangular grid, attempt subset 
                # through .where calls
                for k in subset_params:
                    if k == 'time':
                        ds = ds.where(((ds[k]>=pd.to_datetime(subset_params[k].start)) &
                                       (ds[k]<=pd.to_datetime(subset_params[k].stop))),drop=True)
                    else:
                        ds = ds.where(((ds[k]>=subset_params[k].start) &
                                       (ds[k]<=subset_params[k].stop)),drop=True)
            else:
                ds = ds.sel(**subset_params)

            # Force load, if necessary
            if force_load:
                ds.load()
                
                
            # Put in dummy column (if it doesn't already have values in the file)
            # THE PROBLEM IS THIS FORCE LOADS IT ANYWAYS... AS DOES `.assign_coords()`
            # or `xr.concat()`... added down here to at least be beyond the subsetting
            if len(rows) == 1:
                if len(concat_columns)>0:
                    if not force_load:
                        warnings.warn('Expand dims to index column(s) '+
                                      ', '.join([col for col in concat_columns if col not in ds.sizes])+
                                      ' will force load into memory...')

                    for col in [col for col in concat_columns if col not in ds.sizes]:
                        #ds = ds.expand_dims({col:[0]})
                        if col in rows:
                            ds = ds.expand_dims({col:[rows.loc[col][0]]})
                        else:
                            ds = ds.expand_dims({col:[0]})

            return ds

        except Exception as e:
            if return_errors: 
                return e
            else:
                return None

    
    # Now load
    if len(index_cols)>0:
        # Use indices to load data into separate dictionary
        # keys based on which parameters can't be merged
        # or concatenated
        dss = {idx:load_mfdataset(fns.loc[[idx]]) #THIS IS THE BIG CHANGE 01/10/2024
               for idx in np.unique(fns.index)}
        
        # Remove empty dictionary entries, generated if 
        # there are errors in the loading of an index (for
        # example, if there's subset errors - non-trad grid,
        # etc.)
        dss = {k:v for k,v in dss.items() if v is not None}
    else:
        # Return one dataset if dictionary structure
        # not needed (i.e., all data have compatible
        # dimensions)
        dss = load_mfdataset(fns)
        if dss is None:
            warnings.warn('error in load_mfdataset() in loading, None was returned')
        
        # If the data have to be returned as a dict, 
        # use the desired search_params key to turn
        # it into a dict of one key, value pair
        if force_return_dict:
            if force_key in search_params:
                dss = {search_params[force_key]:dss}
            else:
                warnings.warn("through force_return_dict = True, no '"+force_key+
                              "' entry was found in `search_params`. A dataset will be returned instead.")
    
    # Nest dictionary if desired
    if return_nested_dict:
        if len(index_cols) > 1:  
            # Create dict relating individual key levels to the full multiindex
            # (there's a more elegant way of doing this, but I think this is more 
            # legible)
            index_df = pd.DataFrame(list(np.unique(fns.index)),columns=index_cols)
            index_df['idx'] = np.unique(fns.index)

            if len(index_cols) == 2:
                dss = {idx0:{idx1:dss[index_df.loc[(index_df[index_cols[0]]==idx0) & 
                                                        (index_df[index_cols[1]]==idx1),'idx'].iloc[0]]
                             for idx1 in np.unique(index_df.loc[index_df[index_cols[0]]==idx0,index_cols[1]])}
                       for idx0 in np.unique(index_df[index_cols[0]])}
            else:
                warnings.warn("Can't yet create nested dicts with more than 2 index levels... maybe someday I'll figure out how to do it recursively...")

    #----------- Return -----------
    if return_fns:
        return dss,fns
    else:
        return dss



def load_raw(search_str,
             search_dir=None,  
             rsearch=False,
             fn_ignore_regexs=[],
             subset_params=None, 
             squeeze=True,
             aggregate=False,    
             aggregate_dims = ['latitude','longitude'], 
             load_single=True,
             show_filenames = False,
             return_filenames = False,
             fix360_subset = True
            ):
    ''' Loads and subsets climate data files
    
    Theoretically takes all the back-end work out of loading raw climate
    data. Loads, subsets, and aggregates, without having to remember which
    file suffix corresponds to which geographic subset. 
    
    Depends on the behavior of xr.combine_by_coords().
    
    NOTE: Currently can't deal with files with overlapping domains...
    
    Parameters:
    ------------------------------
    search_str : str
        The string used by glob.glob to search for files to load. 
        
    search_dir : str, by default get_params()['raw']
        The directory in which to look for files. 
        
    rsearch : bool, default False
        NOT YET IMPLEMENTED. If `True`, looks recursively through
        directories within `search_dir` and concatenates along 
        a dimension [X]. 
        
    fn_ignore_regexs : str or list
        
    subset_params : dict, by default None
        If not `None`, then files will be subset to the slices in 
        this dict. Sample: 
            `subset_params = {'lat':slice(-5,5)}`
        Pro-tip: if instead of a slice you put in a single value, 
        then the code may break. Replace with a list, e.g.:
            `subset_params = {'plev':[650]}`
        which will be squeezed out at the end anyways, if 
        `squeeze=True`. 
            
    squeeze : bool, by default True
        If True, then the returned xr.Dataset is squeezed (i.e., 
        singleton dimensions are removed)
            
    aggregate : bool, by default False
        If `True`, then the mean over `aggregate_dims` is taken
        
    aggregate_dims : list, by default `['latitude','longitude']`
        If `aggregate == True`, then the average over these dimensions
        is taken. The code will first look to see if these dimension
        names exist in the merged dataset, or, if that fails, attempt
        to take the mean treating `aggregate_dims` as `cf_xarray` 
        names. 
        
    load_single : bool, by default True
        If `True`, then if no subset is taken (subset_params), 
        if more than one file is found in `search_str`, the code fails.
        Designed as a failsafe if too many files are unintentionally
        caught with `search_str`
        
    show_filenames : bool, by default False
        If `True`, then the matched filenames are printed.
        
    return_filenames : bool, by default False
        If `True`, then the matched filenames are returned.
        
    fix360_subset : bool, by default True
        If `True`, then if the calendar type of a file is cftime.Datetime360Day
        `subset_params` includes a time subset, and that time subset ends 
        on the 31st day of a month, this is replaced with the 30th 
        day of the month instead     
    
    Returns:
    ------------------------------
    ds : xr.Dataset
        A merged dataset, potentially aggregated, of the desired data.
        
    fns_match : list
        If `return_filenames = True`, the list of matched filenames
    
    '''
    
    
    #----------- Setup -----------
    # If no search_dir provided, assume it's the raw 
    # data directory from get_params()
    if search_dir is None:
        dir_list = get_params()
        search_dir = dir_list['raw']
        
    if type(fn_ignore_regexs) != list:
        fn_ignore_regexs = [fn_ignore_regexs]
    
    
    #----------- Find and load files -----------
    # Get files in [search_dir] that match [search_str]
    if rsearch:
        raise NotYetImplementedError()
    
    fns_match = glob.glob(search_dir+search_str)
    if show_filenames:
        print('Files found from search "'+search_dir+search_str+'":\n  '+
              '\n  '.join(fns_match))
            
    if len(fns_match) == 0:
        raise Exception('No files found using search "'+search_dir+search_str+'"')
    
    if len(fn_ignore_regexs) != 0:
        for fn_ignore_regex in fn_ignore_regexs:
            fns_match = list(filter(lambda item: item is not None,
                                    [fn if (re.search(fn_ignore_regex,fn) is None) else None for fn in fns_match]))
        if show_filenames: 
            print('filenames after subsetting:\n'+'\n'.join(fns_match))
    
    # Load them for their dimensions 
    try: 
        dss = [xr.open_dataset(fn) for fn in fns_match]
    except:
        raise Exception('Issue loading one of the following files:'+'\n'.join(fns_match))
    
    # Subset all using subset_params
    if subset_params is not None: 
        subset_params_tmp = subset_params
        if (('time' in subset_params) and 
            (type(dss[0].time.values[0]) == cftime.Datetime360Day) and 
            fix360_subset):
            subset_params_tmp['time'] = slice(*[re.sub('-31$','-30',subset_params['time'].start),
                                            re.sub('-31$','-30',subset_params['time'].stop)])

        dss = [ds.sel(**subset_params_tmp) for ds in dss]
        
        # Test if this subsetting resulted in empty 
        # dimensions in a particular file
        subset_flags = [np.all([ds.sizes[k]!=0 for k in subset_params_tmp])
                        for ds in dss]
        # Remove empty subsets
        dss = list(itertools.compress(dss,subset_flags))
    else:
        if len(dss)>1:
            warnings.warn('Multiple files found, with no '+
                          'desired subset: \n  '+
                          '\n  '.join(fns_match))
            if load_single:
                raise NotUniqueFile('More than one file found, but since '+
                                'load_single==True, no files are loaded '+ 
                                'to avoid memory overloads.')
                
    #----------- Concatenate -----------
    dss = xr.combine_by_coords(dss,combine_attrs='drop_conflicts')
    
    #----------- Additional processing if desired -----------
    if aggregate:
        try:
            # If subset_params dimensions names aren't in the 
            # dataset dimensions, then try using cf_xarray
            # conventions
            if np.any([dim not in dss.sizes for dim in aggregate_dims]):
                dss = dss.cf.mean(aggregate_dims)
            # If subset_params dimensions names are in the 
            # dataset dimensions, aggregate over those dimensions
            else:
                dss = dss.mean(aggregate_dims)
        except ValueError:
            raise Exception('The dimensions on which to aggregate ('+','.join(aggregate_dims)+') '+
                            'were not all found in the dimension list ('+','.join([dim for dim in dss.sizes])+') '+
                            'or as cf_xarray supported dimension names.')
    
    #----------- Return -----------
    # Remove singleton dimensions if desired
    if squeeze:
        dss = dss.squeeze()
    if return_filenames:
        return dss,fns_match
    else:
        return dss
