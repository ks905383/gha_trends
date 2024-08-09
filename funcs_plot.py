import xarray as xr
import dask
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import re
import glob
import copy
import string
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from cartopy import crs as ccrs
import matplotlib.ticker as mticker
import scipy.stats as sstats
import cmocean
import seaborn as sns
import warnings
from tqdm.notebook import tqdm

from funcs_support import get_filepaths,get_params,utility_print,subset_to_srat,area_mean,printRoman,id_timeframe
from funcs_load import load_raws
dir_list = get_params()

extra_fonts = {'lato':mpl.font_manager.FontProperties(fname=dir_list['aux']+'fonts/Lato-Regular.ttf'),
 'lato-bold':mpl.font_manager.FontProperties(fname=dir_list['aux']+'fonts/Lato-Bold.ttf')}

def percentileof_alongdim(da,da_score,nan_policy='omit'):
    ''' apply_ufunc-compatible wrapper of percentileofscore
    '''
    
    return sstats.percentileofscore(da.flatten(),
                                    da_score,
                                    nan_policy=nan_policy)*0.01

def hist_plot(dss_plot,
              start_year = 1986,
              end_year = 2004,
              nruns = None,
              seas = 'MAM',
              plot_type = 'levels',
              plot_exps = ['amip','hist-ssp245'],
              exp_labels = {'amip':'AMIP6','hist-ssp245':'CMIP6','obs':'Obs.'},
              xlims = [-1.5,1.5],
              ylims = 'auto',
              obs_mods = 'auto',
              label_obs = False,
              annotate_args = {'text_vgap':0.05,'text_v0':0.7,'arrow_length':0.1},
              palette = ['tab:blue','tab:green'],
              text_kwargs = {},
              title_str = None,
              fig = None,
              ax = None):
    
    #-------------------- Setup --------------------
    if type(obs_mods) == list:
        if 'idv' in dss_plot['obs'].sizes:
            dss_plot['obs'] = dss_plot['obs'].isel(idv=[mod in obs_mods for mod in dss_plot['obs'].model.values])
        else:
            dss_plot['obs'] = dss_plot['obs'].sel(model=obs_mods)

    if plot_type in ['levels','cv']:
        keep_vars = ['prtrend','pr_std']
    elif plot_type in ['tslevels','tscv']:
        keep_vars = ['pr_tsslope','pr_iqr']
    else:
        raise KeyError('`plot_type` must be one of: levels, cv, tslevels, tscv.')
    dfs = {exp:ds.sel(season=seas,start_year = start_year,end_year = end_year).to_dataframe()[keep_vars].dropna(axis=0) 
           for exp,ds in dss_plot.items()}
        
    annotate_args_ref = {'text_vgap':0.05,'text_v0':0.7,'arrow_length':0.1}
    for kwarg in [kwarg for kwarg in annotate_args if kwarg not in annotate_args_ref]:
        annotate_args[kwarg] = annotate_args_ref[kwarg]

    for exp in dfs:
        # Subset to top # of runs if desired (have to do this in 
        # pandas, because non-nan values are not always in run order
        # and xarray has no easy way to deal with that)
        if nruns is not None:
            dfs[exp] = dfs[exp].groupby('model').head(nruns)
        
        if exp_labels is None:
            dfs[exp]['explabel'] = exp+' ('+str(len(dfs[exp]))+' runs)'
        else:
            dfs[exp]['explabel'] = exp_labels[exp]+' ('+str(len(dfs[exp]))+' runs)'
        dfs[exp]['exp'] = exp

    dfs = pd.concat([df for exp,df in dfs.items()])
    
    
    if plot_type == 'levels':
        dfs['plot_var'] = dfs['prtrend']
        xlabel = r'$P$ trend [mm/day/10yr]'
    elif plot_type == 'cv':
        dfs['plot_var'] = dfs['prtrend'] / dfs['pr_std']
        xlabel = r'$P$ trend [Trend/10yr/SD]'
    elif plot_type == 'tslevels':
        dfs['plot_var'] = dfs['pr_tsslope']
        xlabel = r'$P$ Theil-Sen slope [mm/day/10yr]'
    elif plot_type == 'tscv':
        dfs['plot_var'] = dfs['pr_tsslope'] / dfs['pr_iqr']
        xlabel = r'$P$ Theil-Sen slope [Trend/10yr/IQR]'

    # Change to / decade to match Rowell
    dfs['plot_var'] = dfs['plot_var']*10
    
    if title_str is None:
        title_str = str(start_year)+'-'+str(end_year)+' '+seas+' trend'

    #-------------------- Plot --------------------
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = plt.subplot()

    # Plot KDEs of models
    
    ax = sns.kdeplot(data=dfs.loc[[x in plot_exps for x in dfs.exp],:].reset_index(),
                     x='plot_var',hue = 'explabel',palette = palette,
                     fill=True,ax=ax)

    # Plot obs as vertical lines
    for v in dfs.loc[dfs.exp=='obs'].plot_var.values:
        ax.axvline(v,color='tab:red',linestyle=':')

    # Text annotations
    ax.set_xlabel(xlabel,**text_kwargs)
    ax.set_title(title_str,**text_kwargs,fontweight='bold')
    ax.axvline(0,color='grey',linestyle='-')

    ax.set_xlim(*xlims)
    if type(ylims) is not str:
        ax.set_ylim(ylims)

    # Plot means of model dists
    ylims = ax.get_ylim()
    for exp,color in zip(plot_exps,palette):
        ax.plot([dfs[['plot_var','exp']].groupby('exp').mean().loc[exp]['plot_var']]*2,
            [0,0.1*ylims[1]],
            linewidth=4,color=color)

    ax.set_ylim(ylims)
    
    # Point out obs if desired
    if label_obs:
        # Get data from obs models 
        df_obs = dfs.loc[dfs['exp'] == 'obs']
        df_obs = df_obs.sort_values('plot_var',ascending=False)
        
        for omod,omod_idx in zip(df_obs.reset_index()['model'].values,np.arange(0,len(df_obs))):
            # Get coordinates of arrow point (at the x of the obs_mod,
            # at some reference y)
            arrow_point = [df_obs['plot_var'].loc[omod].values,
                               ax.get_ylim()[0] + np.diff(ax.get_ylim())*(annotate_args['text_v0'] - omod_idx*annotate_args['text_vgap'])]
            # Get text location (to the right of the lines if positive, 
            # to the left if negative
            if df_obs['plot_var'].mean()<=0:
                text_loc = [arrow_point[0]-np.diff(ax.get_xlim())*annotate_args['arrow_length'],
                            arrow_point[1]]
                ha = 'right'
            else:
                text_loc = [arrow_point[0]+np.diff(ax.get_xlim())*annotate_args['arrow_length'],
                            arrow_point[1]]
                ha = 'left'
            
            # Annotate
            ax.annotate(omod,xy=arrow_point,xytext=text_loc,
                    ha=ha,va='center',arrowprops = {'arrowstyle':'-','color':'tab:red'})
            # Add little round circle on the vertical line
            ax.plot(*arrow_point,marker='o',markersize=3.5,markerfacecolor='none',color='tab:red')
    
    #-------------------- Return --------------------
    return ax


def plot_triangle(ds_field,ds_hatch,
                  nruns = None,
                  season = 'MAM',
                  region = 'HoA-bimod',
                  plot_type = 'levels',
                  hatch_lims = [0.05,0.95],
                  cbar_params = {'vmin':-0.15,'vmax':0.15,'cmap':cmocean.cm.balance_r,'levels':21},
                  add_colorbar = False,
                  add_annotation=False,
                  trend_guide_spacing = 10,# or None
                  label_trend_guide = False,
                  factor = 10,
                  fig = None,
                  ax = None,
                  year_lims = None, # otherwise dict piped into .sel()
                  # 'all' = years where every field and hatch model has data
                  # 'any' = years where at least 1 field _and_ 1 hatch product has data 
                  # None = skip this step
                  year_subset = 'all'):
                  
    #------------- Gather data -------------
    plot_data = {'field':ds_field,'hatch':ds_hatch}

    if plot_type == 'levels':
        var_list = ['prtrend']
    elif plot_type == 'cv':
        var_list = ['prtrend','pr_std']
    elif plot_type == 'tslevels':
        var_list = ['pr_tsslope']
    elif plot_type == 'tscv':
        var_list = ['pr_tsslope','pr_iqr']
    else:
        raise ValueError(plot_type+' is not a valid plot_type.')

    # Subset to season, region
    plot_data = {typ:ds.sel(season=season,region=region,drop=True)[var_list]*factor
               for typ,ds in plot_data.items()}

    # Subset manually to specific years, if desired
    if year_lims is not None:
        plot_data = {typ:ds.sel(**year_lims) for typ,ds in plot_data.items()}

    # Subset dynamically to years with data, if desired
    if year_subset is not None:
        with warnings.catch_warnings():
            # Ignores depreciation warning from numpy product of creating array 
            # from ragged lists... 
            warnings.filterwarnings('ignore')
            
            for typ in plot_data:
                for dim in ['run','model']:
                    if dim not in plot_data[typ].sizes:
                        plot_data[typ] = plot_data[typ].expand_dims({dim:['0']})
            
            if year_subset == 'all':
                keep_matrix = xr.concat([(~np.isnan(ds[var_list[0]])).any('run').all('model')
                                             for typ,ds in plot_data.items()],
                                          dim='typ').prod('typ').astype(bool)
            elif year_subset == 'any':
                keep_matrix = xr.concat([(~np.isnan(ds[var_list[0]])).any('run').any('model')
                                             for typ,ds in plot_data.items()],
                                          dim='typ').prod('typ').astype(bool)

            plot_data = {typ:ds.where(keep_matrix,drop=True) for typ,ds in plot_data.items()}

    # Subset by number of runs, if desired
    if nruns is not None:
        for typ in plot_data:
            for var in var_list:
                if 'run' in plot_data[typ][var].sizes:
                    plot_data[typ][var] = (plot_data[typ][var].to_dataframe().dropna().
                                 groupby(['model','start_year','end_year']).head(nruns).
                                 to_xarray()[var])

                    plot_data[typ][var] = plot_data[typ][var].dropna(dim='run',how='all')

    # Average across model, run in field variable if sizes > 0 
    if 'idv' in plot_data['field'].sizes:
        # If using the idv / stacked file format
        plot_data['field'] = plot_data['field'].mean('idv')
    else:
        plot_data['field'] = plot_data['field'].mean([dim for dim in ['model','run'] if dim in plot_data['field'].sizes])

    for typ in plot_data:
        if plot_type in ['levels','tslevels']:
            plot_data[typ] = plot_data[typ][var_list[0]]
        elif plot_type in ['cv','tscv']:
            plot_data[typ] = plot_data[typ][var_list[0]] / plot_data[typ][var_list[1]]
            plot_data[typ] = plot_data[typ]*factor # This otherwise gets divided out

    # Subset to only start/end years that both arrays have
    plot_data = {typ:ds.sel(**{yrtyp:[yr for yr in plot_data['hatch'][yrtyp].values if yr in plot_data['field'][yrtyp]]
                          for yrtyp in ['start_year','end_year']})
                 for typ,ds in plot_data.items()}
            
    #------------- Get hatching -------------
    # Stack hatch variable into one model-run 
    # dimension for %ile calc
    if 'idv' in plot_data['hatch'].sizes:
        # idv format uses different name for model-run dimension than this code
        plot_data['hatch'] = plot_data['hatch'].rename({'idv':'modrun'})
    else:
        plot_data['hatch'] = plot_data['hatch'].stack(modrun = ('model','run'))

    # Get what %ile of the hatch distribution
    # the field distribution is 
    pcts = xr.apply_ufunc(percentileof_alongdim,
                          plot_data['hatch'],
                          plot_data['field'],
                          input_core_dims = [['modrun'],[]],
                          vectorize=True)

    # Get hatching flag
    hatching = (pcts < np.min(hatch_lims)) | (pcts > np.max(hatch_lims))

    # Make sure hatching leaves the stuff blank it should
    hatching = hatching.where(~np.isnan(plot_data['field']))

    #------------- Plot -------------
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = plt.subplot()

    # Shade observations
    plot_data['field'].plot.contourf(x='start_year',**cbar_params,add_colorbar=False,ax=ax)
    # Add 0 line
    plot_data['field'].plot.contour(x='start_year',levels=[0],colors=['grey'],ax=ax)

    # Add hatching for obs outside of CMIP6 AMIP quantiles
    (hatching.plot.contourf(levels=[-1,0],hatches=[None,'///'],cmap='none',add_colorbar=False,
                   x='start_year',ax=ax))

    # Add a dividing line if the last year is before the graph
    # edges (for AMIP, for example, where obs goes beyond) 
    if hatching.end_year.max()<plot_data['field'].end_year.max():
        ax.axhline(hatching.end_year.max(),color='k',linestyle='--')
        if (col_idx == 0):
            ax.annotate('Model limit',(plot_data['field'].start_year.max()-1,hatching.end_year.max()),
                        (plot_data['field'].start_year.max()-1,hatching.end_year.max()-5),ha='right',va='top',
                        arrowprops={'arrowstyle':'->','relpos':(0.5,0.5)},
                        rotation=270)


    #------------- Annotate -------------

    ax.set_xlabel('Trend start year')
    ax.set_ylabel('Trend end year')

    if add_annotation:
        ann_params = {'xy':[0.975,0.005],
                          'va':'bottom'}
        ax.annotate('Hatching: outside of \n'+'-'.join([str(q) for q in qs])+' range',
                    xycoords='axes fraction',ha='right',**ann_params)


    ax.set_aspect('equal')

    # Add trend length guidance
    if trend_guide_spacing is not None:
        lims = ax.get_xlim()
        guide_lines = [[[lims[0],lims[1]-(guide_idx*trend_guide_spacing)],
                         [lims[0]+(guide_idx*trend_guide_spacing),lims[1]]]
                       for guide_idx in np.arange(1,np.diff(lims)[0]/trend_guide_spacing)]
        for guide_idx in range(0,len(guide_lines)):
            ax.plot(*guide_lines[guide_idx],color='grey',linewidth=0.5)
    
        # Now set tick labels with trend length
        if label_trend_guide:
            ax_tmp = ax.secondary_xaxis('top')
            ax_tmp.set_xticks([gl[0][1] for gl in guide_lines],
                             labels=[str(int((guide_idx*trend_guide_spacing)))+' yrs'
                                       for guide_idx in np.arange(1,np.diff(lims)[0]/trend_guide_spacing)],
                              rotation=45,ha='left')
            ax_tmp.tick_params(axis='x',top=False,pad=-2)

    if add_colorbar:
        # Vertical colorbar
        fig.subplots_adjust(right=0.825)
        cax = fig.add_axes([0.875, 0.15, 0.025, 0.7])
        levels = mpl.ticker.MaxNLocator(nbins=cbar_params['levels']).tick_values(cbar_params['vmin'],cbar_params['vmax'])
        norm = mpl.colors.BoundaryNorm(levels, ncolors=cbar_params['cmap'].N ,extend='both')
        sm = plt.cm.ScalarMappable(cmap=cbar_params['cmap'],norm=norm)
        plt.colorbar(sm,cax=cax,label=clabel)

    #------------- Return -------------
    return fig,ax

def plot_triangles(dss,
                   exps = ['hist-ssp245','amip'],
                   obs_mods = 'auto', # or list of models
                   lims = [1980,2022],
                   seas = 'MAM',
                   label_trend_guide = True,
                   trend_guide_spacing = 10,
                   ncol = 5,
                   figsize = (20,15),
                   cbar_params = {'vmin':-1.5,'vmax':1.5,
                       'cmap':cmocean.cm.balance_r,'levels':21},
                   plot_type = 'levels', # or cv, tslevels, tscv
                   exp_titles = {'hist-ssp245':r'CMIP6\ (Hist\ /\ SSP245)','amip':'AMIP6',
                                 'hindcasts05lead':r'0.5-month\ lead','hindcasts1-3lead':r'1.5-3.5-month\ lead'},
                   year_subset = 'all',
                   save_fig = False,
                   output_fn = None):
    ''' Plot a separate triangle plot comparing against
    each observational data product '''
                
    if (type(obs_mods) == str) and (obs_mods == 'auto'):
        obs_mods = dss['obs'].model.values

    #------- Create figure -------
    nrows = ((len(obs_mods)-1) // ncol + 1) * len(exps) 
    nrows = nrows + int((nrows / len(exps)) - 1)
    height_ratios = np.ones(nrows)
    height_ratios[np.arange(0,nrows)[len(exps)::(len(exps)+1)]] = 0.1


    fig,axs = plt.subplots(nrows,ncol,
                           figsize=figsize,
                           gridspec_kw={'height_ratios':height_ratios}
                            )

    #------- Set plot data -------
    ax_params = [[{'ds_field':dss['obs'].sel(model=obs_mod),'ds_hatch':dss[exp],'season':seas,'plot_type':plot_type}
                 for obs_mod in obs_mods] for exp in exps]

    if plot_type == 'levels':
        clabel = r'$P$ trend [mm/day/10yr]'
    elif plot_type == 'cv':
        clabel = r'$P$ trend [Trend/10yr/SD]'
    elif plot_type == 'tslevels':
        clabel = r'$P$ Theil-Sen slope [mm/day/10yr]'
    elif plot_type == 'tscv':
        clabel = r'$P$ Theil-Sen slope [Trend/10yr/IQR]'

    for exp_idx in np.arange(0,len(exps)):
        for mod_idx in np.arange(0,len(obs_mods)):
            #for plt_idx in np.arange(0,len(ax_params)):

            #--------- Plot location ---------
            # Get row index
            row_idx = (mod_idx // ncol)*len(exps) + exp_idx
            # Add offset for dummy axis in the middle 
            row_idx = row_idx + mod_idx // ncol

            # Column index
            col_idx = mod_idx % ncol

            if (row_idx == 0) and (label_trend_guide):
                ex_params = {'label_trend_guide':True}
            else:
                ex_params = {}

            fig,axs[row_idx,col_idx] = plot_triangle(**ax_params[exp_idx][mod_idx],year_subset = 'any',
                                             fig=fig,ax=axs[row_idx,col_idx],cbar_params = cbar_params,
                                                     trend_guide_spacing = trend_guide_spacing,
                                                    **ex_params)


            # Axis lettering
            if exp_idx == 0:
                axs[row_idx,col_idx].set_title(r'$\mathbf{'+str(ax_params[exp_idx][mod_idx]['ds_field'].model.values)+'}$',fontsize=12)

            if (exp_idx == (len(exps)-1)) and ((row_idx == (nrows - 1)) or 
                                              ((row_idx == (nrows - len(exps) - 2)) and 
                                               (col_idx >= (len(obs_mods) % ncol)))):
                axs[row_idx,col_idx].set_xlabel('Trend start year',fontsize=12)
            else:
                axs[row_idx,col_idx].set_xlabel('')

            if (mod_idx % ncol) == 0:
                axs[row_idx,col_idx].set_ylabel(r'$\mathbf{'+exp_titles[exps[exp_idx]]+'}$'+'\nTrend end year',fontsize=12)
            else:
                axs[row_idx,col_idx].set_ylabel('')

            # Subplot lettering
            if len(obs_mods)*ncol <= 26:
                sp_id = string.ascii_lowercase[(ncol*row_idx)+col_idx]
            else:
                sp_id = printRoman((ncol*row_idx)+col_idx + 1).lower()
            if label_trend_guide and (row_idx == 0):
                pos = [-0.05,1.01]
            else:
                pos = [0.01,1.01]
            axs[row_idx,col_idx].annotate(sp_id+'.',
                                pos,xycoords='axes fraction',
                                va='bottom',ha='left',fontsize=12,fontweight='bold')

            # Axis parameters
            if np.diff(lims)<50:
                tick_spacing = 10
            else:
                tick_spacing = 20
            axs[row_idx,col_idx].set_yticks(np.arange(lims[0],lims[1],tick_spacing))
            axs[row_idx,col_idx].set_xticks(np.arange(lims[0],lims[1],tick_spacing))
            axs[row_idx,col_idx].set_xlim(lims)
            axs[row_idx,col_idx].set_ylim(lims)
            axs[row_idx,col_idx].grid()
            # Grid parameters
            if (mod_idx % ncol) == 0:
                axs[row_idx,col_idx].tick_params(axis='y', which='both',left=True,labelleft=True)
            elif (mod_idx % ncol) == (ncol - 1):
                axs[row_idx,col_idx].tick_params(axis='y',which='both',right=True,labelright=True,left=False,labelleft=False)
            else:
                axs[row_idx,col_idx].tick_params(axis='y', which='both',left=False,labelleft=False)

            if (exp_idx == (len(exps)-1)) and ((row_idx == (nrows - 1)) or 
                                              ((row_idx == (nrows - len(exps) - 2)) and 
                                               (col_idx >= (len(obs_mods) % ncol)))):
                axs[row_idx,col_idx].tick_params(axis='x',which='both',bottom=True,labelbottom=True)
            else:
                axs[row_idx,col_idx].tick_params(axis='x',which='both',bottom=False,labelbottom=False)

    #--------- Additional annotations ---------
    # Add legend
    axs[0,0].legend(handles=[mpatches.Patch(facecolor=cmocean.cm.balance_r(80), edgecolor=cmocean.cm.balance_r(80),
                             label='Obs. trends'),
                           mpatches.Patch(facecolor='w',edgecolor='k',hatch='\\\\\\',
                                          label='Obs. outside of\n90% of models/runs')],
                  loc='lower right',fontsize='small')

    # Vertical colorbar
    fig.subplots_adjust(right=0.825)
    cax = fig.add_axes([0.875, 0.15, 0.025, 0.7])
    levels = mpl.ticker.MaxNLocator(nbins=cbar_params['levels']).tick_values(cbar_params['vmin'],cbar_params['vmax'])
    norm = mpl.colors.BoundaryNorm(levels, ncolors=cbar_params['cmap'].N, clip=True)
    sm = plt.cm.ScalarMappable(cmap=cbar_params['cmap'],norm=norm)
    cb = plt.colorbar(sm,cax=cax)
    cb.ax.tick_params(labelsize=15) 
    cb.set_label(clabel,fontsize=15)


    # Blank out rows
    blank_rows = ((len(exps)+1) * np.arange(0,len(obs_mods) // ncol + 1) - 1)
    blank_rows = blank_rows[(blank_rows > 0) & (blank_rows < nrows)]
    for row_idx in blank_rows: 
        for col_idx in np.arange(0,ncol):
            axs[row_idx,col_idx].set_axis_off()

            # Add line visually separating the two sets of rows
            #axs[row_idx,col_idx].plot([-0.5,1.5],[0.25,0.25], clip_on=False,#transform=axs[row_idx,col_idx].transAxes
            #          linewidth=2, color='k')

    # Blank out panels in rows at the end 
    if len(obs_mods) % ncol != 0:
        for col_idx in np.arange(len(obs_mods) % ncol, ncol):
            for row_idx in np.arange((len(axs)-len(exps)),len(axs)):
                axs[row_idx,col_idx].set_visible(False)



    #--------- Save ---------
    if save_fig:
        utility_print(output_fn)