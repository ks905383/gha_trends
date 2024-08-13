[![DOI](https://zenodo.org/badge/840417831.svg)](https://zenodo.org/doi/10.5281/zenodo.13315405)

# Revisiting the `East African Paradox' Replication Code
This repository contains replication code for Schwarzwald, Kevin and Richard Seager (2024), "'Revisiting the “East African Paradox': CMIP6 models also struggle to reproduce strong observed MAM long rain drying trends." (under revision at Journal of Climate). This includes the code necessary to calculate the trends used in the analysis, in addition to the code necessary to reproduce all main text and supplementary figures. 

Replication data can be found [here](https://zenodo.org/doi/10.5281/zenodo.13286726). 

The code requires the following directories, contained in the replication data repository above (the upper-level directories don't necessarily have to be in the same top-level directory or be called "raw" or "proc"; either way, their location and name must be set explicitly in `dir_list.csv`): 

```
project
 |- raw
    |- model1
    |- model2
    |- obsprod1
    |- etc.
 |- proc
    |- model1
    |- model2
    |- obsprod1
    |- etc.
 |- figs
 |- code
 |- aux
```
Where the paths to `raw`, `proc`, etc. are set in `dir_list.csv`. The `raw` directory contains "raw" (but preprocessed$^*$) climate data files. The `proc` directory contains processed files, which are created by the code in this repository. 

This repository also includes an `environment.yaml` file, which can be used to create a conda or mamba environment, which should include all necessary packages to run this code. To create the `gha_trends` environment from file, run: 

```conda env create -f environment.yml```

## File conventions
The code assumes that all climate data files follow a file naming convention akin to the CMIP5/CMIP6 formats (should be able to handle either, whether with or without the grid specification), with a slight modification detailed below. This includes data files from observational or reanalysis data products, which must additionally be preprocessed to follow CMIP* variable name and file structure (one file per variable over all time) conventions. In other words, filenames in `raw` or `proc` must be of the form: 

```
pr_Amon_ACCESS-CM2_historical_r1i1p1f1_18500101-20141231_HoAfrica.nc
[varname]_[freq]_[model]_[experiment]_[ensmember]_[timeframe]_[suffix].nc
```
The (optional) `[suffix]` field is the main difference from the standard CMIP* format; author KS uses it as a field to identify files that span less than global. 

Note that files _must_ be stored in subdirectories under `raw` or `proc` named after their model or data product (so `raw/ACCESS-CM2/` or `proc/CHIRPS`, for example). 

## Other notes
For questions, please reach out to @ks905383 Kevin Schwarzwald. 
