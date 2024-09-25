---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.2
  kernelspec:
    display_name: 1 Python 3 (based on the module python3/2023.01)
    language: python
    name: python3_2023_01
---

<b><i> carbon_cycle_and_amoc </i></b>

Created by Eduardo Alastrue de Asenjo on 2023-11-14

- Purpose: Understand changes in carbon cycle due to AMOC decline
- Methods: Analyse ocean bgc data (e.g., dissic), AMOC, and SAT in MPI-ESM simulations
- Comments: Works with Levante's Jupyterhub kernel python3/2023.1 
- Other resources:



# Load modules, variables, and functions

```python
import xarray as xr
import numpy as np
import dask
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import os; os.environ['PROJ_LIB'] = '/work/uo1075/m300817/phd/conda/share/proj'
#import regionmask
import xesmf as xe
import cdo # Import Cdo-py
cdo = cdo.Cdo(tempdir='/scratch/m/m300817/tmp/cdo-py') # change this to a directory in your scratch
import eccodes
import cfgrib
import zlib
from tqdm import tqdm
```

```python
g_per_molC = 12.0107  # to convert units to g, as dissic units are given in mol m-3
```

```python
def weighted_area_lat(ds):
    """
    Calculate the area-weighted temperature over its domain. 
    In a regular latitude/ longitude grid the grid cell area decreases towards the pole.
    We can use the cosine of the latitude as proxy for the grid cell area.
    Taken from https://docs.xarray.dev/en/stable/examples/area_weighted_temperature.html
    """
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = "weights"

    return ds.weighted(weights)
```

```python
def weighted_mon_to_year_mean(ds, var):
    """
    weight by days in each month when doing annual mean from monthly values in xarray
    taken from https://ncar.github.io/esds/posts/2021/yearly-averages-xarray/
    """
    month_length = ds.time.dt.days_in_month # Determine the month length
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()  # Calculate the weights
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)  # Make sure the weights in each year add up to 1
    obs = ds[var]  # Subset our dataset for our variable
    cond = obs.isnull() # Setup our masking for nan values
    ones = xr.where(cond, 0.0, 1.0)   
    obs_sum = (obs * wgts).resample(time="AS").sum(dim="time")  # Calculate the numerator
    ones_out = (ones * wgts).resample(time="AS").sum(dim="time") # Calculate the denominator
    return obs_sum / ones_out # Return the weighted average
```

```python
colors_scen = {      # IPCC colors for SSPs
    'hist': (0/255, 0/255, 0/255),        
    'ssp126': (23/255, 60/255, 102/255),      
    'ssp245':(247/255, 148/255, 32/255),      
    'ssp585': (149/255, 27/255, 30/255)      
}
```

dask cluster

```python
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import dask
#dask.config.set(scheduler="single-threaded")
dask.config.set(scheduler="threads")
dask.config.config.get('distributed').get('dashboard').update({'link':'{JUPYTERHUB_SERVICE_PREFIX}/proxy/{port}/status'})
from dask.distributed import Lock
lock = Lock('xarray-IO')
```

```python
cluster = SLURMCluster(name='dask-cluster',
                      cores=32,          # number of cores per job / number of cores each worker can use
                      processes=6,      # number of workers per SLURM job
                      memory='100GB',     # total memory per job / memory each worker has
                      account='uo1075',
                      interface='ib0',
                      queue='shared,compute',
                      walltime='7:40:00',
                      local_directory = '/scratch/m/m300817/dask_temp/',
                      log_directory   = '/scratch/m/m300817/dask_temp/log/',
                      asynchronous=0)
                      #, scheduler_options={'dashboard_address': ':8787'})
cluster.scale(18) # number of workers
#cluster.adapt(minimum=12,maximum=36, wait_count=60) # number of workers
```

```python
client = Client(cluster)
```

<!-- #region toc-hr-collapsed=true -->
# Load data (MPI-ESM1.2-LR)
<!-- #endregion -->

<!-- #region toc-hr-collapsed=true -->
## Dissolved inorganic Carbon (dissic)
<!-- #endregion -->

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### piC, historical, scenarios
<!-- #endregion -->

```python
# Load DIC in piC
file_type = 'dissic'
infiles = glob.glob(f'/pool/data/CMIP6/data/CMIP/MPI-M/MPI-ESM1-2-LR/piControl/r1i1p1f1/Omon/{file_type}/gn/v20190710/*{file_type}*.nc')
ds_dissic_pi = xr.open_mfdataset(infiles, use_cftime=True, chunks={"time": 50}, parallel=True)
```

```python
# Load DIC in historical
file_type = 'dissic'
infiles = glob.glob(f'/pool/data/CMIP6/data/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Omon/{file_type}/gn/v20190710/*{file_type}*.nc')
ds_dissic_hist = xr.open_mfdataset(infiles, use_cftime=True, parallel=True)
```

```python
# Load DIC in scenario ssp126
file_type = 'dissic'
infiles = glob.glob(f'/pool/data/CMIP6/data/ScenarioMIP/MPI-M/MPI-ESM1-2-LR/ssp126/r1i1p1f1/Omon/{file_type}/gn/v20190710/*{file_type}*.nc')
ds_dissic_ssp126 = xr.open_mfdataset(infiles, use_cftime=True, parallel=True)
```

```python
# Load DIC in scenario ssp585
file_type = 'dissic'
infiles = glob.glob(f'/pool/data/CMIP6/data/ScenarioMIP/MPI-M/MPI-ESM1-2-LR/ssp585/r1i1p1f1/Omon/{file_type}/gn/v20190710/*{file_type}*.nc')
ds_dissic_ssp585 = xr.open_mfdataset(infiles, use_cftime=True, parallel=True)
```

### 1pctCO2

```python
# Load DIC in 1pctCO2
file_type = 'dissic'
infiles = glob.glob(f'/pool/data/CMIP6/data/CMIP/MPI-M/MPI-ESM1-2-LR/1pctCO2/r1i1p1f1/Omon/{file_type}/gn/v20190710/*{file_type}*.nc')
ds_dissic_1pct = xr.open_mfdataset(infiles, use_cftime=True, parallel=True, chunks={"time": 10})
```

```python
# Load DIC in 1pctCO2-bgc
file_type = 'dissic'
infiles = glob.glob(f'/pool/data/CMIP6/data/C4MIP/MPI-M/MPI-ESM1-2-LR/1pctCO2-bgc/r1i1p1f1/Omon/{file_type}/gn/v20190710/*{file_type}*.nc')
ds_dissic_1pctbgc = xr.open_mfdataset(infiles, use_cftime=True, parallel=True, chunks={"time": 10})
```

```python
# Load DIC in 1pctCO2-rad
file_type = 'dissic'
infiles = glob.glob(f'/pool/data/CMIP6/data/C4MIP/MPI-M/MPI-ESM1-2-LR/1pctCO2-rad/r1i1p1f1/Omon/{file_type}/gn/v20190710/*{file_type}*.nc')
ds_dissic_1pctrad = xr.open_mfdataset(infiles, use_cftime=True, parallel=True, chunks={"time": 10})
```

<!-- #region toc-hr-collapsed=true jp-MarkdownHeadingCollapsed=true -->
### Original hosing
<!-- #endregion -->

u05-LR

<!-- #raw -->
# NO NEED TO RE-RUN --- Pre-process (select and merge) dissic monthly
in_path  = '/work/mh0033/from_Mistral/mh0033/m300817/mpiesm-1.2.01p6-passivesalt_update/experiments/hosing_naa05Sv_FcSV-LR/outdata/hamocc/'
wildcard = "*hamocc_data_3d_mm_*"
ifiles   = in_path+wildcard
outpath = "/work/uo1075/m300817/hosing/post/data/dissic/"
outfile = "hosing_naa05Sv_FcSV-LR_dissic.nc"
if not glob.glob(outpath):
    os.mkdir(outpath)
if not os.path.isfile(outpath+outfile):
    cdo.mergetime(input = "-select,name=dissic "+ifiles, output = outpath+outfile)
<!-- #endraw -->

```python
# Load resulting file (note: now in new location)
ds_dissic_u05hosLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_naa05Sv_FcSV-LR_dissic.nc"
                                       , use_cftime=True, parallel=True)
```

u03-LR

<!-- #raw -->
# NO NEED TO RE-RUN --- Pre-process (select and merge) dissic monthly
in_path  = '/work/mh0287/from_Mistral/mh0287/mh0469/m211054/projects/hosing/mpiesm-1.2.01p6-passivesalt_update/experiments/hosing_naa03Sv_FcSV-LR/outdata/hamocc/'
wildcard = "*hamocc_data_3d_mm_*"
ifiles   = in_path+wildcard
outpath = "/work/uo1075/m300817/hosing/post/data/dissic/"
outfile = "hosing_naa03Sv_FcSV-LR_dissic.nc"
if not glob.glob(outpath):
    os.mkdir(outpath)
if not os.path.isfile(outpath+outfile):
    cdo.mergetime(input = "-select,name=dissic "+ifiles, output = outpath+outfile)
<!-- #endraw -->

```python
# Load resulting file (note: now in new location)
ds_dissic_u03hosLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_naa03Sv_FcSV-LR_dissic.nc"
                                       , use_cftime=True, parallel=True)
```

g03-LR

<!-- #raw -->
# NO NEED TO RE-RUN --- Pre-process (select and merge) dissic monthly
in_path  = '/work/mh0033/from_Mistral/mh0033/m300817/mpiesm-1.2.01p6-passivesalt_update/experiments/hosing_grc03Sv_FcSV-LR/outdata/hamocc/'
wildcard = "*hamocc_data_3d_mm_*"
ifiles   = in_path+wildcard
outpath = "/work/uo1075/m300817/hosing/post/data/dissic/"
outfile = "hosing_grc03Sv_FcSV-LR_dissic.nc"
if not glob.glob(outpath):
    os.mkdir(outpath)
if not os.path.isfile(outpath+outfile):
    cdo.mergetime(input = "-select,name=dissic "+ifiles, output = outpath+outfile)
<!-- #endraw -->

```python
# Load resulting file
ds_dissic_g03hosLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_grc03Sv_FcSV-LR_dissic.nc"
                                       , use_cftime=True, parallel=True)
```

g01-LR

<!-- #raw -->
# NO NEED TO RE-RUN --- Pre-process (select and merge) dissic yearly
in_path  = '/work/uo1075/m300817/hosing/mpiesm-1.2.01p6-passivesalt_update/experiments/hosing_grc01Sv_FcSV-LR/outdata/hamocc/'
wildcard = "*hamocc_data_3d_ym_*"
ifiles   = in_path+wildcard
outpath = "/work/uo1075/m300817/hosing/post/data/dissic/"
outfile = "hosing_grc01Sv_FcSV-LR_dissic.nc"
if not glob.glob(outpath):
    os.mkdir(outpath)
if not os.path.isfile(outpath+outfile):
    cdo.mergetime(input = "-select,name=dissic "+ifiles, output = outpath+outfile)
<!-- #endraw -->

```python
# Load resulting file. NOTE: HERE IS DIRECTLY YEARLY MEANS
ds_dissic_g01hosLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_grc01Sv_FcSV-LR_dissic.nc"
                                       , use_cftime=True, parallel=True)
```

### New hosing 1pct & scenarios


1pctbgc

```python
# NO NEED TO RE-RUN --- Pre-process (select and merge) dissic yearly
wildcard = "*hamocc_data_3d_ym_*"
for exp in ["hosing_naa01Sv_1pctbgc-LR","hosing_naa03Sv_1pctbgc-LR", "hosing_naa05Sv_1pctbgc-LR",
            "hosing_grc01Sv_1pctbgc-LR", "hosing_grc03Sv_1pctbgc-LR", "hosing_grc05Sv_1pctbgc-LR",
            "hosing_naa03Sv_1pct-LR", "hosing_naa05Sv_1pct-LR"]:
    in_path  = f'/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/{exp}/outdata/hamocc/'
    ifiles   = in_path+wildcard
    outpath = "/work/uo1075/m300817/hosing/post/data/dissic/"
    outfile = f"{exp}_dissic.nc"
    if not glob.glob(outpath):
        os.mkdir(outpath)
    if not os.path.isfile(outpath+outfile):
        cdo.mergetime(input = "-select,name=dissic "+ifiles, output = outpath+outfile)
```

```python
# Load resulting file. NOTE: HERE IS DIRECTLY YEARLY MEANS
ds_dissic_u011pctbgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_naa01Sv_1pctbgc-LR_dissic.nc"
                                       , use_cftime=True, parallel=True, chunks={"time": 10})
ds_dissic_u031pctbgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_naa03Sv_1pctbgc-LR_dissic.nc"
                                       , use_cftime=True, parallel=True, chunks={"time": 10})
ds_dissic_u051pctbgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_naa05Sv_1pctbgc-LR_dissic.nc"
                                       , use_cftime=True, parallel=True, chunks={"time": 10})
ds_dissic_g011pctbgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_grc01Sv_1pctbgc-LR_dissic.nc"
                                       , use_cftime=True, parallel=True, chunks={"time": 10})
ds_dissic_g031pctbgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_grc03Sv_1pctbgc-LR_dissic.nc"
                                       , use_cftime=True, parallel=True, chunks={"time": 10})
ds_dissic_g051pctbgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_grc05Sv_1pctbgc-LR_dissic.nc"
                                       , use_cftime=True, parallel=True, chunks={"time": 10})
ds_dissic_u031pctLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_naa03Sv_1pct-LR_dissic.nc"
                                       , use_cftime=True, parallel=True, chunks={"time": 10})
ds_dissic_u051pctLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_naa05Sv_1pct-LR_dissic.nc"
                                       , use_cftime=True, parallel=True, chunks={"time": 10})
```

hist-bgc and u03 bgc-scenarios

```python
# NO NEED TO RE-RUN --- Pre-process (select and merge) dissic yearly
wildcard = "*hamocc_data_3d_ym_*"
for exp in ["histbgc-LR","hosing_naa03Sv_ssp126bgc-LR", "hosing_naa03Sv_ssp245bgc-LR",
            "hosing_naa03Sv_ssp585bgc-LR"]:
    in_path  = f'/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/{exp}/outdata/hamocc/'
    ifiles   = in_path+wildcard
    outpath = "/work/uo1075/m300817/hosing/post/data/dissic/"
    outfile = f"{exp}_dissic.nc"
    if not glob.glob(outpath):
        os.mkdir(outpath)
    if not os.path.isfile(outpath+outfile):
        cdo.mergetime(input = "-select,name=dissic "+ifiles, output = outpath+outfile)
```

```python
# Load resulting file. NOTE: HERE IS DIRECTLY YEARLY MEANS
ds_dissic_histbgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/histbgc-LR_dissic.nc"
                                       , use_cftime=True, parallel=True, chunks={"time": 10})
ds_dissic_u03ssp126bgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_naa03Sv_ssp126bgc-LR_dissic.nc"
                                       , use_cftime=True, parallel=True, chunks={"time": 10})
ds_dissic_u03ssp245bgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_naa03Sv_ssp245bgc-LR_dissic.nc"
                                       , use_cftime=True, parallel=True, chunks={"time": 10})
ds_dissic_u03ssp585bgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_naa03Sv_ssp585bgc-LR_dissic.nc"
                                       , use_cftime=True, parallel=True, chunks={"time": 10})
```

ssp245bgc

```python
# NO NEED TO RE-RUN --- Pre-process (select and merge) dissic yearly
wildcard = "*hamocc_data_3d_ym_*"
for exp in ["hosing_naa01Sv_ssp245bgc-LR","hosing_naa03Sv_ssp245bgc-LR", "hosing_naa05Sv_ssp245bgc-LR",
            "hosing_grc01Sv_ssp245bgc-LR", "hosing_grc03Sv_ssp245bgc-LR", "hosing_grc05Sv_ssp245bgc-LR",
            "ssp245bgc-LR"]:
    in_path  = f'/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/{exp}/outdata/hamocc/'
    ifiles   = in_path+wildcard
    outpath = "/work/uo1075/m300817/hosing/post/data/dissic/"
    outfile = f"{exp}_dissic.nc"
    if not glob.glob(outpath):
        os.mkdir(outpath)
    if not os.path.isfile(outpath+outfile):
        cdo.mergetime(input = "-select,name=dissic "+ifiles, output = outpath+outfile)
```

```python
# Load resulting file. NOTE: HERE IS DIRECTLY YEARLY MEANS
ds_dissic_u01ssp245bgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_naa01Sv_ssp245bgc-LR_dissic.nc"
                                       , use_cftime=True, parallel=True, chunks={"time": 10})
ds_dissic_u03ssp245bgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_naa03Sv_ssp245bgc-LR_dissic.nc"
                                       , use_cftime=True, parallel=True, chunks={"time": 10})
ds_dissic_u05ssp245bgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_naa05Sv_ssp245bgc-LR_dissic.nc"
                                       , use_cftime=True, parallel=True, chunks={"time": 10})
ds_dissic_g01ssp245bgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_grc01Sv_ssp245bgc-LR_dissic.nc"
                                       , use_cftime=True, parallel=True, chunks={"time": 10})
ds_dissic_g03ssp245bgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_grc03Sv_ssp245bgc-LR_dissic.nc"
                                       , use_cftime=True, parallel=True, chunks={"time": 10})
ds_dissic_g05ssp245bgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/hosing_grc05Sv_ssp245bgc-LR_dissic.nc"
                                       , use_cftime=True, parallel=True, chunks={"time": 10})
ds_dissic_ssp245bgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/dissic/ssp245bgc-LR_dissic.nc"
                                       , use_cftime=True, parallel=True, chunks={"time": 10})
```

<!-- #region toc-hr-collapsed=true -->
## SAT (tas)
<!-- #endregion -->

<!-- #region toc-hr-collapsed=true jp-MarkdownHeadingCollapsed=true -->
### Original hosing
<!-- #endregion -->

u03 & u05

```python
ds_tas_u03hosLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/tas/u03-hos-LR_tas_mon.nc", use_cftime=True, parallel=True)
ds_tas_u05hosLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/tas/u05-hos-LR_tas_mon.nc", use_cftime=True, parallel=True)
ds_tas_u03hosLR=ds_tas_u03hosLR.assign_coords(time=xr.cftime_range(start="1850", periods=2400, freq="M", calendar="proleptic_gregorian"))
ds_tas_u05hosLR=ds_tas_u05hosLR.assign_coords(time=xr.cftime_range(start="1850", periods=2400, freq="M", calendar="proleptic_gregorian"))
```

<!-- #raw -->
# For daily data
ds_tas_u03hosLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/tas/u03-hos-LR_tas.nc", use_cftime=True, parallel=True)
ds_tas_u05hosLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/tas/u05-hos-LR_tas.nc", use_cftime=True, parallel=True)
ds_tas_u03hosLR=ds_tas_u03hosLR.assign_coords(time=xr.cftime_range(start="1850", periods=73049, freq="D", calendar="proleptic_gregorian"))
ds_tas_u05hosLR=ds_tas_u05hosLR.assign_coords(time=xr.cftime_range(start="1850", periods=73049, freq="D", calendar="proleptic_gregorian"))
<!-- #endraw -->

g03-LR

```python
# grib to nc 
cdo.setCdo("/sw/spack-levante/cdo-1.9.10-j5frmz/bin/cdo")
inpath  = "/work/mh0033/from_Mistral/mh0033/m300817/mpiesm-1.2.01p6-passivesalt_update/experiments/hosing_grc03Sv_FcSV-LR/outdata/echam6/"
outpath = "/scratch/m/m300817/tmp/"
for year in tqdm(range(1850, 2050)):
    outfile=f"hosing_grc03Sv_FcSV-LR_echam6_echammon_{year}.nc"
    if not os.path.isfile(outpath+outfile):
        cdo.copy(input = f"-selcode,167 {inpath}hosing_grc03Sv_FcSV-LR_echam6_echammon_{year}.grb", 
                 output=outpath+outfile, 
                 options = '-f nc -t echam6 ')
```

```python
# Load resulting file
ifiles  = "/scratch/m/m300817/tmp/hosing_grc03Sv_FcSV-LR_echam6_echammon_*.nc"
ds_tas_g03hosLR = xr.open_mfdataset(ifiles, use_cftime=True)
ds_tas_g03hosLR=ds_tas_g03hosLR.assign_coords(time=xr.cftime_range(start="1850", periods=2400, freq="M", calendar="proleptic_gregorian"))
```

```python
# grib to nc 
cdo.setCdo("/sw/spack-levante/cdo-1.9.10-j5frmz/bin/cdo")
inpath  = "/work/mh0033/from_Mistral/mh0033/m300817/mpiesm-1.2.01p6-passivesalt_update/experiments/hosing_grc03Sv_FcSV-LR/outdata/echam6/"
outpath = "/work/uo1075/m300817/hosing/post/data/tas/g03-hos-LR_tas_mon/"
for year in tqdm(range(1850, 2050)):
    outfile=f"hosing_grc03Sv_FcSV-LR_echam6_echammon_{year}.nc"
    if not os.path.isfile(outpath+outfile):
        cdo.copy(input = f"-selcode,167 {inpath}hosing_grc03Sv_FcSV-LR_echam6_echammon_{year}.grb", 
                 output=outpath+outfile, 
                 options = '-f nc -t echam6 ')
```

```python
cdo.mergetime(input = "-chname,temp2,tas /work/uo1075/m300817/hosing/post/data/tas/g03-hos-LR_tas_mon/*nc", 
                 output="/work/uo1075/m300817/hosing/post/data/tas/g03-hos-LR_tas_mon.nc")
```

g01-LR

```python
# grib to nc 
cdo.setCdo("/sw/spack-levante/cdo-1.9.10-j5frmz/bin/cdo")
inpath  = "/work/uo1075/m300817/hosing/mpiesm-1.2.01p6-passivesalt_update/experiments/hosing_grc01Sv_FcSV-LR/outdata/echam6/"
outpath = "/work/uo1075/m300817/hosing/post/data/tas/g01-hos-LR_tas_mon/"
for year in tqdm(range(1850, 2050)):
    outfile=f"hosing_grc01Sv_FcSV-LR_echam6_echammon_{year}.nc"
    if not os.path.isfile(outpath+outfile):
        cdo.copy(input = f"-selcode,167 {inpath}hosing_grc01Sv_FcSV-LR_echam6_echammon_{year}.grb", 
                 output=outpath+outfile, 
                 options = '-f nc -t echam6 ')
```

```python
cdo.mergetime(input = "-chname,temp2,tas /work/uo1075/m300817/hosing/post/data/tas/g01-hos-LR_tas_mon/*nc", 
                 output="/work/uo1075/m300817/hosing/post/data/tas/g01-hos-LR_tas_mon.nc")
```

```python
# grib to nc 
cdo.setCdo("/sw/spack-levante/cdo-1.9.10-j5frmz/bin/cdo")
inpath  = "/work/uo1075/m300817/hosing/mpiesm-1.2.01p6-passivesalt_update/experiments/hosing_grc01Sv_FcSV-LR/outdata/echam6/"
outpath = "/scratch/m/m300817/tmp/"
for year in tqdm(range(1850, 2050)):
    outfile=f"hosing_grc01Sv_FcSV-LR_echam6_echammon_{year}.nc"
    if not os.path.isfile(outpath+outfile):
        cdo.copy(input = f"-selcode,167 {inpath}hosing_grc01Sv_FcSV-LR_echam6_echammon_{year}.grb", 
                 output=outpath+outfile, 
                 options = '-f nc -t echam6 ')
```

```python
# Load resulting file
ifiles  = "/scratch/m/m300817/tmp/hosing_grc01Sv_FcSV-LR_echam6_echammon_*.nc"
ds_tas_g01hosLR = xr.open_mfdataset(ifiles, use_cftime=True)
ds_tas_g01hosLR=ds_tas_g01hosLR.assign_coords(time=xr.cftime_range(start="1850", periods=2400, freq="M", calendar="proleptic_gregorian"))
```

### New hosing 1pct & scenarios


1pctbgc

```python
# grib to nc 
cdo.setCdo("/sw/spack-levante/cdo-1.9.10-j5frmz/bin/cdo")
for exp in ["hosing_naa01Sv_1pctbgc-LR","hosing_naa03Sv_1pctbgc-LR", "hosing_naa05Sv_1pctbgc-LR",
            "hosing_grc01Sv_1pctbgc-LR", "hosing_grc03Sv_1pctbgc-LR", "hosing_grc05Sv_1pctbgc-LR",
            "hosing_naa03Sv_1pct-LR", "hosing_naa05Sv_1pct-LR"]:
    inpath  = f"/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/{exp}/outdata/echam6/"
    outpath = "/scratch/m/m300817/tmp/"
    for year in range (1850, 1990):
        outfile=f"{exp}_echam6_echam_{year}.nc"
        if not os.path.isfile(outpath+outfile):
            cdo.copy(input = f"-selcode,167 {inpath}{exp}_echam6_echam_{year}.grb", 
                     output=outpath+outfile, 
                     options = '-f nc -t echam6 ')
```

```python
# Load resulting file
ds_tas_new = {}
for exp in ["hosing_naa01Sv_1pctbgc-LR","hosing_naa03Sv_1pctbgc-LR", "hosing_naa05Sv_1pctbgc-LR",
            "hosing_grc01Sv_1pctbgc-LR", "hosing_grc03Sv_1pctbgc-LR", "hosing_grc05Sv_1pctbgc-LR",
            "hosing_naa03Sv_1pct-LR", "hosing_naa05Sv_1pct-LR"]:
    ifiles  = f"/scratch/m/m300817/tmp/{exp}_echam6_echam_*.nc"
    ds_tas_new[exp] = xr.open_mfdataset(ifiles, use_cftime=True)
    ds_tas_new[exp] = ds_tas_new[exp].assign_coords(time=xr.cftime_range(start="1850", periods=1680, freq="M", calendar="proleptic_gregorian"))
```

hist-bgc and u03 bgc-scenarios

```python
cdo.setCdo("/sw/spack-levante/cdo-1.9.10-j5frmz/bin/cdo")
exp = "histbgc-LR"
inpath  = f"/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/{exp}/outdata/echam6/"
outpath = "/scratch/m/m300817/tmp/"
for year in range (1850, 2015):
    outfile=f"{exp}_echam6_echam_{year}.nc"
    if not os.path.isfile(outpath+outfile):
        cdo.copy(input = f"-selcode,167 {inpath}{exp}_echam6_echam_{year}.grb", 
                 output=outpath+outfile, 
                 options = '-f nc -t echam6 ')
for exp in ["hosing_naa03Sv_ssp126bgc-LR", "hosing_naa03Sv_ssp245bgc-LR", "hosing_naa03Sv_ssp585bgc-LR"]:
    inpath  = f"/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/{exp}/outdata/echam6/"
    outpath = "/scratch/m/m300817/tmp/"
    for year in range (2015, 2100):
        outfile=f"{exp}_echam6_echam_{year}.nc"
        if not os.path.isfile(outpath+outfile):
            cdo.copy(input = f"-selcode,167 {inpath}{exp}_echam6_echam_{year}.grb", 
                     output=outpath+outfile, 
                     options = '-f nc -t echam6 ')
```

```python
ds_tas_new_scen = {}

exp = "histbgc-LR"
ifiles  = f"/scratch/m/m300817/tmp/{exp}_echam6_echam_*.nc"
ds_tas_new_scen[exp] = xr.open_mfdataset(ifiles, use_cftime=True)
ds_tas_new_scen[exp] = ds_tas_new_scen[exp].assign_coords(time=xr.cftime_range(start="1850", periods=1980, freq="M", calendar="proleptic_gregorian"))

for exp in ["hosing_naa03Sv_ssp126bgc-LR", "hosing_naa03Sv_ssp245bgc-LR", "hosing_naa03Sv_ssp585bgc-LR"]:
    ifiles  = f"/scratch/m/m300817/tmp/{exp}_echam6_echam_*.nc"
    ds_tas_new_scen[exp] = xr.open_mfdataset(ifiles, use_cftime=True)
    ds_tas_new_scen[exp] = ds_tas_new_scen[exp].assign_coords(time=xr.cftime_range(start="2015", periods=1020, freq="M", calendar="proleptic_gregorian"))
```

ssp245bgc

```python
cdo.setCdo("/sw/spack-levante/cdo-1.9.10-j5frmz/bin/cdo")
for exp in ["hosing_naa01Sv_ssp245bgc-LR","hosing_naa03Sv_ssp245bgc-LR", "hosing_naa05Sv_ssp245bgc-LR",
            "hosing_grc01Sv_ssp245bgc-LR", "hosing_grc03Sv_ssp245bgc-LR", "hosing_grc05Sv_ssp245bgc-LR",
            "ssp245bgc-LR"]:
    inpath  = f"/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/{exp}/outdata/echam6/"
    outpath = "/scratch/m/m300817/tmp/"
    for year in range (2015, 2100):
        outfile=f"{exp}_echam6_echam_{year}.nc"
        if not os.path.isfile(outpath+outfile):
            cdo.copy(input = f"-selcode,167 {inpath}{exp}_echam6_echam_{year}.grb", 
                     output=outpath+outfile, 
                     options = '-f nc -t echam6 ')
```

```python
ds_tas_ssp245 = {}
for exp in ["hosing_naa01Sv_ssp245bgc-LR","hosing_naa03Sv_ssp245bgc-LR", "hosing_naa05Sv_ssp245bgc-LR",
            "hosing_grc01Sv_ssp245bgc-LR", "hosing_grc03Sv_ssp245bgc-LR", "hosing_grc05Sv_ssp245bgc-LR",
            "ssp245bgc-LR"]:
    ifiles  = f"/scratch/m/m300817/tmp/{exp}_echam6_echam_*.nc"
    ds_tas_ssp245[exp] = xr.open_mfdataset(ifiles, use_cftime=True)
    ds_tas_ssp245[exp] = ds_tas_ssp245[exp].assign_coords(time=xr.cftime_range(start="2015", periods=1020, freq="M", calendar="proleptic_gregorian"))
```

<!-- #region toc-hr-collapsed=true -->
## AMOC (msftmz)
<!-- #endregion -->

### 1pctCO2

```python
# Load amoc in 1pctCO2-bgc
file_type = 'msftmz'
infiles = glob.glob(f'/pool/data/CMIP6/data/C4MIP/MPI-M/MPI-ESM1-2-LR/1pctCO2-bgc/r1i1p1f1/Omon/{file_type}/gn/v20190710/*{file_type}*.nc')
ds_msftmz_1pctbgc = xr.open_mfdataset(infiles, use_cftime=True, parallel=True)
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### Original hosing 
<!-- #endregion -->

```python
# Load directly for u03/05, the pre-processing was already done in a different script
ds_msftmz_u03hosLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/msftmz/u03-hos-LR_msftmz.nc", use_cftime=True, parallel=True)
ds_msftmz_u05hosLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/msftmz/u05-hos-LR_msftmz.nc", use_cftime=True, parallel=True)
```

<!-- #raw -->
# For grc-03
# NO NEED TO RE-RUN --- Pre-process (select and merge) msftmz monthly
in_path  = '/work/mh0033/from_Mistral/mh0033/m300817/mpiesm-1.2.01p6-passivesalt_update/experiments/hosing_grc03Sv_FcSV-LR/outdata/mpiom/'
wildcard = "*mpiom_data_moc_mm_*"
ifiles   = in_path+wildcard
outpath = "/work/uo1075/m300817/hosing/post/data/msftmz/"
outfile = "hosing_grc03Sv_FcSV-LR_msftmz.nc"
if not glob.glob(outpath):
    os.mkdir(outpath)
if not os.path.isfile(outpath+outfile):
    cdo.mergetime(input = "-select,name=atlantic_moc "+ifiles, output = outpath+outfile)
<!-- #endraw -->

```python
ds_msftmz_g03hosLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/msftmz/hosing_grc03Sv_FcSV-LR_msftmz.nc", use_cftime=True, parallel=True)
```

<!-- #raw -->
# For grc-01
# NO NEED TO RE-RUN --- Pre-process (select and merge) msftmz monthly
in_path  = '/work/uo1075/m300817/hosing/mpiesm-1.2.01p6-passivesalt_update/experiments/hosing_grc01Sv_FcSV-LR/outdata/mpiom/'
wildcard = "*mpiom_data_moc_mm_*"
ifiles   = in_path+wildcard
outpath = "/work/uo1075/m300817/hosing/post/data/msftmz/"
outfile = "hosing_grc01Sv_FcSV-LR_msftmz.nc"
if not glob.glob(outpath):
    os.mkdir(outpath)
if not os.path.isfile(outpath+outfile):
    cdo.mergetime(input = "-select,name=atlantic_moc "+ifiles, output = outpath+outfile)
<!-- #endraw -->

```python
ds_msftmz_g01hosLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/msftmz/hosing_grc01Sv_FcSV-LR_msftmz.nc", use_cftime=True, parallel=True)
```

### New hosing 1pct & scenario


1pct-bgc

```python
ds_msftmz_u011pctbgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/hosing_naa01Sv_1pctbgc-LR/outdata/mpiom/*moc_mm*",
                                       use_cftime=True)
ds_msftmz_u031pctbgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/hosing_naa03Sv_1pctbgc-LR/outdata/mpiom/*moc_mm*",
                                       use_cftime=True)
ds_msftmz_u031pctLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/hosing_naa03Sv_1pct-LR/outdata/mpiom/*moc_mm*",
                                       use_cftime=True)
ds_msftmz_u051pctbgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/hosing_naa05Sv_1pctbgc-LR/outdata/mpiom/*moc_mm*",
                                       use_cftime=True)
ds_msftmz_u051pctLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/hosing_naa05Sv_1pct-LR/outdata/mpiom/*moc_mm*",
                                       use_cftime=True)
ds_msftmz_g011pctbgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/hosing_grc01Sv_1pctbgc-LR/outdata/mpiom/*moc_mm*",
                                       use_cftime=True)
ds_msftmz_g031pctbgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/hosing_grc03Sv_1pctbgc-LR/outdata/mpiom/*moc_mm*",
                                       use_cftime=True)
ds_msftmz_g051pctbgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/hosing_grc05Sv_1pctbgc-LR/outdata/mpiom/*moc_mm*",
                                       use_cftime=True)
```

hist-bgc and u03 bgc-scenarios

```python
ds_msftmz_histbgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/histbgc-LR/outdata/mpiom/*moc_mm*",
                                        use_cftime=True)
ds_msftmz_u03ssp126bgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/hosing_naa03Sv_ssp126bgc-LR/outdata/mpiom/*moc_mm*",
                                             use_cftime=True)
ds_msftmz_u03ssp245bgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/hosing_naa03Sv_ssp245bgc-LR/outdata/mpiom/*moc_mm*",
                                             use_cftime=True)
ds_msftmz_u03ssp585bgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/hosing_naa03Sv_ssp585bgc-LR/outdata/mpiom/*moc_mm*", 
                                             use_cftime=True)
```

ssp245bgc

```python
ds_msftmz_ssp245bgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/ssp245bgc-LR/outdata/mpiom/*moc_mm*",
                                        use_cftime=True)
ds_msftmz_u05ssp245bgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/hosing_naa05Sv_ssp245bgc-LR/outdata/mpiom/*moc_mm*",
                                             use_cftime=True)
ds_msftmz_u03ssp245bgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/hosing_naa03Sv_ssp245bgc-LR/outdata/mpiom/*moc_mm*",
                                             use_cftime=True)
ds_msftmz_u01ssp245bgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/hosing_naa01Sv_ssp245bgc-LR/outdata/mpiom/*moc_mm*",
                                             use_cftime=True)
ds_msftmz_g05ssp245bgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/hosing_grc05Sv_ssp245bgc-LR/outdata/mpiom/*moc_mm*",
                                             use_cftime=True)
ds_msftmz_g03ssp245bgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/hosing_grc03Sv_ssp245bgc-LR/outdata/mpiom/*moc_mm*",
                                             use_cftime=True)
ds_msftmz_g01ssp245bgcLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/hosing_grc01Sv_ssp245bgc-LR/outdata/mpiom/*moc_mm*",
                                             use_cftime=True)
```

<!-- #region toc-hr-collapsed=true -->
## fx files
<!-- #endregion -->

### mpi-esm output

<!-- #raw -->
# NO NEED TO RE-RUN --- Pre-process volcello mpi output
in_path  = '/work/mh0033/from_Mistral/mh0033/m300817/mpiesm-1.2.01p6-passivesalt_update/experiments/hosing_naa05Sv_FcSV-LR/outdata/mpiom/'
filefx = "hosing_naa05Sv_FcSV-LR_mpiom_fx_18500101_18501231.nc
ifiles   = in_path+filefx
outpath = "/work/uo0122/m300817/hosing/post/data/fx/"
outfile = "hosing_naa05Sv_FcSV-LR_volcello.nc"
if not glob.glob(outpath):
    os.mkdir(outpath)
if not os.path.isfile(outpath+outfile):
    #cdo(input = "expr,’volcello=areacello*thkcello’ "+ifiles, output = outpath+outfile)
    cdo(input = " -expr,'volcello=areacello*thkcello' "+ifiles, output = outpath+outfile)
<!-- #endraw -->

```python
# Load resulting file, same for both hosing configurations (note: now in new location)
ds_volcello_u05hosLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/data/fx/hosing_naa05Sv_FcSV-LR_volcello.nc"
                                       , use_cftime=True, parallel=True)
```

```python
ds_volcello_u05hosLR = ds_volcello_u05hosLR.rename({'x_2':'x', 'y_2':'y', 'lat_2': 'lat','lon_2': 'lon', 'depth_2': 'depth'})
```

```python
# Files for ocean/land mask LR & HR
ds_basin_hosLR = xr.open_mfdataset("/pool/data/MPIOM/GR15/GR15_basin.nc", use_cftime=True, parallel=True)
ds_basin_hosHR = xr.open_mfdataset("/pool/data/MPIOM/TP04/TP04_basin.nc", use_cftime=True, parallel=True)
```

### cmor

```python
infiles = glob.glob('/pool/data/CMIP6/data/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Ofx/volcello/gn/v20190710/*.nc')
ds_volcello = xr.open_mfdataset(infiles, use_cftime=True, parallel=True)

infiles = glob.glob('/pool/data/CMIP6/data/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Ofx/areacello/gn/v20190710/*.nc')
ds_areacello = xr.open_mfdataset(infiles, use_cftime=True, parallel=True)
```

```python
# Files for ocean/land mask LR 
infiles = glob.glob('/pool/data/CMIP6/data/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Ofx/basin/gn/v20190710/*.nc')
ds_basin = xr.open_mfdataset(infiles, use_cftime=True, parallel=True)
```

# Analysis

<!-- #region toc-hr-collapsed=true -->
## Original hosing
<!-- #endregion -->

### u03

```python jupyter={"outputs_hidden": true}
ds_dissic_u03hosLR.load()
```

```python
# Total DIC full depth (0-5720m) u03hosLR last vs first 10 years
u03_total_full = ((ds_dissic_u03hosLR.dissic.isel(time=np.arange(-120,-0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u03hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='y').sum(dim='x').values
```

```python
# Total DIC below 1000m (1085-5720m) u03hosLR last vs first 10 years
u03_total_deep =  ((ds_dissic_u03hosLR.dissic.isel(time=np.arange(-120,-0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(23,40)).sum(dim='depth')
-(ds_dissic_u03hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(23,40)).sum(dim='depth')
).sum(dim='y').sum(dim='x').values
```

```python
# Total DIC above 1000m (0-1085m) u03hosLR last vs first 10 years
u03_total_surf = ((ds_dissic_u03hosLR.dissic.isel(time=np.arange(-120,-0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,23)).sum(dim='depth')
-(ds_dissic_u03hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,23)).sum(dim='depth')
).sum(dim='y').sum(dim='x').values
```

```python
fig=plt.figure(figsize=(14, 9))

ax1 = fig.add_subplot(211, projection=ccrs.Robinson(central_longitude=-60))
ax1.set_global()
((ds_dissic_u03hosLR.dissic.isel(time=np.arange(-120,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u03hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(), x="lon", y="lat", vmin=-0.040, vmax=0.040, cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'})
ax1.coastlines()
ax1.set_title("Full depth (0-5720m)") 
ax1.text(-0.08, 1, str(round(u03_total_full.item(), 3))+" PgC", transform=ax1.transAxes, fontsize=14,
        verticalalignment='top' , fontweight='bold')

ax2 = fig.add_subplot(223, projection=ccrs.Robinson(central_longitude=-60))
ax2.set_global()
((ds_dissic_u03hosLR.dissic.isel(time=np.arange(-120,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,23)).sum(dim='depth')
-(ds_dissic_u03hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,23)).sum(dim='depth')
).plot.pcolormesh(
    ax=ax2, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', vmin=-0.040, vmax=0.040, add_colorbar=False)
ax2.coastlines()
ax2.set_title("Upper ocean (0-1085m)") 
ax2.text(-0.08, 1, str(round(u03_total_surf.item(), 3))+" PgC", transform=ax2.transAxes, fontsize=14,
        verticalalignment='top', fontweight='bold')

ax3 = fig.add_subplot(224, projection=ccrs.Robinson(central_longitude=-60))
ax3.set_global()
((ds_dissic_u03hosLR.dissic.isel(time=np.arange(-120,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(23,40)).sum(dim='depth')
-(ds_dissic_u03hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(23,40)).sum(dim='depth')
).plot.pcolormesh(
   ax=ax3, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', vmin=-0.040, vmax=0.040, add_colorbar=False)
ax3.coastlines()
ax3.set_title("Deep ocean (1085-5720m)") 

ax3.text(-0.08, 1, str(round(u03_total_deep.item(), 3))+" PgC", transform=ax3.transAxes, fontsize=14,
        verticalalignment='top', fontweight='bold')
```

### u05

```python jupyter={"outputs_hidden": true}
ds_dissic_u05hosLR.load()
```

```python
# Total DIC full depth (0-5720m) u05hosLR last vs first 10 years
u05_total_full = ((ds_dissic_u05hosLR.dissic.isel(time=np.arange(-120,-0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u05hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='y').sum(dim='x').values
```

```python
# Total DIC below 1000m (1085-5720m) u05hosLR last vs first 10 years
u05_total_deep = ((ds_dissic_u05hosLR.dissic.isel(time=np.arange(-120,-0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(23,40)).sum(dim='depth')
-(ds_dissic_u05hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(23,40)).sum(dim='depth')
).sum(dim='y').sum(dim='x').values
```

```python
# Total DIC above 1000m (0-1085m) u05hosLR last vs first 10 years
u05_total_surf = ((ds_dissic_u05hosLR.dissic.isel(time=np.arange(-120,-0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,23)).sum(dim='depth')
-(ds_dissic_u05hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,23)).sum(dim='depth')
).sum(dim='y').sum(dim='x').values
```

```python
fig=plt.figure(figsize=(14, 9))

ax1 = fig.add_subplot(211, projection=ccrs.Robinson(central_longitude=-60))
ax1.set_global()
((ds_dissic_u05hosLR.dissic.isel(time=np.arange(-120,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u05hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(), x="lon", y="lat", vmin=-0.040, vmax=0.040, cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'})
ax1.coastlines()
ax1.set_title("Full depth (0-5720m)") 
ax1.text(-0.08, 1.1, str(round(u05_total_full.item(), 3))+" PgC", transform=ax1.transAxes, fontsize=14,
        verticalalignment='top', fontweight='bold')

ax2 = fig.add_subplot(223, projection=ccrs.Robinson(central_longitude=-60))
ax2.set_global()
((ds_dissic_u05hosLR.dissic.isel(time=np.arange(-120,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,23)).sum(dim='depth')
-(ds_dissic_u05hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,23)).sum(dim='depth')
).plot.pcolormesh(
    ax=ax2, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', vmin=-0.040, vmax=0.040, add_colorbar=False)
ax2.coastlines()
ax2.set_title("Upper ocean (0-1085m)") 
ax2.text(-0.08, 1.1, str(round(u05_total_surf.item(), 3))+" PgC", transform=ax2.transAxes, fontsize=14,
        verticalalignment='top', fontweight='bold')

ax3 = fig.add_subplot(224, projection=ccrs.Robinson(central_longitude=-60))
ax3.set_global()
((ds_dissic_u05hosLR.dissic.isel(time=np.arange(-120,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(23,40)).sum(dim='depth')
-(ds_dissic_u05hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(23,40)).sum(dim='depth')
).plot.pcolormesh(
   ax=ax3, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', vmin=-0.040, vmax=0.040, add_colorbar=False)
ax3.coastlines()
ax3.set_title("Deep ocean (1085-5720m)") 
ax3.text(-0.08, 1.1, str(round(u05_total_deep.item(), 3))+" PgC", transform=ax3.transAxes, fontsize=14,
        verticalalignment='top', fontweight='bold')
```

### g03

```python jupyter={"outputs_hidden": true}
ds_dissic_g03hosLR.load()
```

```python
# Total DIC full depth (0-5720m) u05hosLR last vs first 10 years
g03_total_full = ((ds_dissic_g03hosLR.dissic.isel(time=np.arange(-120,-0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g03hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='y').sum(dim='x').values
```

```python
# Total DIC below 1000m (1085-5720m) u05hosLR last vs first 10 years
g03_total_deep = ((ds_dissic_g03hosLR.dissic.isel(time=np.arange(-120,-0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(23,40)).sum(dim='depth')
-(ds_dissic_g03hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(23,40)).sum(dim='depth')
).sum(dim='y').sum(dim='x').values
```

```python
# Total DIC above 1000m (0-1085m) u05hosLR last vs first 10 years
g03_total_surf = ((ds_dissic_g03hosLR.dissic.isel(time=np.arange(-120,-0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,23)).sum(dim='depth')
-(ds_dissic_g03hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,23)).sum(dim='depth')
).sum(dim='y').sum(dim='x').values
```

```python
fig=plt.figure(figsize=(14, 9))

ax1 = fig.add_subplot(211, projection=ccrs.Robinson(central_longitude=-60))
ax1.set_global()
((ds_dissic_g03hosLR.dissic.isel(time=np.arange(-120,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g03hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(), x="lon", y="lat", vmin=-0.040, vmax=0.040, cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'})
ax1.coastlines()
ax1.set_title("Full depth (0-5720m)") 
ax1.text(-0.08, 1, str(round(g03_total_full.item(), 3))+" PgC", transform=ax1.transAxes, fontsize=14,
        verticalalignment='top', fontweight='bold')

ax2 = fig.add_subplot(223, projection=ccrs.Robinson(central_longitude=-60))
ax2.set_global()
((ds_dissic_g03hosLR.dissic.isel(time=np.arange(-120,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,23)).sum(dim='depth')
-(ds_dissic_g03hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,23)).sum(dim='depth')
).plot.pcolormesh(
    ax=ax2, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', vmin=-0.040, vmax=0.040, add_colorbar=False)
ax2.coastlines()
ax2.set_title("Upper ocean (0-1085m)") 
ax2.text(-0.08, 1, str(round(g03_total_surf.item(), 3))+" PgC", transform=ax2.transAxes, fontsize=14,
        verticalalignment='top', fontweight='bold')


ax3 = fig.add_subplot(224, projection=ccrs.Robinson(central_longitude=-60))
ax3.set_global()
((ds_dissic_g03hosLR.dissic.isel(time=np.arange(-120,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(23,40)).sum(dim='depth')
-(ds_dissic_g03hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(23,40)).sum(dim='depth')
).plot.pcolormesh(
   ax=ax3, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', vmin=-0.040, vmax=0.040, add_colorbar=False)
ax3.coastlines()
ax3.set_title("Deep ocean (1085-5720m)")
ax3.text(-0.08, 1, str(round(g03_total_deep.item(), 3))+" PgC", transform=ax3.transAxes, fontsize=14,
        verticalalignment='top', fontweight='bold')
```

### g01

```python jupyter={"outputs_hidden": true}
ds_dissic_g01hosLR.load()
```

```python
# Total DIC full depth (0-5720m) u05hosLR last vs first 10 years
g01_total_full = ((ds_dissic_g01hosLR.dissic.isel(time=np.arange(-10,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g01hosLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='y').sum(dim='x').values
```

```python
# Total DIC below 1000m (1085-5720m) u05hosLR last vs first 10 years
g01_total_deep = ((ds_dissic_g01hosLR.dissic.isel(time=np.arange(-10,-0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(23,40)).sum(dim='depth')
-(ds_dissic_g01hosLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(23,40)).sum(dim='depth')
).sum(dim='y').sum(dim='x').values
```

```python
# Total DIC above 1000m (0-1085m) u05hosLR last vs first 10 years
g01_total_surf = ((ds_dissic_g01hosLR.dissic.isel(time=np.arange(-10,-0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,23)).sum(dim='depth')
-(ds_dissic_g01hosLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,23)).sum(dim='depth')
).sum(dim='y').sum(dim='x').values
```

```python
fig=plt.figure(figsize=(14, 9))

ax1 = fig.add_subplot(211, projection=ccrs.Robinson(central_longitude=-60))
ax1.set_global()
((ds_dissic_g01hosLR.dissic.isel(time=np.arange(-10,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g01hosLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(), x="lon", y="lat", vmin=-0.040, vmax=0.040, cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'})
ax1.coastlines()
ax1.set_title("Full depth (0-5720m)") 
ax1.text(-0.08, 1, str(round(g01_total_full.item(), 3))+" PgC", transform=ax1.transAxes, fontsize=14,
        verticalalignment='top', fontweight='bold')

ax2 = fig.add_subplot(223, projection=ccrs.Robinson(central_longitude=-60))
ax2.set_global()
((ds_dissic_g01hosLR.dissic.isel(time=np.arange(-10,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,23)).sum(dim='depth')
-(ds_dissic_g01hosLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,23)).sum(dim='depth')
).plot.pcolormesh(
    ax=ax2, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', vmin=-0.040, vmax=0.040, add_colorbar=False)
ax2.coastlines()
ax2.set_title("Upper ocean (0-1085m)") 
ax2.text(-0.08, 1, str(round(g01_total_surf.item(), 3))+" PgC", transform=ax2.transAxes, fontsize=14,
        verticalalignment='top', fontweight='bold')

ax3 = fig.add_subplot(224, projection=ccrs.Robinson(central_longitude=-60))
ax3.set_global()
((ds_dissic_g01hosLR.dissic.isel(time=np.arange(-10,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(23,40)).sum(dim='depth')
-(ds_dissic_g01hosLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(23,40)).sum(dim='depth')
).plot.pcolormesh(
   ax=ax3, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', vmin=-0.040, vmax=0.040, add_colorbar=False)
ax3.coastlines()
ax3.set_title("Deep ocean (1085-5720m)")
ax3.text(-0.08, 1, str(round(g01_total_deep.item(), 3))+" PgC", transform=ax3.transAxes, fontsize=14,
        verticalalignment='top', fontweight='bold')
```

### all - full depth

```python
fig=plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(221, projection=ccrs.Robinson(central_longitude=-60))
ax1.set_global()
((ds_dissic_u03hosLR.dissic.isel(time=np.arange(-120,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u03hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(), x="lon", y="lat", vmin=-0.040, vmax=0.040, cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'})
ax1.coastlines()
ax1.set_title("u03")
ax1.text(-0.1, 1.1, str(round(u03_total_full.item(), 3))+" PgC", transform=ax1.transAxes, fontsize=14,
        verticalalignment='top', fontweight='bold')

ax2 = fig.add_subplot(222, projection=ccrs.Robinson(central_longitude=-60))
ax2.set_global()
((ds_dissic_u05hosLR.dissic.isel(time=np.arange(-120,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u05hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).plot.pcolormesh(
    ax=ax2, transform=ccrs.PlateCarree(), x="lon", y="lat", vmin=-0.040, vmax=0.040, cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'})
ax2.coastlines()
ax2.set_title("u05")
ax2.text(-0.1, 1.1, str(round(u05_total_full.item(), 3))+" PgC", transform=ax2.transAxes, fontsize=14,
        verticalalignment='top', fontweight='bold')


ax3 = fig.add_subplot(223, projection=ccrs.Robinson(central_longitude=-60))
ax3.set_global()
((ds_dissic_g03hosLR.dissic.isel(time=np.arange(-120,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g03hosLR.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).plot.pcolormesh(
    ax=ax3, transform=ccrs.PlateCarree(), x="lon", y="lat", vmin=-0.040, vmax=0.040, cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'})
ax3.coastlines()
ax3.set_title("g03")
ax3.text(-0.1, 1.1, str(round(g03_total_full.item(), 3))+" PgC", transform=ax3.transAxes, fontsize=14,
        verticalalignment='top', fontweight='bold')


ax4 = fig.add_subplot(224, projection=ccrs.Robinson(central_longitude=-60))
ax4.set_global()
((ds_dissic_g01hosLR.dissic.isel(time=np.arange(-10,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g01hosLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).plot.pcolormesh(
    ax=ax4, transform=ccrs.PlateCarree(), x="lon", y="lat", vmin=-0.040, vmax=0.040, cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'})
ax4.coastlines()
ax4.set_title("g01")
ax4.text(-0.1, 1.1, str(round(g01_total_full.item(), 3))+" PgC", transform=ax4.transAxes, fontsize=14,
        verticalalignment='top', fontweight='bold')
```

### Time series

```python
diff_dissic_u05 = []
for year in range(1,200):
    diff_dissic_u05.append(
        ((ds_dissic_u05hosLR.dissic.isel(time=np.arange(year*12,year*12+12)).mean(dim="time")*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        -(ds_dissic_u05hosLR.dissic.isel(time=np.arange(0,12)).mean(dim="time")*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        ).sum(dim='y').sum(dim='x').values)
diff_dissic_u03 = []
for year in range(1,200):
    diff_dissic_u03.append(
        ((ds_dissic_u03hosLR.dissic.isel(time=np.arange(year*12,year*12+12)).mean(dim="time")*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        -(ds_dissic_u03hosLR.dissic.isel(time=np.arange(0,12)).mean(dim="time")*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        ).sum(dim='y').sum(dim='x').values)    
diff_dissic_g03 = []
for year in range(1,200):
    diff_dissic_g03.append(
        ((ds_dissic_g03hosLR.dissic.isel(time=np.arange(year*12,year*12+12)).mean(dim="time")*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        -(ds_dissic_g03hosLR.dissic.isel(time=np.arange(0,12)).mean(dim="time")*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        ).sum(dim='y').sum(dim='x').values)        
diff_dissic_g01 = []
for year in range(1,200):
    diff_dissic_g01.append(
        ((ds_dissic_g01hosLR.dissic.isel(time=year)*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        -(ds_dissic_g01hosLR.dissic.isel(time=0)*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        ).sum(dim='y').sum(dim='x').values)  
```

```python
np.save('diff_dissic_u03.npy', diff_dissic_u03)  
np.save('diff_dissic_u05.npy', diff_dissic_u05)  
np.save('diff_dissic_g03.npy', diff_dissic_g03)  
np.save('diff_dissic_g01.npy', diff_dissic_g01)  
```

```python
diff_dissic_u03 = np.load('diff_dissic_u03.npy')
diff_dissic_u05 = np.load('diff_dissic_u05.npy')
diff_dissic_g03 = np.load('diff_dissic_g03.npy')
diff_dissic_g01 = np.load('diff_dissic_g01.npy')
```

```python
fig, ax1 = plt.subplots()

ax1.plot(np.arange(1851, 2050), diff_dissic_u03, label='u03-LR', color='orange')
ax1.plot(np.arange(1851, 2050), diff_dissic_u05, label ='u05-LR', color='red')
ax1.plot(np.arange(1851, 2050), diff_dissic_g03, label ='g03-LR', color='darkblue')
ax1.plot(np.arange(1851, 2050), diff_dissic_g01, label ='g01-LR', color='lightblue')
ax1.legend()
ax1.set_xlabel("Time [year]")
ax1.set_ylabel('Difference Ocean Carbon [PgC]', color='darkgreen')
ax1.tick_params(axis='y', labelcolor="darkgreen")
ax1.spines['top'].set_visible(False)

ax2 = ax1.twinx()
(ds_msftmz_u03hosLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=ax2, color='orange')
(ds_msftmz_u05hosLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=ax2, color='red')
(ds_msftmz_g03hosLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=ax2, color='darkblue')
(ds_msftmz_g01hosLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=ax2, color='lightblue')
ax2.tick_params(axis='y', labelcolor='k')
ax2.set_ylabel("AMOC strength [Sv]")
ax2.spines['top'].set_visible(False)

#plt.legend(['u03','u05','g03','g01'])
fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()
```

```python
fig, axs = plt.subplots(2, 1, figsize=(7,7), sharex=True)
fig.subplots_adjust(hspace=0)

axs[0].plot(np.arange(1851, 2050), diff_dissic_u03, label='u03-LR', color='orange')
axs[0].plot(np.arange(1851, 2050), diff_dissic_u05, label ='u05-LR', color='red')
axs[0].plot(np.arange(1851, 2050), diff_dissic_g03, label ='g03-LR', color='darkblue')
axs[0].plot(np.arange(1851, 2050), diff_dissic_g01, label ='g01-LR', color='lightblue')
axs[0].legend(loc='lower center')
axs[0].set_xlabel("")
axs[0].set_ylabel('Difference Ocean Carbon [PgC]', color='darkgreen')
axs[0].tick_params(axis='y', labelcolor="darkgreen")
axs[0].spines['bottom'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].set_xticks([])
axs[0].get_xaxis().set_visible(False)

(ds_msftmz_u03hosLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='orange')
(ds_msftmz_u05hosLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='red')
(ds_msftmz_g03hosLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='darkblue')
(ds_msftmz_g01hosLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='lightblue')
axs[1].tick_params(axis='y', labelcolor='darkblue')
axs[1].set_ylabel("AMOC strength [Sv]")
axs[1].spines['left'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].set_xlabel("Time [year]")
axs[1].yaxis.set_label_position("right")
axs[1].yaxis.tick_right()
axs[1].set_title("")
axs[1].set_xticks([1850, 1875, 1900, 1925, 1950, 1975, 2000, 2025, 2050], [0, 25, 50, 75, 100, 125, 150, 175, 200])
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
```

```python
amoc_u03=(ds_msftmz_u03hosLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9)
amoc_u05=(ds_msftmz_u05hosLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9)
amoc_g03=(ds_msftmz_g03hosLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9)
amoc_g01=(ds_msftmz_g01hosLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9)
```

```python
plt.plot(amoc_u03[1:], diff_dissic_u03, label='u03-LR')
plt.plot(amoc_u05[1:], diff_dissic_u05, label='u05-LR')
plt.plot(amoc_g03[1:], diff_dissic_g03, label='g03-LR')
plt.plot(amoc_g01[1:], diff_dissic_g01, label='g01-LR')
plt.xlabel('AMOC [Sv]')
plt.ylabel('Difference Ocean Carbon [PgC]')
plt.legend()
plt.savefig('/home/m/m300817/carbon_amoc/plots/dissic_vs_moc_hosing.jpg', bbox_inches='tight')
```

### Temperature

```python
diff_tas_u05 = []
for year in range(1,200):
    diff_tas_u05.append(
        ((ds_tas_u05hosLR.tas.isel(time=np.arange(year*12,year*12+12)).mean(dim="time")
        -(ds_tas_u05hosLR.tas.isel(time=np.arange(0,12)).mean(dim="time"))
        ).mean(dim='lat').mean(dim='lon').values))
diff_tas_u03 = []
for year in range(1,200):
    diff_tas_u03.append(
        ((ds_tas_u03hosLR.tas.isel(time=np.arange(year*12,year*12+12)).mean(dim="time")
        -(ds_tas_u03hosLR.tas.isel(time=np.arange(0,12)).mean(dim="time"))
        ).mean(dim='lat').mean(dim='lon').values))
diff_tas_g03 = []
for year in range(1,200):
    diff_tas_g03.append(
        ((ds_tas_g03hosLR.temp2.isel(time=np.arange(year*12,year*12+12)).mean(dim="time")
        -(ds_tas_g03hosLR.temp2.isel(time=np.arange(0,12)).mean(dim="time"))
        ).mean(dim='lat').mean(dim='lon').values))       
diff_tas_g01 = []
for year in range(1,200):
    diff_tas_g01.append(
        ((ds_tas_g01hosLR.temp2.isel(time=np.arange(year*12,year*12+12)).mean(dim="time")
        -(ds_tas_g01hosLR.temp2.isel(time=np.arange(0,12)).mean(dim="time"))
        ).mean(dim='lat').mean(dim='lon').values))     
```

```python
ds_tas_u03hosLR.groupby('time.year').mean('time').mean(dim='lat').mean(dim='lon').tas.plot(label='u03-LR')
ds_tas_u05hosLR.groupby('time.year').mean('time').mean(dim='lat').mean(dim='lon').tas.plot(label='u05-LR')
ds_tas_g03hosLR.groupby('time.year').mean('time').mean(dim='lat').mean(dim='lon').temp2.plot(label='g03-LR')
ds_tas_g01hosLR.groupby('time.year').mean('time').mean(dim='lat').mean(dim='lon').temp2.plot(label='g01-LR')
```

```python
plt.scatter(amoc_u03[1:], diff_tas_u03, label='u03-LR')
plt.scatter(amoc_u05[1:], diff_tas_u05, label='u05-LR')
plt.scatter(amoc_g03[1:], diff_tas_g03, label='g03-LR')
plt.scatter(amoc_g01[1:], diff_tas_g01, label='g01-LR')
plt.xlabel('AMOC [Sv]')
plt.ylabel('$\Delta$T (°C)')
plt.legend()
```

```python
fig=plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(221, projection=ccrs.Robinson(central_longitude=-60))
ax1.set_global()
(ds_tas_u03hosLR.tas.isel(time=np.arange(-120,0)).mean(dim="time")
-ds_tas_u03hosLR.tas.isel(time=np.arange(0,120)).mean(dim="time")
).plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', cbar_kwargs={'label': '$\Delta$T (°C)'})
ax1.coastlines()
ax1.set_title("u03")

ax2 = fig.add_subplot(222, projection=ccrs.Robinson(central_longitude=-60))
ax2.set_global()
(ds_tas_u05hosLR.tas.isel(time=np.arange(-120,0)).mean(dim="time")
-ds_tas_u05hosLR.tas.isel(time=np.arange(0,120)).mean(dim="time")
).plot.pcolormesh(
    ax=ax2, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', cbar_kwargs={'label': '$\Delta$T (°C)'})
ax2.coastlines()
ax2.set_title("u05")


ax3 = fig.add_subplot(223, projection=ccrs.Robinson(central_longitude=-60))
ax3.set_global()
(ds_tas_g03hosLR.temp2.isel(time=np.arange(-120,0)).mean(dim="time")
-ds_tas_g03hosLR.temp2.isel(time=np.arange(0,120)).mean(dim="time")
).plot.pcolormesh(
    ax=ax3, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', cbar_kwargs={'label': '$\Delta$T (°C)'})
ax3.coastlines()
ax3.set_title("g03")


ax4 = fig.add_subplot(224, projection=ccrs.Robinson(central_longitude=-60))
ax4.set_global()
(ds_tas_g01hosLR.temp2.isel(time=np.arange(-120,0)).mean(dim="time")
-ds_tas_g01hosLR.temp2.isel(time=np.arange(0,120)).mean(dim="time")
).plot.pcolormesh(
    ax=ax4, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', cbar_kwargs={'label': '$\Delta$T (°C)'})
ax4.coastlines()
ax4.set_title("g01")
```

```python
fig=plt.figure(figsize=(14, 9))

ax1 = fig.add_subplot(211, projection=ccrs.Robinson(central_longitude=-60))
ax1.set_global()
((ds_tas_u03hosLR.groupby('time.year').mean('time').isel(year=np.arange(90,100)).mean(dim='year').tas)
-(ds_tas_u03hosLR.groupby('time.year').mean('time').isel(year=np.arange(0,10)).mean(dim='year').tas)
).plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', cbar_kwargs={'label': 'K'})
ax1.coastlines()
ax1.set_title("Anomaly after 100 yrs wrt year 0") 

ax1 = fig.add_subplot(212, projection=ccrs.Robinson(central_longitude=-60))
ax1.set_global()
((ds_tas_u05hosLR.groupby('time.year').mean('time').isel(year=np.arange(-10,0)).mean(dim='year').tas)
-(ds_tas_u05hosLR.groupby('time.year').mean('time').isel(year=np.arange(100,110)).mean(dim='year').tas)
).plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', cbar_kwargs={'label': 'K'})
ax1.coastlines()
ax1.set_title("Anomaly after 200 yrs wrt year 100") 
```

GIF

```python
u05_amoc=(ds_msftmz_u05hosLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).load()
```

```python
for i in range(1,20):
    fig=plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111, projection=ccrs.Robinson(central_longitude=-60))
    ax1.set_global()
    ((ds_tas_u03hosLR.groupby('time.year').mean('time').isel(year=np.arange(i*10,(i+1)*10)).mean(dim='year').tas)
    -(ds_tas_u03hosLR.groupby('time.year').mean('time').isel(year=np.arange(0,10)).mean(dim='year').tas)
    ).plot.pcolormesh(
        ax=ax1, transform=ccrs.PlateCarree(), x="lon", y="lat", vmin=-3.5, vmax=3.5, cmap='RdBu_r', cbar_kwargs={'label': 'K', 'extend': 'both'})
    ax1.coastlines()
    ax1.set_title(f"SAT anomaly after {(i+1)*10} years", fontsize =16) 
    ax1.text(0, 1.2, str(round(u05_amoc.isel(year=(i+1)*10-1).values.item(), 1))+" Sv", transform=ax1.transAxes, fontsize=20,
        verticalalignment='top', fontweight='bold')
    plt.savefig(f'/work/uo1075/m300817/carbon_amoc/plots/Tdiff_u05_y{(i+1)*10}.png')
    plt.close()
```

```python
fig=plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(111, projection=ccrs.Robinson(central_longitude=-60))
ax1.set_global()
((ds_tas_u03hosLR.groupby('time.year').mean('time').isel(year=np.arange(19*10,(19+1)*10)).mean(dim='year').tas)
-(ds_tas_u03hosLR.groupby('time.year').mean('time').isel(year=np.arange(0,10)).mean(dim='year').tas)
).plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(), x="lon", y="lat", vmin=-3.5, vmax=3.5, cmap='RdBu_r', cbar_kwargs={'label': 'K'})
ax1.coastlines()
ax1.set_title(f"SAT anomaly after {(19+1)*10} years", fontsize =16) 
ax1.text(0, 1.2, str(round(u05_amoc.isel(year=(19+1)*10-1).values.item(), 1))+" Sv", transform=ax1.transAxes, fontsize=20,
         verticalalignment='top', fontweight='bold')
```

```python
import imageio
images = []
for i in range(1,20):
    images.append(imageio.imread(f'/work/uo1075/m300817/carbon_amoc/plots/Tdiff_u05_y{(i+1)*10}.png'))
imageio.mimsave('/work/uo1075/m300817/carbon_amoc/plots/Tdiff_u05.gif', images, fps=1.4)  # adjust fps as needed
```

<!-- #region toc-hr-collapsed=true -->
## 1pctCO2 
<!-- #endregion -->

```python
#ds_dissic_1pct.load()
ds_dissic_1pctbgc.load()
#ds_dissic_1pctrad.load()
```

```python
fig=plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(221, projection=ccrs.Robinson(central_longitude=-60))
ax1.set_global()
((ds_dissic_1pct.dissic.isel(time=np.arange(-120,0)).mean(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
-(ds_dissic_1pct.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
).plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(), x="longitude", y="latitude", cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'})
ax1.coastlines()
ax1.set_title("1pct")

ax2 = fig.add_subplot(222, projection=ccrs.Robinson(central_longitude=-60))
ax2.set_global()
((ds_dissic_1pctbgc.dissic.isel(time=np.arange(-120,0)).mean(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
-(ds_dissic_1pctbgc.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
).plot.pcolormesh(
    ax=ax2, transform=ccrs.PlateCarree(), x="longitude", y="latitude", cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'})
ax2.coastlines()
ax2.set_title("1pctbgc")


ax3 = fig.add_subplot(223, projection=ccrs.Robinson(central_longitude=-60))
ax3.set_global()
((ds_dissic_1pctrad.dissic.isel(time=np.arange(-120,0)).mean(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
-(ds_dissic_1pctrad.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
).plot.pcolormesh(
    ax=ax3, transform=ccrs.PlateCarree(), x="longitude", y="latitude", cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'})
ax3.coastlines()
ax3.set_title("1pctrad")
```

```python jupyter={"outputs_hidden": true}
# Total DIC below 1000m (1085-5720m) last 10 years 1pct-1pctbgc
plt.figure(figsize=(14, 6))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=-60))
ax.set_global()
((ds_dissic_1pct.dissic.isel(time=np.arange(-120,-1)).sum(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(23,40)).sum(dim='lev')
-(ds_dissic_1pctbgc.dissic.isel(time=np.arange(-120,-1)).sum(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(23,40)).sum(dim='lev')
).plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(), x="longitude", y="latitude", cmap='RdBu_r', cbar_kwargs={'label': ''})
ax.coastlines()
```

```python jupyter={"outputs_hidden": true}
# Plot DIC below 1000m (1085-5720m) difference last 10 years 1pctbgc-1pctrad
plt.figure(figsize=(14, 6))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=-60))
ax.set_global()
(ds_dissic_1pctbgc.dissic.load().isel(lev=np.arange(23,40)).sum(dim='lev').isel(time=np.arange(-120,-1)).mean(dim='time')-
 ds_dissic_1pctrad.dissic.load().isel(lev=np.arange(23,40)).sum(dim='lev').isel(time=np.arange(-120,-1)).mean(dim='time')).plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(), x="longitude", y="latitude")
ax.coastlines()
#ax.set_xlim([-100, 40])
```

```python jupyter={"outputs_hidden": true}
# Plot DIC below 1000m (1085-5720m) difference last 10 years 1pct-1pctbgc
plt.figure(figsize=(14, 6))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=-60))
ax.set_global()
(ds_dissic_1pct.dissic.load().isel(lev=np.arange(23,40)).sum(dim='lev').isel(time=np.arange(-120,-1)).mean(dim='time')-
 ds_dissic_1pctbgc.dissic.load().isel(lev=np.arange(23,40)).sum(dim='lev').isel(time=np.arange(-120,-1)).mean(dim='time')).plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(), x="longitude", y="latitude")
ax.coastlines()
#ax.set_xlim([-100, 40])
```

```python jupyter={"outputs_hidden": true}
# Plot DIC below 1000m (1085-5720m) difference last 10 years 1pct-1pctbgc
plt.figure(figsize=(14, 6))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=-60))
ax.set_global()
(ds_dissic_1pct.dissic.load().isel(lev=np.arange(23,40)).sum(dim='lev').isel(time=np.arange(-120,-1)).mean(dim='time')-
 ds_dissic_1pctbgc.dissic.load().isel(lev=np.arange(23,40)).sum(dim='lev').isel(time=np.arange(-120,-1)).mean(dim='time')).plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(), x="longitude", y="latitude")
ax.coastlines()
#ax.set_xlim([-100, 40])
```

```python jupyter={"outputs_hidden": true}
# Total DIC below 1000m (1085-5720m) 1pct-1pctbgc
plt.figure(figsize=(14, 6))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=-60))
ax.set_global()
((ds_dissic_1pct.dissic*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(23,40)).sum(dim='lev').isel(time=np.arange(-120,-1)).sum(dim="time")
-(ds_dissic_1pctbgc.dissic*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(23,40)).sum(dim='lev').isel(time=np.arange(-120,-1)).sum(dim="time")
).plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(), x="longitude", y="latitude", cbar_kwargs={'label': 'PgC'})
ax.coastlines()

```

```python jupyter={"outputs_hidden": true}
# Total DIC below 1000m (1085-5720m) 1pct-1pctbgc

((ds_dissic_1pct.dissic*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(23,40)).sum(dim='lev').isel(time=np.arange(-120,-1)).sum(dim="time")
-(ds_dissic_1pctbgc.dissic*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(23,40)).sum(dim='lev').isel(time=np.arange(-120,-1)).sum(dim="time")
).sum(dim='i').sum(dim='j').values

```

```python jupyter={"outputs_hidden": true}
# Plot DIC below 1000m (1085-5720m) difference last 10 years 1pctrad & first 10 years hist
plt.figure(figsize=(14, 6))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=-60))
ax.set_global()
(ds_dissic_1pctrad.dissic.load().isel(lev=np.arange(23,40)).sum(dim='lev').isel(time=np.arange(-120,-1)).mean(dim='time')-
 ds_dissic_hist.dissic.load().isel(lev=np.arange(23,40)).sum(dim='lev').isel(time=np.arange(0,120)).mean(dim='time')).plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(), x="longitude", y="latitude")
ax.coastlines()
#ax.set_xlim([-100, 40])
```

```python jupyter={"outputs_hidden": true}
fig=plt.figure(figsize=(14, 9))

ax1 = fig.add_subplot(211, projection=ccrs.Robinson(central_longitude=-60))
ax1.set_global()
((ds_dissic_1pctbgc.dissic.isel(time=np.arange(-12,0)).mean(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
-(ds_dissic_1pctbgc.dissic.isel(time=np.arange(0,12)).mean(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
).plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(), x="longitude", y="latitude", vmin=-0.01, vmax=0.01, cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'})
ax1.coastlines()
ax1.set_title("Full depth (0-5720m)") 

ax2 = fig.add_subplot(223, projection=ccrs.Robinson(central_longitude=-60))
ax2.set_global()
((ds_dissic_1pctbgc.dissic.isel(time=np.arange(-12,0)).mean(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,23)).sum(dim='lev')
-(ds_dissic_1pctbgc.dissic.isel(time=np.arange(0,12)).mean(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,23)).sum(dim='lev')
).plot.pcolormesh(
    ax=ax2, transform=ccrs.PlateCarree(), x="longitude", y="latitude", vmin=-0.01, vmax=0.01, cmap='RdBu_r', add_colorbar=False)
ax2.coastlines()
ax2.set_title("Upper ocean (0-1085m)") 

ax3 = fig.add_subplot(224, projection=ccrs.Robinson(central_longitude=-60))
ax3.set_global()
((ds_dissic_1pctbgc.dissic.isel(time=np.arange(-12,0)).mean(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(23,40)).sum(dim='lev')
-(ds_dissic_1pctbgc.dissic.isel(time=np.arange(0,12)).mean(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(23,40)).sum(dim='lev')
).plot.pcolormesh(
   ax=ax3, transform=ccrs.PlateCarree(), x="longitude", y="latitude", vmin=-0.01, vmax=0.01, cmap='RdBu_r', add_colorbar=False)
ax3.coastlines()
ax3.set_title("Deep ocean (1085-5720m)") 
```

### Other variables

```python
# Load wfo in 1pctCO2-bgc
file_type = 'wfo'
infiles = glob.glob(f'/pool/data/CMIP6/data/C4MIP/MPI-M/MPI-ESM1-2-LR/1pctCO2-bgc/r1i1p1f1/Omon/{file_type}/gn/v20190710/*{file_type}*.nc')
ds_wfo_1pctbgc = xr.open_mfdataset(infiles, use_cftime=True, parallel=True)
```

```python
fig=plt.figure(figsize=(14, 9))
ax1 = fig.add_subplot(211, projection=ccrs.Robinson(central_longitude=-60))
ax1.set_global()
ds_wfo_1pctbgc.mean(dim="time").wfo.plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(), vmin=-0.0001, vmax=0.0001, x="longitude", y="latitude", cmap='RdBu_r')
ax1.coastlines()
```

```python
# Surface net water into ocean flux FIELD
ifiles  = "/work/mh0033/from_Mistral/mh0033/m300817/mpiesm-1.2.01p6-passivesalt_update/experiments/hosing_naa05Sv_FcSV-LR/outdata/mpiom/*2d_mm_*"
outfile = "/scratch/m/m300817/tmp/hosing_naa05Sv_FcSV-LR_wfo.nc"
if not os.path.isfile(outfile):
    cdo.yearmean(input = "-mergetime -select,name=wfo "+ifiles , output = outfile)
```

```python
ds_wfo_u05hosLR = xr.open_mfdataset("/scratch/m/m300817/tmp/hosing_naa05Sv_FcSV-LR_wfo.nc", use_cftime=True, parallel=True)
```

```python jupyter={"outputs_hidden": true}
ds_wfo_u05hosLR
```

```python
fig=plt.figure(figsize=(14, 9))
ax1 = fig.add_subplot(211, projection=ccrs.Robinson(central_longitude=-60))
ax1.set_global()
ds_wfo_u05hosLR.mean(dim="time").mean(dim='depth').wfo.plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(), vmin=-0.0001, vmax=0.0001, x="lon", y="lat", cmap='RdBu_r')
ax1.coastlines()
```

```python
ds_fwmask_u03 = xr.open_mfdataset("/work/mh0287/m211054/mpiesm/hosing/masks/HOSING_NAA_03SV_GR15.nc", use_cftime=True, parallel=True)
```

```python
fig=plt.figure(figsize=(14, 9))
ax1 = fig.add_subplot(211, projection=ccrs.Robinson(central_longitude=-60))
(ds_fwmask_u03.hosing.where((ds_fwmask_u03.hosing>0).compute(), drop=True)*1e6).plot(
    ax=ax1, transform=ccrs.PlateCarree(), x="lon", y="lat", vmin=0)
ax1.coastlines()
```

```python
import matplotlib.colors as mcolors
blue_cmap = mcolors.LinearSegmentedColormap.from_list('blue_cmap', ['white', 'steelblue'])
dblue_cmap = mcolors.LinearSegmentedColormap.from_list('dblue_cmap', ['steelblue', 'steelblue'])
```

```python
fig=plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(111, projection=ccrs.NearsidePerspective(central_longitude=-20, central_latitude= 35, satellite_height=35785831)) #frame_on=False
ds_fwmask_u03.hosing.where(ds_fwmask_u03.hosing>0).plot(
    ax=ax1, transform=ccrs.PlateCarree(), x="lon", y="lat", add_colorbar=False, cmap='Blues', alpha=1, zorder=-1)
ds_fwmask_u03.hosing.where(ds_fwmask_u03.hosing==0).plot(
    ax=ax1, transform=ccrs.PlateCarree(), x="lon", y="lat", add_colorbar=False, cmap=dblue_cmap, alpha=1, zorder=1)
ax1.coastlines(color='black')
for spine in ax1.spines.values(): # color of circle
    spine.set_edgecolor('white')
ax1.stock_img()
ax1.gridlines(color='black')
```

```python
ds_fwmask_g01 = xr.open_mfdataset("/work/mh0287/m211054/mpiesm/hosing/masks/HOSING_GRC_01SV_GR15.nc", use_cftime=True, parallel=True)
```

```python
fig=plt.figure(figsize=(14, 9))
ax1 = fig.add_subplot(211, projection=ccrs.Robinson(central_longitude=-60))
(ds_fwmask_g01.flux.where((ds_fwmask_g01.flux>0).compute(), drop=True)*1e6).plot(
    ax=ax1, transform=ccrs.PlateCarree(), x="longitude", y="latitude", vmin=0)
ax1.coastlines()
```

so

```python
ds_sss_u03hosLR = xr.open_mfdataset("/work/uo1075/m300817/hosing/post/from_Mistral/data/SSS/hosing_naa03Sv_FcSV-LR_SSS.nc"
                                    , use_cftime=True, parallel=True)
```

```python
ds_sss_u03hosLR.mean(dim='x').mean(dim='y').sos.plot()
```

```python
# Surface net water into ocean flux FIELD
ifiles  = "/work/mh0033/from_Mistral/mh0033/m300817/mpiesm-1.2.01p6-passivesalt_update/experiments/hosing_naa05Sv_FcSV-LR/outdata/mpiom/*3d_mm_*"
outfile = "/scratch/m/m300817/tmp/hosing_naa05Sv_FcSV-LR_so.nc"
if not os.path.isfile(outfile):
    cdo.yearmean(input = "-mergetime -select,name=so "+ifiles , output = outfile)
```

```python
ds_so_u05hosLR = xr.open_mfdataset("/scratch/m/m300817/tmp/hosing_naa05Sv_FcSV-LR_so.nc", use_cftime=True, parallel=True)
```

```python
# Surface net water into ocean flux FIELD
ifiles  = "/work/uo1075/m300817/hosing/mpiesm-1.2.01p7-passivesalt-hosing/experiments/hosing_naa03Sv_1pctbgc-LR/outdata/mpiom/*3d_mm_*"
outfile = "/scratch/m/m300817/tmp/hosing_naa03Sv_1pctbgc-LR_so.nc"
if not os.path.isfile(outfile):
    cdo.yearmean(input = "-mergetime -select,name=so "+ifiles , output = outfile)
```

```python
ds_so_u031pctbgcLR= xr.open_mfdataset("/scratch/m/m300817/tmp/hosing_naa03Sv_1pctbgc-LR_so.nc", use_cftime=True, parallel=True)
```

```python
# Load so in 1pctCO2-bgc
file_type = 'so'
infiles = glob.glob(f'/pool/data/CMIP6/data/C4MIP/MPI-M/MPI-ESM1-2-LR/1pctCO2-bgc/r1i1p1f1/Omon/{file_type}/gn/v20190710/*{file_type}*.nc')
ds_so_1pctbgc = xr.open_mfdataset(infiles, use_cftime=True, parallel=True)
```

```python
fig=plt.figure(figsize=(14, 9))
ax1 = fig.add_subplot(211, projection=ccrs.Robinson(central_longitude=-60))
ax1.set_global()
ds_so_1pctbgc.isel(lev=0).isel(time=0).so.plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(), x="longitude", y="latitude", cmap='RdBu_r')
ax1.coastlines()
```

```python
fig=plt.figure(figsize=(14, 9))
ax1 = fig.add_subplot(211, projection=ccrs.Robinson(central_longitude=-60))
ax1.set_global()
ds_so_u05hosLR.isel(depth=0).isel(time=0).so.plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r')
ax1.coastlines()
```

```python
ds_so_1pctbgc.isel(lev=0).mean(dim='i').mean(dim='j').groupby('time.year').mean('time').so.plot()
```

### plot concentrations 1pct

```python
import matplotlib.pyplot as plt
import numpy as np

# Initial concentration
concentration = 284.7
# List to store concentrations
concentrations = [concentration]

# Calculate concentrations for 140 years
for year in range(140):
    concentration *= 1.01  # Increase by 1%
    concentrations.append(concentration)

# Create an array for the years
years = np.arange(0, 141, 1)

# Create the plot
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(years, concentrations, linewidth=3)
ax.set_xlabel('Years')
ax.set_ylabel(r'CO$_2$ concentration [ppm]')
ax.set_xlim(0,140)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()
```

1 PgC = 44.01/12.01 = 3.664 GtCO2

```python
5*3.664
```

```python
3.664*11.6
```

<!-- #region toc-hr-collapsed=true -->
## New hosing bgc
<!-- #endregion -->

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### Time series
<!-- #endregion -->

AMOC

```python
(ds_msftmz_g011pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(color='lightsteelblue')
(ds_msftmz_g031pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(color='powderblue')
(ds_msftmz_g051pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(color='skyblue')
(ds_msftmz_u011pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(color='lightsalmon')
(ds_msftmz_u031pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(color='indianred')
(ds_msftmz_u051pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(color='lightcoral')
(ds_msftmz_u031pctLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(color='darkgrey')
(ds_msftmz_u051pctLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(color='dimgrey')
plt.legend(['g01-1pctbgc', 'g03-1pctbgc', 'g05-1pctbgc', 'u01-1pctbgc','u03-1pctbgc', 'u05-1pctbgc', 'u03-1pct','u05-1pct'])
```

```python
(ds_msftmz_histbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(color=colors_scen['hist'])
(ds_msftmz_u03ssp126bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(color=colors_scen['ssp126'])
(ds_msftmz_u03ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(color=colors_scen['ssp245'])
(ds_msftmz_u03ssp585bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(color=colors_scen['ssp585'])
plt.legend(['hist', 'ssp126', 'ssp245', 'ssp585'])
```

```python
(ds_msftmz_ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(color='k')
(ds_msftmz_u05ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(color='darkred')
(ds_msftmz_u03ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(color='salmon')
(ds_msftmz_u01ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(color='peachpuff')
(ds_msftmz_g05ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(color='lightblue')
(ds_msftmz_g03ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(color='dodgerblue')
(ds_msftmz_g01ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(color='darkblue')
plt.legend(['ssp245-bgc', 'u05-ssp245-bgc', 'u03-ssp245-bgc', 'u01-ssp245-bgc', 'g05-ssp245-bgc', 'g03-ssp245-bgc', 'g01-ssp245-bgc'])
```

Temperature

```python
ds_tas_new["hosing_grc01Sv_1pctbgc-LR"].groupby('time.year').mean('time').mean(dim='lat').mean(dim='lon').temp2.plot(color='lightblue')
ds_tas_new["hosing_grc03Sv_1pctbgc-LR"].groupby('time.year').mean('time').mean(dim='lat').mean(dim='lon').temp2.plot(color='khaki')
ds_tas_new["hosing_naa03Sv_1pctbgc-LR"].groupby('time.year').mean('time').mean(dim='lat').mean(dim='lon').temp2.plot(color='orange')
ds_tas_new["hosing_naa05Sv_1pctbgc-LR"].groupby('time.year').mean('time').mean(dim='lat').mean(dim='lon').temp2.plot(color='red')
ds_tas_new["hosing_naa03Sv_1pct-LR"].groupby('time.year').mean('time').mean(dim='lat').mean(dim='lon').temp2.plot(color='lightgrey')
ds_tas_new["hosing_naa05Sv_1pct-LR"].groupby('time.year').mean('time').mean(dim='lat').mean(dim='lon').temp2.plot(color='darkgrey')
plt.legend(['g01-1pctbgc','g03-1pctbgc','u03-1pctbgc', 'u05-1pctbgc', 'u03-1pct','u05-1pct'])
```

```python
ds_tas_new_scen["histbgc-LR"].groupby('time.year').mean('time').mean(dim='lat').mean(dim='lon').temp2.plot(color=colors_scen['hist'])
ds_tas_new_scen["hosing_naa03Sv_ssp126bgc-LR"].groupby('time.year').mean('time').mean(dim='lat').mean(dim='lon').temp2.plot(color=colors_scen['ssp126'])
ds_tas_new_scen["hosing_naa03Sv_ssp245bgc-LR"].groupby('time.year').mean('time').mean(dim='lat').mean(dim='lon').temp2.plot(color=colors_scen['ssp245'])
ds_tas_new_scen["hosing_naa03Sv_ssp585bgc-LR"].groupby('time.year').mean('time').mean(dim='lat').mean(dim='lon').temp2.plot(color=colors_scen['ssp585'])
plt.legend(['hist', 'ssp126', 'ssp245', 'ssp585'])
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### dissic maps
<!-- #endregion -->

```python
# u03-1pctbgc dissic map
fig=plt.figure(figsize=(14, 9))

ax1 = fig.add_subplot(211, projection=ccrs.Robinson(central_longitude=-60))
ax1.set_global()
((ds_dissic_u031pctbgcLR.dissic.isel(time=np.arange(-10,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u031pctbgcLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(), x="lon", y="lat", vmin=-0.1, vmax=0.1, cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'})
ax1.coastlines()
ax1.set_title("Full depth (0-5720m)") 

ax2 = fig.add_subplot(223, projection=ccrs.Robinson(central_longitude=-60))
ax2.set_global()
((ds_dissic_u031pctbgcLR.dissic.isel(time=np.arange(-10,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,23)).sum(dim='depth')
-(ds_dissic_u031pctbgcLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,23)).sum(dim='depth')
).plot.pcolormesh(
    ax=ax2, transform=ccrs.PlateCarree(), x="lon", y="lat", vmin=-0.1, vmax=0.1, cmap='RdBu_r', add_colorbar=False)
ax2.coastlines()
ax2.set_title("Upper ocean (0-1085m)") 

ax3 = fig.add_subplot(224, projection=ccrs.Robinson(central_longitude=-60))
ax3.set_global()
((ds_dissic_u031pctbgcLR.dissic.isel(time=np.arange(-10,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(23,40)).sum(dim='depth')
-(ds_dissic_u031pctbgcLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(23,40)).sum(dim='depth')
).plot.pcolormesh(
   ax=ax3, transform=ccrs.PlateCarree(), x="lon", y="lat", vmin=-0.1, vmax=0.1, cmap='RdBu_r', add_colorbar=False)
ax3.coastlines()
ax3.set_title("Deep ocean (1085-5720m)") 
```

```python
# Total DIC full depth (0-5720m) last vs first 10 years
dissic_u011pctbgc_total_full = ((ds_dissic_u011pctbgcLR.dissic.isel(time=np.arange(-10,-0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u011pctbgcLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='y').sum(dim='x').values

dissic_u031pctbgc_total_full = ((ds_dissic_u031pctbgcLR.dissic.isel(time=np.arange(-10,-0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u031pctbgcLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='y').sum(dim='x').values

dissic_u051pctbgc_total_full = ((ds_dissic_u051pctbgcLR.dissic.isel(time=np.arange(-10,-0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u051pctbgcLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='y').sum(dim='x').values

dissic_g011pctbgc_total_full = ((ds_dissic_g011pctbgcLR.dissic.isel(time=np.arange(-10,-0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g011pctbgcLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='y').sum(dim='x').values

dissic_g031pctbgc_total_full = ((ds_dissic_g031pctbgcLR.dissic.isel(time=np.arange(-10,-0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g031pctbgcLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='y').sum(dim='x').values

dissic_g051pctbgc_total_full = ((ds_dissic_g051pctbgcLR.dissic.isel(time=np.arange(-10,-0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g051pctbgcLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='y').sum(dim='x').values

dissic_u031pct_total_full = ((ds_dissic_u031pctLR.dissic.isel(time=np.arange(-10,-0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u031pctLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='y').sum(dim='x').values

dissic_u051pct_total_full = ((ds_dissic_u051pctLR.dissic.isel(time=np.arange(-10,-0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u051pctLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='y').sum(dim='x').values

# Now with 1pct simulations without hosing
dissic_1pctbgc_total_full = ((ds_dissic_1pctbgc.dissic.isel(time=np.arange(-120,-0)).mean(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
-(ds_dissic_1pctbgc.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
).sum(dim='i').sum(dim='j').values

dissic_1pct_total_full = ((ds_dissic_1pct.dissic.isel(time=np.arange(-120,-0)).mean(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
-(ds_dissic_1pct.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
).sum(dim='i').sum(dim='j').values
```

```python jupyter={"outputs_hidden": true}
fig=plt.figure(figsize=(18, 8))

ax1 = fig.add_subplot(231, projection=ccrs.Robinson(central_longitude=-60))
ax1.set_global()
((ds_dissic_1pctbgc.dissic.isel(time=np.arange(-120,0)).mean(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
-(ds_dissic_1pctbgc.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
).plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(), x="longitude", y="latitude", cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'}, vmin=-0.15, vmax=0.15)
ax1.coastlines()
ax1.set_title("1pctbgc")
ax1.text(-0.1, 1.3, str(round(dissic_1pctbgc_total_full.item(), 1))+" PgC", transform=ax1.transAxes, fontsize=14, verticalalignment='top', fontweight='bold')

ax2 = fig.add_subplot(232, projection=ccrs.Robinson(central_longitude=-60))
ax2.set_global()
((ds_dissic_u031pctbgcLR.dissic.isel(time=np.arange(-10,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u031pctbgcLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).plot.pcolormesh(
    ax=ax2, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'}, vmin=-0.15, vmax=0.15)
ax2.coastlines()
ax2.set_title("u03-1pctbgc")
ax2.text(-0.1, 1.3, str(round(dissic_u031pctbgc_total_full.item(), 1))+" PgC", transform=ax2.transAxes, fontsize=14, verticalalignment='top', fontweight='bold')

ax3 = fig.add_subplot(233, projection=ccrs.Robinson(central_longitude=-60))
ax3.set_global()
((ds_dissic_u051pctbgcLR.dissic.isel(time=np.arange(-10,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u051pctbgcLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).plot.pcolormesh(
    ax=ax3, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'}, vmin=-0.15, vmax=0.15)
ax3.coastlines()
ax3.set_title("u05-1pctbgc")
ax3.text(-0.1, 1.3, str(round(dissic_u051pctbgc_total_full.item(), 1))+" PgC", transform=ax3.transAxes, fontsize=14, verticalalignment='top', fontweight='bold')

ax4 = fig.add_subplot(234, projection=ccrs.Robinson(central_longitude=-60))
ax4.set_global()
((ds_dissic_g011pctbgcLR.dissic.isel(time=np.arange(-10,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g011pctbgcLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).plot.pcolormesh(
    ax=ax4, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'}, vmin=-0.15, vmax=0.15)
ax4.coastlines()
ax4.set_title("g01-1pctbgc")
ax4.text(-0.1, 1.3, str(round(dissic_g011pctbgc_total_full.item(), 1))+" PgC", transform=ax6.transAxes, fontsize=14, verticalalignment='top', fontweight='bold')

ax5 = fig.add_subplot(235, projection=ccrs.Robinson(central_longitude=-60))
ax5.set_global()
((ds_dissic_g031pctbgcLR.dissic.isel(time=np.arange(-10,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g031pctbgcLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).plot.pcolormesh(
    ax=ax5, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'}, vmin=-0.15, vmax=0.15)
ax5.coastlines()
ax5.set_title("g03-1pctbgc")
ax5.text(-0.1, 1.3, str(round(dissic_g031pctbgc_total_full.item(), 1))+" PgC", transform=ax5.transAxes, fontsize=14, verticalalignment='top', fontweight='bold')

ax6 = fig.add_subplot(236, projection=ccrs.Robinson(central_longitude=-60))
ax6.set_global()
((ds_dissic_g011pctbgcLR.dissic.isel(time=np.arange(-10,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g011pctbgcLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).plot.pcolormesh(
    ax=ax6, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'}, vmin=-0.15, vmax=0.15)
ax6.coastlines()
ax6.set_title("g01-1pctbgc")
ax6.text(-0.1, 1.3, str(round(dissic_g011pctbgc_total_full.item(), 1))+" PgC", transform=ax6.transAxes, fontsize=14, verticalalignment='top', fontweight='bold')
```

```python
diff10y_dissic_1pctbgc = ((ds_dissic_1pctbgc.dissic.isel(time=np.arange(-120,0)).mean(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
               -(ds_dissic_1pctbgc.dissic.isel(time=np.arange(0,120)).mean(dim="time")*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')).rename({'i': 'x','j': 'y','latitude':'lat', 'longitude':'lon'})
diff10y_dissic_1pctbgc.coords['lon'] = (diff10y_dissic_1pctbgc.coords['lon'] + 180) % 360 - 180

diff10y_dissic_u011pctbgc = ((ds_dissic_u011pctbgcLR.dissic.isel(time=np.arange(-10,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u011pctbgcLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth'))
diff10y_dissic_u011pctbgc.coords['lon'] = (diff10y_dissic_u011pctbgc.coords['lon'] + 180) % 360 - 180

diff10y_dissic_u031pctbgc = ((ds_dissic_u031pctbgcLR.dissic.isel(time=np.arange(-10,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u031pctbgcLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth'))
diff10y_dissic_u031pctbgc.coords['lon'] = (diff10y_dissic_u031pctbgc.coords['lon'] + 180) % 360 - 180

diff10y_dissic_u051pctbgc = ((ds_dissic_u051pctbgcLR.dissic.isel(time=np.arange(-10,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u051pctbgcLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth'))
diff10y_dissic_u051pctbgc.coords['lon'] = (diff10y_dissic_u051pctbgc.coords['lon'] + 180) % 360 - 180

diff10y_dissic_g051pctbgc = ((ds_dissic_g051pctbgcLR.dissic.isel(time=np.arange(-10,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g051pctbgcLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth'))
diff10y_dissic_g051pctbgc.coords['lon'] = (diff10y_dissic_g051pctbgc.coords['lon'] + 180) % 360 - 180

diff10y_dissic_g031pctbgc = ((ds_dissic_g031pctbgcLR.dissic.isel(time=np.arange(-10,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g031pctbgcLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth'))
diff10y_dissic_g031pctbgc.coords['lon'] = (diff10y_dissic_g031pctbgc.coords['lon'] + 180) % 360 - 180

diff10y_dissic_g011pctbgc =((ds_dissic_g011pctbgcLR.dissic.isel(time=np.arange(-10,0)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g011pctbgcLR.dissic.isel(time=np.arange(0,10)).mean(dim="time")*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth'))
diff10y_dissic_g011pctbgc.coords['lon'] = (diff10y_dissic_g011pctbgc.coords['lon'] + 180) % 360 - 180
```

```python
# Now difference plot 1pctbgc MINUS EACH HOSING 1pctbgc
fig=plt.figure(figsize=(18, 8))

ax1 = fig.add_subplot(231, projection=ccrs.Robinson(central_longitude=-40))
ax1.set_global()
(xr.DataArray(diff10y_dissic_1pctbgc.variable - diff10y_dissic_u011pctbgc.variable, coords=diff10y_dissic_1pctbgc.coords)).plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'}, vmin=-0.03, vmax=0.03)
ax1.coastlines()
ax1.set_title("1pctbgc - u01-1pctbgc")
ax1.text(-0.1, 1.1, str(round(dissic_1pctbgc_total_full-dissic_u011pctbgc_total_full.item(), 1))+" PgC", transform=ax1.transAxes, fontsize=14, verticalalignment='top', fontweight='bold')

ax2 = fig.add_subplot(232, projection=ccrs.Robinson(central_longitude=-40))
ax2.set_global()
(xr.DataArray(diff10y_dissic_1pctbgc.variable - diff10y_dissic_u031pctbgc.variable, coords=diff10y_dissic_1pctbgc.coords)).plot.pcolormesh(
    ax=ax2, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'}, vmin=-0.03, vmax=0.03)
ax2.coastlines()
ax2.set_title("1pctbgc - u03-1pctbgc")
ax2.text(-0.1, 1.1, str(round(dissic_1pctbgc_total_full-dissic_u031pctbgc_total_full.item(), 1))+" PgC", transform=ax2.transAxes, fontsize=14, verticalalignment='top', fontweight='bold')

ax3 = fig.add_subplot(233, projection=ccrs.Robinson(central_longitude=-60))
ax3.set_global()
(xr.DataArray(diff10y_dissic_1pctbgc.variable - diff10y_dissic_u051pctbgc.variable, coords=diff10y_dissic_1pctbgc.coords)).plot.pcolormesh(
    ax=ax3, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'}, vmin=-0.03, vmax=0.03)
ax3.coastlines()
ax3.set_title("1pctbgc - u05-1pctbgc")
ax3.text(-0.1, 1.1, str(round(dissic_1pctbgc_total_full-dissic_u051pctbgc_total_full.item(), 1))+" PgC", transform=ax3.transAxes, fontsize=14, verticalalignment='top', fontweight='bold')


ax4 = fig.add_subplot(234, projection=ccrs.Robinson(central_longitude=-60))
ax4.set_global()
(xr.DataArray(diff10y_dissic_1pctbgc.variable - diff10y_dissic_g011pctbgc.variable, coords=diff10y_dissic_1pctbgc.coords)).plot.pcolormesh(
    ax=ax4, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'}, vmin=-0.03, vmax=0.03)
ax4.coastlines()
ax4.set_title("1pctbgc - g01-1pctbgc")
ax4.text(-0.1, 1.1, str(round(dissic_1pctbgc_total_full-dissic_g011pctbgc_total_full.item(), 1))+" PgC", transform=ax4.transAxes, fontsize=14, verticalalignment='top', fontweight='bold')

ax5 = fig.add_subplot(235, projection=ccrs.Robinson(central_longitude=-60))
ax5.set_global()
(xr.DataArray(diff10y_dissic_1pctbgc.variable - diff10y_dissic_g031pctbgc.variable, coords=diff10y_dissic_1pctbgc.coords)).plot.pcolormesh(
    ax=ax5, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'}, vmin=-0.03, vmax=0.03)
ax5.coastlines()
ax5.set_title("1pctbgc - g03-1pctbgc")
ax5.text(-0.1, 1.1, str(round(dissic_1pctbgc_total_full-dissic_g031pctbgc_total_full.item(), 1))+" PgC", transform=ax5.transAxes, fontsize=14, verticalalignment='top', fontweight='bold')

ax6 = fig.add_subplot(236, projection=ccrs.Robinson(central_longitude=-60))
ax6.set_global()
(xr.DataArray(diff10y_dissic_1pctbgc.variable - diff10y_dissic_g051pctbgc.variable, coords=diff10y_dissic_1pctbgc.coords)).plot.pcolormesh(
    ax=ax6, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', cbar_kwargs={'label': 'PgC, column integrated'}, vmin=-0.03, vmax=0.03)
ax6.coastlines()
ax6.set_title("1pctbgc - g05-1pctbgc")
ax6.text(-0.1, 1.1, str(round(dissic_1pctbgc_total_full-dissic_g051pctbgc_total_full.item(), 1))+" PgC", transform=ax6.transAxes, fontsize=14, verticalalignment='top', fontweight='bold')
```

### Calculation feedbacks


First, calculate the total difference in tas, dissic and msftmz; comparing initial to last states

```python
# Total diff tas 1pctbgc
total_tas_diff = {}
for exp in ["hosing_naa01Sv_1pctbgc-LR","hosing_naa03Sv_1pctbgc-LR", "hosing_naa05Sv_1pctbgc-LR",
            "hosing_grc01Sv_1pctbgc-LR", "hosing_grc03Sv_1pctbgc-LR", "hosing_grc05Sv_1pctbgc-LR",
            "hosing_naa03Sv_1pct-LR", "hosing_naa05Sv_1pct-LR"]:
    total_tas_diff[exp] = (ds_tas_new[exp].groupby('time.year').mean('time').isel(year=-1).mean(dim='lat').mean(dim='lon').temp2-ds_tas_new[exp].groupby(
        'time.year').mean('time').isel(year=0).mean(dim='lat').mean(dim='lon').temp2).values
    print(exp+": "+str(total_tas_diff[exp]))
```

```python
# Total diff dissic bgc hist & scenarios
total_tas_diff_scen = {}
for exp in ["histbgc-LR", "hosing_naa03Sv_ssp126bgc-LR", "hosing_naa03Sv_ssp245bgc-LR", "hosing_naa03Sv_ssp585bgc-LR"]:
    total_tas_diff_scen[exp] = (ds_tas_new_scen[exp].groupby('time.year').mean('time').isel(year=-1).mean(dim='lat').mean(dim='lon').temp2-ds_tas_new_scen[exp].groupby(
        'time.year').mean('time').isel(year=0).mean(dim='lat').mean(dim='lon').temp2).values
    print(exp+": "+str(total_tas_diff_scen[exp]))
```

```python
# Total diff tas ssp245bgc
total_tas_diff_ssp245 = {}
for exp in ["hosing_naa01Sv_ssp245bgc-LR","hosing_naa03Sv_ssp245bgc-LR", "hosing_naa05Sv_ssp245bgc-LR",
            "hosing_grc01Sv_ssp245bgc-LR", "hosing_grc03Sv_ssp245bgc-LR", "hosing_grc05Sv_ssp245bgc-LR",
            "ssp245bgc-LR"]:
    total_tas_diff_ssp245[exp] = (ds_tas_ssp245[exp].groupby('time.year').mean('time').isel(year=-1).mean(dim='lat').mean(dim='lon').temp2-ds_tas_ssp245[exp].groupby(
        'time.year').mean('time').isel(year=0).mean(dim='lat').mean(dim='lon').temp2).values
    print(exp+": "+str(total_tas_diff_ssp245[exp]))
```

```python

```

```python
# Total diff dissic 1pctbgc
dissic_1pctbgc_diff = ((ds_dissic_1pctbgc.dissic.groupby('time.year').mean('time').isel(year=-1)*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
-(ds_dissic_1pctbgc.dissic.groupby('time.year').mean('time').isel(year=0)*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
).sum(dim='i').sum(dim='j').values
print(dissic_1pctbgc_diff)

dissic_u011pctbgc_diff = ((ds_dissic_u011pctbgcLR.dissic.isel(time=-1)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u011pctbgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='x').sum(dim='y').values
print(dissic_u011pctbgc_diff)

dissic_u031pctbgc_diff = ((ds_dissic_u031pctbgcLR.dissic.isel(time=-1)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u031pctbgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='x').sum(dim='y').values
print(dissic_u031pctbgc_diff)

dissic_u051pctbgc_diff = ((ds_dissic_u051pctbgcLR.dissic.isel(time=-1)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u051pctbgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='x').sum(dim='y').values
print(dissic_u051pctbgc_diff)

dissic_g011pctbgc_diff = ((ds_dissic_g011pctbgcLR.dissic.isel(time=-1)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g011pctbgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='x').sum(dim='y').values
print(dissic_g011pctbgc_diff)

dissic_g031pctbgc_diff = ((ds_dissic_g031pctbgcLR.dissic.isel(time=-1)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g031pctbgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='x').sum(dim='y').values
print(dissic_g031pctbgc_diff)

dissic_g051pctbgc_diff = ((ds_dissic_g051pctbgcLR.dissic.isel(time=-1)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g051pctbgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='x').sum(dim='y').values
print(dissic_g051pctbgc_diff)

dissic_1pct_diff = ((ds_dissic_1pct.dissic.groupby('time.year').mean('time').isel(year=-1)*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
-(ds_dissic_1pct.dissic.groupby('time.year').mean('time').isel(year=0)*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
).sum(dim='i').sum(dim='j').values
print(dissic_1pct_diff)

dissic_u031pct_diff = ((ds_dissic_u031pctLR.dissic.isel(time=-1)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u031pctLR.dissic.isel(time=0)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='x').sum(dim='y').values
print(dissic_u031pct_diff)

dissic_u051pct_diff = ((ds_dissic_u051pctLR.dissic.isel(time=-1)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u051pctLR.dissic.isel(time=0)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='x').sum(dim='y').values
print(dissic_u051pct_diff)
```

```python
# Total diff dissic bgc hist & scenarios
dissic_histbgc_diff = ((ds_dissic_histbgcLR.dissic.isel(time=-1)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_histbgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='x').sum(dim='y').values
print(dissic_histbgc_diff)

dissic_u03ssp126bgc_diff = ((ds_dissic_u03ssp126bgcLR.dissic.isel(time=-1)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u03ssp126bgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='x').sum(dim='y').values
print(dissic_u03ssp126bgc_diff)

dissic_u03ssp245bgc_diff = ((ds_dissic_u03ssp245bgcLR.dissic.isel(time=-1)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u03ssp245bgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='x').sum(dim='y').values
print(dissic_u03ssp245bgc_diff)

dissic_u03ssp585bgc_diff = ((ds_dissic_u03ssp585bgcLR.dissic.isel(time=-1)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u03ssp585bgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='x').sum(dim='y').values
print(dissic_u03ssp585bgc_diff)
```

```python
# Total diff dissic ssp245bgc
dissic_ssp245bgc_diff = ((ds_dissic_ssp245bgcLR.dissic.isel(time=-1)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_ssp245bgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='x').sum(dim='y').values
print(dissic_ssp245bgc_diff)

dissic_u01ssp245bgc_diff = ((ds_dissic_u01ssp245bgcLR.dissic.isel(time=-1)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u01ssp245bgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='x').sum(dim='y').values
print(dissic_u01ssp245bgc_diff)

dissic_u03ssp245bgc_diff = ((ds_dissic_u03ssp245bgcLR.dissic.isel(time=-1)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u03ssp245bgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='x').sum(dim='y').values
print(dissic_u03ssp245bgc_diff)

dissic_u05ssp245bgc_diff = ((ds_dissic_u05ssp245bgcLR.dissic.isel(time=-1)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_u05ssp245bgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='x').sum(dim='y').values
print(dissic_u05ssp245bgc_diff)

dissic_g01ssp245bgc_diff = ((ds_dissic_g01ssp245bgcLR.dissic.isel(time=-1)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g01ssp245bgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='x').sum(dim='y').values
print(dissic_g01ssp245bgc_diff)

dissic_g03ssp245bgc_diff = ((ds_dissic_g03ssp245bgcLR.dissic.isel(time=-1)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g03ssp245bgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='x').sum(dim='y').values
print(dissic_g03ssp245bgc_diff)

dissic_g05ssp245bgc_diff = ((ds_dissic_g05ssp245bgcLR.dissic.isel(time=-1)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
-(ds_dissic_g05ssp245bgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.mean(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
).sum(dim='x').sum(dim='y').values
print(dissic_g05ssp245bgc_diff)
```

```python

```

```python
# Total diff msftmz 1pct
msftmz_1pctbgc_diff = ((ds_msftmz_1pctbgc.sel(lat=26.5).sel(lev=1020).isel(basin=1).groupby('time.year').mean('time').isel(year=-1).msftmz.values/1e9)) - ((ds_msftmz_1pctbgc.sel(
    lat=26.5).sel(lev=1020).isel(basin=1).groupby('time.year').mean('time').isel(year=0).msftmz.values/1e9))
print(msftmz_1pctbgc_diff)

msftmz_u011pctbgc_diff = ((ds_msftmz_u011pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=-1).atlantic_moc.values)/1e9) - ((ds_msftmz_u011pctbgcLR.sel(
    lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=0).atlantic_moc.values)/1e9)
print(msftmz_u011pctbgc_diff)

msftmz_u031pctbgc_diff = ((ds_msftmz_u031pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=-1).atlantic_moc.values)/1e9) - ((ds_msftmz_u031pctbgcLR.sel(
    lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=0).atlantic_moc.values)/1e9)
print(msftmz_u031pctbgc_diff)

msftmz_u051pctbgc_diff = ((ds_msftmz_u051pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=-1).atlantic_moc.values)/1e9) - ((ds_msftmz_u051pctbgcLR.sel(
    lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=0).atlantic_moc.values)/1e9)
print(msftmz_u051pctbgc_diff)

msftmz_g011pctbgc_diff = ((ds_msftmz_g011pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=-1).atlantic_moc.values)/1e9) - ((ds_msftmz_g011pctbgcLR.sel(
    lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=0).atlantic_moc.values)/1e9)
print(msftmz_g011pctbgc_diff)

msftmz_g031pctbgc_diff = ((ds_msftmz_g031pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=-1).atlantic_moc.values)/1e9) - ((ds_msftmz_g031pctbgcLR.sel(
    lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=0).atlantic_moc.values)/1e9)
print(msftmz_g031pctbgc_diff)

msftmz_g051pctbgc_diff = ((ds_msftmz_g051pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=-1).atlantic_moc.values)/1e9) - ((ds_msftmz_g051pctbgcLR.sel(
    lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=0).atlantic_moc.values)/1e9)
print(msftmz_g051pctbgc_diff)

msftmz_u031pct_diff = ((ds_msftmz_u031pctLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=-1).atlantic_moc.values)/1e9) - ((ds_msftmz_u031pctLR.sel(
    lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=0).atlantic_moc.values)/1e9)
print(msftmz_u031pct_diff)

msftmz_u051pct_diff = ((ds_msftmz_u051pctLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=-1).atlantic_moc.values)/1e9) - ((ds_msftmz_u051pctLR.sel(
    lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=0).atlantic_moc.values)/1e9)
print(msftmz_u051pct_diff)
```

```python
# Total diff msftmz scen
msftmz_histbgc_diff = ((ds_msftmz_histbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=-1).atlantic_moc.values)/1e9) - ((ds_msftmz_histbgcLR.sel(
    lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=0).atlantic_moc.values)/1e9)
print(msftmz_histbgc_diff)

msftmz_u03ssp126bgc_diff = ((ds_msftmz_u03ssp126bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=-1).atlantic_moc.values)/1e9) - ((ds_msftmz_u03ssp126bgcLR.sel(
    lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=0).atlantic_moc.values)/1e9)
print(msftmz_u03ssp126bgc_diff)

msftmz_u03ssp245bgc_diff = ((ds_msftmz_u03ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=-1).atlantic_moc.values)/1e9) - ((ds_msftmz_u03ssp245bgcLR.sel(
    lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=0).atlantic_moc.values)/1e9)
print(msftmz_u03ssp245bgc_diff)

msftmz_u03ssp585bgc_diff = ((ds_msftmz_u03ssp585bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=-1).atlantic_moc.values)/1e9) - ((ds_msftmz_u03ssp585bgcLR.sel(
    lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=0).atlantic_moc.values)/1e9)
print(msftmz_u03ssp585bgc_diff)
```

```python
# Total diff msftmz 1pct
msftmz_ssp245bgc_diff = ((ds_msftmz_ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=-1).atlantic_moc.values)/1e9) - ((ds_msftmz_ssp245bgcLR.sel(
    lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=0).atlantic_moc.values)/1e9)
print(msftmz_ssp245bgc_diff)

msftmz_u01ssp245bgc_diff = ((ds_msftmz_u01ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=-1).atlantic_moc.values)/1e9) - ((ds_msftmz_u01ssp245bgcLR.sel(
    lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=0).atlantic_moc.values)/1e9)
print(msftmz_u01ssp245bgc_diff)

msftmz_u03ssp245bgc_diff = ((ds_msftmz_u03ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=-1).atlantic_moc.values)/1e9) - ((ds_msftmz_u03ssp245bgcLR.sel(
    lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=0).atlantic_moc.values)/1e9)
print(msftmz_u03ssp245bgc_diff)

msftmz_u05ssp245bgc_diff = ((ds_msftmz_u05ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=-1).atlantic_moc.values)/1e9) - ((ds_msftmz_u05ssp245bgcLR.sel(
    lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=0).atlantic_moc.values)/1e9)
print(msftmz_u05ssp245bgc_diff)

msftmz_g01ssp245bgc_diff = ((ds_msftmz_g01ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=-1).atlantic_moc.values)/1e9) - ((ds_msftmz_g01ssp245bgcLR.sel(
    lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=0).atlantic_moc.values)/1e9)
print(msftmz_g01ssp245bgc_diff)

msftmz_g03ssp245bgc_diff = ((ds_msftmz_g03ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=-1).atlantic_moc.values)/1e9) - ((ds_msftmz_g03ssp245bgcLR.sel(
    lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=0).atlantic_moc.values)/1e9)
print(msftmz_g03ssp245bgc_diff)

msftmz_g05ssp245bgc_diff = ((ds_msftmz_g05ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=-1).atlantic_moc.values)/1e9) - ((ds_msftmz_g05ssp245bgcLR.sel(
    lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=0).atlantic_moc.values)/1e9)
print(msftmz_g05ssp245bgc_diff)
```

Linear regression

<!-- #raw jupyter={"outputs_hidden": true} -->
from sklearn.linear_model import LinearRegression
X = [msftmz_1pctbgc_diff, msftmz_g011pctbgc_diff, msftmz_u031pctbgc_diff, msftmz_u051pctbgc_diff]
Y = [dissic_1pctbgc_diff*10, dissic_g011pctbgc_diff*10, dissic_u031pctbgc_diff*10, dissic_u051pctbgc_diff*10]
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  
<!-- #endraw -->

```python
def estimate_coef(x, y):
  # number of observations/points
  n = np.size(x)
  # mean of x and y vector
  m_x = np.mean(x)
  m_y = np.mean(y)
  # calculating cross-deviation and deviation about x
  SS_xy = np.sum(y*x) - n*m_y*m_x
  SS_xx = np.sum(x*x) - n*m_x*m_x
  # calculating regression coefficients
  b_1 = SS_xy / SS_xx
  b_0 = m_y - b_1*m_x
 
  return (b_0, b_1)
```

```python
X = np.array([msftmz_1pctbgc_diff, msftmz_g011pctbgc_diff, msftmz_g031pctbgc_diff, msftmz_g051pctbgc_diff, msftmz_u011pctbgc_diff, msftmz_u031pctbgc_diff, msftmz_u051pctbgc_diff])
Y = np.array([dissic_1pctbgc_diff, dissic_g011pctbgc_diff, dissic_g031pctbgc_diff, dissic_g051pctbgc_diff, dissic_u011pctbgc_diff, dissic_u031pctbgc_diff, dissic_u051pctbgc_diff])
b = estimate_coef(X,Y)
y_pred = b[0] + b[1]*X
```

```python
plt.scatter(msftmz_1pctbgc_diff, dissic_1pctbgc_diff, label='1pctbgc', color='dimgrey')
plt.scatter(msftmz_g011pctbgc_diff, dissic_g011pctbgc_diff, label='1pctbgc-g01', color='darkblue')
plt.scatter(msftmz_g031pctbgc_diff, dissic_g031pctbgc_diff, label='1pctbgc-g03', color='dodgerblue')
plt.scatter(msftmz_g051pctbgc_diff, dissic_g051pctbgc_diff, label='1pctbgc-g05', color='lightblue')
plt.scatter(msftmz_u011pctbgc_diff, dissic_u011pctbgc_diff, label='1pctbgc-u01', color='peachpuff')
plt.scatter(msftmz_u031pctbgc_diff, dissic_u031pctbgc_diff, label='1pctbgc-u03', color='salmon')
plt.scatter(msftmz_u051pctbgc_diff, dissic_u051pctbgc_diff, label='1pctbgc-u05', color='darkred')
plt.plot(X, y_pred, color = "k")
plt.xlabel("$\Delta$AMOC [Sv]")
plt.ylabel("$\Delta$C [PgC]")
plt.text(-13,633, f"slope={b[1].round(3)}",fontsize=11, rotation=34)
plt.legend()
#plt.text(-11,635, "2.57", fontweight='bold')
#plt.savefig('../plots/carbon_vs_amoc_regression.png', transparent=True)
```

```python
X = np.array([msftmz_u03ssp126bgc_diff, msftmz_u03ssp245bgc_diff, msftmz_u03ssp245bgc_diff])
Y = np.array([dissic_u03ssp126bgc_diff, dissic_u03ssp245bgc_diff, dissic_u03ssp585bgc_diff])
b = estimate_coef(X,Y)
y_pred = b[0] + b[1]*X
```

```python
plt.scatter(msftmz_u03ssp126bgc_diff, dissic_u03ssp126bgc_diff, label='ssp126bgc-u03', color=colors_scen['ssp126'])
plt.scatter(msftmz_u03ssp245bgc_diff, dissic_u03ssp245bgc_diff, label='ssp245bgc-u03', color=colors_scen['ssp245'])
plt.scatter(msftmz_u03ssp585bgc_diff, dissic_u03ssp585bgc_diff, label='ssp585bgc-u03', color=colors_scen['ssp585'])
#plt.plot(X, y_pred, color = "k")
plt.xlabel("$\Delta$AMOC [Sv]")
plt.ylabel("$\Delta$C [PgC]")
plt.xlim(-16,-1)
#plt.ylim(625,665)
#plt.text(-13,633, f"slope={b[1].round(3)}",fontsize=11, rotation=34)
plt.legend()
#plt.text(-11,635, "2.57", fontweight='bold')
#plt.savefig('../plots/carbon_vs_amoc_regression.png', transparent=True)
```

```python
X = np.array([msftmz_ssp245bgc_diff, msftmz_g01ssp245bgc_diff, msftmz_g03ssp245bgc_diff, msftmz_g05ssp245bgc_diff, msftmz_u01ssp245bgc_diff, msftmz_u03ssp245bgc_diff, msftmz_u05ssp245bgc_diff])
Y = np.array([dissic_ssp245bgc_diff, dissic_g01ssp245bgc_diff, dissic_g03ssp245bgc_diff, dissic_g05ssp245bgc_diff, dissic_u01ssp245bgc_diff, dissic_u03ssp245bgc_diff, dissic_u05ssp245bgc_diff])
b = estimate_coef(X,Y)
y_pred = b[0] + b[1]*X
```

```python
plt.scatter(msftmz_ssp245bgc_diff, dissic_ssp245bgc_diff, label='ssp245bgc', color='dimgrey')
plt.scatter(msftmz_g01ssp245bgc_diff, dissic_g01ssp245bgc_diff, label='ssp245bgc-g01', color='darkblue')
plt.scatter(msftmz_g03ssp245bgc_diff, dissic_g03ssp245bgc_diff, label='ssp245bgc-g03', color='dodgerblue')
plt.scatter(msftmz_g05ssp245bgc_diff, dissic_g05ssp245bgc_diff, label='ssp245bgc-g05', color='lightblue')
plt.scatter(msftmz_u01ssp245bgc_diff, dissic_u01ssp245bgc_diff, label='ssp245bgc-u01', color='peachpuff')
plt.scatter(msftmz_u03ssp245bgc_diff, dissic_u03ssp245bgc_diff, label='ssp245bgc-u03', color='salmon')
plt.scatter(msftmz_u05ssp245bgc_diff, dissic_u05ssp245bgc_diff, label='ssp245bgc-u05', color='darkred')
plt.plot(X, y_pred, color = "k")
plt.xlabel("$\Delta$AMOC [Sv]")
plt.ylabel("$\Delta$C [PgC]")
plt.text(-9, 277, f"slope={b[1].round(3)}",fontsize=11, rotation=34)
plt.legend()
#plt.text(-11,635, "2.57", fontweight='bold')
#plt.savefig('../plots/carbon_vs_amoc_regression.png', transparent=True)
```

### Yearly differences plots

```python
ds_dissic_1pctbgc_year = ds_dissic_1pctbgc.dissic.groupby('time.year').mean('time')
```

```python jupyter={"outputs_hidden": true}
ds_dissic_1pctbgc_year.load()
```

```python jupyter={"outputs_hidden": true}
ds_dissic_u051pctbgcLR.load()
ds_dissic_u031pctbgcLR.load()
ds_dissic_u011pctbgcLR.load()
ds_dissic_g011pctbgcLR.load()
ds_dissic_g031pctbgcLR.load()
ds_dissic_g051pctbgcLR.load()
```

```python
diff_dissic_u051pctbgc = []
diff_dissic_u031pctbgc = []
diff_dissic_u011pctbgc = []
diff_dissic_g011pctbgc = []
diff_dissic_g031pctbgc = []
diff_dissic_g051pctbgc = []
diff_dissic_1pctbgc    = []
for year in tqdm(range(1,140)):
    diff_dissic_u051pctbgc.append(
        ((ds_dissic_u051pctbgcLR.dissic.isel(time=np.arange(year,year+1))*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        -(ds_dissic_u051pctbgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        ).sum(dim='y').sum(dim='x').values)
    diff_dissic_u031pctbgc.append(
        ((ds_dissic_u031pctbgcLR.dissic.isel(time=np.arange(year,year+1))*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        -(ds_dissic_u031pctbgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        ).sum(dim='y').sum(dim='x').values)
    diff_dissic_u011pctbgc.append(
        ((ds_dissic_u011pctbgcLR.dissic.isel(time=np.arange(year,year+1))*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        -(ds_dissic_u011pctbgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        ).sum(dim='y').sum(dim='x').values)
    diff_dissic_g011pctbgc.append(
        ((ds_dissic_g011pctbgcLR.dissic.isel(time=np.arange(year,year+1))*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        -(ds_dissic_g011pctbgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        ).sum(dim='y').sum(dim='x').values)
    diff_dissic_g031pctbgc.append(
        ((ds_dissic_g031pctbgcLR.dissic.isel(time=np.arange(year,year+1))*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        -(ds_dissic_g031pctbgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        ).sum(dim='y').sum(dim='x').values)
    diff_dissic_g051pctbgc.append(
        ((ds_dissic_g051pctbgcLR.dissic.isel(time=np.arange(year,year+1))*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        -(ds_dissic_g051pctbgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        ).sum(dim='y').sum(dim='x').values)
    diff_dissic_1pctbgc.append(
        ((ds_dissic_1pctbgc_year.isel(year=np.arange(year,year+1))*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
        -(ds_dissic_1pctbgc_year.isel(year=0)*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev')
        ).sum(dim='j').sum(dim='i').values)
```

```python jupyter={"outputs_hidden": true}
ds_dissic_u03ssp126bgcLR.load()
ds_dissic_u03ssp245bgcLR.load()
ds_dissic_u03ssp585bgcLR.load()
```

```python
diff_dissic_ = []
diff_dissic_u03ssp126bgc = []
diff_dissic_u03ssp245bgc = []
diff_dissic_u03ssp585bgc = []

for year in tqdm(range(1,85)):
    diff_dissic_u03ssp126bgc.append(
        ((ds_dissic_u03ssp126bgcLR.dissic.isel(time=np.arange(year,year+1))*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        -(ds_dissic_u03ssp126bgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        ).sum(dim='y').sum(dim='x').values)
    diff_dissic_u03ssp245bgc.append(
        ((ds_dissic_u03ssp245bgcLR.dissic.isel(time=np.arange(year,year+1))*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        -(ds_dissic_u03ssp245bgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        ).sum(dim='y').sum(dim='x').values)
    diff_dissic_u03ssp585bgc.append(
        ((ds_dissic_u03ssp585bgcLR.dissic.isel(time=np.arange(year,year+1))*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        -(ds_dissic_u03ssp585bgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        ).sum(dim='y').sum(dim='x').values)
```

```python
ds_dissic_u05ssp245bgcLR.load()
ds_dissic_u03ssp245bgcLR.load()
ds_dissic_u01ssp245bgcLR.load()
ds_dissic_g01ssp245bgcLR.load()
ds_dissic_g03ssp245bgcLR.load()
ds_dissic_g05ssp245bgcLR.load()
ds_dissic_ssp245bgcLR.load()
```

```python
diff_dissic_u05ssp245bgc = []
diff_dissic_u03ssp245bgc = []
diff_dissic_u01ssp245bgc = []
diff_dissic_g01ssp245bgc = []
diff_dissic_g03ssp245bgc = []
diff_dissic_g05ssp245bgc = []
diff_dissic_ssp245bgc    = []
for year in tqdm(range(1,85)):
    diff_dissic_u05ssp245bgc.append(
        ((ds_dissic_u05ssp245bgcLR.dissic.isel(time=np.arange(year,year+1))*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        -(ds_dissic_u05ssp245bgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        ).sum(dim='y').sum(dim='x').values)
    diff_dissic_u03ssp245bgc.append(
        ((ds_dissic_u03ssp245bgcLR.dissic.isel(time=np.arange(year,year+1))*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        -(ds_dissic_u03ssp245bgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        ).sum(dim='y').sum(dim='x').values)
    diff_dissic_u01ssp245bgc.append(
        ((ds_dissic_u01ssp245bgcLR.dissic.isel(time=np.arange(year,year+1))*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        -(ds_dissic_u01ssp245bgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        ).sum(dim='y').sum(dim='x').values)
    diff_dissic_g01ssp245bgc.append(
        ((ds_dissic_g01ssp245bgcLR.dissic.isel(time=np.arange(year,year+1))*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        -(ds_dissic_g01ssp245bgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        ).sum(dim='y').sum(dim='x').values)
    diff_dissic_g03ssp245bgc.append(
        ((ds_dissic_g03ssp245bgcLR.dissic.isel(time=np.arange(year,year+1))*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        -(ds_dissic_g03ssp245bgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        ).sum(dim='y').sum(dim='x').values)
    diff_dissic_g05ssp245bgc.append(
        ((ds_dissic_g05ssp245bgcLR.dissic.isel(time=np.arange(year,year+1))*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        -(ds_dissic_g05ssp245bgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        ).sum(dim='y').sum(dim='x').values)
    diff_dissic_ssp245bgc.append(
        ((ds_dissic_ssp245bgcLR.dissic.isel(time=np.arange(year,year+1))*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        -(ds_dissic_ssp245bgcLR.dissic.isel(time=0)*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth')
        ).sum(dim='y').sum(dim='x').values)
```

```python
fig, axs = plt.subplots(2, 1, figsize=(7,7), sharex=True)
fig.subplots_adjust(hspace=0)

axs[0].plot(np.arange(1851, 1990), diff_dissic_1pctbgc, label ='1pct-bgc', color='dimgrey')
axs[0].plot(np.arange(1851, 1990), diff_dissic_g011pctbgc, label ='g01-1pct-bgc', color='darkblue')
axs[0].plot(np.arange(1851, 1990), diff_dissic_g031pctbgc,  label='g03-1pct-bgc', color='dodgerblue')
axs[0].plot(np.arange(1851, 1990), diff_dissic_g051pctbgc,  label='g05-1pct-bgc', color='lightblue')
axs[0].plot(np.arange(1851, 1990), diff_dissic_u011pctbgc, label ='u01-1pct-bgc', color='peachpuff')
axs[0].plot(np.arange(1851, 1990), diff_dissic_u031pctbgc, label ='u03-1pct-bgc', color='salmon')
axs[0].plot(np.arange(1851, 1990), diff_dissic_u051pctbgc, label ='u05-1pct-bgc', color='darkred')
axs[0].legend(loc='lower center')
axs[0].set_xlabel("")
axs[0].set_ylabel('$\Delta$C [PgC]')
axs[0].tick_params(axis='y')
axs[0].spines['bottom'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].set_xticks([])
axs[0].get_xaxis().set_visible(False)
(ds_msftmz_1pctbgc.isel(basin=1).sel(lat=26.5).sel(lev=1020).groupby('time.year').mean('time').msftmz/1e9).plot(ax=axs[1], color='dimgrey')
(ds_msftmz_g011pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='darkblue')
(ds_msftmz_g031pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='dodgerblue')
(ds_msftmz_g051pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='lightblue')
(ds_msftmz_u011pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='peachpuff')
(ds_msftmz_u031pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='salmon')
(ds_msftmz_u051pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='darkred')
axs[1].tick_params(axis='y')
axs[1].set_ylabel("AMOC strength [Sv]")
axs[1].spines['left'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].set_xlabel("Time [year]")
axs[1].yaxis.set_label_position("right")
axs[1].yaxis.tick_right()
axs[1].set_title("")
axs[1].set_xticks([1850, 1875, 1900, 1925, 1950, 1975, 2000], [0, 25, 50, 75, 100, 125, 150])
fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.show()
#plt.savefig('../plots/hosing_amoc_carbon_timeseries.png', transparent=True)
```

```python
fig, axs = plt.subplots(2, 1, figsize=(7,7), sharex=True)
fig.subplots_adjust(hspace=0)
plt.rcParams['font.size'] = 13

axs[0].plot(np.arange(2015, 2099), diff_dissic_u03ssp126bgc, label ='ssp126', color=colors_scen['ssp126'])
axs[0].plot(np.arange(2015, 2099), diff_dissic_u03ssp245bgc, label ='ssp245', color=colors_scen['ssp245'])
axs[0].plot(np.arange(2015, 2099), diff_dissic_u03ssp585bgc, label ='ssp585', color=colors_scen['ssp585'])
axs[0].set_ylabel('$\Delta$C in ocean [PgC]')
axs[0].set_title("")
axs[0].set_xlabel("")
axs[0].tick_params(axis='y')
axs[0].spines['bottom'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].set_xticks([])
axs[0].get_xaxis().set_visible(False)
axs[0].legend(loc=3)


(ds_msftmz_u03ssp126bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color=colors_scen['ssp126'], label ='ssp126')
(ds_msftmz_u03ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color=colors_scen['ssp245'], label ='ssp245')
(ds_msftmz_u03ssp585bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color=colors_scen['ssp585'], label ='ssp585')
axs[1].set_ylabel("AMOC strength [Sv]")
axs[1].tick_params(axis='y')
axs[1].spines['left'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].set_xlabel("Time [year]")
axs[1].yaxis.set_label_position("right")
axs[1].yaxis.tick_right()
axs[1].set_title("")
axs[1].set_xticks([2020, 2040, 2060, 2080])
fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.show()
#plt.savefig('../plots/hosing_amoc_carbon_timeseries.png', transparent=True)
```

```python
diff_dissic_1pctbgc = np.array(diff_dissic_1pctbgc)
diff_dissic_g011pctbgc = np.array(diff_dissic_g011pctbgc)
diff_dissic_g031pctbgc = np.array(diff_dissic_g031pctbgc)
diff_dissic_g051pctbgc = np.array(diff_dissic_g051pctbgc)
diff_dissic_u011pctbgc = np.array(diff_dissic_u011pctbgc)
diff_dissic_u031pctbgc = np.array(diff_dissic_u031pctbgc)
diff_dissic_u051pctbgc = np.array(diff_dissic_u051pctbgc)
```

```python
diff_dissic_u03ssp126bgc = np.array(diff_dissic_u03ssp126bgc)
diff_dissic_u03ssp245bgc = np.array(diff_dissic_u03ssp245bgc)
diff_dissic_u03ssp585bgc = np.array(diff_dissic_u03ssp585bgc)
```

```python
diff_dissic_ssp245bgc = np.array(diff_dissic_ssp245bgc)
diff_dissic_g01ssp245bgc = np.array(diff_dissic_g01ssp245bgc)
diff_dissic_g03ssp245bgc = np.array(diff_dissic_g03ssp245bgc)
diff_dissic_g05ssp245bgc = np.array(diff_dissic_g05ssp245bgc)
diff_dissic_u01ssp245bgc = np.array(diff_dissic_u01ssp245bgc)
diff_dissic_u03ssp245bgc = np.array(diff_dissic_u03ssp245bgc)
diff_dissic_u05ssp245bgc = np.array(diff_dissic_u05ssp245bgc)
```

```python
fig, axs = plt.subplots(2, 1, figsize=(7,7), sharex=True)
fig.subplots_adjust(hspace=0)
plt.rcParams['font.size'] = 13

#axs[0].plot(np.arange(1851, 1990), -(diff_dissic_1pctbgc-diff_dissic_1pctbgc), label ='1pct-bgc', color='dimgrey')
axs[0].plot(np.arange(1851, 1990), -(diff_dissic_1pctbgc-diff_dissic_g011pctbgc), label ='g01-1pct-bgc', color='lightblue')
axs[0].plot(np.arange(1851, 1990), -(diff_dissic_1pctbgc-diff_dissic_g031pctbgc), label ='g03-1pct-bgc', color='dodgerblue')
axs[0].plot(np.arange(1851, 1990), -(diff_dissic_1pctbgc-diff_dissic_g051pctbgc), label ='g05-1pct-bgc', color='darkblue')
axs[0].plot(np.arange(1851, 1990), -(diff_dissic_1pctbgc-diff_dissic_u011pctbgc), label ='u01-1pct-bgc', color='peachpuff')
axs[0].plot(np.arange(1851, 1990), -(diff_dissic_1pctbgc-diff_dissic_u031pctbgc), label ='u03-1pct-bgc', color='salmon')
axs[0].plot(np.arange(1851, 1990), -(diff_dissic_1pctbgc-diff_dissic_u051pctbgc), label ='u05-1pct-bgc', color='darkred')
axs[0].set_ylabel('$\Delta$C in ocean w.r.t. 1pct-bgc [PgC]')
axs[0].set_title("")
axs[0].set_xlabel("")
axs[0].tick_params(axis='y')
axs[0].spines['bottom'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].set_xticks([])
axs[0].get_xaxis().set_visible(False)
#axs[0].legend(loc=3)

#(ds_msftmz_1pctbgc.isel(basin=1).sel(lat=26.5).sel(lev=1020).groupby('time.year').mean('time').msftmz/1e9).plot(ax=axs[1], color='dimgrey', label ='1pct-bgc')
(ds_msftmz_g011pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='lightblue', label ='g01-1pct-bgc')
(ds_msftmz_g031pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='dodgerblue', label ='g03-1pct-bgc')
(ds_msftmz_g051pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='darkblue', label ='g05-1pct-bgc')
(ds_msftmz_u011pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='peachpuff', label ='u01-1pct-bgc')
(ds_msftmz_u031pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='salmon', label ='u03-1pct-bgc')
(ds_msftmz_u051pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='darkred', label ='u05-1pct-bgc')
axs[1].set_ylabel("AMOC strength [Sv]")
axs[1].tick_params(axis='y')
axs[1].spines['left'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].set_xlabel("Time [year]")
axs[1].yaxis.set_label_position("right")
axs[1].yaxis.tick_right()
axs[1].set_title("")
axs[1].set_xticks([1850, 1875, 1900, 1925, 1950, 1975, 2000], [0, 25, 50, 75, 100, 125, 150])
fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.show()
plt.savefig('/work/uo1075/m300817/carbon_amoc/plots/hosing_amoc_carbon_timeseries.pdf', transparent=True)
```

```python
fig, axs = plt.subplots(2, 1, figsize=(7,7), sharex=True)
fig.subplots_adjust(hspace=0)
plt.rcParams['font.size'] = 13

axs[0].plot(np.arange(1851, 1990), -(diff_dissic_1pctbgc-diff_dissic_1pctbgc), label ='1pct-bgc', color='dimgrey')
axs[0].plot(np.arange(1851, 1990), -(diff_dissic_1pctbgc-diff_dissic_g011pctbgc), label ='g01-1pct-bgc', color='lightblue')
axs[0].plot(np.arange(1851, 1990), -(diff_dissic_1pctbgc-diff_dissic_g031pctbgc), label ='g03-1pct-bgc', color='dodgerblue')
axs[0].plot(np.arange(1851, 1990), -(diff_dissic_1pctbgc-diff_dissic_g051pctbgc), label ='g05-1pct-bgc', color='darkblue')
axs[0].plot(np.arange(1851, 1990), -(diff_dissic_1pctbgc-diff_dissic_u011pctbgc), label ='u01-1pct-bgc', color='peachpuff')
axs[0].plot(np.arange(1851, 1990), -(diff_dissic_1pctbgc-diff_dissic_u031pctbgc), label ='u03-1pct-bgc', color='salmon')
axs[0].plot(np.arange(1851, 1990), -(diff_dissic_1pctbgc-diff_dissic_u051pctbgc), label ='u05-1pct-bgc', color='darkred')
axs[0].set_ylabel('$\Delta$C in ocean w.r.t. 1pct-bgc [PgC]')
axs[0].set_title("")
axs[0].set_xlabel("")
axs[0].tick_params(axis='y')
axs[0].spines['bottom'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].set_xticks([])
axs[0].get_xaxis().set_visible(False)
axs[0].legend(loc=3)

(ds_msftmz_1pctbgc.isel(basin=1).sel(lat=26.5).sel(lev=1020).groupby('time.year').mean('time').msftmz/1e9).plot(ax=axs[1], color='dimgrey', label ='1pct-bgc')
(ds_msftmz_g011pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='lightblue', label ='g01-1pct-bgc')
(ds_msftmz_g031pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='dodgerblue', label ='g03-1pct-bgc')
(ds_msftmz_g051pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='darkblue', label ='g05-1pct-bgc')
(ds_msftmz_u011pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='peachpuff', label ='u01-1pct-bgc')
(ds_msftmz_u031pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='salmon', label ='u03-1pct-bgc')
(ds_msftmz_u051pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='darkred', label ='u05-1pct-bgc')
axs[1].set_ylabel("AMOC strength [Sv]")
axs[1].tick_params(axis='y')
axs[1].spines['left'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].set_xlabel("Time [year]")
axs[1].yaxis.set_label_position("right")
axs[1].yaxis.tick_right()
axs[1].set_title("")
axs[1].set_xticks([1850, 1875, 1900, 1925, 1950, 1975, 2000], [0, 25, 50, 75, 100, 125, 150])
fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.show()
#plt.savefig('../plots/hosing_amoc_carbon_timeseries.png', transparent=True)
```

```python
fig, axs = plt.subplots(2, 1, figsize=(7,7), sharex=True)
fig.subplots_adjust(hspace=0)
plt.rcParams['font.size'] = 13

axs[0].plot(np.arange(2015, 2099), -(diff_dissic_ssp245bgc-diff_dissic_ssp245bgc), label ='ssp245-bgc', color='dimgrey')
axs[0].plot(np.arange(2015, 2099), -(diff_dissic_ssp245bgc-diff_dissic_g01ssp245bgc), label ='g01-ssp245-bgc', color='lightblue')
axs[0].plot(np.arange(2015, 2099), -(diff_dissic_ssp245bgc-diff_dissic_g03ssp245bgc), label ='g03-ssp245-bgc', color='dodgerblue')
axs[0].plot(np.arange(2015, 2099), -(diff_dissic_ssp245bgc-diff_dissic_g05ssp245bgc), label ='g05-ssp245-bgc', color='darkblue')
axs[0].plot(np.arange(2015, 2099), -(diff_dissic_ssp245bgc-diff_dissic_u01ssp245bgc), label ='u01-ssp245-bgc', color='peachpuff')
axs[0].plot(np.arange(2015, 2099), -(diff_dissic_ssp245bgc-diff_dissic_u03ssp245bgc), label ='u03-ssp245-bgc', color='salmon')
axs[0].plot(np.arange(2015, 2099), -(diff_dissic_ssp245bgc-diff_dissic_u05ssp245bgc), label ='u05-ssp245-bgc', color='darkred')
axs[0].set_ylabel('$\Delta$C in ocean w.r.t. ssp245-bgc [PgC]')
axs[0].set_title("")
axs[0].set_xlabel("")
axs[0].tick_params(axis='y')
axs[0].spines['bottom'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].set_xticks([])
axs[0].get_xaxis().set_visible(False)
axs[0].legend(loc=3)

(ds_msftmz_ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='dimgrey', label ='ssp245-bgc')
(ds_msftmz_g01ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='lightblue', label ='g01-ssp245-bgc')
(ds_msftmz_g03ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='dodgerblue', label ='g03-ssp245-bgc')
(ds_msftmz_g05ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='darkblue', label ='g05-ssp245-bgc')
(ds_msftmz_u01ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='peachpuff', label ='u01-ssp245-bgc')
(ds_msftmz_u03ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='salmon', label ='u03-ssp245-bgc')
(ds_msftmz_u05ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[1], color='darkred', label ='u05-ssp245-bgc')
axs[1].set_ylabel("AMOC strength [Sv]")
axs[1].tick_params(axis='y')
axs[1].spines['left'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].set_xlabel("Time [year]")
axs[1].yaxis.set_label_position("right")
axs[1].yaxis.tick_right()
axs[1].set_title("")
axs[1].set_xticks([2020, 2040, 2060, 2080, 2100], [2020, 2040, 2060, 2080, 2100])
fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.show()
#plt.savefig('../plots/hosing_amoc_carbon_timeseries.png', transparent=True)
```

```python
fig, axs = plt.subplots(2, 1, figsize=(7,7), sharex=True)
fig.subplots_adjust(hspace=0)

(ds_msftmz_1pctbgc.isel(basin=1).sel(lat=26.5).sel(lev=1020).groupby('time.year').mean('time').msftmz/1e9).plot(ax=axs[0], color='dimgrey', label ='1pct-bgc')
(ds_msftmz_g011pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[0], color='darkblue', label ='g01-1pct-bgc')
(ds_msftmz_g031pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[0], color='dodgerblue', label ='g03-1pct-bgc')
(ds_msftmz_g051pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[0], color='lightblue', label ='g05-1pct-bgc')
(ds_msftmz_u011pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[0], color='peachpuff', label ='u01-1pct-bgc')
(ds_msftmz_u031pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[0], color='salmon', label ='u03-1pct-bgc')
(ds_msftmz_u051pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).plot(ax=axs[0], color='darkred', label ='u05-1pct-bgc')
#axs[0].legend(loc='lower center')
axs[0].set_title("")
axs[0].set_xlabel("")
axs[0].set_ylabel("AMOC strength [Sv]")
axs[0].tick_params(axis='y')
axs[0].spines['bottom'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].set_xticks([])
axs[0].get_xaxis().set_visible(False)

axs[1].plot(np.arange(1851, 1990), diff_dissic_1pctbgc-diff_dissic_1pctbgc, label ='1pct-bgc', color='dimgrey')
axs[1].plot(np.arange(1851, 1990), diff_dissic_1pctbgc-diff_dissic_g011pctbgc, label ='g01-1pct-bgc', color='darkblue')
axs[1].plot(np.arange(1851, 1990), diff_dissic_1pctbgc-diff_dissic_g031pctbgc, label ='g03-1pct-bgc', color='dodgerblue')
axs[1].plot(np.arange(1851, 1990), diff_dissic_1pctbgc-diff_dissic_g051pctbgc, label ='g05-1pct-bgc', color='lightblue')
axs[1].plot(np.arange(1851, 1990), diff_dissic_1pctbgc-diff_dissic_u011pctbgc, label ='u01-1pct-bgc', color='peachpuff')
axs[1].plot(np.arange(1851, 1990), diff_dissic_1pctbgc-diff_dissic_u031pctbgc, label ='u03-1pct-bgc', color='salmon')
axs[1].plot(np.arange(1851, 1990), diff_dissic_1pctbgc-diff_dissic_u051pctbgc, label ='u05-1pct-bgc', color='darkred')
axs[1].tick_params(axis='y')
axs[1].set_ylabel('$\Delta$C w.r.t. 1pct-bgc [PgC]')
axs[1].spines['left'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].set_xlabel("Time [year]")
axs[1].yaxis.set_label_position("right")
axs[1].yaxis.tick_right()
axs[1].set_title("")
axs[1].legend(loc='upper center')
axs[1].set_xticks([1850, 1875, 1900, 1925, 1950, 1975, 2000], [0, 25, 50, 75, 100, 125, 150])
fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.show()
#plt.savefig('../plots/hosing_amoc_carbon_timeseries.png', transparent=True)
```

```python
diff_dissic_u051pctbgc_array=[]
diff_dissic_u031pctbgc_array=[]
diff_dissic_u011pctbgc_array=[]
diff_dissic_g051pctbgc_array=[]
diff_dissic_g031pctbgc_array=[]
diff_dissic_g011pctbgc_array=[]
diff_dissic_1pctbgc_array=[]
for i in range(len(diff_dissic_u051pctbgc)):
    diff_dissic_u051pctbgc_array.append(diff_dissic_u051pctbgc[i].item())
    diff_dissic_u031pctbgc_array.append(diff_dissic_u031pctbgc[i].item())
    diff_dissic_u011pctbgc_array.append(diff_dissic_u011pctbgc[i].item())
    diff_dissic_g011pctbgc_array.append(diff_dissic_g011pctbgc[i].item())
    diff_dissic_g031pctbgc_array.append(diff_dissic_g031pctbgc[i].item())
    diff_dissic_g051pctbgc_array.append(diff_dissic_g051pctbgc[i].item())
    diff_dissic_1pctbgc_array.append(diff_dissic_1pctbgc[i].item())
```

```python
diff_dissic_u05ssp245bgc_array=[]
diff_dissic_u03ssp245bgc_array=[]
diff_dissic_u01ssp245bgc_array=[]
diff_dissic_g05ssp245bgc_array=[]
diff_dissic_g03ssp245bgc_array=[]
diff_dissic_g01ssp245bgc_array=[]
diff_dissic_ssp245bgc_array=[]
for i in range(len(diff_dissic_u05ssp245bgc)):
    diff_dissic_u05ssp245bgc_array.append(diff_dissic_u05ssp245bgc[i].item())
    diff_dissic_u03ssp245bgc_array.append(diff_dissic_u03ssp245bgc[i].item())
    diff_dissic_u01ssp245bgc_array.append(diff_dissic_u01ssp245bgc[i].item())
    diff_dissic_g01ssp245bgc_array.append(diff_dissic_g01ssp245bgc[i].item())
    diff_dissic_g03ssp245bgc_array.append(diff_dissic_g03ssp245bgc[i].item())
    diff_dissic_g05ssp245bgc_array.append(diff_dissic_g05ssp245bgc[i].item())
    diff_dissic_ssp245bgc_array.append(diff_dissic_ssp245bgc[i].item())
```

```python
fig, axs = plt.subplots(figsize=(14, 6))
axs.plot(np.arange(1851, 1990), diff_dissic_u051pctbgc_array/(ds_msftmz_u051pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=np.arange(0,139)).atlantic_moc.values/1e9), color='darkred')
axs.plot(np.arange(1851, 1990), diff_dissic_u031pctbgc_array/(ds_msftmz_u031pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=np.arange(0,139)).atlantic_moc.values/1e9), color='salmon')
axs.plot(np.arange(1851, 1990), diff_dissic_u011pctbgc_array/(ds_msftmz_u011pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=np.arange(0,139)).atlantic_moc.values/1e9), color='peachpuff')
axs.plot(np.arange(1851, 1990), diff_dissic_g011pctbgc_array/(ds_msftmz_g011pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=np.arange(0,139)).atlantic_moc.values/1e9), color='lightblue')
axs.plot(np.arange(1851, 1990), diff_dissic_g031pctbgc_array/(ds_msftmz_g031pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=np.arange(0,139)).atlantic_moc.values/1e9), color='dodgerblue')
axs.plot(np.arange(1851, 1990), diff_dissic_g051pctbgc_array/(ds_msftmz_g051pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').isel(year=np.arange(0,139)).atlantic_moc.values/1e9), color='darkblue')
axs.plot(np.arange(1851, 1990), diff_dissic_1pctbgc_array/(ds_msftmz_1pctbgc.sel(lat=26.5).sel(lev=1020).isel(basin=1).groupby('time.year').mean('time').isel(year=np.arange(0,139)).msftmz.values/1e9), color='dimgrey')
axs.legend(loc='lower center')
axs.set_ylabel('Difference Ocean Carbon/AMOC strength [PgC/Sv]', color='k')
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.set_xlabel("Time [year]")
plt.show()
```

u03

```python
dissic_u031pct_diff/total_tas_diff['hosing_naa03Sv_1pct-LR']
```

```python
(dissic_u031pct_diff - dissic_u031pctbgc_diff)/total_tas_diff['hosing_naa03Sv_1pct-LR']
```

```python
(dissic_u031pct_diff - dissic_u031pctbgc_diff)/(total_tas_diff['hosing_naa03Sv_1pct-LR']*msftmz_u031pct_diff)
```

u05

```python
(dissic_u051pct_diff - dissic_u051pctbgc_diff)/total_tas_diff['hosing_naa05Sv_1pct-LR']
```

```python
(dissic_u051pct_diff - dissic_u051pctbgc_diff)/(total_tas_diff['hosing_naa05Sv_1pct-LR']*msftmz_u051pct_diff)
```

# Export data to csv

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## 1pct-bgc
<!-- #endregion -->

<!-- #raw -->
Dissic totals
<!-- #endraw -->

```python
(ds_dissic_u051pctbgcLR.dissic*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth').sum(dim='y').sum(dim='x').values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_dissic_u051pctbgcLR.csv', sep=',')
```

```python
(ds_dissic_u031pctbgcLR.dissic*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth').sum(dim='y').sum(dim='x').values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_dissic_u031pctbgcLR.csv', sep=',')
```

```python
(ds_dissic_u011pctbgcLR.dissic*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth').sum(dim='y').sum(dim='x').values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_dissic_u011pctbgcLR.csv', sep=',')
```

```python
(ds_dissic_g011pctbgcLR.dissic*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth').sum(dim='y').sum(dim='x').values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_dissic_g011pctbgcLR.csv', sep=',')
```

```python
(ds_dissic_g031pctbgcLR.dissic*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth').sum(dim='y').sum(dim='x').values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_dissic_g031pctbgcLR.csv', sep=',')
```

```python
(ds_dissic_g051pctbgcLR.dissic*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth').sum(dim='y').sum(dim='x').values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_dissic_g051pctbgcLR.csv', sep=',')
```

```python
(ds_dissic_1pctbgc.groupby('time.year').mean().dissic*ds_volcello.volcello*g_per_molC/1e15).isel(lev=np.arange(0,40)).sum(dim='lev').sum(dim='i').sum(dim='j').values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_dissic_1pctbgcLR.csv', sep=',')
```

```python
Dissic diff
```

```python
np.array(diff_dissic_u051pctbgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_dissic_u051pctbgcLR.csv', sep=',')
```

```python
np.array(diff_dissic_u031pctbgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_dissic_u031pctbgcLR.csv', sep=',')
```

```python
np.array(diff_dissic_u011pctbgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_dissic_u011pctbgcLR.csv', sep=',')
```

```python
np.array(diff_dissic_g011pctbgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_dissic_g011pctbgcLR.csv', sep=',')
```

```python
np.array(diff_dissic_g031pctbgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_dissic_g031pctbgcLR.csv', sep=',')
```

```python
np.array(diff_dissic_g051pctbgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_dissic_g051pctbgcLR.csv', sep=',')
```

```python
np.array(diff_dissic_1pctbgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_dissic_1pctbgcLR.csv', sep=',')
```

```python

```

```python
msftmz totals 
```

```python
(ds_msftmz_u051pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_amoc26.5N_u051pctbgcLR.csv', sep=',')
```

```python
(ds_msftmz_u031pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_amoc26.5N_u031pctbgcLR.csv', sep=',')
```

```python
(ds_msftmz_u011pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_amoc26.5N_u011pctbgcLR.csv', sep=',')
```

```python
(ds_msftmz_g011pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_amoc26.5N_g011pctbgcLR.csv', sep=',')
```

```python
(ds_msftmz_g031pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_amoc26.5N_g031pctbgcLR.csv', sep=',')
```

```python
(ds_msftmz_g051pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_amoc26.5N_g051pctbgcLR.csv', sep=',')
```

```python
(ds_msftmz_1pctbgc.isel(basin=1).sel(lat=26.5).sel(lev=1020).groupby('time.year').mean('time').msftmz/1e9).values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_amoc26.5N_1pctbgcLR.csv', sep=',')
```

```python
msftmz diff
```

```python
ds_msftmz_u051pctbgcLR.load()
ds_msftmz_u031pctbgcLR.load()
ds_msftmz_u011pctbgcLR.load()
ds_msftmz_g011pctbgcLR.load()
ds_msftmz_g031pctbgcLR.load()
ds_msftmz_g011pctbgcLR.load()
ds_msftmz_1pctbgc.load()
```

```python
# not the same as msftmt_..._diff, which just calculate the total difference at the end of the period
from tqdm import tqdm
diff_msftmz_u051pctbgc = []
diff_msftmz_u031pctbgc = []
diff_msftmz_u011pctbgc = []
diff_msftmz_g011pctbgc = []
diff_msftmz_g031pctbgc = []
diff_msftmz_g051pctbgc = []
diff_msftmz_1pctbgc    = []
amoc26_u051pctbgc_yearly = ds_msftmz_u051pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time')
amoc26_u031pctbgc_yearly = ds_msftmz_u031pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time')
amoc26_u011pctbgc_yearly = ds_msftmz_u011pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time')
amoc26_g011pctbgc_yearly = ds_msftmz_g011pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time')
amoc26_g031pctbgc_yearly = ds_msftmz_g031pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time')
amoc26_g051pctbgc_yearly = ds_msftmz_g051pctbgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time')
amoc26_1pctbgc_yearly = ds_msftmz_1pctbgc.sel(lat=26.5).sel(lev=1020).isel(basin=1).groupby('time.year').mean('time')

for year in tqdm(range(1,140)): 
    diff_msftmz_u051pctbgc.append(
        ((amoc26_u051pctbgc_yearly.isel(year=np.arange(year,year+1)).atlantic_moc.values)/1e9)
        -((amoc26_u051pctbgc_yearly.isel(year=0).atlantic_moc.values)/1e9))
    
    diff_msftmz_u031pctbgc.append(
        ((amoc26_u031pctbgc_yearly.isel(year=np.arange(year,year+1)).atlantic_moc.values)/1e9)
        -((amoc26_u031pctbgc_yearly.isel(year=0).atlantic_moc.values)/1e9))
    
    diff_msftmz_u011pctbgc.append(
        ((amoc26_u011pctbgc_yearly.isel(year=np.arange(year,year+1)).atlantic_moc.values)/1e9)
        -((amoc26_u011pctbgc_yearly.isel(year=0).atlantic_moc.values)/1e9))
    
    diff_msftmz_g011pctbgc.append(
        ((amoc26_g011pctbgc_yearly.isel(year=np.arange(year,year+1)).atlantic_moc.values)/1e9)
        -((amoc26_g011pctbgc_yearly.isel(year=0).atlantic_moc.values)/1e9))
    
    diff_msftmz_g031pctbgc.append(
        ((amoc26_g031pctbgc_yearly.isel(year=np.arange(year,year+1)).atlantic_moc.values)/1e9)
        -((amoc26_g031pctbgc_yearly.isel(year=0).atlantic_moc.values)/1e9))
    
    diff_msftmz_g051pctbgc.append(
        ((amoc26_g051pctbgc_yearly.isel(year=np.arange(year,year+1)).atlantic_moc.values)/1e9)
        -((amoc26_g051pctbgc_yearly.isel(year=0).atlantic_moc.values)/1e9))
    
    diff_msftmz_1pctbgc.append(
        ((amoc26_1pctbgc_yearly.isel(year=np.arange(year,year+1)).msftmz.values)/1e9)
        -((amoc26_1pctbgc_yearly.isel(year=0).msftmz.values)/1e9))
```

```python
diff_amoc26_u051pctbgc_array=[]
diff_amoc26_u031pctbgc_array=[]
diff_amoc26_u011pctbgc_array=[]
diff_amoc26_g011pctbgc_array=[]
diff_amoc26_g031pctbgc_array=[]
diff_amoc26_g051pctbgc_array=[]
diff_amoc26_1pctbgc_array=[]
for i in range(len(diff_msftmz_u051pctbgc)):
    diff_amoc26_u051pctbgc_array.append(diff_msftmz_u051pctbgc[i].item())
    diff_amoc26_u031pctbgc_array.append(diff_msftmz_u031pctbgc[i].item())
    diff_amoc26_u011pctbgc_array.append(diff_msftmz_u011pctbgc[i].item())
    diff_amoc26_g011pctbgc_array.append(diff_msftmz_g011pctbgc[i].item())
    diff_amoc26_g031pctbgc_array.append(diff_msftmz_g031pctbgc[i].item())
    diff_amoc26_g051pctbgc_array.append(diff_msftmz_g051pctbgc[i].item())
    diff_amoc26_1pctbgc_array.append(diff_msftmz_1pctbgc[i].item())
```

```python
np.array(diff_amoc26_u051pctbgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_amoc26.5N_u051pctbgcLR.csv', sep=',')
```

```python
np.array(diff_amoc26_u031pctbgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_amoc26.5N_u031pctbgcLR.csv', sep=',')
```

```python
np.array(diff_amoc26_u011pctbgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_amoc26.5N_u011pctbgcLR.csv', sep=',')
```

```python
np.array(diff_amoc26_g011pctbgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_amoc26.5N_g011pctbgcLR.csv', sep=',')
```

```python
np.array(diff_amoc26_g031pctbgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_amoc26.5N_g031pctbgcLR.csv', sep=',')
```

```python
np.array(diff_amoc26_g051pctbgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_amoc26.5N_g051pctbgcLR.csv', sep=',')
```

```python
np.array(diff_amoc26_1pctbgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_amoc26.5N_1pctbgcLR.csv', sep=',')
```

```python
amoc26_u051pctbgc_yearly.year.values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/time.csv', sep=',')
```

```python
import csv
with open('/work/uo1075/m300817/carbon_amoc/files/diff_amoc26.5N_1pctbgcLR.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
data_array = np.array(data)
```

## ssp245-bgc

<!-- #raw -->
Dissic totals
<!-- #endraw -->

```python
(ds_dissic_u05ssp245bgcLR.dissic*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth').sum(dim='y').sum(dim='x').values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_dissic_u05ssp245bgcLR.csv', sep=',')
```

```python
(ds_dissic_u03ssp245bgcLR.dissic*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth').sum(dim='y').sum(dim='x').values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_dissic_u03ssp245bgcLR.csv', sep=',')
```

```python
(ds_dissic_u01ssp245bgcLR.dissic*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth').sum(dim='y').sum(dim='x').values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_dissic_u01ssp245bgcLR.csv', sep=',')
```

```python
(ds_dissic_g01ssp245bgcLR.dissic*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth').sum(dim='y').sum(dim='x').values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_dissic_g01ssp245bgcLR.csv', sep=',')
```

```python
(ds_dissic_g03ssp245bgcLR.dissic*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth').sum(dim='y').sum(dim='x').values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_dissic_g03ssp245bgcLR.csv', sep=',')
```

```python
(ds_dissic_g05ssp245bgcLR.dissic*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth').sum(dim='y').sum(dim='x').values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_dissic_g05ssp245bgcLR.csv', sep=',')
```

```python
(ds_dissic_ssp245bgcLR.dissic*ds_volcello_u05hosLR.sum(dim="time").volcello*g_per_molC/1e15).isel(depth=np.arange(0,40)).sum(dim='depth').sum(dim='y').sum(dim='x').values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_dissic_ssp245bgcLR.csv', sep=',')
```

```python
Dissic diff
```

```python
np.array(diff_dissic_u05ssp245bgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_dissic_u05ssp245bgcLR.csv', sep=',')
```

```python
np.array(diff_dissic_u03ssp245bgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_dissic_u03ssp245bgcLR.csv', sep=',')
```

```python
np.array(diff_dissic_u01ssp245bgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_dissic_u01ssp245bgcLR.csv', sep=',')
```

```python
np.array(diff_dissic_g01ssp245bgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_dissic_g01ssp245bgcLR.csv', sep=',')
```

```python
np.array(diff_dissic_g03ssp245bgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_dissic_g03ssp245bgcLR.csv', sep=',')
```

```python
np.array(diff_dissic_g05ssp245bgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_dissic_g05ssp245bgcLR.csv', sep=',')
```

```python
np.array(diff_dissic_ssp245bgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_dissic_ssp245bgcLR.csv', sep=',')
```

```python

```

```python
msftmz totals 
```

```python
(ds_msftmz_u05ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_amoc26.5N_u05ssp245bgcLR.csv', sep=',')
```

```python
(ds_msftmz_u03ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_amoc26.5N_u03ssp245bgcLR.csv', sep=',')
```

```python
(ds_msftmz_u01ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_amoc26.5N_u01ssp245bgcLR.csv', sep=',')
```

```python
(ds_msftmz_g01ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_amoc26.5N_g01ssp245bgcLR.csv', sep=',')
```

```python
(ds_msftmz_g03ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_amoc26.5N_g03ssp245bgcLR.csv', sep=',')
```

```python
(ds_msftmz_g05ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_amoc26.5N_g05ssp245bgcLR.csv', sep=',')
```

```python
(ds_msftmz_ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time').atlantic_moc/1e9).values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/total_amoc26.5N_ssp245bgcLR.csv', sep=',')
```

msftmz diff

```python jupyter={"outputs_hidden": true}
ds_msftmz_u05ssp245bgcLR.load()
ds_msftmz_u03ssp245bgcLR.load()
ds_msftmz_u01ssp245bgcLR.load()
ds_msftmz_g01ssp245bgcLR.load()
ds_msftmz_g03ssp245bgcLR.load()
ds_msftmz_g01ssp245bgcLR.load()
ds_msftmz_ssp245bgcLR.load()
```

```python
# not the same as msftmt_..._diff, which just calculate the total difference at the end of the period
from tqdm import tqdm
diff_msftmz_u05ssp245bgc = []
diff_msftmz_u03ssp245bgc = []
diff_msftmz_u01ssp245bgc = []
diff_msftmz_g01ssp245bgc = []
diff_msftmz_g03ssp245bgc = []
diff_msftmz_g05ssp245bgc = []
diff_msftmz_ssp245bgc    = []
amoc26_u05ssp245bgc_yearly = ds_msftmz_u05ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time')
amoc26_u03ssp245bgc_yearly = ds_msftmz_u03ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time')
amoc26_u01ssp245bgc_yearly = ds_msftmz_u01ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time')
amoc26_g01ssp245bgc_yearly = ds_msftmz_g01ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time')
amoc26_g03ssp245bgc_yearly = ds_msftmz_g03ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time')
amoc26_g05ssp245bgc_yearly = ds_msftmz_g05ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time')
amoc26_ssp245bgc_yearly = ds_msftmz_ssp245bgcLR.sel(lat=26.5).sel(depth_2=1020).sel(lon=0).groupby('time.year').mean('time')
for year in tqdm(range(1,85)): 
    diff_msftmz_u05ssp245bgc.append(
        ((amoc26_u05ssp245bgc_yearly.isel(year=np.arange(year,year+1)).atlantic_moc.values)/1e9)
        -((amoc26_u05ssp245bgc_yearly.isel(year=0).atlantic_moc.values)/1e9))
    
    diff_msftmz_u03ssp245bgc.append(
        ((amoc26_u03ssp245bgc_yearly.isel(year=np.arange(year,year+1)).atlantic_moc.values)/1e9)
        -((amoc26_u03ssp245bgc_yearly.isel(year=0).atlantic_moc.values)/1e9))
    
    diff_msftmz_u01ssp245bgc.append(
        ((amoc26_u01ssp245bgc_yearly.isel(year=np.arange(year,year+1)).atlantic_moc.values)/1e9)
        -((amoc26_u01ssp245bgc_yearly.isel(year=0).atlantic_moc.values)/1e9))
    
    diff_msftmz_g01ssp245bgc.append(
        ((amoc26_g01ssp245bgc_yearly.isel(year=np.arange(year,year+1)).atlantic_moc.values)/1e9)
        -((amoc26_g01ssp245bgc_yearly.isel(year=0).atlantic_moc.values)/1e9))
    
    diff_msftmz_g03ssp245bgc.append(
        ((amoc26_g03ssp245bgc_yearly.isel(year=np.arange(year,year+1)).atlantic_moc.values)/1e9)
        -((amoc26_g03ssp245bgc_yearly.isel(year=0).atlantic_moc.values)/1e9))
    
    diff_msftmz_g05ssp245bgc.append(
        ((amoc26_g05ssp245bgc_yearly.isel(year=np.arange(year,year+1)).atlantic_moc.values)/1e9)
        -((amoc26_g05ssp245bgc_yearly.isel(year=0).atlantic_moc.values)/1e9))
    
    diff_msftmz_ssp245bgc.append(
        ((amoc26_ssp245bgc_yearly.isel(year=np.arange(year,year+1)).atlantic_moc.values)/1e9)
        -((amoc26_ssp245bgc_yearly.isel(year=0).atlantic_moc.values)/1e9))
    
```

```python
diff_amoc26_u05ssp245bgc_array=[]
diff_amoc26_u03ssp245bgc_array=[]
diff_amoc26_u01ssp245bgc_array=[]
diff_amoc26_g01ssp245bgc_array=[]
diff_amoc26_g03ssp245bgc_array=[]
diff_amoc26_g05ssp245bgc_array=[]
diff_amoc26_ssp245bgc_array=[]
for i in range(len(diff_msftmz_u05ssp245bgc)):
    diff_amoc26_u05ssp245bgc_array.append(diff_msftmz_u05ssp245bgc[i].item())
    diff_amoc26_u03ssp245bgc_array.append(diff_msftmz_u03ssp245bgc[i].item())
    diff_amoc26_u01ssp245bgc_array.append(diff_msftmz_u01ssp245bgc[i].item())
    diff_amoc26_g01ssp245bgc_array.append(diff_msftmz_g01ssp245bgc[i].item())
    diff_amoc26_g03ssp245bgc_array.append(diff_msftmz_g03ssp245bgc[i].item())
    diff_amoc26_g05ssp245bgc_array.append(diff_msftmz_g05ssp245bgc[i].item())
    diff_amoc26_ssp245bgc_array.append(diff_msftmz_ssp245bgc[i].item())
```

```python
np.array(diff_amoc26_u05ssp245bgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_amoc26.5N_u05ssp245bgcLR.csv', sep=',')
```

```python
np.array(diff_amoc26_u03ssp245bgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_amoc26.5N_u03ssp245bgcLR.csv', sep=',')
```

```python
np.array(diff_amoc26_u01ssp245bgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_amoc26.5N_u01ssp245bgcLR.csv', sep=',')
```

```python
np.array(diff_amoc26_g01ssp245bgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_amoc26.5N_g01ssp245bgcLR.csv', sep=',')
```

```python
np.array(diff_amoc26_g03ssp245bgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_amoc26.5N_g03ssp245bgcLR.csv', sep=',')
```

```python
np.array(diff_amoc26_g05ssp245bgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_amoc26.5N_g05ssp245bgcLR.csv', sep=',')
```

```python
np.array(diff_amoc26_ssp245bgc_array).tofile(
    '/work/uo1075/m300817/carbon_amoc/files/diff_amoc26.5N_ssp245bgcLR.csv', sep=',')
```

```python
amoc26_u05ssp245bgc_yearly.year.values.tofile(
    '/work/uo1075/m300817/carbon_amoc/files/time.csv', sep=',')
```

# Country masks

```python
# Use kernel "env_24", as regionmask is not installed in the default environments
```

```python
import xarray as xr
import numpy as np
import dask
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import os; os.environ['PROJ_LIB'] = '/work/uo1075/m300817/phd/conda/share/proj'
#import regionmask
import cdo # Import Cdo-py
cdo = cdo.Cdo(tempdir='/scratch/m/m300817/tmp/cdo-py') # change this to a directory in your scratch
import zlib
from tqdm import tqdm
import regionmask
import cf_xarray
import pickle
```

```python
ds_tas_new = {}
for exp in ["hosing_naa03Sv_1pctbgc-LR", "hosing_naa05Sv_1pctbgc-LR", "hosing_grc01Sv_1pctbgc-LR", "hosing_naa03Sv_1pct-LR", "hosing_naa05Sv_1pct-LR"]:
    ifiles  = f"/scratch/m/m300817/tmp/{exp}_echam6_echam_*.nc"
    ds_tas_new[exp] = xr.open_mfdataset(ifiles, use_cftime=True)
    ds_tas_new[exp] = ds_tas_new[exp].assign_coords(time=xr.cftime_range(start="1850", periods=1680, freq="M", calendar="proleptic_gregorian"))
```

```python
keys_countries={}
for i in range(0, 177):
    keys_countries[i]=regionmask.defined_regions.natural_earth_v5_0_0.countries_110.names[i]
```

```python jupyter={"outputs_hidden": true}
keys_countries 
```

```python
ds_tas_new_yearly = {}
for exp in ["hosing_naa03Sv_1pctbgc-LR", "hosing_naa05Sv_1pctbgc-LR", "hosing_grc01Sv_1pctbgc-LR", "hosing_naa03Sv_1pct-LR", "hosing_naa05Sv_1pct-LR"]:
    ds_tas_new_yearly[exp] = ds_tas_new[exp].groupby('time.year').mean('time')
```

```python
for exp in ["hosing_naa03Sv_1pctbgc-LR", "hosing_naa05Sv_1pctbgc-LR", "hosing_grc01Sv_1pctbgc-LR", "hosing_naa03Sv_1pct-LR", "hosing_naa05Sv_1pct-LR"]:
    ds_tas_new_yearly[exp].load()
```

```python
tas_series_countries = {}
for exp in ds_tas_new.keys():
    mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(ds_tas_new[exp])
    tas_series_countries[exp] = {}
    for i in range(0, 177):
        mask_country = mask == i
        tas_series_countries[exp][i] = ds_tas_new_yearly[exp].where(mask_country).mean(dim='lon').mean(dim='lat').temp2
```

```python
with open('/work/uo1075/m300817/carbon_amoc/files/tas_series_countries.pkl', 'wb') as f:
    pickle.dump(tas_series_countries, f)
```

```python jupyter={"outputs_hidden": true}
mask.plot()
```

```python jupyter={"outputs_hidden": true}
for i in range(0, 177):
    plt.plot(tas_series_countries['hosing_naa03Sv_1pctbgc-LR'][i].year,
             tas_series_countries['hosing_naa03Sv_1pctbgc-LR'][i].values-273.15, label=keys_countries[i])
    plt.ylabel("T(°C)")
    plt.xlabel("Year")
    plt.legend()
```

```python
for i in range(0, 177):
    plt.plot(tas_series_countries['hosing_naa03Sv_1pctbgc-LR'][i].year,
             tas_series_countries['hosing_naa03Sv_1pctbgc-LR'][i].values-273.15, label=keys_countries[i])
    plt.ylabel("T(°C)")
    plt.xlabel("Year")
    #plt.legend()
```

```python
for i in range(0, 177):
    plt.plot(tas_series_countries['hosing_naa03Sv_1pct-LR'][i].year,
             tas_series_countries['hosing_naa03Sv_1pct-LR'][i].values-273.15, label=keys_countries[i])
    plt.ylabel("T(°C)")
    plt.xlabel("Year")
    #plt.legend()
```

```python
tas_timemean_countries = {}
for exp in ds_tas_new.keys():
    mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(ds_tas_new[exp])
    tas_timemean_countries[exp] = {}
    for i in range(0, 177):
        mask_country = mask == i
        tas_timemean_countries[exp][i] = ds_tas_new_yearly[exp].where(mask_country).mean(dim='year').temp2
```

```python jupyter={"outputs_hidden": true}
for i in range(0, 177):
    plt.figure(figsize=(9, 6))
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=-60))
    ax.set_global()
    tas_timemean_countries['hosing_naa03Sv_1pct-LR'][i].plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(), x="lon", y="lat", cmap='RdBu_r', vmin=260,  vmax=305)
ax.coastlines()
#ax.set_title("") 
```

```python

```

now try to do in xarray and then to pandas

```python
tas_series_countries = {}
for exp in ds_tas_new.keys():
    mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(ds_tas_new[exp])
    tas_series_countries[exp] = {}
    for i in range(0, 177):
        mask_country = mask == i
        tas_series_countries[exp][keys_countries[i]] = ds_tas_new_yearly[exp].where(mask_country).mean(dim='lon').mean(dim='lat').temp2.values
```

```python
import pandas as pd
```

```python
tas_countries_series_pandas_newexp = pd.DataFrame.from_dict(tas_series_countries)
```

```python
tas_countries_series_pandas_newexp
```

```python
tas_countries_series_pandas_newexp.to_csv('/work/uo1075/m300817/carbon_amoc/files/tas_countries_series_pandas_newexp.csv')
```

Now for 1pctbgc (cmor)

```python
# Load tas in 1pctCO2-bgc
file_type = 'tas'
infiles = glob.glob(f'/pool/data/CMIP6/data/C4MIP/MPI-M/MPI-ESM1-2-LR/1pctCO2-bgc/r1i1p1f1/Amon/{file_type}/gn/v20190710/*{file_type}*.nc')
ds_tas_1pctbgc = xr.open_mfdataset(infiles, use_cftime=True, parallel=True)
```

```python
ds_tas_1pctbgc_yearly = ds_tas_1pctbgc.groupby('time.year').mean('time')
```

```python jupyter={"outputs_hidden": true}
ds_tas_1pctbgc_yearly.load()
```

```python
tas_series_countries_cmor = {}
for exp in ['1pctbgc']:
    mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(ds_tas_1pctbgc_yearly)
    tas_series_countries_cmor[exp] = {}
    for i in range(0, 177):
        mask_country = mask == i
        tas_series_countries_cmor[exp][i] = ds_tas_1pctbgc_yearly.where(mask_country).mean(dim='lon').mean(dim='lat').tas
```

```python jupyter={"outputs_hidden": true}
tas_series_countries_cmor['1pctbgc'][0]
```

```python
with open('/work/uo1075/m300817/carbon_amoc/files/tas_series_countries_cmor.pkl', 'wb') as f:
    pickle.dump(tas_series_countries_cmor, f)
```

```python
# now in pandas
```

```python
tas_series_countries_cmor = {}
for exp in ['1pctbgc']:
    mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(ds_tas_1pctbgc_yearly)
    tas_series_countries_cmor[exp] = {}
    for i in range(0, 177):
        mask_country = mask == i
        tas_series_countries_cmor[exp][keys_countries[i]] = ds_tas_1pctbgc_yearly.where(mask_country).mean(dim='lon').mean(dim='lat').tas.values
```

```python
tas_countries_series_pandas_1pctbgc = pd.DataFrame.from_dict(tas_series_countries_cmor)
```

```python
tas_countries_series_pandas_1pctbgc.to_csv('/work/uo1075/m300817/carbon_amoc/files/tas_countries_series_pandas_1pctbgc.csv')
```

# SSPs and piC in CMIP6 models


## ssp245

```python
# Paths for ssp245 scenario
paths={'MPI-ESM1.2-LR':'/work/ik1017/CMIP6/data/CMIP6/ScenarioMIP/MPI-M/MPI-ESM1-2-LR/ssp245/',
       'CanESM5':'/work/ik1017/CMIP6/data/CMIP6/ScenarioMIP/CCCma/CanESM5/ssp245',
       'CESM2':'/work/ik1017/CMIP6/data/CMIP6/ScenarioMIP/NCAR/CESM2/ssp245',
       'NorESM2-LM':'/work/ik1017/CMIP6/data/CMIP6/ScenarioMIP/NCC/NorESM2-LM/ssp245',
       'ACCESS-ESM1-5':'/work/ik1017/CMIP6/data/CMIP6/ScenarioMIP/CSIRO/ACCESS-ESM1-5/ssp245',
       'MIROC-ES2L': '/work/ik1017/CMIP6/data/CMIP6/ScenarioMIP/MIROC/MIROC-ES2L/ssp245', 
       'GISS-E2-1-G': '/work/ik1017/CMIP6/data/CMIP6/ScenarioMIP/NASA-GISS/GISS-E2-1-G//ssp245/'
      }
```

### MPI-ESM1.2-LR

```python
array = []
for i in range(1,51):
    rea = "r"+str(i)+"i1p1f1"
    infiles = glob.glob(f'{paths["MPI-ESM1.2-LR"]}/{rea}/Omon/msftmz/gn/**/*.nc', recursive=True)
    array.append(xr.open_mfdataset(infiles, use_cftime=True, data_vars="minimal", coords="minimal", compat="override", parallel=True).assign_coords(realiz=rea))
msftmz_mpiesmlr = xr.concat(array, dim='realiz')
```

```python jupyter={"outputs_hidden": true}
year_amoc_levs= weighted_mon_to_year_mean(msftmz_mpiesmlr.sel(lat=26.5, method='nearest').isel(basin=1), 'msftmz')/(1025.0 * 10**6)
max_lev_indices=year_amoc_levs.idxmax(dim='lev').compute()
year_amoc = year_amoc_levs.sel(lev=max_lev_indices)
for r in range(50):
    year_amoc.isel(realiz=r).plot()
```

```python
year_amoc= weighted_mon_to_year_mean(msftmz_mpiesmlr.sel(lat=26.5).sel(lev=1020).isel(basin=1), 'msftmz')/(1025.0 * 10**6)
for r in range(50):
    year_amoc.isel(realiz=r).plot()
```

```python
msftmz_mpiesmlr_mean = (weighted_mon_to_year_mean(msftmz_mpiesmlr.sel(lat=26.5).sel(lev=1020).isel(basin=1), 'msftmz').mean(dim="realiz"))/(1025.0 * 10**6)
msftmz_mpiesmlr_std = (weighted_mon_to_year_mean(msftmz_mpiesmlr.sel(lat=26.5).sel(lev=1020).isel(basin=1), 'msftmz').std(dim="realiz"))/(1025.0 * 10**6)
```

```python
time = msftmz_mpiesmlr_mean['time'].dt.year.values
mean = msftmz_mpiesmlr_mean
std = msftmz_mpiesmlr_std

plt.figure(figsize=(10, 6))
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.2, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.show()
```

```python
# Save to .nc file
ds = xr.Dataset({
    'mean': msftmz_mpiesmlr_mean,
    'std': msftmz_mpiesmlr_std
})
file_path = '/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/MPI-ESM1.2-LR/msftmz_mpiesmlr_ssp245.nc'
ds.to_netcdf(path=file_path)
```

```python
loaded_ds = xr.open_dataset('/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/MPI-ESM1.2-LR/msftmz_mpiesmlr_ssp245.nc')
```

```python
time = loaded_ds['mean']['time'].dt.year.values
mean = loaded_ds['mean']
std = loaded_ds['std']

plt.figure(figsize=(10, 6))
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.2, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.show()
```

3 ens members only

```python
msftmz_mpiesmlr_mean = (weighted_mon_to_year_mean(msftmz_mpiesmlr.sel(lat=26.5).sel(lev=1020).isel(basin=1), 'msftmz').isel(realiz=slice(0, 3)).mean(dim="realiz"))/(1025.0 * 10**6)
msftmz_mpiesmlr_std = (weighted_mon_to_year_mean(msftmz_mpiesmlr.sel(lat=26.5).sel(lev=1020).isel(basin=1), 'msftmz').isel(realiz=slice(0, 3)).std(dim="realiz"))/(1025.0 * 10**6)
```

```python
time = msftmz_mpiesmlr_mean['time'].dt.year.values
mean = msftmz_mpiesmlr_mean
std = msftmz_mpiesmlr_std

plt.figure(figsize=(10, 6))
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.2, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.show()
```

```python
# Save to .nc file
ds = xr.Dataset({
    'mean': msftmz_mpiesmlr_mean,
    'std': msftmz_mpiesmlr_std
})
file_path = '/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/MPI-ESM1.2-LR/msftmz_mpiesmlr_ssp245_r1-3i1p2f1.nc'
ds.to_netcdf(path=file_path)
```

### CanESM5

```python
array = []
#for i in range(1,26):
#    rea = "r"+str(i)+"i1p1f1"
#    infiles = glob.glob(f'{paths["CanESM5"]}/{rea}/Omon/msftmz/gn/**/*.nc', recursive=True)
#    array.append(xr.open_mfdataset(infiles, use_cftime=True, data_vars="minimal", coords="minimal", compat="override", parallel=True).assign_coords(realiz=rea))
for i in range(1,26):
    rea = "r"+str(i)+"i1p2f1"
    infiles = glob.glob(f'{paths["CanESM5"]}/{rea}/Omon/msftmz/gn/**/*.nc', recursive=True)
    array.append(xr.open_mfdataset(infiles, use_cftime=True, data_vars="minimal", coords="minimal", compat="override", parallel=True).assign_coords(realiz=rea))
ssp245_msftmz_canesm5 = xr.concat(array, dim='realiz')
```

```python
year_amoc_levs= weighted_mon_to_year_mean(ssp245_msftmz_canesm5.sel(lat=26.5, method='nearest').isel(basin=0), 'msftmz').load()/(1025.0 * 10**6)
max_lev_indices=year_amoc_levs.idxmax(dim='lev')
year_amoc = year_amoc_levs.sel(lev=max_lev_indices)
ssp245_msftmz_canesm5_mean = year_amoc.mean(dim="realiz")
ssp245_msftmz_canesm5_std = year_amoc.std(dim="realiz")
```

```python
time = ssp245_msftmz_canesm5_mean['time'].dt.year.values
mean = ssp245_msftmz_canesm5_mean
std = ssp245_msftmz_canesm5_std

plt.figure(figsize=(10, 6))
for r in range(25):
    plt.plot(time, year_amoc.isel(realiz=r), color='lightblue', alpha=1, zorder=-1)
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.3, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.title('CanESM5')
plt.show()
```

```python
# Save to .nc file
ds = xr.Dataset({
    'mean': ssp245_msftmz_canesm5_mean,
    'std': ssp245_msftmz_canesm5_std
})
file_path = '/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/CanESM5/msftmz_canesm5_ssp245_r1-25i1p2f1.nc'
ds.to_netcdf(path=file_path)
```

```python
loaded_ds = xr.open_dataset('/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/CanESM5/msftmz_canesm5_ssp245_r1-25i1p2f1.nc')
```

```python jupyter={"outputs_hidden": true}
time = loaded_ds['mean']['time'].dt.year.values
mean = loaded_ds['mean']
std = loaded_ds['std']

plt.figure(figsize=(10, 6))
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.2, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.show()
```

3 ens members only

```python
ssp245_msftmz_canesm5_mean = year_amoc.isel(realiz=slice(0, 3)).mean(dim="realiz")
ssp245_msftmz_canesm5_std = year_amoc.isel(realiz=slice(0, 3)).std(dim="realiz")
```

```python
time = ssp245_msftmz_canesm5_mean['time'].dt.year.values
mean = ssp245_msftmz_canesm5_mean
std = ssp245_msftmz_canesm5_std

plt.figure(figsize=(10, 6))
for r in range(3):
    plt.plot(time, year_amoc.isel(realiz=r), color='lightblue', alpha=1, zorder=-1)
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.3, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.title('CanESM5')
plt.show()
```

```python
# Save to .nc file
ds = xr.Dataset({
    'mean': ssp245_msftmz_canesm5_mean,
    'std': ssp245_msftmz_canesm5_std
})
file_path = '/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/CanESM5/msftmz_canesm5_ssp245_r1-3i1p2f1.nc'
ds.to_netcdf(path=file_path)
```

### CESM2

```python jupyter={"outputs_hidden": true}
array = []
for i in [4, 10, 11]:
    rea = "r"+str(i)+"i1p1f1"
    infiles = glob.glob(f'{paths["CESM2"]}/{rea}/Omon/msftmz/gn/**/*.nc', recursive=True)
    array.append(xr.open_mfdataset(infiles, use_cftime=True, data_vars="minimal", coords="minimal", compat="override", parallel=True).assign_coords(realiz=rea))
ssp245_msftmz_cesm2 = xr.concat(array, dim='realiz')
```

```python
year_amoc_levs= weighted_mon_to_year_mean(ssp245_msftmz_cesm2.sel(lat=26.5, method='nearest').isel(basin=0), 'msftmz').load()/(1025.0 * 10**6)
max_lev_indices=year_amoc_levs.idxmax(dim='lev')
year_amoc = year_amoc_levs.sel(lev=max_lev_indices)
ssp245_msftmz_cesm2_mean = year_amoc.mean(dim="realiz")
ssp245_msftmz_cesm2_std = year_amoc.std(dim="realiz")
```

```python
time = ssp245_msftmz_cesm2_mean['time'].dt.year.values
mean = ssp245_msftmz_cesm2_mean
std = ssp245_msftmz_cesm2_std

plt.figure(figsize=(10, 6))
for r in range(3):
    plt.plot(time, year_amoc.isel(realiz=r), color='lightblue', alpha=1, zorder=-1)
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.3, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.title('CESM2')
plt.show()
```

```python
# Save to .nc file
ds = xr.Dataset({
    'mean': ssp245_msftmz_cesm2_mean,
    'std': ssp245_msftmz_cesm2_std
})
file_path = '/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/CESM2/msftmz_cesm2_ssp245_r4_r10_r11i1p1f1.nc'
ds.to_netcdf(path=file_path)
```

```python
loaded_ds = xr.open_dataset('/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/CESM2/msftmz_cesm2_ssp245_r4_r10_r11i1p1f1.nc')
```

```python jupyter={"outputs_hidden": true}
time = loaded_ds['mean']['time'].dt.year.values
mean = loaded_ds['mean']
std = loaded_ds['std']

plt.figure(figsize=(10, 6))
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.2, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.show()
```

### NorESM2-LM

```python
array = []
for i in [1, 2, 3]:
    rea = "r"+str(i)+"i1p1f1"
    infiles = glob.glob(f'{paths["NorESM2-LM"]}/{rea}/Omon/msftmz/grz/**/*.nc', recursive=True)
    array.append(xr.open_mfdataset(infiles, use_cftime=True, data_vars="minimal", coords="minimal", compat="override", parallel=True).assign_coords(realiz=rea))
ssp245_msftmz_noresm2lm = xr.concat(array, dim='realiz')
```

```python
year_amoc_levs= weighted_mon_to_year_mean(ssp245_msftmz_noresm2lm.sel(lat=26.5, method='nearest').isel(basin=0), 'msftmz').load()/(1025.0 * 10**6)
max_lev_indices=year_amoc_levs.idxmax(dim='lev')
year_amoc = year_amoc_levs.sel(lev=max_lev_indices)
ssp245_msftmz_noresm2lm_mean = year_amoc.mean(dim="realiz")
ssp245_msftmz_noresm2lm_std = year_amoc.std(dim="realiz")
```

```python
time = ssp245_msftmz_noresm2lm_mean['time'].dt.year.values
mean = ssp245_msftmz_noresm2lm_mean
std = ssp245_msftmz_noresm2lm_std

plt.figure(figsize=(10, 6))
for r in range(3):
    plt.plot(time, year_amoc.isel(realiz=r), color='lightblue', alpha=1, zorder=-1)
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.3, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.title('NorESM2-LM')
plt.show()
```

```python
# Save to .nc file
ds = xr.Dataset({
    'mean': ssp245_msftmz_noresm2lm_mean,
    'std': ssp245_msftmz_noresm2lm_std
})
file_path = '/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/NorESM2-LM/msftmz_noresm2lm_ssp245_r1-3i1p1f1.nc'
ds.to_netcdf(path=file_path)
```

```python
loaded_ds = xr.open_dataset('/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/NorESM2-LM/msftmz_noresm2lm_ssp245_r1-3i1p1f1.nc')
```

```python jupyter={"outputs_hidden": true}
time = loaded_ds['mean']['time'].dt.year.values
mean = loaded_ds['mean']
std = loaded_ds['std']

plt.figure(figsize=(10, 6))
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.2, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.show()
```

### ACCESS-ESM1-5

```python
array = []
for i in [1,2,3,4,6,7,8]:
    rea = "r"+str(i)+"i1p1f1"
    infiles = glob.glob(f'{paths["ACCESS-ESM1-5"]}/{rea}/Omon/msftmz/gn/**/*.nc', recursive=True)
    array.append(xr.open_mfdataset(infiles, use_cftime=True, data_vars="minimal", coords="minimal", compat="override", parallel=True).assign_coords(realiz=rea))
ssp245_msftmz_accessesm15 = xr.concat(array, dim='realiz')
```

```python
year_amoc_levs= weighted_mon_to_year_mean(ssp245_msftmz_accessesm15.sel(lat=26.5, method='nearest').isel(basin=0), 'msftmz').load()/(1025.0 * 10**6)
max_lev_indices=year_amoc_levs.idxmax(dim='lev')
year_amoc = year_amoc_levs.sel(lev=max_lev_indices)
ssp245_msftmz_accessesm15_mean = year_amoc.mean(dim="realiz")
ssp245_msftmz_accessesm15_std = year_amoc.std(dim="realiz")
```

```python
time = ssp245_msftmz_accessesm15_mean['time'].dt.year.values
mean = ssp245_msftmz_accessesm15_mean
std = ssp245_msftmz_accessesm15_std

plt.figure(figsize=(10, 6))
for r in range(7):
    plt.plot(time, year_amoc.isel(realiz=r), color='lightblue', alpha=1, zorder=-1)
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.3, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.title('ACCESS-ESM1-5')
plt.show()
```

```python
# Save to .nc file
ds = xr.Dataset({
    'mean': ssp245_msftmz_accessesm15_mean,
    'std': ssp245_msftmz_accessesm15_std
})
file_path = '/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/ACCESS-ESM1-5/msftmz_accessesm15_ssp245_r1-4_6-8i1p1f1.nc'
ds.to_netcdf(path=file_path)
```

```python
loaded_ds = xr.open_dataset('/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/ACCESS-ESM1-5/msftmz_accessesm15_ssp245_r1-4_6-8i1p1f1.nc')
```

```python
time = loaded_ds['mean']['time'].dt.year.values
mean = loaded_ds['mean']
std = loaded_ds['std']

plt.figure(figsize=(10, 6))
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.2, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.show()
```

3 ens members only

```python
ssp245_msftmz_accessesm15_mean = year_amoc.isel(realiz=slice(0, 3)).mean(dim="realiz")
ssp245_msftmz_accessesm15_std = year_amoc.isel(realiz=slice(0, 3)).std(dim="realiz")
```

```python
time = ssp245_msftmz_accessesm15_mean['time'].dt.year.values
mean = ssp245_msftmz_accessesm15_mean
std = ssp245_msftmz_accessesm15_std

plt.figure(figsize=(10, 6))
for r in range(3):
    plt.plot(time, year_amoc.isel(realiz=r), color='lightblue', alpha=1, zorder=-1)
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.3, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.title('ACCESS-ESM1-5')
plt.show()
```

```python
# Save to .nc file
ds = xr.Dataset({
    'mean': ssp245_msftmz_accessesm15_mean,
    'std': ssp245_msftmz_accessesm15_std
})
file_path = '/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/ACCESS-ESM1-5/msftmz_accessesm15_ssp245_r1-3i1p1f1.nc'
ds.to_netcdf(path=file_path)
```

### MIROC-ES2L

```python
array = []
for i in range(1,31):
    rea = "r"+str(i)+"i1p1f2"
    infiles = glob.glob(f'{paths["MIROC-ES2L"]}/{rea}/Omon/msftmz/gr/**/*.nc', recursive=True)
    array.append(xr.open_mfdataset(infiles, use_cftime=True, data_vars="minimal", coords="minimal", compat="override", parallel=True).assign_coords(realiz=rea))
ssp245_msftmz_miroces2l = xr.concat(array, dim='realiz')
```

```python
year_amoc_levs= weighted_mon_to_year_mean(ssp245_msftmz_miroces2l.sel(lat=26.5, method='nearest').isel(basin=0), 'msftmz').load()/(1025.0 * 10**6)
max_lev_indices=year_amoc_levs.idxmax(dim='lev')
year_amoc = year_amoc_levs.sel(lev=max_lev_indices)
ssp245_msftmz_miroces2l_mean = year_amoc.mean(dim="realiz")
ssp245_msftmz_miroces2l_std = year_amoc.std(dim="realiz")
```

```python
time = ssp245_msftmz_miroces2l_mean['time'].dt.year.values
mean = ssp245_msftmz_miroces2l_mean
std = ssp245_msftmz_miroces2l_std

plt.figure(figsize=(10, 6))
for r in range(30):
    plt.plot(time, year_amoc.isel(realiz=r), color='lightblue', alpha=1, zorder=-1)
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.3, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.title('MIROC-ES2L')
plt.show()
```

```python
# Save to .nc file
ds = xr.Dataset({
    'mean': ssp245_msftmz_miroces2l_mean,
    'std': ssp245_msftmz_miroces2l_std
})
file_path = '/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/MIROC-ES2L/msftmz_miroces2l_ssp245_r1-30i1p1f2.nc'
ds.to_netcdf(path=file_path)
```

```python
loaded_ds = xr.open_dataset('/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/MIROC-ES2L/msftmz_miroces2l_ssp245_r1-30i1p1f2.nc')
```

```python
time = loaded_ds['mean']['time'].dt.year.values
mean = loaded_ds['mean']
std = loaded_ds['std']

plt.figure(figsize=(10, 6))
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.2, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.show()
```

3 ens members only

```python
ssp245_msftmz_miroces2l_mean = year_amoc.isel(realiz=slice(0, 3)).mean(dim="realiz")
ssp245_msftmz_miroces2l_std = year_amoc.isel(realiz=slice(0, 3)).std(dim="realiz")
```

```python
time = ssp245_msftmz_miroces2l_mean['time'].dt.year.values
mean = ssp245_msftmz_miroces2l_mean
std = ssp245_msftmz_miroces2l_std

plt.figure(figsize=(10, 6))
for r in range(3):
    plt.plot(time, year_amoc.isel(realiz=r), color='lightblue', alpha=1, zorder=-1)
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.3, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.title('MIROC-ES2L')
plt.show()
```

```python
# Save to .nc file
ds = xr.Dataset({
    'mean': ssp245_msftmz_miroces2l_mean,
    'std': ssp245_msftmz_miroces2l_std
})
file_path = '/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/MIROC-ES2L/msftmz_miroces2l_ssp245_r1-3i1p1f2.nc'
ds.to_netcdf(path=file_path)
```

### GISS-E2-1-G

```python
array = []
for i in range(2,11):
    rea = "r"+str(i)+"i1p1f2"
    infiles = glob.glob(f'{paths["GISS-E2-1-G"]}/{rea}/Omon/msftmz/gn/**/*.nc', recursive=True)
    array.append(xr.open_mfdataset(infiles, use_cftime=True, data_vars="minimal", coords="minimal", compat="override", parallel=True).assign_coords(realiz=rea))
ssp245_msftmz_gisse21g = xr.concat(array, dim='realiz')
```

```python
year_amoc_levs= weighted_mon_to_year_mean(ssp245_msftmz_gisse21g.sel(lat=26.5, method='nearest').isel(basin=0), 'msftmz').load()/(1025.0 * 10**6)
max_lev_indices=year_amoc_levs.idxmax(dim='lev')
year_amoc = year_amoc_levs.sel(lev=max_lev_indices)
ssp245_msftmz_gisse21g_mean = year_amoc.mean(dim="realiz")
ssp245_msftmz_gisse21g_std = year_amoc.std(dim="realiz")
```

```python
time = ssp245_msftmz_gisse21g_mean['time'].dt.year.values
mean = ssp245_msftmz_gisse21g_mean
std = ssp245_msftmz_gisse21g_std

plt.figure(figsize=(10, 6))
for r in range(9):
    plt.plot(time, year_amoc.isel(realiz=r), color='lightblue', alpha=1, zorder=-1)
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.3, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.title('GISS-E2-1-G')
plt.show()
```

```python
# Save to .nc file
ds = xr.Dataset({
    'mean': ssp245_msftmz_gisse21g_mean,
    'std': ssp245_msftmz_gisse21g_std
})
file_path = '/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/GISS-E2-1-G/msftmz_gisse21g_ssp245_r2-10i1p1f2.nc'
ds.to_netcdf(path=file_path)
```

```python
loaded_ds = xr.open_dataset('/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/GISS-E2-1-G/msftmz_gisse21g_ssp245_r2-10i1p1f2.nc')
```

```python
time = loaded_ds['mean']['time'].dt.year.values
mean = loaded_ds['mean']
std = loaded_ds['std']

plt.figure(figsize=(10, 6))
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.2, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.show()
```

3 ens members only

```python
ssp245_msftmz_gisse21g_mean = year_amoc.isel(realiz=slice(0, 3)).mean(dim="realiz")
ssp245_msftmz_gisse21g_std = year_amoc.isel(realiz=slice(0, 3)).std(dim="realiz")
```

```python
time = ssp245_msftmz_gisse21g_mean['time'].dt.year.values
mean = ssp245_msftmz_gisse21g_mean
std = ssp245_msftmz_gisse21g_std

plt.figure(figsize=(10, 6))
for r in range(3):
    plt.plot(time, year_amoc.isel(realiz=r), color='lightblue', alpha=1, zorder=-1)
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.3, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.title('GISS-E2-1-G')
plt.show()
```

```python
# Save to .nc file
ds = xr.Dataset({
    'mean': ssp245_msftmz_gisse21g_mean,
    'std': ssp245_msftmz_gisse21g_std
})
file_path = '/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/GISS-E2-1-G/msftmz_gisse21g_ssp245_r2-4i1p1f2.nc'
ds.to_netcdf(path=file_path)
```

## ssp126

```python
# Paths for ssp245 scenario
paths={'MPI-ESM1.2-LR':'/work/ik1017/CMIP6/data/CMIP6/ScenarioMIP/MPI-M/MPI-ESM1-2-LR/ssp126/',
      'NorESM2-LM':'/work/ik1017/CMIP6/data/CMIP6/ScenarioMIP/NCC/NorESM2-LM/ssp126',
      }
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### MPI-ESM1.2-LR
<!-- #endregion -->

```python
array = []
for i in range(1,51):
    rea = "r"+str(i)+"i1p1f1"
    infiles = glob.glob(f'{paths["MPI-ESM1.2-LR"]}/{rea}/Omon/msftmz/gn/**/*.nc', recursive=True)
    array.append(xr.open_mfdataset(infiles, use_cftime=True, data_vars="minimal", coords="minimal", compat="override", parallel=True).assign_coords(realiz=rea))
msftmz_mpiesmlr = xr.concat(array, dim='realiz')
```

```python
year_amoc= weighted_mon_to_year_mean(msftmz_mpiesmlr.sel(lat=26.5).sel(lev=1020).isel(basin=1), 'msftmz')/(1025.0 * 10**6)
for r in range(50):
    year_amoc.isel(realiz=r).plot()
```

```python
msftmz_mpiesmlr_mean = (weighted_mon_to_year_mean(msftmz_mpiesmlr.sel(lat=26.5).sel(lev=1020).isel(basin=1), 'msftmz').mean(dim="realiz"))/(1025.0 * 10**6)
msftmz_mpiesmlr_std = (weighted_mon_to_year_mean(msftmz_mpiesmlr.sel(lat=26.5).sel(lev=1020).isel(basin=1), 'msftmz').std(dim="realiz"))/(1025.0 * 10**6)
```

```python
time = msftmz_mpiesmlr_mean['time'].dt.year.values
mean = msftmz_mpiesmlr_mean
std = msftmz_mpiesmlr_std

plt.figure(figsize=(10, 6))
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.2, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.show()
```

```python
# Save to .nc file
ds = xr.Dataset({
    'mean': msftmz_mpiesmlr_mean,
    'std': msftmz_mpiesmlr_std
})
file_path = '/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/MPI-ESM1.2-LR/msftmz_mpiesmlr_ssp126.nc'
ds.to_netcdf(path=file_path)
```

### NorESM2-LM

```python
array = []
for i in [1]:
    rea = "r"+str(i)+"i1p1f1"
    infiles = glob.glob(f'{paths["NorESM2-LM"]}/{rea}/Omon/msftmz/**/**/*.nc', recursive=True)
    array.append(xr.open_mfdataset(infiles, use_cftime=True, data_vars="minimal", coords="minimal", compat="override", parallel=True).assign_coords(realiz=rea))
ssp126_msftmz_noresm2lm = xr.concat(array, dim='realiz')
```

```python
year_amoc_levs= weighted_mon_to_year_mean(ssp126_msftmz_noresm2lm.sel(lat=26.5, method='nearest').isel(basin=0), 'msftmz').load()/(1025.0 * 10**6)
max_lev_indices=year_amoc_levs.idxmax(dim='lev')
year_amoc = year_amoc_levs.sel(lev=max_lev_indices)
ssp126_msftmz_noresm2lm_mean = year_amoc.mean(dim="realiz")
```

```python
time = ssp126_msftmz_noresm2lm_mean['time'].dt.year.values
mean = ssp126_msftmz_noresm2lm_mean

plt.figure(figsize=(10, 6))
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.title('NorESM2-LM')
plt.show()
```

```python
# Save to .nc file
ds = xr.Dataset({
    'r1i1p1f1': ssp126_msftmz_noresm2lm_mean,
})
file_path = '/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/NorESM2-LM/msftmz_noresm2lm_ssp126_r1i1p1f1.nc'
ds.to_netcdf(path=file_path)
```

```python
loaded_ds = xr.open_dataset('/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/NorESM2-LM/msftmz_noresm2lm_ssp126_r1i1p1f1.nc')
```

```python
time = loaded_ds['r1i1p1f1']['time'].dt.year.values
mean = loaded_ds['r1i1p1f1']

plt.figure(figsize=(10, 6))
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.show()
```

## ssp585

```python
# Paths for ssp245 scenario
paths={'MPI-ESM1.2-LR':'/work/ik1017/CMIP6/data/CMIP6/ScenarioMIP/MPI-M/MPI-ESM1-2-LR/ssp585/',
       'NorESM2-LM':'/work/ik1017/CMIP6/data/CMIP6/ScenarioMIP/NCC/NorESM2-LM/ssp585',
      }
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### MPI-ESM1.2-LR
<!-- #endregion -->

```python
array = []
for i in range(1,51):
    rea = "r"+str(i)+"i1p1f1"
    infiles = glob.glob(f'{paths["MPI-ESM1.2-LR"]}/{rea}/Omon/msftmz/gn/**/*.nc', recursive=True)
    array.append(xr.open_mfdataset(infiles, use_cftime=True, data_vars="minimal", coords="minimal", compat="override", parallel=True).assign_coords(realiz=rea))
msftmz_mpiesmlr = xr.concat(array, dim='realiz')
```

```python
year_amoc= weighted_mon_to_year_mean(msftmz_mpiesmlr.sel(lat=26.5).sel(lev=1020).isel(basin=1), 'msftmz')/(1025.0 * 10**6)
for r in range(50):
    year_amoc.isel(realiz=r).plot()
```

```python
msftmz_mpiesmlr_mean = (weighted_mon_to_year_mean(msftmz_mpiesmlr.sel(lat=26.5).sel(lev=1020).isel(basin=1), 'msftmz').mean(dim="realiz"))/(1025.0 * 10**6)
msftmz_mpiesmlr_std = (weighted_mon_to_year_mean(msftmz_mpiesmlr.sel(lat=26.5).sel(lev=1020).isel(basin=1), 'msftmz').std(dim="realiz"))/(1025.0 * 10**6)
```

```python
time = msftmz_mpiesmlr_mean['time'].dt.year.values
mean = msftmz_mpiesmlr_mean
std = msftmz_mpiesmlr_std

plt.figure(figsize=(10, 6))
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.fill_between(time, mean - std, mean + std, color='blue', alpha=0.2, label='Ens. std')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.show()
```

```python
# Save to .nc file
ds = xr.Dataset({
    'mean': msftmz_mpiesmlr_mean,
    'std': msftmz_mpiesmlr_std
})
file_path = '/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/MPI-ESM1.2-LR/msftmz_mpiesmlr_ssp585.nc'
ds.to_netcdf(path=file_path)
```

### NorESM2-LM

```python
array = []
for i in [1]:
    rea = "r"+str(i)+"i1p1f1"
    infiles = glob.glob(f'{paths["NorESM2-LM"]}/{rea}/Omon/msftmz/**/**/*.nc', recursive=True)
    array.append(xr.open_mfdataset(infiles, use_cftime=True, data_vars="minimal", coords="minimal", compat="override", parallel=True).assign_coords(realiz=rea))
ssp585_msftmz_noresm2lm = xr.concat(array, dim='realiz')
```

```python
year_amoc_levs= weighted_mon_to_year_mean(ssp585_msftmz_noresm2lm.sel(lat=26.5, method='nearest').isel(basin=0), 'msftmz').load()/(1025.0 * 10**6)
max_lev_indices=year_amoc_levs.idxmax(dim='lev')
year_amoc = year_amoc_levs.sel(lev=max_lev_indices)
ssp585_msftmz_noresm2lm_mean = year_amoc.mean(dim="realiz")
```

```python
time = ssp585_msftmz_noresm2lm_mean['time'].dt.year.values
mean = ssp585_msftmz_noresm2lm_mean

plt.figure(figsize=(10, 6))
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.title('NorESM2-LM')
plt.show()
```

```python
# Save to .nc file
ds = xr.Dataset({
    'r1i1p1f1': ssp126_msftmz_noresm2lm_mean,
})
file_path = '/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/NorESM2-LM/msftmz_noresm2lm_ssp126_r1i1p1f1.nc'
ds.to_netcdf(path=file_path)
```

```python
loaded_ds = xr.open_dataset('/work/uo1075/m300817/carbon_amoc/amoc-carbon/data/CMIP6_amoc/NorESM2-LM/msftmz_noresm2lm_ssp126_r1i1p1f1.nc')
```

```python
time = loaded_ds['r1i1p1f1']['time'].dt.year.values
mean = loaded_ds['r1i1p1f1']

plt.figure(figsize=(10, 6))
plt.plot(time, mean, label='Ens. mean', color='blue', linestyle='-')
plt.xlabel('Time')
plt.ylabel('AMOC strength 26°N')
plt.show()
```

## piControl

```python
# Paths for ssp245 scenario
paths={'MPI-ESM1.2-LR':'/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/piControl/',
       'CESM2':'/work/ik1017/CMIP6/data/CMIP6/CMIP/NCAR/CESM2/piControl',
       'NorESM2-LM':'/work/ik1017/CMIP6/data/CMIP6/CMIP/NCC/NorESM2-LM/piControl',
       'ACCESS-ESM1-5':'/work/ik1017/CMIP6/data/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/piControl',
       'CanESM5':'/work/ik1017/CMIP6/data/CMIP6/CMIP/CCCma/CanESM5/piControl',
       'GISS-E2-1-G':'/work/ik1017/CMIP6/data/CMIP6/CMIP/NASA-GISS/GISS-E2-1-G/piControl/',
       'MIROC6':'/work/ik1017/CMIP6/data/CMIP6/CMIP/MIROC/MIROC6/piControl/'
      }

paths_r1i1p1f1={'MPI-ESM1.2-LR':'/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/piControl/',
       'CESM2':'/work/ik1017/CMIP6/data/CMIP6/CMIP/NCAR/CESM2/piControl',
       'NorESM2-LM':'/work/ik1017/CMIP6/data/CMIP6/CMIP/NCC/NorESM2-LM/piControl',
       'ACCESS-ESM1-5':'/work/ik1017/CMIP6/data/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/piControl',
      }
paths_r1i1p2f1 = {'CanESM5':'/work/ik1017/CMIP6/data/CMIP6/CMIP/CCCma/CanESM5/piControl'}
paths_r1i1p1f2 = {'GISS-E2-1-G':'/work/ik1017/CMIP6/data/CMIP6/CMIP/NASA-GISS/GISS-E2-1-G/piControl/'} 
path_MIROC = {'MIROC6':'/work/ik1017/CMIP6/data/CMIP6/CMIP/MIROC/MIROC6/piControl/'} 
```

```python
mod
```

```python
xr.open_mfdataset("/work/ik1017/CMIP6/data/CMIP6/CMIP/MIROC/MIROC6/piControl/r1i1p1f1/Omon/msftmz/gr/v20200421/msftmz_Omon_MIROC6_piControl_r1i1p1f1_gr_320001-399912.nc",use_cftime=True)
```

```python
msftmz_models_dict = {}
for mod in paths_r1i1p1f1:
    infiles = glob.glob(f'{paths_r1i1p1f1[mod]}/r1i1p1f1/Omon/msftmz/**/**/*.nc', recursive=True)
    msftmz_models_dict[mod]=(xr.open_mfdataset(infiles, use_cftime=True, data_vars="minimal", coords="minimal", compat="override").assign_coords(model=mod))
for mod in paths_r1i1p2f1:
    infiles = glob.glob(f'{paths_r1i1p2f1[mod]}/r1i1p2f1/Omon/msftmz/**/**/*.nc', recursive=True)
    msftmz_models_dict[mod]=(xr.open_mfdataset(infiles, use_cftime=True, data_vars="minimal", coords="minimal", compat="override").assign_coords(model=mod))
for mod in paths_r1i1p1f2:
    infiles = glob.glob(f'{paths_r1i1p1f2[mod]}/r1i1p1f2/Omon/msftmz/**/**/*.nc', recursive=True)
    msftmz_models_dict[mod]=(xr.open_mfdataset(infiles, use_cftime=True, data_vars="minimal", coords="minimal", compat="override").assign_coords(model=mod))
for mod in path_MIROC:
    infiles = "/work/ik1017/CMIP6/data/CMIP6/CMIP/MIROC/MIROC6/piControl/r1i1p1f1/Omon/msftmz/gr/v20200421/msftmz_Omon_MIROC6_piControl_r1i1p1f1_gr_320001-399912.nc"
    msftmz_models_dict[mod]=(xr.open_mfdataset(infiles).assign_coords(model=mod))
```

```python jupyter={"outputs_hidden": true}
msftmz_models_dict[]
```

```python
picontrol_msftmz_cmip6 = {'mean':{}, 'std':{}}
for mod in paths:
    if mod=='MPI-ESM1.2-LR': # atlantic is basin=1 
        year_amoc_levs=weighted_mon_to_year_mean(msftmz_models_dict[mod].sel(lat=26.5, method='nearest').isel(basin=1), 'msftmz').load()/(1025.0 * 10**6)
        max_lev_indices=year_amoc_levs.idxmax(dim='lev')
        year_amoc = year_amoc_levs.sel(lev=max_lev_indices)
        picontrol_msftmz_cmip6['mean'][mod]=year_amoc.mean(dim='time')
        picontrol_msftmz_cmip6['std'][mod]=year_amoc.std(dim='time')

    else: # atlantic basin=0
        year_amoc_levs=weighted_mon_to_year_mean(msftmz_models_dict[mod].sel(lat=26.5, method='nearest').isel(basin=0), 'msftmz').load()/(1025.0 * 10**6)
        max_lev_indices=year_amoc_levs.idxmax(dim='lev')
        year_amoc = year_amoc_levs.sel(lev=max_lev_indices)
        picontrol_msftmz_cmip6['mean'][mod]=year_amoc.mean(dim='time')
        picontrol_msftmz_cmip6['std'][mod]=year_amoc.std(dim='time')
```

```python
picontrol_msftmz_cmip6
```

```python
## historical 
```

<!-- #region toc-hr-collapsed=true -->
# Extra stuff
<!-- #endregion -->

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## Regridding
<!-- #endregion -->

Regridding might be necessary for next steps, such as lat-depth plots

```python
ds_dissic_pi = ds_dissic_pi.rename({"longitude": "lon", "latitude": "lat"})
ds_dissic_hist = ds_dissic_hist.rename({"longitude": "lon", "latitude": "lat"})
ds_dissic_ssp126 = ds_dissic_ssp126.rename({"longitude": "lon", "latitude": "lat"})
ds_dissic_ssp585 = ds_dissic_ssp585.rename({"longitude": "lon", "latitude": "lat"})
ds_dissic_1pct= ds_dissic_1pct.rename({"longitude": "lon", "latitude": "lat"})
ds_dissic_1pctbgc = ds_dissic_1pctbgc.rename({"longitude": "lon", "latitude": "lat"})
ds_dissic_1pctrad = ds_dissic_1pctrad.rename({"longitude": "lon", "latitude": "lat"})
```

```python
ds_out = xe.util.grid_global(1, 1)
```

```python
regridder = xe.Regridder(ds_dissic_hist, ds_out, "bilinear")
ds_dissic_hist_out = regridder(ds_dissic_hist['dissic'])
```

```python
regridder = xe.Regridder(ds_dissic_1pctrad, ds_out, "bilinear")
ds_dissic_1pctrad_out = regridder(ds_dissic_1pctrad['dissic'])
```

```python
# Plot DIC below 1000m (1085-5720m) difference last 10 years 1pctrad & first 10 years hist
plt.figure(figsize=(14, 6))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=-60))
ax.set_global()
(ds_dissic_1pctrad_out.isel(lev=np.arange(23,40)).sum(dim='lev').isel(time=np.arange(-120,-1)).mean(dim='time')-
 ds_dissic_hist_out.isel(lev=np.arange(23,40)).sum(dim='lev').isel(time=np.arange(0,120)).mean(dim='time')).plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(), x="lon", y="lat")
ax.coastlines()
#ax.set_xlim([-100, 40])
```

## Mask land

```python
ds_tas_u03hosLR.where(ds_basin_hosLR_d.basin==0).mean(dim"time").tas.plot()
```

```python
ds_tas_u03hosLR_year.where(ds_basin_hosLR.basin==0).mean(dim="time").isel(depth=0).tas
```

```python
ds_basin.where(ds_basin.basin==0)
```

```python
land=ds_basin.where(ds_basin.basin==0)
```

```python jupyter={"outputs_hidden": true}
fig=plt.figure(figsize=(14, 9))

ax1 = fig.add_subplot(211, projection=ccrs.Robinson(central_longitude=-60))
land.basin.plot(ax=ax1, transform=ccrs.PlateCarree(), x="longitude", y="latitude")
```

```python jupyter={"outputs_hidden": true}
land=ds_basin.basin.where(ds_basin.basin==0, drop=True)
```

```python
fig=plt.figure(figsize=(14, 9))

ax1 = fig.add_subplot(211, projection=ccrs.Robinson(central_longitude=-60))
ds_basin.basin.where(ds_basin.basin==0).plot(ax=ax1, transform=ccrs.PlateCarree(), x="longitude", y="latitude")
```

```python
infiles = glob.glob('/work/mh0033/from_Mistral/mh0033/m300817/mpiesm-1.2.01p6-passivesalt_update/experiments/hosing_naa05Sv_FcSV-LR/outdata/mpiom/*fx*.nc')
ds_fx = xr.open_mfdataset(infiles, use_cftime=True, parallel=True)
```

```python
fig=plt.figure(figsize=(14, 9))
ax1 = fig.add_subplot(211, projection=ccrs.Robinson(central_longitude=-60))
ds_fx.weto.isel(depth_2=0).plot(ax=ax1, transform=ccrs.PlateCarree(), x="lon_2", y="lat_2")
```

<!-- #region toc-hr-collapsed=true -->
## Basin means
<!-- #endregion -->

```python
regionmask.defined_regions.natural_earth_v5_0_0.ocean_basins_50
```

## Scratch

```python
#weighted_mon_to_year_mean(his.sel(depth_2=1020).sel(lat=26.5)/(1025.0 * 10**6), "atlantic_moc").plot(label='hist')
weighted_mon_to_year_mean(s26.sel(depth_2=1020).sel(lat=26.5)/(1025.0 * 10**6), "atlantic_moc").plot(label='ssp126')
weighted_mon_to_year_mean(s45.sel(depth_2=1020).sel(lat=26.5)/(1025.0 * 10**6), "atlantic_moc").plot(label='ssp245')
weighted_mon_to_year_mean(s58.sel(depth_2=1020).sel(lat=26.5)/(1025.0 * 10**6), "atlantic_moc").plot(label='ssp585')
plt.ylim(0,25)
plt.legend()
```

```python
ds= xr.open_mfdataset("/pool/data/ECHAM6/input/r0008/greenhouse_ssp245.nc", use_cftime=True, compat='override')
```
