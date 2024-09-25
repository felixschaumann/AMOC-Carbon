# AMOC-Carbon

This repository provides the code for reproducing the results of _Schaumann & Alastrué de Asenjo (2024): Weakening AMOC reduces carbon drawdown and increases the social cost of carbon_. 

## Structure of the repository

There are three main folders:

1. `data`
2. `scripts`
3. `src`

### `data` folder

This folder contains original ESM simulation output in the subfolders `CMIP6_amoc` and `Carbon_time_series`.

`CMIP6_amoc` contains AMOC projections and preindustrial values of AMOC strengths at 26.5°N for the seven CMIP6 models in netCDF format. 

`Carbon_time_series` contains annual results of MPI-ESM hosing simulations, for AMOC strength at 26.5°N and ocean carbon storage (DIC). They are subdivided into results for the `1pct` experiment and the `ssp245` scenario; they are given as CSV files.

`Katavouta2021_Atl_gammas.csv` and `Katavouta2021_Atl_betas.csv` are values for the carbon-climate feedback and carbon-concentration feedback, respectively. They are only focusing on tht Atlantic ocean basin, and they are all taken from [Katavouta & Williams (2021)](https://doi.org/10.5194/bg-18-3189-2021).

All the remaining CSV files are intermediate results of our analysis; they are generated by the scripts themselves.

### `scripts` folder

The scripts are subdivided into those that analyse Earth System Model (ESM) output: (`ESM_analysis`), as well as those that analyse output from the [META model](https://github.com/openmodels/META) (`META_analysis`).

`assemble_table.jl` produced Table 1 of the paper; it contains both ESM and META model output.

### `src` folder

Here, `script_functions.jl` contains functions that are used by scripts. This is the folder where the [META-AC model](https://github.com/felixschaumann/META-AC) (AC: AMOC-Carbon) should be cloned to (see below).

## Running the scripts

> ⚠️ **IMPORTANT**  
> This repository is meant to be nested. Most scripts require that the [META-AC model](https://github.com/felixschaumann/META-AC)  is cloned to the `src` folder:  
> `git clone https://github.com/felixschaumann/META-AC`
>
> Importantly, this folder has to be renamed from `META-AC` to `META` for all the scripts to work seamlessly. Of course, the scripts could also be adapted to with with the folder name `META-AC`, if you prefer that.
>
> Most scripts start with the line `using DrWatson; @quickactivate "AMOC-Carbon"; cd(srcdir("META/src"))`. If this line runs without problems, you'll know that the repository structure is nested correctly.

We use the [`DrWatson`](https://juliadynamics.github.io/DrWatson.jl/dev/) package to manage environments, file paths, and simulation runs in Julia. If you clone this repository, you should be able to recreate our Julia environment by running  
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```
and all the `DrWatson` functionalities should work from there onwards.