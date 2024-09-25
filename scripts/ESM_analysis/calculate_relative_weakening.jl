#%%
using DrWatson; @quickactivate "AMOC-Carbon"; cd(srcdir("META/src"))

using Dates
using Statistics

include(srcdir("script_functions.jl"))
include(srcdir("META/src/lib/AMOC_Carbon_functions.jl"))

#%%

AMOC_pi_vals = CSV.read(srcdir("META/data/CMIP6_amoc/AMOC_pi_values.csv"), DataFrame)

for (i, name) in enumerate(names(AMOC_pi_vals)[2:end])
    amoc_ds = get_projection_ds(name)
    amoc_mean = amoc_ds["mean"] |> Array
    amoc_std = amoc_ds["std"] |> Array
    weakening_pi = round((1 - amoc_mean[end] ./ AMOC_pi_vals[:, name][1]) * 100, digits=1)
    weakening_2015 = round((1 - amoc_mean[end] ./ amoc_mean[1]) * 100, digits=1)
    println(name, " in 2100 wrt pi for ssp245: ", weakening_pi, "%")
    AMOC_pi_vals[3, name] = weakening_pi
    println(name, " in 2100 wrt 2015 for ssp245: ", weakening_2015, "%")
    AMOC_pi_vals[4, name] = weakening_2015
    AMOC_pi_vals[9, name] = amoc_mean[end]
    AMOC_pi_vals[10, name] = amoc_std[end]
end

for name in ["MPI-ESM1.2-LR", "NorESM2-LM"]
    for scenario in ["ssp126", "ssp585"]
        amoc_ds = get_projection_ds(name, scenario=scenario)
        amoc_mean = amoc_ds[name=="MPI-ESM1.2-LR" ? "mean" : "r1i1p1f1"] |> Array
        weakening_pi = round((1 - amoc_mean[end] ./ AMOC_pi_vals[:, name][1]) * 100, digits=1)
        weakening_2015 = round((1 - amoc_mean[end] ./ amoc_mean[1]) * 100, digits=1)
        println(name, " in 2100 wrt pi for $(scenario): ", weakening_pi, "%")
        AMOC_pi_vals[scenario=="ssp126" ? 5 : 7, name] = weakening_pi
        println(name, " in 2100 wrt 2015 for $(scenario): ", weakening_2015, "%")
        AMOC_pi_vals[scenario=="ssp126" ? 6 : 8, name] = weakening_2015
        AMOC_pi_vals[scenario=="ssp126" ? 11 : 13, name] = amoc_mean[end]
        if name == "MPI-ESM1.2-LR"
            amoc_std = amoc_ds["std"] |> Array
            AMOC_pi_vals[scenario=="ssp126" ? 12 : 14, name] = amoc_std[end]
        end
    end
end

CSV.write(datadir("CMIP6_amoc/AMOC_pi_values.csv"), AMOC_pi_vals)