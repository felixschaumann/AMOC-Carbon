#%% GET PACKAGES
using DrWatson; @quickactivate "AMOC-Carbon"; cd(srcdir("META/src"))

using DelimitedFiles
using DataFrames
using CSV

#%% read in data

n_MC = 10000
MC_proj_res = CSV.read(datadir("MC_proj_and_SCC_results_$(n_MC)_samples.csv"), DataFrame)
AMOC_pi_vals = CSV.read(srcdir("META/data/CMIP6_amoc/AMOC_pi_values.csv"), DataFrame)
T_AT_2100 = 2.289
T_AT_2100_26 = 1.381
T_AT_2100_85 = 4.539

function relative_weakening_std(mean_pi, mean_2100, std_pi, std_2100)
    term1 = std_2100^2 / mean_pi^2 
    term2 = mean_2100^2 * std_pi^2 / mean_pi^4
    return sqrt(term1 + term2) * 100
    
end

table = ""
for name in names(AMOC_pi_vals)[2:end]
    row = "{\\color{darkred}$(name)} & $(round(AMOC_pi_vals[1, name], digits=1)) ± $(round(AMOC_pi_vals[2, name], digits=1)) Sv & $(round(AMOC_pi_vals[9, name], digits=1)) ± $(round(AMOC_pi_vals[10, name], digits=1)) Sv & $(round(AMOC_pi_vals[3, name], digits=1)) ± $(round(relative_weakening_std(AMOC_pi_vals[1, name], AMOC_pi_vals[9, name], AMOC_pi_vals[2, name], AMOC_pi_vals[10, name]), digits=1)) \\% & $(round(MC_proj_res[1, name], digits=1)) ± $(round(MC_proj_res[2, name], digits=1)) PgC & $(round(MC_proj_res[3, name]/T_AT_2100*100, digits=2)) ± $(round(MC_proj_res[4, name]/T_AT_2100*100, digits=2)) \\% & $(round(MC_proj_res[5, name]/1e12, digits=1)) ± $(round(MC_proj_res[6, name]/1e12, digits=1)) trillion \\\$ & +$(round(MC_proj_res[9, name], digits=2)) ± $(round(MC_proj_res[10, name], digits=2)) \\% "*"\\"*"\\"*"\n"
    table *= row
end

table_scenarios = ""
for name in ["MPI-ESM1.2-LR", "NorESM2-LM"]
    for scenario in ["ssp126", "ssp585"]
        row = "{\\color{$(scenario=="ssp126" ? "darkorange" : "rebeccapurple")}$(name)} & $(round(AMOC_pi_vals[1, name], digits=1)) ± $(round(AMOC_pi_vals[2, name], digits=1)) Sv & $(round(AMOC_pi_vals[scenario=="ssp126" ? 11 : 13, name], digits=1)) ± $(round(AMOC_pi_vals[scenario=="ssp126" ? 12 : 14, name], digits=1)) Sv & $(round(AMOC_pi_vals[scenario=="ssp126" ? 5 : 7, name], digits=1)) ± $(round(relative_weakening_std(AMOC_pi_vals[1, name], AMOC_pi_vals[scenario=="ssp126" ? 11 : 13, name], AMOC_pi_vals[2, name], AMOC_pi_vals[scenario=="ssp126" ? 12 : 14, name]), digits=1)) \\% & $(round(MC_proj_res[1, "$(name)_$(scenario)"], digits=1)) ± $(round(MC_proj_res[2, "$(name)_$(scenario)"], digits=1)) PgC & $(round(MC_proj_res[3, "$(name)_$(scenario)"]/(scenario=="ssp126" ? T_AT_2100_26 : T_AT_2100_85)*100, digits=2)) ± $(round(MC_proj_res[4, "$(name)_$(scenario)"]/(scenario=="ssp126" ? T_AT_2100_26 : T_AT_2100_85)*100, digits=2)) \\% & $(round(MC_proj_res[5, "$(name)_$(scenario)"]/1e12, digits=1)) ± $(round(MC_proj_res[6, "$(name)_$(scenario)"]/1e12, digits=1)) trillion \\\$ & +$(round(MC_proj_res[9, "$(name)_$(scenario)"], digits=2)) ± $(round(MC_proj_res[10, "$(name)_$(scenario)"], digits=2)) \\% "*"\\"*"\\"*"\n"
        table_scenarios *= row
    end
end

println(table)
println(table_scenarios)