#%% GET PACKAGES
using DrWatson; @quickactivate "AMOC-Carbon"; cd(srcdir("META/src"))

using DelimitedFiles
using Distributions

using Mimi
include(srcdir("META/src/scc.jl"))

include(srcdir("script_functions.jl"))

#%% calculate additional damages

inflation_2010_to_2024 = 1.44 # from https://www.minneapolisfed.org/about-us/monetary-policy/inflation-calculator#dw

model_245 = base_model(rcp="RCP4.5", ssp="SSP2", tdamage="pointestimate", slrdamage="mode"); run(model_245)
cpc_growth_rate_global_245 = (sum(model_245[:Consumption, :gdppc_growth][266:end, :] .* model_245[:Consumption, :gdppc][266:end, :], dims=2) ./ sum(model_245[:Consumption, :gdppc][266:end, :], dims=2) .* 100)[:, 1]

function add_damages(dam_amoc, dam_base, emuc, prtp)
    n_MC = length(dam_amoc)

    sdr_245 = 100*prtp .+ cpc_growth_rate_global_245 .* emuc
    disc_factor_245 = accumulate(*, 1 .- 0.01.*sdr_245)

    add_dam = [sum((dam_amoc[j] .- dam_base[j]) .* disc_factor_245) .* inflation_2010_to_2024 for j in 1:n_MC] # in 2024 USD

    return add_dam
end

#%% read in simulations

n_MC = 1000

df_scc = collect_results(datadir("simulations/META"))
df_proj = collect_results(datadir("simulations/META_proj"))

df_scc = df_scc[(length.(df_scc.MC_samples).==n_MC) .& (df_scc.AMOC_runs.==="C"), :]
df_proj = df_proj[(length.(df_proj.MC_samples).==n_MC) .& (df_proj.AMOC_runs.==="C"), :]

#%%

emucs = unique(df_scc.emuc)
prtps = unique(df_scc.prtp)
persists = unique(df_scc.persist)
dams = ["low", "BHM", "high"]

col_names = ["emuc", "prtp", "persistence", "damages", 
"Add. damages MPI", "Add. damages Nor",
"Abs. SCC MPI", "Abs. SCC Nor",
"Rel. SCC MPI", "Rel. SCC Nor",
]

df_sens = DataFrame([Symbol(col) => Vector{Any}() for col in col_names]...)

persist, dam = 0.25, "BHM"
for e in emucs
    for p in prtps
        df_scc_temp = df_scc[(df_scc.emuc.===e) .& (df_scc.prtp.===p) .& (df_scc.persist.===persist) .& (df_scc.Dam.===dam), :]
        df_proj_temp = df_proj[(df_proj.emuc.===e) .& (df_proj.prtp.===p) .& (df_proj.persist.===persist) .& (df_proj.Dam.===dam), :]
        if size(df_scc_temp, 1) > 0
            scc_mpi = df_scc_temp[(df_scc_temp.DecreaseCalib.==="MPI-ESM1.2-LR"), :].sccs[1]
            scc_nor = df_scc_temp[(df_scc_temp.DecreaseCalib.==="NorESM2-LM"), :].sccs[1]
            scc_base = mean(df_scc_temp[(df_scc_temp.AMOC.==="none"), :].sccs[1])

            dam_mpi = [df_proj_temp[df_proj_temp.DecreaseCalib.==="MPI-ESM1.2-LR", :].mcres[1][j][:TotalDamages_total_damages_global_peryear][6:end] for j in 1:n_MC]
            dam_nor = [df_proj_temp[df_proj_temp.DecreaseCalib.==="NorESM2-LM", :].mcres[1][j][:TotalDamages_total_damages_global_peryear][6:end] for j in 1:n_MC]
            dam_base = [df_proj_temp[df_proj_temp.AMOC.==="none", :].mcres[1][j][:TotalDamages_total_damages_global_peryear][6:end] for j in 1:n_MC]
            
            row = [e, p, persist, dam, 
            mean(add_damages(dam_mpi, dam_base, e, p)), mean(add_damages(dam_nor, dam_base, e, p)),
            mean(scc_mpi), mean(scc_nor),
            mean((scc_mpi .- scc_base) ./ scc_base .* 100), mean((scc_nor .- scc_base) ./ scc_base .* 100),
            ]
            push!(df_sens, row)
        end
    end
end

emuc, prtp = 1.05, 0.005
for per in persists
    for d in dams
        df_scc_temp = df_scc[(df_scc.emuc.===emuc) .& (df_scc.prtp.===prtp) .& (df_scc.persist.===per) .& (df_scc.Dam.===d), :]
        df_proj_temp = df_proj[(df_proj.emuc.===emuc) .& (df_proj.prtp.===prtp) .& (df_proj.persist.===per) .& (df_proj.Dam.===d), :]
        if size(df_scc_temp, 1) > 0
            scc_mpi = df_scc_temp[(df_scc_temp.DecreaseCalib.==="MPI-ESM1.2-LR"), :].sccs[1]
            scc_nor = df_scc_temp[(df_scc_temp.DecreaseCalib.==="NorESM2-LM"), :].sccs[1]
            scc_base = mean(df_scc_temp[(df_scc_temp.AMOC.==="none"), :].sccs[1])

            dam_mpi = [df_proj_temp[df_proj_temp.DecreaseCalib.==="MPI-ESM1.2-LR", :].mcres[1][j][:TotalDamages_total_damages_global_peryear][6:end] for j in 1:n_MC];
            dam_nor = [df_proj_temp[df_proj_temp.DecreaseCalib.==="NorESM2-LM", :].mcres[1][j][:TotalDamages_total_damages_global_peryear][6:end] for j in 1:n_MC];
            dam_base = [df_proj_temp[df_proj_temp.AMOC.==="none", :].mcres[1][j][:TotalDamages_total_damages_global_peryear][6:end] for j in 1:n_MC];
            
            row = [emuc, prtp, per, d, 
            mean(add_damages(dam_mpi, dam_base, emuc, prtp)), mean(add_damages(dam_nor, dam_base, emuc, prtp)),
            mean(scc_mpi), mean(scc_nor),
            mean((scc_mpi .- scc_base) ./ scc_base .* 100), mean((scc_nor .- scc_base) ./ scc_base .* 100),
            ]
            push!(df_sens, row)
        end

    end
end

#%% write to CSV

CSV.write(datadir("sensitivity_table.csv"), df_sens)

#%% read from CSV

df_sens = CSV.read(datadir("sensitivity_table.csv"), DataFrame)

#%%

table = ""
for r in eachrow(df_sens)
    row = "$(r.emuc) & $(r.prtp*100)\\% & $(r.persistence) & $(r.damages.=="BHM" ? "central" : r.damages) & $(round((r."Add. damages MPI")/1e12, digits=1)) trillion \\\$ | $(round((r."Add. damages Nor")/1e12, digits=1)) trillion \\\$ & $(round(r."Abs. SCC MPI", digits=1)) \\\$ | $(round(r."Abs. SCC Nor", digits=1)) \\\$ & $(round(r."Rel. SCC MPI", digits=2)) \\% | $(round(r."Rel. SCC Nor", digits=2)) \\% "*"\\"*"\\"*"\n"
    table *= row
end

println(table)