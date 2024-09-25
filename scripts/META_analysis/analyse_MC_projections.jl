#%% GET PACKAGES
using DrWatson; @quickactivate "AMOC-Carbon"; cd(srcdir("META/src"))

using DelimitedFiles
using DataFrames
using Distributions
using Statistics

include(srcdir("META/src/scc.jl"))

#%% read in simulations

df = collect_results(datadir("simulations/META_proj"))

#%%

n_MC = 10000
df = df[length.(df.MC_samples).==n_MC, :]

df_26 = df[(df.Scenario .=== "ssp126") .& (df.AMOC .=== "none"), :]
df_45 = df[(df.Scenario .=== "ssp245") .& (df.AMOC .=== "none"), :]
df_85 = df[(df.Scenario .=== "ssp585") .& (df.AMOC .=== "none"), :]
df_C_26 = df[(df.Scenario .=== "ssp126") .& (df.AMOC .=== "C"), :]
df_C_45 = df[(df.Scenario .=== "ssp245") .& (df.AMOC .=== "C"), :]
df_C_85 = df[(df.Scenario .=== "ssp585") .& (df.AMOC .=== "C"), :]

cum_C_AMOC_2100 = []
for i in 1:size(df_C_45, 1)
    cum_C = [df_C_45.mcres[i][j][:AMOC_Carbon_cum_CO2_AMOC]/22*6 for j in 1:n_MC]
    push!(cum_C_AMOC_2100, cum_C)
end
for i in 1:size(df_C_26, 1)
    cum_C = [df_C_26.mcres[i][j][:AMOC_Carbon_cum_CO2_AMOC]/22*6 for j in 1:n_MC]
    push!(cum_C_AMOC_2100, cum_C)
end
for i in 1:size(df_C_85, 1)
    cum_C = [df_C_85.mcres[i][j][:AMOC_Carbon_cum_CO2_AMOC]/22*6 for j in 1:n_MC]
    push!(cum_C_AMOC_2100, cum_C)
end

dT_AT_2100 = []
for i in 1:size(df_C_45, 1)
    dT = [df_C_45.mcres[i][j][:TemperatureConverter_T_AT][91] - df_45.mcres[1][j][:TemperatureConverter_T_AT][91] for j in 1:n_MC]
    push!(dT_AT_2100, dT)
end
for i in 1:size(df_C_26, 1)
    dT = [df_C_26.mcres[i][j][:TemperatureConverter_T_AT][91] - df_26.mcres[1][j][:TemperatureConverter_T_AT][91] for j in 1:n_MC]
    push!(dT_AT_2100, dT)
end
for i in 1:size(df_C_85, 1)
    dT = [df_C_85.mcres[i][j][:TemperatureConverter_T_AT][91] - df_85.mcres[1][j][:TemperatureConverter_T_AT][91] for j in 1:n_MC]
    push!(dT_AT_2100, dT)
end

#%% analyse additional damages (two ways: 1. from utility and 2. from damages)

prtp, emuc = 0.5, 1.05 # CAREFUL: prtp here in %, not so in model
inflation_2010_to_2024 = 1.44 # from https://www.minneapolisfed.org/about-us/monetary-policy/inflation-calculator#dw

model_245 = base_model(rcp="RCP4.5", ssp="SSP2", tdamage="pointestimate", slrdamage="mode"); run(model_245)
model_126 = base_model(rcp="RCP3-PD/2.6", ssp="SSP1", tdamage="pointestimate", slrdamage="mode"); run(model_126)
model_585 = base_model(rcp="RCP8.5", ssp="SSP5", tdamage="pointestimate", slrdamage="mode"); run(model_585)

cons_pc_2015 = sum(model_245[:Consumption, :conspc][266, :] .* model_245[:Utility, :pop][266, :]) / model_245[:Utility, :world_population][266]

cpc_growth_rate_global_245 = (sum(model_245[:Consumption, :gdppc_growth][266:end, :] .* model_245[:Consumption, :gdppc][266:end, :], dims=2) ./ sum(model_245[:Consumption, :gdppc][266:end, :], dims=2) .* 100)[:, 1]
sdr_245 = prtp .+ cpc_growth_rate_global_245 .* emuc
disc_factor_245 = accumulate(*, 1 .- 0.01.*sdr_245)

cpc_growth_rate_global_126 = (sum(model_126[:Consumption, :gdppc_growth][266:end, :] .* model_126[:Consumption, :gdppc][266:end, :], dims=2) ./ sum(model_126[:Consumption, :gdppc][266:end, :], dims=2) .* 100)[:, 1]
sdr_126 = prtp .+ cpc_growth_rate_global_126 .* emuc
disc_factor_126 = accumulate(*, 1 .- 0.01.*sdr_126)

cpc_growth_rate_global_585 = (sum(model_585[:Consumption, :gdppc_growth][266:end, :] .* model_585[:Consumption, :gdppc][266:end, :], dims=2) ./ sum(model_585[:Consumption, :gdppc][266:end, :], dims=2) .* 100)[:, 1]
sdr_585 = prtp .+ cpc_growth_rate_global_585 .* emuc
disc_factor_585 = accumulate(*, 1 .- 0.01.*sdr_585)

add_damages = []
for i in 1:size(df_C_45, 1)
    dam_amoc = [df_C_45.mcres[i][j][:TotalDamages_total_damages_global_peryear][6:end] for j in 1:n_MC]
    dam_noamoc = [df_45.mcres[1][j][:TotalDamages_total_damages_global_peryear][6:end] for j in 1:n_MC]
    add_dam = [sum((dam_amoc[j] .- dam_noamoc[j]) .* disc_factor_245) .* inflation_2010_to_2024 for j in 1:n_MC] # in 2024 USD
    push!(add_damages, add_dam)
end
println("SSP2-4.5 additional damages in trillion 2024 USD:")
for i in 1:7
    println(df_C_45.DecreaseCalib[i], ": ", round(mean(add_damages[i])/1e12, digits=1), " ± ", round(std(add_damages[i])/1e12, digits=1))
end
for i in 1:size(df_C_26, 1)
    dam_amoc = [df_C_26.mcres[i][j][:TotalDamages_total_damages_global_peryear][6:end] for j in 1:n_MC]
    dam_noamoc = [df_26.mcres[1][j][:TotalDamages_total_damages_global_peryear][6:end] for j in 1:n_MC]
    add_dam = [sum((dam_amoc[j] .- dam_noamoc[j]) .* disc_factor_126) .* inflation_2010_to_2024 for j in 1:n_MC] # in 2024 USD
    push!(add_damages, add_dam)
end
for i in 1:size(df_C_85, 1)
    dam_amoc = [df_C_85.mcres[i][j][:TotalDamages_total_damages_global_peryear][6:end] for j in 1:n_MC]
    dam_noamoc = [df_85.mcres[1][j][:TotalDamages_total_damages_global_peryear][6:end] for j in 1:n_MC]
    add_dam = [sum((dam_amoc[j] .- dam_noamoc[j]) .* disc_factor_585) .* inflation_2010_to_2024 for j in 1:n_MC] # in 2024 USD
    push!(add_damages, add_dam)
end

welfare_loss = []
for i in 1:size(df_C_45, 1)
    util_amoc = [df_C_45.mcres[i][j][:Utility_world_disc_utility][6:end] for j in 1:n_MC]
    util_noamoc = [df_45.mcres[1][j][:Utility_world_disc_utility][6:end] for j in 1:n_MC]
    welfare = [- sum(util_amoc[j] .- util_noamoc[j]) / (cons_pc_2015^(-emuc)) .* inflation_2010_to_2024 for j in 1:n_MC] # given in 2024 USD
    push!(welfare_loss, welfare)
end
println("SSP2-4.5 welfare loss in trillion 2024 USD:")
for i in 1:7
    println(df_C_45.DecreaseCalib[i], ": ", round(mean(welfare_loss[i])/1e12, digits=1), " ± ", round(std(welfare_loss[i])/1e12, digits=1))
end
for i in 1:size(df_C_26, 1)
    util_amoc = [df_C_26.mcres[i][j][:Utility_world_disc_utility][6:end] for j in 1:n_MC]
    util_noamoc = [df_26.mcres[1][j][:Utility_world_disc_utility][6:end] for j in 1:n_MC]
    welfare = [- sum(util_amoc[j] .- util_noamoc[j]) / (cons_pc_2015^(-emuc)) .* inflation_2010_to_2024 for j in 1:n_MC] # given in 2024 USD
    push!(welfare_loss, welfare)
end
for i in 1:size(df_C_85, 1)
    util_amoc = [df_C_85.mcres[i][j][:Utility_world_disc_utility][6:end] for j in 1:n_MC]
    util_noamoc = [df_85.mcres[1][j][:Utility_world_disc_utility][6:end] for j in 1:n_MC]
    welfare = [- sum(util_amoc[j] .- util_noamoc[j]) / (cons_pc_2015^(-emuc)) .* inflation_2010_to_2024 for j in 1:n_MC] # given in 2024 USD
    push!(welfare_loss, welfare)
end

#%%

df_MC_proj = DataFrame()

row_names = ["mean_cum_C_AMOC_2100", "std_cum_C_AMOC_2100",
             "mean_dT_AT_2100", "std_dT_AT_2100",
             "mean_add_damages", "std_add_damages",
             "mean_welfare_loss", "std_welfare_loss",
             ]
df_MC_proj[!, :RowNames] = row_names             

for i in 1:size(df_C_45, 1)
    col_name = Symbol(df_C_45[i, :DecreaseCalib])
    df_MC_proj[!, col_name] = [
        mean(cum_C_AMOC_2100[i]), std(cum_C_AMOC_2100[i]), 
        mean(dT_AT_2100[i]), std(dT_AT_2100[i]), 
        mean(add_damages[i]), std(add_damages[i]),
        mean(welfare_loss[i]), std(welfare_loss[i]),
    ]
end    
for i in 1:size(df_C_26, 1)
    col_name = Symbol(df_C_26[i, :DecreaseCalib] * ("_ssp126"))
    df_MC_proj[!, col_name] = [mean(cum_C_AMOC_2100[i+size(df_C_45, 1)]), std(cum_C_AMOC_2100[i+size(df_C_45, 1)]),
                                mean(dT_AT_2100[i+size(df_C_45, 1)]), std(dT_AT_2100[i+size(df_C_45, 1)]),
                                mean(add_damages[i+size(df_C_45, 1)]), std(add_damages[i+size(df_C_45, 1)]),
                                mean(welfare_loss[i+size(df_C_45, 1)]), std(welfare_loss[i+size(df_C_45, 1)])]
end    
for i in 1:size(df_C_85, 1)
    col_name = Symbol(df_C_85[i, :DecreaseCalib] * ("_ssp585"))
    df_MC_proj[!, col_name] = [mean(cum_C_AMOC_2100[i+size(df_C_45, 1)+size(df_C_26, 1)]), std(cum_C_AMOC_2100[i+size(df_C_45, 1)+size(df_C_26, 1)]), 
                                mean(dT_AT_2100[i+size(df_C_45, 1)+size(df_C_26, 1)]), std(dT_AT_2100[i+size(df_C_45, 1)+size(df_C_26, 1)]),
                                mean(add_damages[i+size(df_C_45, 1)+size(df_C_26, 1)]), std(add_damages[i+size(df_C_45, 1)+size(df_C_26, 1)]),
                                mean(welfare_loss[i+size(df_C_45, 1)+size(df_C_26, 1)]), std(welfare_loss[i+size(df_C_45, 1)+size(df_C_26, 1)])]
end

#%% write out SCC results as well (10,000 samples)

CMIP6_SCC_results = CSV.read(datadir("CMIP6_SCC_results_10000_samples.csv"), DataFrame)

function sort_columns_alphabetically_keep_first(df::DataFrame)
    first_col_name = names(df)[1]
    first_col = df[:, first_col_name]
    remaining_cols = df[:, Not(first_col_name)]
    sorted_indices = sortperm(names(remaining_cols))
    sorted_cols = remaining_cols[:, sorted_indices]
    return hcat(DataFrame(first_col_name => first_col), sorted_cols)
end

CMIP6_SCC_results = sort_columns_alphabetically_keep_first(CMIP6_SCC_results)
df_MC_proj = sort_columns_alphabetically_keep_first(df_MC_proj)

all_results = vcat(df_MC_proj, CMIP6_SCC_results)

#%% save as csv

CSV.write(datadir("MC_proj_results_$(n_MC)_samples.csv"), df_MC_proj)
CSV.write(datadir("MC_proj_and_SCC_results_$(n_MC)_samples.csv"), all_results)