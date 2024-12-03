#%%
using DrWatson; @quickactivate "AMOC-Carbon"; cd(srcdir("META/src"))
include(srcdir("META/src/scc.jl"))
include(srcdir("META/src/lib/AMOC_Carbon_functions.jl"))
include(srcdir("script_functions.jl"))

#%%

model = base_model(); run(model)

df_pattern_results = DataFrame(
    Model=collect(keys(cmip6_model_names)), 
    SCC_IPSL=Vector{Union{Missing, Float64}}(missing, length(cmip6_model_names)),
    SCC_BCM=Vector{Union{Missing, Float64}}(missing, length(cmip6_model_names)),
    SCC_HADCM=Vector{Union{Missing, Float64}}(missing, length(cmip6_model_names)),
    SCC_Hadley=Vector{Union{Missing, Float64}}(missing, length(cmip6_model_names)),
    SCC_nopattern=Vector{Union{Missing, Float64}}(missing, length(cmip6_model_names)),
    ∆SCC_IPSL=Vector{Union{Missing, Float64}}(missing, length(cmip6_model_names)),
    ∆SCC_BCM=Vector{Union{Missing, Float64}}(missing, length(cmip6_model_names)),
    ∆SCC_HADCM=Vector{Union{Missing, Float64}}(missing, length(cmip6_model_names)),
    ∆SCC_Hadley=Vector{Union{Missing, Float64}}(missing, length(cmip6_model_names)),
    ∆SCC_nopattern=Vector{Union{Missing, Float64}}(missing, length(cmip6_model_names)),
    )

for (j, pattern) in enumerate(["IPSL", "BCM", "HADCM", "Hadley", false])
    println("Running pattern: $pattern")
    for (i, key) in enumerate(keys(cmip6_model_names))
        m_amoc_pattern = get_amoc_carbon_model(key; temp_pattern=pattern)
        update_param!(m_amoc_pattern, :AMOC_Carbon, :Delta_AMOC, 35)
        run(m_amoc_pattern)
        scc_amoc_pattern = calculate_scc(m_amoc_pattern, 2020, 10., 1.05)
            
        m_base_pattern = full_model(rcp="RCP4.5", ssp="SSP2", saf=false, interaction=false, pcf=false, omh=false, amaz=false, gis=false, ais=false, ism=false, amoc=pattern)
        if pattern != false
            p_vector = ones(dim_count(m_base_pattern, :time))
            p_vector[266] = 0
            update_param!(m_base_pattern, :AMOC_uniforms, p_vector)
            update_param!(m_base_pattern, :AMOC, :Delta_AMOC, 35)
        end
        update_param!(m_base_pattern, :Consumption, :damagepersist, 0.25)
        run(m_base_pattern)
        scc_base_pattern = calculate_scc(m_base_pattern, 2020, 10., 1.05)
            
        scc_change = delta_scc(scc_amoc_pattern, scc_base_pattern)
        df_pattern_results[i, 1+j] = round(scc_amoc_pattern, digits=2)
        df_pattern_results[i, 6+j] = round(scc_change, digits=2)
    end
end

CSV.write(datadir("exogenous_temp_patterns.csv"), df_pattern_results)

#%% Write to table

df_pattern_results = CSV.read(datadir("exogenous_temp_patterns.csv"), DataFrame)

table = ""
for (i, r) in enumerate(eachrow(df_pattern_results))
    row = "$(r.Model) & $(r.∆SCC_nopattern) \\% & $(r.∆SCC_IPSL) \\% & $(r.∆SCC_BCM) \\% & $(r.∆SCC_HADCM) \\% & $(r.∆SCC_Hadley) \\% "*"\\"*"\\"*"\n"
    table *= row
end

println(table)

#%% separate issue, raised by the same reviewer: can the difference in SCC changes be explained by timescales?

AMOC_pi_vals = CSV.read(srcdir("META/data/CMIP6_amoc/AMOC_pi_values.csv"), DataFrame)

weakening_wrt_2015 = filter(row -> row."AMOC variable" == "2100_weakening_to_2015", AMOC_pi_vals)
weakening_wrt_pi = filter(row -> row."AMOC variable" == "2100_weakening_to_pi", AMOC_pi_vals)

ratio_SCC_to_weakening = Dict()

for key in keys(cmip6_model_names)
    ∆SCC = df_pattern_results[findfirst(df_pattern_results.Model .== key), :∆SCC_nopattern]
    println("$(key) SCC change: $(∆SCC) %")
    weakening = weakening_wrt_2015[!, key][1]
    weakening_pi = weakening_wrt_pi[!, key][1]
    # filter(row -> row."Model" == key, weakening_wrt_2015)
    ratio_2015 = round(∆SCC/weakening, digits=3)
    ratio_SCC_to_weakening[key] = ratio_2015
    println("$(key) AMOC weakening: $(weakening) %")
    println("$(key) ratio: $(ratio_2015)")
    println("$(key) AMOC weakening to pi: $(weakening_pi) %")
    println("$(key) ratio: $(round(∆SCC/weakening_pi, digits=3))")
    println("-----------------------")
end

mean_ratio = mean(collect(values(ratio_SCC_to_weakening)))