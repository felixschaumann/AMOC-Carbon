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