using CSV, DataFrames

#%%
# FUNCTIONS FOR RUNNING SCRIPTS
##########################################################################################

function get_carbon_data(co2_scenario::String)
    # Read the CSV file as a string
    data = read(datadir("Carbon_time_series/$(co2_scenario)/time.csv"), String)
    # Split the string into an array
    data = split(data, ",")
    # Convert the array elements to the desired type (e.g., Float64)
    data = parse.(Int64, data)
    # Create a DataFrame with the array as a column
    df = DataFrame(Year = data)

    for file in readdir(datadir("Carbon_time_series/$(co2_scenario)/"))
        if occursin("csv", file) && !occursin("time", file) && !occursin("flux", file) && !occursin("co2_conc", file)
            data = read(datadir("Carbon_time_series/$(co2_scenario)/") * file, String)
            data = split(data, ",")
            data = parse.(Float64, data)
            if length(data) == length(df.Year)-1
                df[!, Symbol(chop(file, tail=4))] = vcat(missing, data)
            elseif length(data) == length(df.Year)
                df[!, Symbol(chop(file, tail=4))] = data
            end
        end
    end
    return df
end

function running_mean(x::Vector, n::Int)
    if n==0
        return x
    end
    cumsum_values = cumsum(skipmissing(x))
    rm_vector = (cumsum_values[n:end] .- vcat([0], cumsum_values[1:end-n])) / (n)
    if n%2 == 0
        return vcat(fill(missing, Int((n-2)/2)), rm_vector, fill(missing, Int(n/2)))
    else
        return vcat(fill(missing, Int((n-1)/2)), rm_vector, fill(missing, Int((n-1)/2)))
    end
end

function delta_scc(mod_sccs::Union{Vector, Float64}, base_sccs::Union{Vector, Float64})
    return (mod_sccs .- base_sccs) ./ base_sccs .* 100
end

function get_sccs(df::DataFrame; N_samples::Int, AMOC::String, Scenario::String, AMOC_runs::Union{String, Nothing}=nothing)
    
    condition = (length.(df.MC_samples) .== N_samples) .& (df.AMOC .== AMOC) .& (df.Scenario .== Scenario)
    
    # Add AMOC_runs condition if the column exists and the parameter is provided
    if !isnothing(AMOC_runs)
        condition = condition .& (df.AMOC_runs .== AMOC_runs)
    end

    return df[condition, :].sccs, df[condition, :].MC_samples
end