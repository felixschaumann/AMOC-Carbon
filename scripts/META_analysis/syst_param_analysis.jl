#%%
println("Starting script")
flush(stdout)
using DrWatson; @quickactivate "AMOC-Carbon"; cd(srcdir("META/src"))
println("Activated project environment")
flush(stdout)

using Sockets # for checking hostname

include(srcdir("META/src/scc.jl"))

include(srcdir("META/src/lib/AMOC_Carbon_functions.jl"))
include(srcdir("script_functions.jl"))

#%%

model = base_model(rcp="RCP4.5", ssp="SSP2", tdamage="pointestimate", slrdamage="mode")
run(model)

#%% functions for restarting from saved data

function check_existence_of_runs(n_samples::Int; type::String="scc")
    df = collect_results(datadir("simulations/META" * (type=="scc" ? "" : "_proj")))
    df = df[length.(df.MC_samples).==n_samples, :]
    samples = df.MC_samples
    if length(samples) == 0
        return []
    end
    if all(s -> s == samples[1], samples)
        return samples[1]
    else
        println("Not all MC samples of length $(n_samples) and type $(type) are the same")
    end
end

#%%

n_run = parse(Int, ENV["N_RUN"])
# n_run = 1
println("Calculating only run $n_run.")
n_samples = parse(Int, ENV["N_SAMPLES"])
# n_samples = 1000
run_mc_proj = parse(Bool, ENV["RUN_MC_PROJ"])
# run_mc_proj = false
run_mc_scc = parse(Bool, ENV["RUN_MC_SCC"])
# run_mc_scc = true

existing_samples = []
if run_mc_proj && run_mc_scc
    println("Running both MC types - not advised!")
    existing_samples = check_existence_of_runs(n_samples, type=(run_mc_scc ? "scc" : "proj"))
elseif !run_mc_proj && !run_mc_scc
    println("Running neither MC type - nothing should happen.")
else 
    existing_samples = check_existence_of_runs(n_samples, type=(run_mc_scc ? "scc" : "proj"))
end

sample_id_subset = []

if existing_samples == [] && (run_mc_proj || run_mc_scc)
    if gethostname()=="FSch"
        data_dir = joinpath("/Users/fsch/.julia/packages/MimiFAIRv2/7ivpN/data", "large_constrained_parameter_files");
    else
        data_dir = joinpath("/home/m/m300940/.julia/packages/MimiFAIRv2/7ivpN/src/../data", "large_constrained_parameter_files");
    end
    thermal_params = DataFrame(load(joinpath(data_dir, "julia_constrained_thermal_parameters_average_probs.csv")));
    rand_indices = sort(sample(1:93995, n_samples, replace=false));
    sample_id_subset = thermal_params[rand_indices, :sample_id]
    println("Selected random sample of $(n_samples) samples")
    flush(stdout)
else
    sample_id_subset = existing_samples
    println("Using existing sample of $(n_samples) samples")
    if !run_mc_proj && !run_mc_scc
        println("No MC type selected, so sample will be empty.")
    end
    flush(stdout)
end

#%%

regressions_df = CSV.read(datadir("carbon_flux_regressions_10yrs.csv"), DataFrame)
Eta = regressions_df[1, :].estimate
Eta_stderr = regressions_df[1, :].stderror_est

#%%

allparams = Dict(
    "MC" => ["MC"],
    "MC_samples" => @onlyif("MC"=="MC", [sample_id_subset]),
    "Dam" => ["BHM"],
    "emuc" => [1.05],
    "prtp" => [0.005],
    "persist" => [0.25],
    "AMOC_runs" => [run_mc_proj ? "C" : "C", "T"],
    "AMOC" => ["none", @onlyif("AMOC_runs"=="T", "T"), @onlyif("AMOC_runs"=="C", "C")],
    "DecreaseCalib" => @onlyif("AMOC" in ["C", "T&C" ], ["CanESM5", "NorESM2-LM","CESM2","MPI-ESM1.2-LR","GISS-E2-1-G","ACCESS-ESM1-5","MIROC-ES2L"]), # ESM name string for AMOC projection, number for stylised decrease from 19Sv in 2010
    "Scenario" => ["ssp245", @onlyif("DecreaseCalib" in ["NorESM2-LM", "MPI-ESM1.2-LR"], "ssp126"), @onlyif("DecreaseCalib" in ["NorESM2-LM", "MPI-ESM1.2-LR"], "ssp585"), @onlyif(("AMOC_runs"=="C") & ("AMOC"=="none"), "ssp126"), @onlyif(("AMOC_runs"=="C") & ("AMOC"=="none"), "ssp585")],
    "Calib" => @onlyif("AMOC" in ["T", "T&C"], ["IPSL", "BCM", "Hadley", "HADCM"]), # AMOC calibration (Anthoff et al. 2016)
    "Eta" => @onlyif("AMOC" in ["C", "T&C"], [Eta*22/6]), # CO2 flux imbalance per Sv AMOC decrease [GtCO2/Sv/yr]
    "Eta_stderr" => @onlyif("AMOC" in ["C", "T&C"], [Eta_stderr*22/6]), # standard error of CO2 flux imbalance per Sv AMOC decrease [GtCO2/Sv/yr]
)


dicts = dict_list(allparams)
println("Defined parameter combinations: ", length(dicts))
flush(stdout)

#%% produce or load

if run_mc_scc
    for (i, d) in enumerate(dicts)

        filename_d = savename("META_SCC_$(n_samples)_samples", d)

        if d["MC"] == "500preset" # function doesn't yet exist
            produce_or_load(run_amoc_preset_sccs, d, datadir("simulations/META"); filename=filename_d)
            println("Finished run $i: $d")
            flush(stdout)
        elseif d["MC"] == "MC"
            if i==n_run
                println("Running SCC for run $i.")
                flush(stdout)
                produce_or_load(run_amoc_mc_sccs, d, datadir("simulations/META"); filename=filename_d)
                println("Finished run $i: $d")
                flush(stdout)
            end
        else
            println("Unknown MC type")
            flush(stdout)
        end
    end
end

#%% produce or load

if run_mc_proj
    for (i, d) in enumerate(dicts)

        filename_d = savename("META_proj_$(n_samples)_samples", d)

        if i==n_run && d["AMOC_runs"] !== "T"
            println("Running MC projection for run $i.")
            flush(stdout)
            produce_or_load(run_amoc_carbon_mc, d, datadir("simulations/META_proj"); filename=filename_d)
            println("Finished run $i: $d")
            flush(stdout)
        end
    end
end

if n_run > length(dicts)
    println("Run number $n_run is larger than the number of parameter combinations. Nothing will be run.")
    flush(stdout)
end

println("Finished Julia script")
flush(stdout)