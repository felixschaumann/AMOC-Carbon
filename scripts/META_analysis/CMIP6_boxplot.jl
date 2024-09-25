#%% GET PACKAGES
using DrWatson; @quickactivate "AMOC-Carbon"; cd(srcdir("META/src"))

using DelimitedFiles
using Distributions
using PythonPlot
using PythonCall
mpath = pyimport("matplotlib.path")

include(srcdir("script_functions.jl"))

#%% read in simulations

df = collect_results(datadir("simulations/META"))
df.sccs_mean = [mean(sccs) for sccs in df.sccs]

AMOC_pi_vals = CSV.read(srcdir("META/data/CMIP6_amoc/AMOC_pi_values.csv"), DataFrame)

#%%

n_MC = 10000
df = df[length.(df.MC_samples).==n_MC, :]
df = sort(df, :AMOC_runs)
df = sort(df, :sccs_mean)

noamoc_sccs_C_45, noamoc_MC_samples_C_45 = get_sccs(df, N_samples=n_MC, AMOC="none", Scenario="ssp245", AMOC_runs="C")
noamoc_sccs_C_26, noamoc_MC_samples_C_26 = get_sccs(df, N_samples=n_MC, AMOC="none", Scenario="ssp126", AMOC_runs="C")
noamoc_sccs_C_85, noamoc_MC_samples_C_85 = get_sccs(df, N_samples=n_MC, AMOC="none", Scenario="ssp585", AMOC_runs="C")
noamoc_sccs_T_45 , noamoc_MC_samples_T_45 = get_sccs(df, N_samples=n_MC, AMOC="none", Scenario="ssp245", AMOC_runs="T")

amoc_sccs_C_45, amoc_MC_samples_C_45 = get_sccs(df, N_samples=n_MC, AMOC="C", Scenario="ssp245")
amoc_sccs_C_26, amoc_MC_samples_C_26 = get_sccs(df, N_samples=n_MC, AMOC="C", Scenario="ssp126")
amoc_sccs_C_85, amoc_MC_samples_C_85 = get_sccs(df, N_samples=n_MC, AMOC="C", Scenario="ssp585")
amoc_sccs_T_45, amoc_MC_samples_T_45 = get_sccs(df, N_samples=n_MC, AMOC="T", Scenario="ssp245")

emucs = df[(length.(df.MC_samples).===n_MC) .& (df.AMOC.==="none") .& (df.AMOC_runs.==="C") .& (df.Scenario.==="ssp245"), :].emuc
emuc = length(emucs) == 1 ? emucs[1] : "various"
prtps = df[(length.(df.MC_samples).===n_MC) .& (df.AMOC.==="none") .& (df.AMOC_runs.==="C") .& (df.Scenario.==="ssp245"), :].prtp
prtp = length(prtps) == 1 ? prtps[1] : "various"
persists = df[(length.(df.MC_samples).===n_MC) .& (df.AMOC.==="none") .& (df.AMOC_runs.==="C") .& (df.Scenario.==="ssp245"), :].persist
persist = length(persists) == 1 ? persists[1] : "various"
Dams = df[(length.(df.MC_samples).===n_MC) .& (df.AMOC.==="none") .& (df.AMOC_runs.==="C") .& (df.Scenario.==="ssp245"), :].Dam
Dam = length(Dams) == 1 ? Dams[1] : "various"

#%% make SCCs relative

for i in 1:length(amoc_sccs_C_45)
    if noamoc_MC_samples_C_45[1] ==  amoc_MC_samples_C_45[i]
        println("ssp245 Carbon MC samples for run $i are identical to the baseline.")
    else
        error("ssp245 Carbon MC samples for run $i are not identical to the baseline.")
    end
end

for i in 1:length(amoc_sccs_C_26)
    if noamoc_MC_samples_C_26[1] ==  amoc_MC_samples_C_26[i]
        println("ssp126 Carbon MC samples for run $i are identical to the baseline.")
    else
        error("ss126 Carbon MC samples for run $i are not identical to the baseline.")
    end
end

for i in 1:length(amoc_sccs_C_85)
    if noamoc_MC_samples_C_85[1] ==  amoc_MC_samples_C_85[i]
        println("ssp585 Carbon MC samples for run $i are identical to the baseline.")
    else
        error("ssp585 Carbon MC samples for run $i are not identical to the baseline.")
    end
end

for i in 1:length(amoc_sccs_T_45)
    if noamoc_MC_samples_T_45[1] ==  amoc_MC_samples_T_45[i]
        println("Temperature MC samples for run $i are identical to the baseline.")
    else
        error("Temperature MC samples for run $i are not identical to the baseline.")
    end
end

Delta_amoc_sccs_C_45 = [delta_scc(amoc_sccs_C_45[i], noamoc_sccs_C_45[1]) for i in 1:length(amoc_sccs_C_45)]
Delta_amoc_sccs_C_26 = [delta_scc(amoc_sccs_C_26[i], noamoc_sccs_C_26[1]) for i in 1:length(amoc_sccs_C_26)]
Delta_amoc_sccs_C_85 = [delta_scc(amoc_sccs_C_85[i], noamoc_sccs_C_85[1]) for i in 1:length(amoc_sccs_C_85)]
Delta_amoc_sccs_T_45 = [delta_scc(amoc_sccs_T_45[i], noamoc_sccs_T_45[1]) for i in 1:length(amoc_sccs_T_45)]

#%% save SCC results to CSV

df_SCC_results = DataFrame()

row_names = ["mean_Delta_SCC", "std_Delta_SCC"]
df_SCC_results[!, :RowNames] = row_names             

column_names_45 = [
        "MPI-ESM1.2-LR",
        "ACCESS-ESM1-5",
        "CanESM5",
        "GISS-E2-1-G",
        "MIROC-ES2L",
        "CESM2",
        "NorESM2-LM"
        ]
column_names_26 = ["MPI-ESM1.2-LR_ssp126","NorESM2-LM_ssp126"]
column_names_85 = ["MPI-ESM1.2-LR_ssp585","NorESM2-LM_ssp585"]

for i in 1:7
    col_name = Symbol(column_names_45[i])
    df_SCC_results[!, col_name] = [
    mean(Delta_amoc_sccs_C_45[i]), std(Delta_amoc_sccs_C_45[i]), 
    ]
end
for i in 1:2
    col_name = Symbol(column_names_26[i])
    df_SCC_results[!, col_name] = [
    mean(Delta_amoc_sccs_C_26[i]), std(Delta_amoc_sccs_C_26[i]), 
    ]
end
for i in 1:2
    col_name = Symbol(column_names_85[i])
    df_SCC_results[!, col_name] = [
    mean(Delta_amoc_sccs_C_85[i]), std(Delta_amoc_sccs_C_85[i]), 
    ]
end

CSV.write(datadir("CMIP6_SCC_results_$(n_MC)_samples.csv"), df_SCC_results)

#%% boxplots in matplotlib

PythonPlot.rc("font", size = 15)
mpi_color = (0, 108, 102)./255

fig, ax = subplots(figsize=(17.8, 9.9));

x_positions = [
                0.6, .8, 1., 1.2,
                1.6, 1.7, 1.8,
                2.05, 2.4, 2.75, 3.1, 3.45, 
                3.7, 3.8, 3.9,
                ]

ax.text(1.6, -7.9, "SSP5-8.5", horizontalalignment="center", verticalalignment="bottom", color="rebeccapurple", fontsize=14, rotation=90)
ax.text(1.7, -7.9, "SSP2-4.5", horizontalalignment="center", verticalalignment="bottom", color="darkred", fontsize=14, rotation=90)
ax.text(1.8, -7.9, "SSP1-2.6", horizontalalignment="center", verticalalignment="bottom", color="darkorange", fontsize=14, rotation=90)
ax.text(3.7, -7.9, "SSP5-8.5", horizontalalignment="center", verticalalignment="bottom", color="rebeccapurple", fontsize=14, rotation=90)
ax.text(3.8, -7.9, "SSP2-4.5", horizontalalignment="center", verticalalignment="bottom", color="darkred", fontsize=14, rotation=90)
ax.text(3.9, -7.9, "SSP1-2.6", horizontalalignment="center", verticalalignment="bottom", color="darkorange", fontsize=14, rotation=90)

boxplot_artists = ax.boxplot(
    vcat(Delta_amoc_sccs_T_45, 
    [Delta_amoc_sccs_C_85[1]], 
    [Delta_amoc_sccs_C_45[1]], 
    [Delta_amoc_sccs_C_26[1]], 
    Delta_amoc_sccs_C_45[2:end-1], 
    [Delta_amoc_sccs_C_85[2]],
    [Delta_amoc_sccs_C_45[end]], 
    [Delta_amoc_sccs_C_26[2]], 
    ), 
    vert=true, sym="o", positions=x_positions, 
    showmeans=true, meanline=false, widths=0.3, patch_artist=true, 
    medianprops=Dict("color"=>"gray", "linewidth"=>2.5), 
    meanprops=Dict("marker"=>"x", "markersize"=>10, "markeredgecolor"=>"darkred", "markeredgewidth" => 3.5),
    boxprops=Dict("color"=>"black", 
    "linewidth"=>2.5),
    whiskerprops=Dict("color"=>"black", "linewidth"=>2.5),
    capprops=Dict("color"=>"black", "linewidth"=>2.5),
    flierprops=Dict("markerfacecolor"=>"gray", "markersize"=>2.5, "markeredgecolor"=>"gray"),
    showfliers=false,
    whis=[5, 95]
);

for i in 0:length(boxplot_artists["boxes"])-1
    PythonPlot.setp(boxplot_artists["boxes"][i]);
    PythonPlot.setp(boxplot_artists["whiskers"][i]);
    boxplot_artists["boxes"][i].set_facecolor("white");
    boxplot_artists["boxes"][i].set_alpha(1.);
    if i < 4
        boxplot_artists["means"][i].set_markeredgecolor("black");
    end
    if i == 4 || i == 12
        boxplot_artists["means"][i].set_markeredgecolor("rebeccapurple");
    end
    if i == 6 || i == 14
        boxplot_artists["means"][i].set_markeredgecolor("darkorange");
    end
    if i in [4, 5, 6, 12, 13, 14]
        box_coords = boxplot_artists["boxes"][i].get_path().vertices
        box_center = (box_coords[0, 0] + box_coords[2, 0]) / 2
        box_coords[0:5, 0] = box_center + (box_coords[0:5, 0] - box_center) / 3
        boxplot_artists["boxes"][i].set_path(mpath.Path(box_coords, boxplot_artists["boxes"][i].get_path().codes))
        boxplot_artists["medians"][i].set_xdata([box_coords[0, 0], box_coords[1, 0]])
        boxplot_artists["caps"][2*i].set_xdata([box_coords[0, 0]+0.02, box_coords[1, 0]-0.02])
        boxplot_artists["caps"][2*i+1].set_xdata([box_coords[0, 0]+0.02, box_coords[1, 0]-0.02])
    end
end

ax.set_ylabel("∆SCC [%]");
ylims = (-8., 6.);
ax.set_ylim(ylims);
ax.set_yticks(collect(-7:1:5.));
# ax.set_yticks(collect(-7.5:0.5:5.5), minor=true);
ax.tick_params(axis="both", which="major", width=1.5)
# ax.tick_params(axis="both", which="minor", width=1.)


ax.set_xlim(0.4, 4.1);
ax.set_xticklabels(vcat([
                        "67%\nHadley", 
                        "27%\nIPSL",
                        "24%\nBCM", 
                        "7%\nHADCM", 
                        ], 
                        "\n",
                        "32.3% - 12.4%\nMPI-ESM", # 22.5%
                        "\n",
                        "23.5%\nACCESS-ESM ",
                        "23.3%\nCanESM",
                        "42.5%\nGISS",
                        "42.1%\nMIROC",
                        "47.9%\nCESM",
                        "\n",
                        "60.1% - 39.8%\nNorESM", # 49.3%
                        "\n",
                        ), fontsize=15);

ax.text(0.9, 1.02*ylims[2], "META temperature effect", horizontalalignment="center", verticalalignment="bottom", color="black", weight="bold", alpha=1.);
ax.text(2.75, 1.02*ylims[2], "CMIP6-calibrated carbon effect", horizontalalignment="center", verticalalignment="bottom", color="darkred", weight="bold", alpha=1.);

ax.text(0.1, -8.95, "Weakening\nModel", horizontalalignment="left", verticalalignment="bottom", fontsize=15)
ax.text(0.1, -9.6, "Method", horizontalalignment="left", verticalalignment="bottom", fontsize=15)
ax.text(0.9, -9.6, "──── Stochastic tipping ────", horizontalalignment="center", verticalalignment="bottom", fontsize=15)
ax.text(2.75, -9.6, "─────────────────────────── Gradual weakening ───────────────────────────", horizontalalignment="center", verticalalignment="bottom", fontsize=15)

bbox_props = Dict("boxstyle" => "round,pad=0.3", "edgecolor" => "black", "facecolor" => "white", "linewidth" => 1.5)

ax.text(0.7, 5.5, latexstring("SCC for year 2020\nalong SSP2-4.5 (SSP5-8.5, SSP1-2.6)\nwith \$\\eta=$(emuc)\$ and \$\\delta=$(prtp*100)\\%\$,\n$(Dam) damages, \$\\varphi=$(persist)\$:\n\$ SCC_\\text{no AMOC} = $(Int(round(mean([mean(noamoc_sccs_T_45[1]), mean(noamoc_sccs_C_45[1])]), digits=0))) \\\$ \$"), horizontalalignment="left", verticalalignment="top", fontsize=15, bbox=bbox_props)

ax.text(0.873, 4.7, "SSP2-4.5", horizontalalignment="left", verticalalignment="bottom", color="darkred", fontsize=15)
ax.text(1.162, 4.7, "SSP5-8.5", horizontalalignment="left", verticalalignment="bottom", color="rebeccapurple", fontsize=15)
ax.text(1.446, 4.7, "SSP1-2.6", horizontalalignment="left", verticalalignment="bottom", color="darkorange", fontsize=15)

ax.axhline(0, color="black", linewidth=3., alpha=0.9);
ax.axhline(1, color="black", linestyle="--", linewidth=1., alpha=1.);
ax.axhline(-1, color="black", linestyle="--", linewidth=1., alpha=1.);
ax.axhline(2, color="black", linestyle="--", linewidth=1., alpha=1.);
ax.axhline(-2, color="black", linestyle="--", linewidth=1., alpha=1.);
ax.axhline(3, color="black", linestyle="--", linewidth=1., alpha=1.);
ax.axhline(-3, color="black", linestyle="--", linewidth=1., alpha=1.);
ax.axhline(4, color="black", linestyle="--", linewidth=1., alpha=1.);
ax.axhline(-4, color="black", linestyle="--", linewidth=1., alpha=1.);
ax.axhline(5, color="black", linestyle="--", linewidth=1., alpha=1.);
ax.axhline(-5, color="black", linestyle="--", linewidth=1., alpha=1.);
ax.axhline(-6, color="black", linestyle="--", linewidth=1., alpha=1.);
ax.axhline(-7, color="black", linestyle="--", linewidth=1., alpha=1.);

for spine in ax.spines.values()
    spine.set_linewidth(1.5)
end

ax.fill_between([0, 1.4], -100, 50, color="lightgray", alpha=1.)

fig.savefig(plotsdir("CMIP6_boxplot.pdf"), dpi=400, bbox_inches="tight");
fig