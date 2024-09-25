#%% GET PACKAGES
using DrWatson; @quickactivate "AMOC-Carbon"; cd(srcdir("META/src"))

using DelimitedFiles
using Distributions
using PythonPlot
using PythonCall
mpath = pyimport("matplotlib.path")

include(srcdir("script_functions.jl"))

#%% read in simulations

n_MC = 1000

df = collect_results(datadir("simulations/META"))
df = df[(length.(df.MC_samples).==n_MC) .& (df.AMOC_runs.==="T"), :]

#%%

noamoc_sccs_T_26 , noamoc_MC_samples_T_26 = get_sccs(df, N_samples=n_MC, AMOC="none", Scenario="ssp126", AMOC_runs="T")
noamoc_sccs_T_45 , noamoc_MC_samples_T_45 = get_sccs(df, N_samples=n_MC, AMOC="none", Scenario="ssp245", AMOC_runs="T")
noamoc_sccs_T_85 , noamoc_MC_samples_T_85 = get_sccs(df, N_samples=n_MC, AMOC="none", Scenario="ssp585", AMOC_runs="T")

amoc_sccs_T_26, amoc_MC_samples_T_26 = get_sccs(df, N_samples=n_MC, AMOC="T", Scenario="ssp126")
amoc_sccs_T_45, amoc_MC_samples_T_45 = get_sccs(df, N_samples=n_MC, AMOC="T", Scenario="ssp245")
amoc_sccs_T_85, amoc_MC_samples_T_85 = get_sccs(df, N_samples=n_MC, AMOC="T", Scenario="ssp585")

emucs = df[(length.(df.MC_samples).===n_MC) .& (df.AMOC.==="none") .& (df.AMOC_runs.==="T") .& (df.Scenario.==="ssp245"), :].emuc
emuc = length(emucs) == 1 ? emucs[1] : "various"
prtps = df[(length.(df.MC_samples).===n_MC) .& (df.AMOC.==="none") .& (df.AMOC_runs.==="T") .& (df.Scenario.==="ssp245"), :].prtp
prtp = length(prtps) == 1 ? prtps[1] : "various"
persists = df[(length.(df.MC_samples).===n_MC) .& (df.AMOC.==="none") .& (df.AMOC_runs.==="T") .& (df.Scenario.==="ssp245"), :].persist
persist = length(persists) == 1 ? persists[1] : "various"
Dams = df[(length.(df.MC_samples).===n_MC) .& (df.AMOC.==="none") .& (df.AMOC_runs.==="T") .& (df.Scenario.==="ssp245"), :].Dam
Dam = length(Dams) == 1 ? Dams[1] : "various"

#%% make SCCs relative

for i in 1:length(amoc_sccs_T_26)
    if noamoc_MC_samples_T_26[1] ==  amoc_MC_samples_T_26[i]
        println("ssp126 Temperature MC samples for run $i are identical to the baseline.")
    else
        error("ss126 Temperature MC samples for run $i are not identical to the baseline.")
    end
end

for i in 1:length(amoc_sccs_T_45)
    if noamoc_MC_samples_T_45[1] ==  amoc_MC_samples_T_45[i]
        println("ssp245 Temperature MC samples for run $i are identical to the baseline.")
    else
        error("ssp245 Temperature MC samples for run $i are not identical to the baseline.")
    end
end

for i in 1:length(amoc_sccs_T_85)
    if noamoc_MC_samples_T_85[1] ==  amoc_MC_samples_T_85[i]
        println("ssp585 Temperature MC samples for run $i are identical to the baseline.")
    else
        error("ssp585 Temperature MC samples for run $i are not identical to the baseline.")
    end
end

Delta_amoc_sccs_T_26 = [delta_scc(amoc_sccs_T_26[i], noamoc_sccs_T_26[1]) for i in 1:length(amoc_sccs_T_26)]
Delta_amoc_sccs_T_45 = [delta_scc(amoc_sccs_T_45[i], noamoc_sccs_T_45[1]) for i in 1:length(amoc_sccs_T_45)]
Delta_amoc_sccs_T_85 = [delta_scc(amoc_sccs_T_85[i], noamoc_sccs_T_85[1]) for i in 1:length(amoc_sccs_T_85)]

#%% boxplots in matplotlib

PythonPlot.rc("font", size = 15)
mpi_color = (0, 108, 102)./255

fig, ax = subplots(figsize=(17.8, 9.9));

x_positions = [
                .6, .8, 1.,
                1.4, 1.6, 1.8,
                2.2, 2.4, 2.6, 
                3., 3.2, 3.4,
                ]

ax.text(.6, -10.75, "SSP5-8.5", horizontalalignment="center", verticalalignment="bottom", color="rebeccapurple", fontsize=14, rotation=90)
ax.text(.8, -10.75, "SSP2-4.5", horizontalalignment="center", verticalalignment="bottom", color="darkred", fontsize=14, rotation=90)
ax.text(1., -10.75, "SSP1-2.6", horizontalalignment="center", verticalalignment="bottom", color="darkorange", fontsize=14, rotation=90)
ax.text(1.4, -10.75, "SSP5-8.5", horizontalalignment="center", verticalalignment="bottom", color="rebeccapurple", fontsize=14, rotation=90)
ax.text(1.6, -10.75, "SSP2-4.5", horizontalalignment="center", verticalalignment="bottom", color="darkred", fontsize=14, rotation=90)
ax.text(1.8, -10.75, "SSP1-2.6", horizontalalignment="center", verticalalignment="bottom", color="darkorange", fontsize=14, rotation=90)
ax.text(2.2, -10.75, "SSP5-8.5", horizontalalignment="center", verticalalignment="bottom", color="rebeccapurple", fontsize=14, rotation=90)
ax.text(2.4, -10.75, "SSP2-4.5", horizontalalignment="center", verticalalignment="bottom", color="darkred", fontsize=14, rotation=90)
ax.text(2.6, -10.75, "SSP1-2.6", horizontalalignment="center", verticalalignment="bottom", color="darkorange", fontsize=14, rotation=90)
ax.text(3., -10.75, "SSP5-8.5", horizontalalignment="center", verticalalignment="bottom", color="rebeccapurple", fontsize=14, rotation=90)
ax.text(3.2, -10.75, "SSP2-4.5", horizontalalignment="center", verticalalignment="bottom", color="darkred", fontsize=14, rotation=90)
ax.text(3.4, -10.75, "SSP1-2.6", horizontalalignment="center", verticalalignment="bottom", color="darkorange", fontsize=14, rotation=90)

boxplot_artists = ax.boxplot(
    vcat(
        [Delta_amoc_sccs_T_85[3], Delta_amoc_sccs_T_45[3], Delta_amoc_sccs_T_26[3]],
        [Delta_amoc_sccs_T_85[4], Delta_amoc_sccs_T_45[4], Delta_amoc_sccs_T_26[4]],
        [Delta_amoc_sccs_T_85[1], Delta_amoc_sccs_T_45[1], Delta_amoc_sccs_T_26[1]],
        [Delta_amoc_sccs_T_85[2], Delta_amoc_sccs_T_45[2], Delta_amoc_sccs_T_26[2]],
    ), 
    vert=true, sym="o", positions=x_positions, 
    showmeans=true, meanline=false, widths=0.1, patch_artist=true, 
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
    if i == 0 || i == 3 || i == 6 || i == 9
        boxplot_artists["means"][i].set_markeredgecolor("rebeccapurple");
    end
    if i == 1 || i == 4 || i == 7 || i == 10
        boxplot_artists["means"][i].set_markeredgecolor("darkred");
    end
    if i == 2 || i == 5 || i == 8 || i == 11
        boxplot_artists["means"][i].set_markeredgecolor("darkorange");
    end
end

ax.set_ylabel("∆SCC [%]");
ylims = (-11, 0.);
ax.set_ylim(ylims);
ax.set_yticks(collect(-10:1:-1.));
ax.tick_params(axis="both", which="major", width=1.5)


ax.set_xlim(0.4, 3.6);
ax.set_xticklabels([
                        "", 
                        "67%\nHadley", 
                        "", 
                        "",
                        "27%\nIPSL",
                        "",
                        "", 
                        "24%\nBCM", 
                        "", 
                        "", 
                        "7%\nHADCM", 
                        ""], 
                        fontsize=15);

ax.text(2, 1.03*ylims[2], "Sensitivity of temperature effect to scenario", horizontalalignment="center", verticalalignment="bottom", color="black", weight="bold", alpha=1.);

ax.text(0.1, -11.75, "Weakening\nModel", horizontalalignment="left", verticalalignment="bottom", fontsize=15)

bbox_props = Dict("boxstyle" => "round,pad=0.3", "edgecolor" => "black", "facecolor" => "white", "linewidth" => 1.5)
ax.text(2.5, -6.5, latexstring("SCC for year 2020\nalong SSP2-4.5 (SSP5-8.5, SSP1-2.6)\nwith \$\\eta=$(emuc)\$ and \$\\delta=$(prtp*100)\\%\$,\n$(Dam) damages, \$\\varphi=$(persist)\$."), horizontalalignment="left", verticalalignment="top", fontsize=15, bbox=bbox_props)

ax.text(2.649, -7.125, "SSP2-4.5", horizontalalignment="left", verticalalignment="bottom", color="darkred", fontsize=15)
ax.text(2.899, -7.125, "SSP5-8.5", horizontalalignment="left", verticalalignment="bottom", color="rebeccapurple", fontsize=15)
ax.text(3.145, -7.125, "SSP1-2.6", horizontalalignment="left", verticalalignment="bottom", color="darkorange", fontsize=15)

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
ax.axhline(-8, color="black", linestyle="--", linewidth=1., alpha=1.);
ax.axhline(-9, color="black", linestyle="--", linewidth=1., alpha=1.);
ax.axhline(-10, color="black", linestyle="--", linewidth=1., alpha=1.);

for spine in ax.spines.values()
    spine.set_linewidth(1.5)
end

ax.fill_between([0, 4], -100, 50, color="lightgray", alpha=1.)

fig.savefig(plotsdir("T_scenario_sensitivity_boxplot.pdf"), dpi=400, bbox_inches="tight");
fig