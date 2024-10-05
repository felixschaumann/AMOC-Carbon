#%%
using DrWatson; @quickactivate "AMOC-Carbon"; cd(srcdir("META/src"))

using PythonPlot

include(srcdir("script_functions.jl"))

#%% Load data

df_ssp = get_carbon_data("ssp245")
cutoff = 140 # 140 for full 1pct run
df_pct = get_carbon_data("1pct")[1:cutoff, :]

####################################################################################
####################################################################################
####################################################################################
#
#%% Plot carbon storage over AMOC strength for both scenarios
#
####################################################################################
####################################################################################
####################################################################################

PythonPlot.rc("font", size = 15)
fig, axs = subplots(2, 2, figsize=(17.8,10.4), width_ratios=[28, 17])
axs[0][0].sharex(axs[1][0])
axs[0][1].sharex(axs[1][1])

legend_colors = ["dimgrey", "lightblue", "peachpuff", "dodgerblue", "salmon", "darkblue", "darkred",]

lines = []
push!(lines, axs[0][1].plot(df_ssp.Year[2:end], -collect(skipmissing(df_ssp.diff_dissic_ssp245bgcLR - df_ssp.diff_dissic_ssp245bgcLR)), color=legend_colors[1], linewidth=4))
push!(lines, axs[0][1].plot(df_ssp.Year[2:end], -collect(skipmissing(df_ssp.diff_dissic_ssp245bgcLR - df_ssp.diff_dissic_g01ssp245bgcLR)), color=legend_colors[2], linewidth=3, clip_on=false))
push!(lines, axs[0][1].plot(df_ssp.Year[2:end], -collect(skipmissing(df_ssp.diff_dissic_ssp245bgcLR - df_ssp.diff_dissic_u01ssp245bgcLR)), color=legend_colors[3], linewidth=3, clip_on=false))
push!(lines, axs[0][1].plot(df_ssp.Year[2:end], -collect(skipmissing(df_ssp.diff_dissic_ssp245bgcLR - df_ssp.diff_dissic_g03ssp245bgcLR)), color=legend_colors[4], linewidth=3, clip_on=false))
push!(lines, axs[0][1].plot(df_ssp.Year[2:end], -collect(skipmissing(df_ssp.diff_dissic_ssp245bgcLR - df_ssp.diff_dissic_u03ssp245bgcLR)), color=legend_colors[5], linewidth=3, clip_on=false))
push!(lines, axs[0][1].plot(df_ssp.Year[2:end], -collect(skipmissing(df_ssp.diff_dissic_ssp245bgcLR - df_ssp.diff_dissic_g05ssp245bgcLR)), color=legend_colors[6], linewidth=3, clip_on=false))
push!(lines, axs[0][1].plot(df_ssp.Year[2:end], -collect(skipmissing(df_ssp.diff_dissic_ssp245bgcLR - df_ssp.diff_dissic_u05ssp245bgcLR)), color=legend_colors[7], linewidth=3, clip_on=false))

axs[0][1].set_ylim(-35, 0)
axs[0][1].spines["bottom"].set_visible(false)
axs[0][1].spines["top"].set_visible(false)
axs[0][1].spines["left"].set_visible(false)
axs[0][1].spines["right"].set_visible(false)
axs[0][1].get_xaxis().set_visible(false)
axs[0][1].get_yaxis().set_visible(false)

legend_labels = ["no hosing", "g01 hosing", "u01 hosing", "g03 hosing", "u03 hosing", "g05 hosing", "u05 hosing",]

for (label, color) in zip(legend_labels, legend_colors)
    axs[0][0].text(0, 0.462 - 0.066 * findfirst(==(label), legend_labels), label, transform=axs[0][0].transAxes, fontsize=15, color=color, va="bottom", weight="bold")
    axs[0][1].text(1, 0.462 - 0.066 * findfirst(==(label), legend_labels), label, transform=axs[0][1].transAxes, fontsize=15, color=color, va="bottom", ha="right", weight="bold")
end

axs[1][0].text(0.5, 0.05, "1% per year \$\\mathbf{CO_2}\$ increase", transform=axs[1][0].transAxes, fontsize=15, weight="bold", ha="center")
axs[1][1].text(0.5, 0.05, "SSP2-4.5 \$\\mathbf{CO_2}\$ increase", transform=axs[1][1].transAxes, fontsize=15, weight="bold", ha="center")

fig.text(0.03, 0.95, "Hosing-induced difference in ocean carbon storage", ha="left", fontsize=15, weight="bold")
fig.text(0.03, 0.48, "Hosing-induced AMOC strength", ha="left", fontsize=15, weight="bold")

fig.text(0.0645, 0.905, "a", ha="left", fontsize=20, weight="bold")
fig.text(0.0645, 0.42, "b", ha="left", fontsize=20, weight="bold")
fig.text(0.65, 0.905, "c", ha="left", fontsize=20, weight="bold")
fig.text(0.65, 0.42, "d", ha="left", fontsize=20, weight="bold")

axs[1][1].plot(df_ssp.Year, df_ssp.var"total_amoc26.5N_ssp245bgcLR", color=legend_colors[1], linewidth=3)
axs[1][1].plot(df_ssp.Year, df_ssp.var"total_amoc26.5N_g01ssp245bgcLR", color=legend_colors[2], linewidth=3, clip_on=false)
axs[1][1].plot(df_ssp.Year, df_ssp.var"total_amoc26.5N_u01ssp245bgcLR", color=legend_colors[3], linewidth=3, clip_on=false)
axs[1][1].plot(df_ssp.Year, df_ssp.var"total_amoc26.5N_g03ssp245bgcLR", color=legend_colors[4], linewidth=3, clip_on=false)
axs[1][1].plot(df_ssp.Year, df_ssp.var"total_amoc26.5N_u03ssp245bgcLR", color=legend_colors[5], linewidth=3, clip_on=false)
axs[1][1].plot(df_ssp.Year, df_ssp.var"total_amoc26.5N_g05ssp245bgcLR", color=legend_colors[6], linewidth=3, clip_on=false)
axs[1][1].plot(df_ssp.Year, df_ssp.var"total_amoc26.5N_u05ssp245bgcLR", color=legend_colors[7], linewidth=3, clip_on=false)

axs[1][1].set_ylim(0, 23)
axs[1][1].tick_params(axis="y")
axs[1][1].spines["left"].set_visible(false)
axs[1][1].spines["top"].set_visible(false)
axs[1][1].spines["right"].set_visible(false)

axs[1][1].set_xlabel("Year")
axs[1][1].set_xlim(2015, 2015+85)
axs[1][1].get_yaxis().set_visible(false)
axs[1][1].set_xticks(collect(2010:15:2100), collect(2010:15:2100))

push!(lines, axs[0][0].plot(df_pct.Year[2:end].-1850, -collect(skipmissing(df_pct.diff_dissic_1pctbgcLR - df_pct.diff_dissic_1pctbgcLR)), color=legend_colors[1], linewidth=4))
push!(lines, axs[0][0].plot(df_pct.Year[2:end].-1850, -collect(skipmissing(df_pct.diff_dissic_1pctbgcLR - df_pct.diff_dissic_g011pctbgcLR)), color=legend_colors[2], linewidth=3, clip_on=false))
push!(lines, axs[0][0].plot(df_pct.Year[2:end].-1850, -collect(skipmissing(df_pct.diff_dissic_1pctbgcLR - df_pct.diff_dissic_u011pctbgcLR)), color=legend_colors[3], linewidth=3, clip_on=false))
push!(lines, axs[0][0].plot(df_pct.Year[2:end].-1850, -collect(skipmissing(df_pct.diff_dissic_1pctbgcLR - df_pct.diff_dissic_g031pctbgcLR)), color=legend_colors[4], linewidth=3, clip_on=false))
push!(lines, axs[0][0].plot(df_pct.Year[2:end].-1850, -collect(skipmissing(df_pct.diff_dissic_1pctbgcLR - df_pct.diff_dissic_u031pctbgcLR)), color=legend_colors[5], linewidth=3, clip_on=false))
push!(lines, axs[0][0].plot(df_pct.Year[2:end].-1850, -collect(skipmissing(df_pct.diff_dissic_1pctbgcLR - df_pct.diff_dissic_g051pctbgcLR)), color=legend_colors[6], linewidth=3, clip_on=false))
push!(lines, axs[0][0].plot(df_pct.Year[2:end].-1850, -collect(skipmissing(df_pct.diff_dissic_1pctbgcLR - df_pct.diff_dissic_u051pctbgcLR)), color=legend_colors[7], linewidth=3, clip_on=false))

axs[0][0].set_ylabel("PgC")
axs[0][0].set_xlabel("")
axs[0][0].set_ylim(-35, 0)
axs[0][0].tick_params(axis="y")
axs[0][0].spines["bottom"].set_visible(false)
axs[0][0].spines["top"].set_visible(false)
axs[0][0].spines["right"].set_visible(false)
axs[0][0].spines["left"].set_position(("axes", -0.05))
axs[0][0].get_xaxis().set_visible(false)

axs[1][0].plot(df_pct.Year.-1850, df_pct.var"total_amoc26.5N_1pctbgcLR", color=legend_colors[1], linewidth=3, clip_on=false)
axs[1][0].plot(df_pct.Year.-1850, df_pct.var"total_amoc26.5N_g011pctbgcLR", color=legend_colors[2], linewidth=3, clip_on=false)
axs[1][0].plot(df_pct.Year.-1850, df_pct.var"total_amoc26.5N_u011pctbgcLR", color=legend_colors[3], linewidth=3, clip_on=false)
axs[1][0].plot(df_pct.Year.-1850, df_pct.var"total_amoc26.5N_g031pctbgcLR", color=legend_colors[4], linewidth=3, clip_on=false)
axs[1][0].plot(df_pct.Year.-1850, df_pct.var"total_amoc26.5N_u031pctbgcLR", color=legend_colors[5], linewidth=3, clip_on=false)
axs[1][0].plot(df_pct.Year.-1850, df_pct.var"total_amoc26.5N_g051pctbgcLR", color=legend_colors[6], linewidth=3, clip_on=false)
axs[1][0].plot(df_pct.Year.-1850, df_pct.var"total_amoc26.5N_u051pctbgcLR", color=legend_colors[7], linewidth=3, clip_on=false)

axs[1][0].set_ylabel("Sv")
axs[1][0].set_ylim(0, 25)
axs[1][0].tick_params(axis="y")
axs[1][0].spines["right"].set_visible(false)
axs[1][0].spines["top"].set_visible(false)
axs[1][0].spines["left"].set_position(("axes", -0.05))

axs[1][0].set_xlabel("Year")
axs[1][0].set_xlim(0, 140)
axs[1][0].set_xticks(collect(0:20:140), collect(0:20:140))

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.subplots_adjust(hspace=0.3)
fig.savefig(plotsdir("carbon_storage_plot_combined.pdf"), dpi=400, bbox_inches="tight")
fig

####################################################################################
####################################################################################
####################################################################################
#
#%% Plot carbon storage over AMOC strength for SSP2-4.5 only
#
####################################################################################
####################################################################################
####################################################################################

PythonPlot.rc("font", size=18)
fig, axs = subplots(2, 1, figsize=(8.5, 6.0), sharex=true)

minor_fontsize = 15

legend_colors = ["dimgrey", "lightblue", "peachpuff", "dodgerblue", "salmon", "darkblue", "darkred",]

lines = []
push!(lines, axs[0].plot(df_ssp.Year[2:end], -collect(skipmissing(df_ssp.diff_dissic_ssp245bgcLR - df_ssp.diff_dissic_ssp245bgcLR)), color=legend_colors[1], linewidth=2.5, clip_on=false))
push!(lines, axs[0].plot(df_ssp.Year[2:end], -collect(skipmissing(df_ssp.diff_dissic_ssp245bgcLR - df_ssp.diff_dissic_g01ssp245bgcLR)), color=legend_colors[2], linewidth=3, clip_on=false))
push!(lines, axs[0].plot(df_ssp.Year[2:end], -collect(skipmissing(df_ssp.diff_dissic_ssp245bgcLR - df_ssp.diff_dissic_u01ssp245bgcLR)), color=legend_colors[3], linewidth=3, clip_on=false))
push!(lines, axs[0].plot(df_ssp.Year[2:end], -collect(skipmissing(df_ssp.diff_dissic_ssp245bgcLR - df_ssp.diff_dissic_g03ssp245bgcLR)), color=legend_colors[4], linewidth=3, clip_on=false))
push!(lines, axs[0].plot(df_ssp.Year[2:end], -collect(skipmissing(df_ssp.diff_dissic_ssp245bgcLR - df_ssp.diff_dissic_u03ssp245bgcLR)), color=legend_colors[5], linewidth=3, clip_on=false))
push!(lines, axs[0].plot(df_ssp.Year[2:end], -collect(skipmissing(df_ssp.diff_dissic_ssp245bgcLR - df_ssp.diff_dissic_g05ssp245bgcLR)), color=legend_colors[6], linewidth=3, clip_on=false))
push!(lines, axs[0].plot(df_ssp.Year[2:end], -collect(skipmissing(df_ssp.diff_dissic_ssp245bgcLR - df_ssp.diff_dissic_u05ssp245bgcLR)), color=legend_colors[7], linewidth=3, clip_on=false))

axs[0].set_ylabel("PgC")
axs[0].set_ylim(-15, 0)
axs[0].tick_params(axis="y", width=1.5)
axs[0].spines["left"].set_linewidth(1.5)
axs[0].spines["left"].set_position(("axes", -0.09))
axs[0].spines["bottom"].set_visible(false)
axs[0].spines["top"].set_visible(false)
axs[0].spines["right"].set_visible(false)
axs[0].get_xaxis().set_visible(false)

legend_labels = ["no hosing", "g01 hosing", "u01 hosing", "g03 hosing", "u03 hosing", "g05 hosing", "u05 hosing",]

for (label, color) in zip(legend_labels, legend_colors)
    axs[0].text(-0.05, 0.75 - 0.11 * findfirst(==(label), legend_labels), label, transform=axs[0].transAxes, fontsize=minor_fontsize, color=color, va="bottom", weight="bold")
end

fig.text(0.03, 0.95, "Hosing-induced difference in ocean carbon storage", ha="left", weight="bold", fontsize=minor_fontsize)
fig.text(0.03, 0.5, "Hosing-induced AMOC strength", ha="left", weight="bold", fontsize=minor_fontsize)

axs[1].plot(df_ssp.Year, df_ssp.var"total_amoc26.5N_ssp245bgcLR", color=legend_colors[1], linewidth=3, clip_on=false)
axs[1].plot(df_ssp.Year, df_ssp.var"total_amoc26.5N_g01ssp245bgcLR", color=legend_colors[2], linewidth=3, clip_on=false)
axs[1].plot(df_ssp.Year, df_ssp.var"total_amoc26.5N_u01ssp245bgcLR", color=legend_colors[3], linewidth=3, clip_on=false)
axs[1].plot(df_ssp.Year, df_ssp.var"total_amoc26.5N_g03ssp245bgcLR", color=legend_colors[4], linewidth=3, clip_on=false)
axs[1].plot(df_ssp.Year, df_ssp.var"total_amoc26.5N_u03ssp245bgcLR", color=legend_colors[5], linewidth=3, clip_on=false)
axs[1].plot(df_ssp.Year, df_ssp.var"total_amoc26.5N_g05ssp245bgcLR", color=legend_colors[6], linewidth=3, clip_on=false)
axs[1].plot(df_ssp.Year, df_ssp.var"total_amoc26.5N_u05ssp245bgcLR", color=legend_colors[7], linewidth=3, clip_on=false)

axs[1].spines["top"].set_visible(false)
axs[1].spines["right"].set_visible(false)
axs[1].spines["left"].set_position(("axes", -0.09))
axs[1].spines["left"].set_linewidth(1.5)
axs[1].spines["bottom"].set_linewidth(1.5)

axs[1].set_xlabel("Year")
axs[1].set_xlim(2020, 2100)
axs[1].set_xticks(collect(2020:20:2100), collect(2020:20:2100))
axs[1].tick_params(axis="x", width=1.5)

axs[1].set_ylabel("Sv")
axs[1].set_ylim(0, 20)
axs[1].tick_params(axis="y", width=1.5)


fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.subplots_adjust(hspace=0.5)
fig.savefig(plotsdir("carbon_storage_plot_ssp245.pdf"), dpi=400, bbox_inches="tight")
fig