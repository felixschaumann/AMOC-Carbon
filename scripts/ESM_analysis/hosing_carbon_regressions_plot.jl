#%%
using DrWatson; @quickactivate "AMOC-Carbon"; cd(srcdir("META/src"))

using GLM
using Statistics
using PythonPlot
using PythonCall
patches = pyimport("matplotlib.patches")

include(srcdir("script_functions.jl"))

exps = ["g01", "u01", "g03", "u03", "g05", "u05",]

#%% CREATE DATAFRAME

df_ssp = get_carbon_data("ssp245")
cutoff = 140 # 140 for full 1pct run
df_pct = get_carbon_data("1pct")[1:cutoff, :]

#%% CALCULATE CARBON FLUXES

diff_dissic_cols_ssp = [occursin("diff_dissic_u", col) || occursin("diff_dissic_g", col) for col in names(df_ssp)]
diff_dissic_cols_pct = [occursin("diff_dissic_u", col) || occursin("diff_dissic_g", col) for col in names(df_pct)]

for col in names(df_ssp[!, diff_dissic_cols_ssp])
    df_ssp[!, Symbol(chop(col, head=12, tail=11) * "_flux")] = -(df_ssp[!, :diff_dissic_ssp245bgcLR] .- df_ssp[!, col] .- circshift(df_ssp[!, :diff_dissic_ssp245bgcLR] .- df_ssp[!, col], 1))
end    
for col in names(df_pct[!, diff_dissic_cols_pct])
    df_pct[!, Symbol(chop(col, head=12, tail=9) * "_flux")] = -(df_pct[!, :diff_dissic_1pctbgcLR] .- df_pct[!, col] .- circshift(df_pct[!, :diff_dissic_1pctbgcLR] .- df_pct[!, col], 1))
end    

#%% ADD RUNNING MEANS FOR CARBON FLUXES AND AMOC

running_window = 10 # years

for exp in exps
    df_ssp[!, Symbol("$(exp)_flux_rm$(running_window)yrs")] = vcat(fill(missing, length(findall(ismissing, df_ssp[!, "$(exp)_flux"]))), running_mean(df_ssp[!, "$(exp)_flux"], running_window))
    df_pct[!, Symbol("$(exp)_flux_rm$(running_window)yrs")] = vcat(fill(missing, length(findall(ismissing, df_pct[!, "$(exp)_flux"]))), running_mean(df_pct[!, "$(exp)_flux"], running_window))
end

total_amoc_cols_ssp = [occursin("total_amoc26.5N_u", col) || occursin("total_amoc26.5N_g", col) for col in names(df_ssp)]
total_amoc_cols_pct = [occursin("total_amoc26.5N_u", col) || occursin("total_amoc26.5N_g", col) for col in names(df_pct)]

for col in names(df_ssp[!, total_amoc_cols_ssp])
    df_ssp[!, Symbol("anom_" * chop(col, head=6, tail=0) * "_rm$(running_window)yrs")] = running_mean(df_ssp[!, col].- df_ssp[!, "total_amoc26.5N_ssp245bgcLR"], running_window)
end

for col in names(df_pct[!, total_amoc_cols_pct])
    df_pct[!, Symbol("anom_" * chop(col, head=6, tail=0) * "_rm$(running_window)yrs")] = running_mean(df_pct[!, col].- df_pct[!, "total_amoc26.5N_1pctbgcLR"], running_window)
end

df_ssp[!, Symbol("amoc26.5N_ssp245bgcLR_rm$(running_window)yrs")] = running_mean(df_ssp[!, "total_amoc26.5N_ssp245bgcLR"], running_window)
df_pct[!, Symbol("amoc26.5N_1pctbgcLR_rm$(running_window)yrs")] = running_mean(df_pct[!, "total_amoc26.5N_1pctbgcLR"], running_window)

#%% REGRESSIONS

AMOC_pi_ssp = mean(df_ssp.var"total_amoc26.5N_ssp245bgcLR")
AMOC_pi_pct = mean(df_pct.var"total_amoc26.5N_1pctbgcLR")

combined_df_ssp = DataFrame(
    flux_all_runs_10yrs = vcat(
        df_ssp[!, "g01_flux_rm$(running_window)yrs"],
        df_ssp[!, "g03_flux_rm$(running_window)yrs"],
        df_ssp[!, "g05_flux_rm$(running_window)yrs"],
        df_ssp[!, "u01_flux_rm$(running_window)yrs"],
        df_ssp[!, "u03_flux_rm$(running_window)yrs"],
        df_ssp[!, "u05_flux_rm$(running_window)yrs"],
    ),
    anom_amoc_all_runs_10yrs = vcat(
        df_ssp[!, "anom_amoc26.5N_g01ssp245bgcLR_rm$(running_window)yrs"],
        df_ssp[!, "anom_amoc26.5N_g03ssp245bgcLR_rm$(running_window)yrs"],
        df_ssp[!, "anom_amoc26.5N_g05ssp245bgcLR_rm$(running_window)yrs"],
        df_ssp[!, "anom_amoc26.5N_u01ssp245bgcLR_rm$(running_window)yrs"],
        df_ssp[!, "anom_amoc26.5N_u03ssp245bgcLR_rm$(running_window)yrs"],
        df_ssp[!, "anom_amoc26.5N_u05ssp245bgcLR_rm$(running_window)yrs"],
    )
)

combined_df_pct = DataFrame(
    flux_all_runs_10yrs = vcat(
        df_pct[!, "g01_flux_rm$(running_window)yrs"],
        df_pct[!, "g03_flux_rm$(running_window)yrs"],
        df_pct[!, "g05_flux_rm$(running_window)yrs"],
        df_pct[!, "u01_flux_rm$(running_window)yrs"],
        df_pct[!, "u03_flux_rm$(running_window)yrs"],
        df_pct[!, "u05_flux_rm$(running_window)yrs"],
    ),
    anom_amoc_all_runs_10yrs = vcat(
        df_pct[!, "anom_amoc26.5N_g011pctbgcLR_rm$(running_window)yrs"],
        df_pct[!, "anom_amoc26.5N_g031pctbgcLR_rm$(running_window)yrs"],
        df_pct[!, "anom_amoc26.5N_g051pctbgcLR_rm$(running_window)yrs"],
        df_pct[!, "anom_amoc26.5N_u011pctbgcLR_rm$(running_window)yrs"],
        df_pct[!, "anom_amoc26.5N_u031pctbgcLR_rm$(running_window)yrs"],
        df_pct[!, "anom_amoc26.5N_u051pctbgcLR_rm$(running_window)yrs"],
    )
)

combined_df_ssp = dropmissing(combined_df_ssp, :flux_all_runs_10yrs)
combined_df_pct = dropmissing(combined_df_pct, :flux_all_runs_10yrs)

combined_df_ssp.anom_amoc_shifted_10yrs = combined_df_ssp.anom_amoc_all_runs_10yrs .+ AMOC_pi_ssp
combined_df_pct.anom_amoc_shifted_10yrs = combined_df_pct.anom_amoc_all_runs_10yrs .+ AMOC_pi_pct

combined_model_ssp = @eval lm(@formula(flux_all_runs_10yrs ~ anom_amoc_shifted_10yrs), combined_df_ssp)
combined_model_pct = @eval lm(@formula(flux_all_runs_10yrs ~ anom_amoc_shifted_10yrs), combined_df_pct)

combined_model_0_ssp = @eval lm(@formula(flux_all_runs_10yrs ~ 0 + anom_amoc_all_runs_10yrs), combined_df_ssp)
combined_model_0_pct = @eval lm(@formula(flux_all_runs_10yrs ~ 0 + anom_amoc_all_runs_10yrs), combined_df_pct)

confint_99_0_ssp = confint(combined_model_0_ssp, 0.99)
confint_99_0_pct = confint(combined_model_0_pct, 0.99)
confint_99_ssp = confint(combined_model_ssp, 0.99)
confint_99_pct = confint(combined_model_pct, 0.99)

regressions_df_ssp = DataFrame(
    name="combined_model_0_ssp_$(running_window)yrs",
    intercept = missing,
    estimate = round(coef(combined_model_0_ssp)[1], digits=5),
    stderror_int = missing,
    stderror_est = round(stderror(combined_model_0_ssp)[1], digits=5),
    confint_99_low = round(confint_99_0_ssp[1], digits=5),
    confint_99_high = round(confint_99_0_ssp[2], digits=5),
    intercept_wo_0 = round(coef(combined_model_ssp)[1] + AMOC_pi_ssp * coef(combined_model_ssp)[2], digits=5),
)

regressions_df_pct = DataFrame(
    name="combined_model_0_pct_$(running_window)yrs",
    intercept = round(missing, digits=5),
    estimate = round(coef(combined_model_0_pct)[1], digits=5),
    stderror_int = round(missing, digits=5),
    stderror_est = round(stderror(combined_model_0_pct)[1], digits=5),
    confint_99_low = round(confint_99_0_pct[1], digits=5),
    confint_99_high = round(confint_99_0_pct[2], digits=5),
    intercept_wo_0 = round(coef(combined_model_pct)[1] + AMOC_pi_pct * coef(combined_model_pct)[2], digits=5),
)

regressions_df_ssp_wo_0 = DataFrame(
    name="combined_model_ssp_$(running_window)yrs",
    intercept = round(coef(combined_model_ssp)[1], digits=5),
    estimate = round(coef(combined_model_ssp)[2], digits=5),
    stderror_int = round(stderror(combined_model_ssp)[1], digits=5),
    stderror_est = round(stderror(combined_model_ssp)[2], digits=5),
    confint_99_low = round(confint_99_ssp[2, 1], digits=5),
    confint_99_high = round(confint_99_ssp[2, 2], digits=5),
    intercept_wo_0 = round(coef(combined_model_ssp)[1] + AMOC_pi_ssp * coef(combined_model_ssp)[2], digits=5),
)

regressions_df_pct_wo_0 = DataFrame(
    name="combined_model_pct_$(running_window)yrs",
    intercept = round(coef(combined_model_pct)[1], digits=5),
    estimate = round(coef(combined_model_pct)[2], digits=5),
    stderror_int = round(stderror(combined_model_pct)[1], digits=5),
    stderror_est = round(stderror(combined_model_pct)[2], digits=5),
    confint_99_low = round(confint_99_pct[2, 1], digits=5),
    confint_99_high = round(confint_99_pct[2, 2], digits=5),
    intercept_wo_0 = round(coef(combined_model_pct)[1] + AMOC_pi_pct * coef(combined_model_pct)[2], digits=5),
)

regressions_df = vcat(regressions_df_ssp, regressions_df_pct, regressions_df_ssp_wo_0, regressions_df_pct_wo_0)

CSV.write(datadir("carbon_flux_regressions_$(running_window)yrs.csv"), regressions_df)

####################################################################################
####################################################################################
####################################################################################
#
#%% PLOT BOTH SCENARIOS WITH MATPLOTLIB
#
####################################################################################
####################################################################################
####################################################################################

PythonPlot.rc("font", size = 15)
fig, axs = subplots(2, 1, figsize=(8.5,9.9), height_ratios=[8, 5], sharex=true)

legend_colors = ["lightblue", "peachpuff", "dodgerblue", "salmon", "darkblue", "darkred"]

# drop missing values for plotting (u03 only as example)
plotting_df_ssp = dropmissing(df_ssp, Symbol("u03_flux_rm$(running_window)yrs"))
plotting_df_pct = dropmissing(df_pct, Symbol("u03_flux_rm$(running_window)yrs"))

for exp in exps
    axs[1].scatter(plotting_df_ssp[!, "anom_amoc26.5N_$(exp)ssp245bgcLR_rm$(running_window)yrs"].+AMOC_pi_ssp, plotting_df_ssp[!, "$(exp)_flux_rm$(running_window)yrs"], color=legend_colors[findfirst(==(exp), exps)], s=25)
    axs[0].scatter(plotting_df_pct[!, "anom_amoc26.5N_$(exp)1pctbgcLR_rm$(running_window)yrs"].+AMOC_pi_pct, plotting_df_pct[!, "$(exp)_flux_rm$(running_window)yrs"], color=legend_colors[findfirst(==(exp), exps)], s=25)
end

axs[1].set_ylim(-0.4, 0.1)
axs[1].set_xlabel("AMOC strength [Sv]")
axs[1].set_xlim(4, 20)
axs[1].set_xticks(collect(4:2:20))
axs[1].spines["top"].set_visible(false)
axs[1].spines["bottom"].set_position(("axes", -0.1))
axs[1].spines["right"].set_visible(false)
axs[1].spines["left"].set_position(("axes", -0.05))

axs[0].tick_params(axis="x", which="both", bottom=false, top=false, labelbottom=false)
axs[0].set_ylim(-0.6, 0.2)
axs[0].tick_params(axis="y")
axs[0].spines["top"].set_visible(false)
axs[0].spines["bottom"].set_visible(false)
axs[0].spines["left"].set_position(("axes", -0.05))
axs[0].spines["right"].set_visible(false)

fig.text(-0.03, 0.5, latexstring("Carbon flux change \$ \\Delta F\\ \\left[\\frac{\\mathrm{PgC}}{\\mathrm{yr}}\\right]\$"), va="center", rotation="vertical")

axs[1].text(0.45, 0.95, "SSP2-4.5 \$\\mathbf{CO_2}\$ increase", transform=axs[1].transAxes, weight="bold", ha="center")
axs[0].text(0.45, 0.95, "1% per year \$\\mathbf{CO_2}\$ increase", transform=axs[0].transAxes, weight="bold", ha="center")

fig.text(0.095, 0.865, "a", ha="left", fontsize=20, weight="bold")
fig.text(0.095, 0.358, "b", ha="left", fontsize=20, weight="bold")

legend_labels = ["g01 hosing", "u01 hosing", "g03 hosing", "u03 hosing", "g05 hosing", "u05 hosing"]
for (label, color) in zip(legend_labels, legend_colors)
    axs[0].text(0, .95 - 0.048 * findfirst(==(label), legend_labels), label, transform=axs[0].transAxes, fontsize=12, color=color, va="top", weight="bold")
end

axs[1].axvline(AMOC_pi_ssp, color="dimgrey", linewidth=1.5, ymin=-0.1, clip_on=false);
axs[0].axvline(AMOC_pi_pct, color="dimgrey", linewidth=1.5);

axs[1].plot([3.3, AMOC_pi_ssp], [0, 0], color="black", linewidth=1., linestyle=":", clip_on=false)
x_ranges = [(3.3, 3.9), (6.7, AMOC_pi_pct)]
for (x_start, x_end) in x_ranges
    axs[0].plot([x_start, x_end], [0, 0], color="black", linewidth=1., linestyle=":", clip_on=false)
end


arrow = patches.FancyArrowPatch(
    (AMOC_pi_ssp, -0.52), 
    (AMOC_pi_ssp, -0.45),
    mutation_scale=11,
    color="dimgrey",
    arrowstyle="-|>",
    linewidth=1.,
    clip_on=false 
)
axs[1].add_patch(arrow)

axs[1].text(
    AMOC_pi_ssp, -0.52,
    latexstring("\$AMOC_\\text{0,ssp245}\$"), 
    fontsize=12, 
    horizontalalignment="center", 
    verticalalignment="top", 
    color="dimgrey", 
    clip_on=false, 
)

axs[0].annotate(latexstring("\$AMOC_\\text{0,1pct}\$"), xy=(AMOC_pi_pct, -0.6), xytext=(AMOC_pi_pct, -0.67), textcoords="data", fontsize=12, arrowprops=Dict("facecolor"=>"dimgrey", "edgecolor"=>"dimgrey", "width"=>1, "headwidth"=>6, "headlength"=> 6, "linewidth"=>.01), horizontalalignment="center", verticalalignment="center", color="dimgrey")

x_range = collect(4.:0.1:AMOC_pi_ssp)
axs[1].plot(x_range, (x_range.-AMOC_pi_ssp) .* coef(combined_model_0_ssp)[1], label=latexstring("\\Delta F_\\text{ssp245} = $(round(coef(combined_model_0_ssp)[1], digits=3))\\ (AMOC-AMOC_\\text{0,ssp245})"), color="k", lw=1.5)
axs[1].fill_between(x_range, confint_99_0_ssp[1] .* (x_range.-AMOC_pi_ssp), confint_99_0_ssp[2] .* (x_range.-AMOC_pi_ssp), alpha=0.5, color="gray", label=latexstring("\\mathrm{99\\% \\ CI \\ around}\\ \\Delta F_\\text{ssp245}"))
x_range = collect(4.:0.1:AMOC_pi_pct)
axs[0].plot(x_range, (x_range.-AMOC_pi_pct) .* coef(combined_model_0_pct)[1], color="k", lw=1.5, label=latexstring("\\Delta F_\\text{1pct} = $(round(coef(combined_model_0_pct)[1], digits=3))\\ (AMOC-AMOC_\\text{0,1pct})"))
axs[0].fill_between(x_range, confint_99_0_pct[1] .* (x_range.-AMOC_pi_pct), confint_99_0_pct[2] .* (x_range.-AMOC_pi_pct), alpha=0.5, color="gray", label=latexstring("\\mathrm{99\\% \\ CI \\ around}\\ \\Delta F_\\text{1pct}"))

axs[1].legend(loc=4, fontsize=12, ncol=1, bbox_to_anchor=(0.92, -0.1), borderaxespad=0., framealpha=0)
axs[0].legend(loc=4, fontsize=12, ncol=1, bbox_to_anchor=(0.945, 0.025), borderaxespad=0., framealpha=0)

fig.savefig(plotsdir("hosing_carbon_regressions_combined_$(running_window)rm.pdf"), dpi=400, bbox_inches="tight");
fig

####################################################################################
####################################################################################
####################################################################################
#
#%% SSP2-4.5-only plot
#
####################################################################################
####################################################################################
####################################################################################


PythonPlot.rc("font", size=18)
fig, ax = subplots(1, 1, figsize=(8.5,5.0))

minor_fontsize = 15

legend_colors = ["lightblue", "peachpuff", "dodgerblue", "salmon", "darkblue", "darkred"]

# drop missing values for plotting (u03 only as example)
plotting_df_ssp = dropmissing(df_ssp, Symbol("u03_flux_rm$(running_window)yrs"))

for exp in exps
    ax.scatter(plotting_df_ssp[!, "anom_amoc26.5N_$(exp)ssp245bgcLR_rm$(running_window)yrs"].+AMOC_pi_ssp, plotting_df_ssp[!, "$(exp)_flux_rm$(running_window)yrs"], color=legend_colors[findfirst(==(exp), exps)], s=25)
end

ax.set_ylim(-0.4, 0.1)
ax.set_xlabel("AMOC strength [Sv]")
ax.set_xlim(4, 20)
ax.set_xticks(collect(4:2:20))
ax.spines["top"].set_visible(false)
ax.spines["bottom"].set_position(("axes", -0.1))
ax.spines["right"].set_visible(false)
ax.spines["left"].set_position(("axes", -0.05))

fig.text(-0.04, 0.5, latexstring("Carbon flux change \$ \\Delta F\\ \\left[\\frac{\\mathrm{PgC}}{\\mathrm{yr}}\\right]\$"), va="center", rotation="vertical")

# ax.text(0.45, 0.97, "SSP2-4.5 \$\\mathbf{CO_2}\$ increase", transform=ax.transAxes, weight="bold", ha="center")

legend_labels = ["g01 hosing", "u01 hosing", "g03 hosing", "u03 hosing", "g05 hosing", "u05 hosing"]
for (label, color) in zip(legend_labels, legend_colors)
    ax.text(0, 1.04 - 0.06 * findfirst(==(label), legend_labels), label, transform=ax.transAxes, fontsize=minor_fontsize, color=color, va="top", weight="bold")
end

ax.axvline(AMOC_pi_ssp, color="dimgrey", linewidth=1.5, ymin=-0.1, clip_on=false);

x_ranges = [(3.3, 3.9), (7.3, AMOC_pi_ssp)]
for (x_start, x_end) in x_ranges
    ax.plot([x_start, x_end], [0, 0], color="black", linewidth=1., linestyle=":", clip_on=false)
end

arrow = patches.FancyArrowPatch(
    (AMOC_pi_ssp, -0.505), 
    (AMOC_pi_ssp, -0.45),
    mutation_scale=11,
    color="dimgrey",
    arrowstyle="-|>",
    linewidth=1.,
    clip_on=false 
)
ax.add_patch(arrow)

ax.text(
    AMOC_pi_ssp, -0.505,
    latexstring("\$AMOC_0\$"), 
    fontsize=minor_fontsize, 
    horizontalalignment="center", 
    verticalalignment="top", 
    color="dimgrey", 
    clip_on=false, 
)

x_range = collect(4.:0.1:AMOC_pi_ssp)
ax.plot(x_range, (x_range.-AMOC_pi_ssp) .* coef(combined_model_0_ssp)[1], label=latexstring("\\Delta F = $(round(coef(combined_model_0_ssp)[1], digits=3))\\ (AMOC-AMOC_0)"), color="k", lw=1.5)
ax.fill_between(x_range, confint_99_0_ssp[1] .* (x_range.-AMOC_pi_ssp), confint_99_0_ssp[2] .* (x_range.-AMOC_pi_ssp), alpha=0.5, color="gray", label=latexstring("\\mathrm{99\\% \\ CI \\ around}\\ \\Delta F"))

ax.legend(loc=4, fontsize=minor_fontsize, ncol=1, bbox_to_anchor=(0.92, -0.08), borderaxespad=0., framealpha=0)

fig.savefig(plotsdir("hosing_carbon_regressions_ssp245_$(running_window)rm.pdf"), dpi=400, bbox_inches="tight");
fig