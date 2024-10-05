#%%
using DrWatson; @quickactivate "AMOC-Carbon"; cd(srcdir("META/src"))

using Dates
using Statistics
using PythonPlot

include(srcdir("script_functions.jl"))
include(srcdir("META/src/lib/AMOC_Carbon_functions.jl"))

#%%

AMOC_pi_vals = CSV.read(srcdir("META/data/CMIP6_amoc/AMOC_pi_values.csv"), DataFrame)

colours = [
    "mediumturquoise",
    "cadetblue",
    "steelblue",
    "darkslateblue",
    "mediumorchid",
    "plum",
    "mediumvioletred",
    ]

mpi_ds = get_projection_ds("mpiesmlr");
years = year.(mpi_ds["time"] |> Array);

####################################################################################
####################################################################################
####################################################################################
#
#%% make absolute and relative weakening plot alongside
#
####################################################################################
####################################################################################
####################################################################################

PythonPlot.rc("font", size = 15)
fig, ax = subplots(1, 2, figsize=(20, 8))

for (i, name) in enumerate(names(AMOC_pi_vals)[2:end])
    amoc_ds = get_projection_ds(name)
    amoc_mean = amoc_ds["mean"] |> Array
    amoc_std = amoc_ds["std"] |> Array
    weakening_pi = 1 - amoc_mean[end] ./ AMOC_pi_vals[:, name][1]
    weakening_2015 = 1 - amoc_mean[end] ./ amoc_mean[1]
    println(name, " in 2100 wrt pi: ", round(weakening_pi*100, digits=1), "%")
    
    ax[0].plot(years, amoc_mean, label=name, lw=3, color=colours[i])
    ax[0].fill_between(years, amoc_mean .- amoc_std, amoc_mean .+ amoc_std, alpha=0.1, color=colours[i])

    ax[1].plot(years, amoc_mean ./ AMOC_pi_vals[:, name][1] .*100 , label=name, lw=3, color=colours[i])
    ax[1].fill_between(years, (amoc_mean .- amoc_std) ./ AMOC_pi_vals[:, name][1] .*100, (amoc_mean .+ amoc_std) ./ AMOC_pi_vals[:, name][1] .*100, alpha=0.1, color=colours[i])
end

ax[0].set_title("Absolute AMOC strength")
ax[1].set_title("Relative AMOC strength")

ax[0].set_ylim(5, 25)
ax[1].set_ylim(40, 120)
ax[0].set_xlim(2010, 2100)
ax[1].set_xlim(2010, 2100)

ax[0].set_ylabel("Sv")
ax[1].set_ylabel("% of preindustrial")

ax[0].set_xlabel("Year")
ax[1].set_xlabel("Year")

ax[0].spines["right"].set_visible(false)
ax[0].spines["top"].set_visible(false)
ax[1].spines["right"].set_visible(false)
ax[1].spines["top"].set_visible(false)
ax[0].spines["left"].set_position(("axes", -0.03))
ax[1].spines["left"].set_position(("axes", -0.03))

for (label, color) in zip(collect(names(AMOC_pi_vals)[2:end]), colours)
    for i in 0:1
        ax[i].text(1, 1 - 0.04 * findfirst(==(label), collect(names(AMOC_pi_vals)[2:end])), label, transform=ax[i].transAxes, fontsize=18, color=color, va="top", ha="right")#, weight="bold")
    end
end

fig

#%% relative weakening panel for overview figure

PythonPlot.rc("font", size = 25)
fig, ax = subplots(1, figsize=(10, 8))

y_lines = [50, 60, 70, 80, 90, 100, 110]
for y in y_lines
    ax.plot([2004.5, y > 90 ? 2072 : 2100], [y, y], color="black", linewidth=2.5, linestyle=":", clip_on=false)
    ax.text(2014.5, y+2.5, "$(y)%", fontsize=30, color="black", ha="right", va="center", clip_on=false)
end

for (i, name) in enumerate(names(AMOC_pi_vals)[2:end])
    amoc_ds = get_projection_ds(name)
    amoc_mean = amoc_ds["mean"] |> Array
    amoc_std = amoc_ds["std"] |> Array
    
    ax.plot(years, amoc_mean ./ AMOC_pi_vals[:, name][1] .*100 , label=name, lw=5, color=colours[i], clip_on=false)
    ax.fill_between(years, (amoc_mean .- amoc_std) ./ AMOC_pi_vals[:, name][1] .*100, (amoc_mean .+ amoc_std) ./ AMOC_pi_vals[:, name][1] .*100, alpha=0.2, color=colours[i], clip_on=false)
end

ax.set_ylim(43.5, 120)
ax.set_xlim(2020, 2100)

x_ticks = collect(2020:20:2100)
ax.set_xticks(x_ticks)

for x in x_ticks
    ax.text(x, 38., "$(x)", fontsize=30, color="black", ha="center", clip_on=false)
end

fig.text(0.47, 0.835, "AMOC strength", color="black", va="top", ha="center", fontsize=30)

ax.spines["right"].set_visible(false)
ax.spines["top"].set_visible(false)
ax.spines["left"].set_visible(false)
ax.spines["bottom"].set_color("black")
ax.tick_params(axis="x", colors="black", width=2.)
ax.spines["bottom"].set_linewidth(2.)
ax.set_xticklabels([])
ax.get_yaxis().set_visible(false)

for (label, color) in zip(collect(names(AMOC_pi_vals)[2:end]), colours)
    ax.text(1, 0.98 - 0.045 * findfirst(==(label), collect(names(AMOC_pi_vals)[2:end])), label, transform=ax.transAxes, fontsize=20, color=color, va="top", ha="right", weight="bold")
end

fig.savefig(plotsdir("relative_amoc_projection.pdf"), dpi=400, bbox_inches="tight")
fig

####################################################################################
####################################################################################
####################################################################################
#
#%% carbon storage difference projection
#
####################################################################################
####################################################################################
####################################################################################

n_MC = 10000
MC_proj_res = CSV.read(datadir("MC_proj_results_$(n_MC)_samples.csv"), DataFrame)

PythonPlot.rc("font", size=15)
fig, ax = subplots(2, 4, figsize=(17.8, 9.9))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

minor_fontsize = 12

for (i, name) in enumerate(names(AMOC_pi_vals)[2:end])
    amoc_ds = get_projection_ds(name)
    amoc_mean = amoc_ds["mean"] |> Array
    amoc_std = amoc_ds["std"] |> Array
    
    ax[0, 0].plot(years, amoc_mean, lw=3, color=colours[i])
    ax[0, 0].fill_between(years, amoc_mean .- amoc_std, amoc_mean .+ amoc_std, alpha=0.1, color=colours[i])
end

ax[0, 0].set_xlim(2015, 2100)
ax[0, 0].set_ylim(5, 25)
ax[0, 0].spines["right"].set_visible(false)
ax[0, 0].spines["top"].set_visible(false)
ax[0, 0].spines["bottom"].set_color("k")
ax[0, 0].tick_params(axis="x", colors="k", width=1.5)
ax[0, 0].spines["bottom"].set_linewidth(1.)
ax[0, 0].spines["left"].set_color("k")
ax[0, 0].tick_params(axis="y", colors="k", width=1.5)
ax[0, 0].spines["left"].set_linewidth(1.)
ax[0, 0].spines["left"].set_position(("axes", -0.0))
ax[0, 0].set_ylabel("AMOC strength [Sv]", color="k")

for (label, color) in zip(collect(names(AMOC_pi_vals)[2:end]), colours)
    ax[0, 0].text(1, 1.05 - 0.045 * findfirst(==(label), collect(names(AMOC_pi_vals)[2:end])), label, transform=ax[0, 0].transAxes, fontsize=12, color=color, va="top", ha="right", weight="bold")
end

for (i, name) in enumerate(names(AMOC_pi_vals)[2:end])
    
    amoc_ds = get_projection_ds(name)
    amoc_mean = amoc_ds["mean"] |> Array
    amoc_std = mean(amoc_ds["std"] |> Array) .* ones(length(amoc_mean))
    
    ax.flatten()[i].plot(years, amoc_mean, lw=3, color=colours[i], clip_on=false)
    ax.flatten()[i].fill_between(years, amoc_mean .- amoc_std, amoc_mean .+ amoc_std, alpha=0.3, color=colours[i], clip_on=false)
    
    ax.flatten()[i].text(2060, 25.6, name, color=colours[i], weight="bold", va="top", ha="center")

    ax.flatten()[i].set_xlim(2015, 2100)
    ax.flatten()[i].set_ylim(5, 25)

    ax.flatten()[i].spines["right"].set_visible(false)
    ax.flatten()[i].spines["top"].set_visible(false)
    ax.flatten()[i].spines["bottom"].set_color("k")
    ax.flatten()[i].tick_params(axis="x", colors="k", width=1.5)
    ax.flatten()[i].spines["bottom"].set_linewidth(1.)
    ax.flatten()[i].spines["left"].set_color("k")
    ax.flatten()[i].tick_params(axis="y", colors="k", width=1.5)
    ax.flatten()[i].spines["left"].set_linewidth(1.)
    ax.flatten()[i].spines["left"].set_position(("axes", -0.0))

    if i > 3
        ax.flatten()[i].set_xlabel("Year", color="k")
    end
    if i == 0 || i == 4
        ax.flatten()[i].set_ylabel("AMOC strength [Sv]", color="k")
    end

end
fig
fig.savefig(plotsdir("proj_carbon_storage_diff_$(n_MC)_samples_proj.pdf"), dpi=400, bbox_inches="tight")

for (i, name) in enumerate(names(AMOC_pi_vals)[2:end])

    AMOC_pi_array = ones(length(years)) .* AMOC_pi_vals[:, name][1];
    ax.flatten()[i].plot(years, AMOC_pi_array, color=colours[i], linestyle="--", lw=3, clip_on=false)
    ax.flatten()[i].fill_between(years, AMOC_pi_array .- AMOC_pi_vals[:, name][2], AMOC_pi_array .+ AMOC_pi_vals[:, name][2], alpha=0.3, color=colours[i], clip_on=false)
    
end
fig
fig.savefig(plotsdir("proj_carbon_storage_diff_$(n_MC)_samples_proj+pi.pdf"), dpi=400, bbox_inches="tight")

for (i, name) in enumerate(names(AMOC_pi_vals)[2:end])
    
    amoc_ds = get_projection_ds(name)
    amoc_mean = amoc_ds["mean"] |> Array
    AMOC_pi_array = ones(length(years)) .* AMOC_pi_vals[:, name][1];
    
    years = convert(Vector{Float64}, years)
    amoc_mean = convert(Vector{Float64}, amoc_mean)
    AMOC_pi_array = convert(Vector{Float64}, AMOC_pi_array)
    ax.flatten()[i].fill_between(years, amoc_mean, AMOC_pi_array, where=(amoc_mean .< AMOC_pi_array), interpolate=true, color="none", hatch="//", edgecolor="gray", alpha=1., clip_on=false)

    mean_storage_diff = round(MC_proj_res[1, name], digits=1)
    std_storage_diff = round(MC_proj_res[2, name], digits=1)

    text_x = 2098
    text_y = (AMOC_pi_array[end] + amoc_mean[end]) / 1.95
    ax.flatten()[i].text(text_x, text_y, "-$(mean_storage_diff)±$(std_storage_diff) PgC", fontsize=minor_fontsize, color=colours[i], weight="bold", va="center", ha="right", clip_on=false,
    bbox=Dict("facecolor" => "white", "edgecolor" => "gray", "boxstyle" => "round,pad=0.2"))
end

fig.savefig(plotsdir("proj_carbon_storage_diff_$(n_MC)_samples.pdf"), dpi=400, bbox_inches="tight")
fig