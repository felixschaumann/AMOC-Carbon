using DrWatson; @quickactivate "AMOC-Carbon"; cd(srcdir("META/src"))

using CSV, DataFrames, GLM
using PythonPlot

#%% Load Atlantic gamma and AMOC weakening data

gamma_data = CSV.read(datadir("Katavouta2021_Atl_gammas.csv"), DataFrame)[:, :]

#%%
gamma_data.gamma_dis_reg = gamma_data.gamma_dis + gamma_data.gamma_reg

zero_model = @eval lm(@formula($(Symbol("gamma_dis_reg")) ~ 0 + AMOC_weakening), gamma_data)
intercept_model = @eval lm(@formula($(Symbol("gamma_dis_reg")) ~ AMOC_weakening), gamma_data)

#%% exclude CNRM-ESM2-1 model
gamma_data_wo_CNRM = gamma_data[gamma_data.Model .!= "CNRM-ESM2-1", :]

zero_model_wo_CNRM = @eval lm(@formula($(Symbol("gamma_dis_reg")) ~ 0 + AMOC_weakening), gamma_data_wo_CNRM)

intercept_model_wo_CNRM = @eval lm(@formula($(Symbol("gamma_dis_reg")) ~ AMOC_weakening), gamma_data_wo_CNRM)

#%% same for relative AMOC strength

gamma_data.AMOC_relative_strength = gamma_data.AMOC_weakening ./ gamma_data.Preindustrial_AMOC
gamma_data_wo_CNRM.AMOC_relative_strength = gamma_data_wo_CNRM.AMOC_weakening ./ gamma_data_wo_CNRM.Preindustrial_AMOC

intercept_model_rel = @eval lm(@formula($(Symbol("gamma_dis_reg")) ~ AMOC_relative_strength), gamma_data)
intercept_model_rel_wo_CNRM = @eval lm(@formula($(Symbol("gamma_dis_reg")) ~ AMOC_relative_strength), gamma_data_wo_CNRM)

#%% plot using PythonPlot

AMOC_weakening_wo_CNRM = gamma_data_wo_CNRM.AMOC_weakening
AMOC_rel_wo_CNRM = gamma_data_wo_CNRM.AMOC_relative_strength
gamma_dis_reg_wo_CNRM = gamma_data_wo_CNRM.gamma_dis_reg
Models_wo_CNRM = gamma_data_wo_CNRM.Model

fig, ax = subplots(figsize=(8, 5))

ax.set_ylim(-12, 0)
ax.set_xlim(20, 0)
ax.set_xticks(collect(0:4:20))
ax.set_xlabel("∆AMOC (Sv)")
ax.set_ylabel(latexstring("\$\\gamma_\\mathrm{AMOC}\$ (\$\\frac{\\mathrm{PgC}}{\\mathrm{K}}\$)"))

ax.spines["right"].set_visible(false)
ax.spines["top"].set_visible(false)

ax.vlines(0, -12, 1, color="gray", ls="--", clip_on=false)
ax.hlines(0, 20, -1, color="gray", ls="--", clip_on=false)

for i in eachindex(AMOC_weakening_wo_CNRM)
    ax.scatter(-AMOC_weakening_wo_CNRM[i], gamma_dis_reg_wo_CNRM[i:i], label=Models_wo_CNRM[i])
end

zero_value = -coef(intercept_model_wo_CNRM)[1]/coef(intercept_model_wo_CNRM)[2]

# add regression line
x_range = .- collect(-20:0.1:1)
# ax.plot(x_range, x_range .* -coef(intercept_model_wo_CNRM)[2] .+ coef(intercept_model_wo_CNRM)[1], label="γ_AMOC = $(round(coef(intercept_model_wo_CNRM)[1], digits=2)) \$\\frac{\\mathrm{PgC}}{\\mathrm{K}}\$ + $(round(coef(intercept_model_wo_CNRM)[2], digits=2)) \$\\frac{\\mathrm{PgC}}{\\mathrm{Sv K}}\$ * ∆AMOC", color="k")
ax.plot(x_range[1:end-33], (x_range[1:end-33] .+ zero_value) .* -coef(intercept_model_wo_CNRM)[2], label="γ_AMOC = -$(round(coef(intercept_model_wo_CNRM)[2], digits=2)) \$\\frac{\\mathrm{PgC}}{\\mathrm{Sv K}}\$ * (∆AMOC - $(-round(zero_value, digits=2)) Sv)", color="k", clip_on=false)
ax.plot(x_range, x_range .* -coef(zero_model_wo_CNRM)[1], label="γ_AMOC = -$(round(coef(zero_model_wo_CNRM)[1], digits=2)) \$\\frac{\\mathrm{PgC}}{\\mathrm{Sv K}}\$ * ∆AMOC", color="k", ls="--", clip_on=false)

ax.legend(loc="lower right", fontsize=7)

fig.savefig(plotsdir("gamma_regressions.pdf"), dpi=400, bbox_inches="tight");
fig

#%% carbon storage reductions in 2100

# 1pct experiment

weakening = 15

T_140 = 4.87 # based on Arora et al. (2020)

C_storage_reduction_intercept = coef(intercept_model_wo_CNRM)[2] * (weakening + zero_value) * T_140

C_storage_reduction_zero = coef(zero_model_wo_CNRM)[1] * weakening * T_140

# SSP2-4.5 (less reliable method - see WP)

T_increase_yearly = 0.02 # should be properly calibrated

weakening = 15

C_storage_reduction_intercept_ssp245 = 2 * coef(intercept_model_wo_CNRM)[2] * T_increase_yearly * (weakening+zero_value) * 85 / 2
C_storage_reduction_zero_ssp245 = 2 * coef(zero_model_wo_CNRM)[1] * T_increase_yearly * weakening * 85 / 2