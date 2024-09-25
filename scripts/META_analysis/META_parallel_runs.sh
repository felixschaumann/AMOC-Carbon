#!/bin/bash
#SBATCH -J parallel_META_runs    # Specify job name
#SBATCH -p compute         # Use partition
#SBATCH -t 8:00:00        # Set a limit on the total run time
#SBATCH -A mh0033          # Charge resources on this project account
#SBATCH -o ./diagn_output/diagn_output_parallel_META.o%j       # File name for standard and error output
#SBATCH -N 1

set -e

echo "Start new parallel META julia script execution at $(date)"

runs=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19) # 14 for only C, 19 for C and T

export N_SAMPLES=17
export RUN_MC_PROJ=false
export RUN_MC_SCC=true

echo "Projection: $RUN_MC_PROJ; SCC: $RUN_MC_SCC."

job_id=$SLURM_JOB_ID

for run in ${runs[@]}; do
    export N_RUN=$run

    # Create a temporary SLURM script for each run
    cat <<EOT > temp_slurm_script_$run.sh
#!/bin/bash
#SBATCH -J META_run_$run
#SBATCH -p compute
#SBATCH -t 8:00:00
#SBATCH -A mh0033
#SBATCH -o ./diagn_output/diagn_output_parallel_META.o${job_id}_$run.txt
#SBATCH -N 1

set -e

export N_SAMPLES=$N_SAMPLES
export RUN_MC_PROJ=$RUN_MC_PROJ
export RUN_MC_SCC=$RUN_MC_SCC
export N_RUN=$run

~/.juliaup/bin/julia --project=/home/m/m300940/amoc-carbon /home/m/m300940/amoc-carbon/scripts/META_analysis/syst_param_analysis.jl

echo "Ran Julia script for run $run."
EOT
    
    sbatch temp_slurm_script_$run.sh
    echo "Submitted run $run for $N_SAMPLES samples."

    rm temp_slurm_script_$run.sh
done
