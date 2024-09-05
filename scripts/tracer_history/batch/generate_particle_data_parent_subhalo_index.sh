#!/bin/bash
#SBATCH --job-name=PDCL-PSI                   # Main job name for tracking
#SBATCH --time=24:00:00                       # Total time limit for the main job (24 hours)
#SBATCH --partition=p.large                   # Specify your cluster partition (adjust as needed)
#SBATCH --cpus-per-task=72
#SBATCH --ntasks-per-node=1
#SBATCH -D ./
#SBATCH -o /vera/u/mista/thesisProject/scripts/tracer_history/batch/output/generate_parent_subhalo_index_glob.out.%j
#SBATCH -e /vera/u/mista/thesisProject/scripts/tracer_history/batch/output/generate_parent_subhalo_index_glob.err.%j

module purge
module load gcc/13 impi/2021.9
module load anaconda/3/2023.03

export OMP_NUM_THREADS=1

source ~/venvs/illustris/bin/activate
which python3

# Loop to execute all tasks with srun in parallel
for i in {0..351}
do
    # Each srun command launches a job with specific resource requests
    srun --time=2:00:00 \
         --output=output/generate_parent_subhalo_index_z"$i".out \
         --error=output/generate_parent_subhalo_index_z"$i".err \
         python3 ~/thesisProject/scripts/tracer_history/plot_quantity_with_time.py parent-subhalo -p "$SLURM_CPUS_PER_TASK" -u -fo -x -f -z "$i" &
done
# Wait for all background srun tasks to complete
wait
