#!/bin/bash
#SBATCH --job-name=PD-PHI                     # Main job name for tracking
#SBATCH --time=24:00:00                       # Total time limit for the main job (24 hours)
#SBATCH --partition=p.large                   # Specify your cluster partition (adjust as needed)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --array=0-351%8
#SBATCH -D ./
#SBATCH -o /vera/u/mista/thesisProject/scripts/tracer_history/batch/output/generate_parent_halo_index_z%a.%A.out
#SBATCH -e /vera/u/mista/thesisProject/scripts/tracer_history/batch/output/generate_parent_halo_index_z%a.%A.err

module purge
module load gcc/13 impi/2021.9
module load anaconda/3/2023.03

export OMP_NUM_THREADS=1

source ~/venvs/illustris/bin/activate
which python3

# Each srun command launches a job with specific resource requests
srun python3 ~/thesisProject/scripts/tracer_history/plot_quantity_with_time.py parent-halo -p "$SLURM_CPUS_PER_TASK" -u -fo -x -f -z "$SLURM_ARRAY_TASK_ID"
