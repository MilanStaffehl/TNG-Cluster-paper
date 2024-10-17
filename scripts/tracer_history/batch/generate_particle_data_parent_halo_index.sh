#!/bin/bash
#SBATCH --job-name=PHI                        # Main job name for tracking
#SBATCH --time=30:00:00                       # Total time limit per job
#SBATCH --partition=p.large
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -D ./
#SBATCH -o /vera/u/mista/thesisProject/scripts/tracer_history/batch/output/generate_parent_halo_index.%j.out
#SBATCH -e /vera/u/mista/thesisProject/scripts/tracer_history/batch/output/generate_parent_halo_index.%j.err

module purge
module load gcc/13 impi/2021.9
module load anaconda/3/2023.03

export OMP_NUM_THREADS=1

source ~/venvs/illustris/bin/activate
which python3

# Each srun command launches a job with specific resource requests
srun python3 ~/thesisProject/scripts/tracer_history/plot_quantity_with_time.py parent-halo -p "$SLURM_CPUS_PER_TASK" -u -fo -x -f
