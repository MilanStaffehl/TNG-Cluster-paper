#!/bin/bash -l
#
#SBATCH -o ./output/out.%j
#SBATCH -e ./output/err.%j
#SBATCH -D ./
#SBATCH -J M3I3TTS
#SBATCH --partition=p.large
#SBATCH --ntasks-per-node=1   # only start 1 task via srun because Python multiprocessing starts more tasks internally
#SBATCH --cpus-per-task=1    # assign all the cores to that first task to make room for Python's multiprocessing tasks
#SBATCH --time=10:00:00       # maximum time the job is allowed to take

module purge
module load gcc/13 impi/2021.9
module load anaconda/3/2023.03

# Important:
# Set the number of OMP threads *per process* to avoid overloading of the node!
export OMP_NUM_THREADS=1

# verify number of threads used afterwards
source ~/venvs/illustris/bin/activate
which python3

# Use the environment variable SLURM_CPUS_PER_TASK to have multiprocessing
# spawn exactly as many processes as the node has CPUs available:
srun python3 ~/thesisProject/scripts/radial_profiles/plot_individual_radial_profiles.py -s TNG300-1 --cluster-core --forbid-tree -f --log -v