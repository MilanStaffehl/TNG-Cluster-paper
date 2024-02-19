#!/bin/bash -l
#
#SBATCH -o ./output/out.%j
#SBATCH -e ./output/err.%j
#SBATCH -D ./
#SBATCH -J M3CC
#SBATCH --partition=p.large
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00       # maximum time the job is allowed to take

module purge
module load gcc/13 impi/2021.9
module load anaconda/3/2023.03

# Important:
# Set the number of OMP threads *per process* to avoid overloading of the node!
export OMP_NUM_THREADS=1

# verify number of threads used afterwards
echo "Number of cores: $SLURM_CPUS_PER_TASK"
source ~/venvs/illustris/bin/activate
which python3

# Use the environment variable SLURM_CPUS_PER_TASK to have multiprocessing
# spawn exactly as many processes as the node has CPUs available:
srun python3 ~/thesisProject/scripts/mass_trends/plot_cool_gas_mass_trends.py --to-file -v --log