#!/bin/bash -l
#
#SBATCH -o ./output/out.%j
#SBATCH -e ./output/err.%j
#SBATCH -D ./
#SBATCH -J TABCLSTR
#SBATCH --partition=p.huge
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=900GB
#SBATCH --time=3:00:00

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
srun python3 ~/thesisProject/scripts/aux/tabulate_cluster_data.py --forbid-tree -vv
