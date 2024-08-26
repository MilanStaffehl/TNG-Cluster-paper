#!/bin/bash -l
#
#SBATCH -o /vera/u/mista/thesisProject/scripts/tracer_history/batch/output/out_generate_distance_z018.%j
#SBATCH -e /vera/u/mista/thesisProject/scripts/tracer_history/batch/output/err_generate_distance_z018.%j
#SBATCH -D ./
#SBATCH -J DMP-018
#SBATCH --partition=p.large   # request (part of) a node with 500GB memory
#SBATCH --ntasks-per-node=1   # only start 1 task via srun because Python multiprocessing starts more tasks internally
#SBATCH --cpus-per-task=36    # assign all the cores to that first task to make room for Python's multiprocessing tasks
#SBATCH --time=2:00:00        # maximum time the job is allowed to take

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
srun python3 ~/thesisProject/scripts/tracer_history/plot_quantity_with_time.py distance -p $SLURM_CPUS_PER_TASK -u -fo -x -f --zoom-in 18
