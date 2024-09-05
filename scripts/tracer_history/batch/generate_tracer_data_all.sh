#!/bin/bash -l
#
#SBATCH -o /vera/u/mista/thesisProject/scripts/tracer_history/batch/output/tracer_job_global_out.%j
#SBATCH -e /vera/u/mista/thesisProject/scripts/tracer_history/batch/output/tracer_job_global_err.%j
#SBATCH -D ./
#SBATCH -J TDCL-GLOB
#SBATCH --time=20:00:00       # maximum time the job is allowed to take

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

# Launch individual jobs
for i in $(seq 0 351)
do
  srun --ntasks-per-node=1 \
       --partition=p.large \
       --cpus-per-task=36 \
       --time=1:00:00 \
       --output=/vera/u/mista/thesisProject/scripts/tracer_history/batch/output/tracer_job_"$i"_out.%j \
       --error=/vera/u/mista/thesisProject/scripts/tracer_history/batch/output/tracer_job_"$i"_err.%j \
       python3 ~/thesisProject/scripts/tracer_history/generate_tracer_data.py trace-back -s TNG-Cluster -p "$SLURM_CPUS_PER_TASK" -n "$i" &
done
wait  # wait for all jobs to complete