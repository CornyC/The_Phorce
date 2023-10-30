#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=38
#SBATCH --time=10:00:00
#SBATCH --output="mp0mp0_opt-%j.out"
#SBATCH --error="mp0mp0_opt-%j.err"
#SBATCH --job-name="mp0mp0_opt"
#SBATCH --export=ALL,MPI_MODULE=mpi/openmpi/4.1,EXECUTABLE=./ompi_omp_program

module purge
module load chem/cp2k/8.2

export EXECUTABLE="cp2k.psmp -o geom_opt_rst.out geom_opt_rst.inp"
export MPIRUN_OPTIONS="--bind-to core --map-by socket:PE=${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUM_CORES=${SLURM_NTASKS}*${SLURM_CPUS_PER_TASK}

echo "${EXECUTABLE} running on ${NUM_CORES} cores with ${SLURM_NTASKS} MPI-tasks and ${OMP_NUM_THREADS} threads"
startexe="mpirun -n ${SLURM_NTASKS} ${MPIRUN_OPTIONS} ${EXECUTABLE}"
echo $startexe
exec $startexe

