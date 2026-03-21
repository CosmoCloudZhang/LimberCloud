#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --mail-type=END
#SBATCH --time=04:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH -J JAX_GPU_Y1_DOUBLE
#SBATCH --output=LOG/%x_%j.out
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load gpu
module load conda
module load cray-mpich
module load PrgEnv-gnu
module load cray-hdf5-parallel

# Activate the conda environment
source $HOME/.bashrc
conda activate $CosmoENV

# Environment variables
export JAX_PLATFORMS=cuda
export SLURM_CPU_BIND=cores

export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Paths
TAG="Y1"
LABEL="DOUBLE"
BASE_PATH="/pscratch/sd/y/yhzhang/LimberCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/LimberCloud/"

# Run the script
srun -n 1 -c $SLURM_CPUS_PER_TASK -G 1 python -u "${BASE_PATH}JAX/GPU/${TAG}/${LABEL}.py" --tag=$TAG --path=$BASE_PATH --label=$LABEL --folder=$BASE_FOLDER --number=$SLURM_CPUS_PER_TASK
