#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=256
#SBATCH --ntasks-per-node=1
#SBATCH -J PYTHON_CCL_Y1_TRIPLE
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load conda
module load cray-mpich
module load PrgEnv-gnu
module load cray-hdf5-parallel

# Activate the conda environment
source $HOME/.bashrc
conda activate $CosmoENV

# Set environment
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export OPENBLAS_NUM_THREADS=1

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

export NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=omp

# Initialize the process
TAG="Y1"
LABEL="TRIPLE"
BASE_PATH="/pscratch/sd/y/yhzhang/LimberCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/LimberCloud/"

# Run applications
python -u "${BASE_PATH}PYTHON/CCL/${TAG}/${LABEL}.py" --tag=$TAG --path=$BASE_PATH --label=$LABEL --folder=$BASE_FOLDER