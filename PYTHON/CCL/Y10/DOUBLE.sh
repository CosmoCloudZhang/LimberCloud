#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH -J PYTHON_CCL_Y10_DOUBLE
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Initialize the process
TAG="Y10"
LABEL="DOUBLE"
BASE_PATH="/pscratch/sd/y/yhzhang/LimberCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/LimberCloud/"

# Run applications
srun -n 1 -c $SLURM_CPUS_PER_TASK python -u "${BASE_PATH}PYTHON/CCL/${TAG}/${LABEL}.py" --tag=$TAG --path=$BASE_PATH --label=$LABEL --folder=$BASE_FOLDER --number=$SLURM_CPUS_PER_TASK