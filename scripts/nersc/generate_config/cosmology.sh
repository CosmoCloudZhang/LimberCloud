#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o logs/%x_%j.out
#SBATCH -J CONFIG_COSMOLOGY
#SBATCH --cpus-per-task=256
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

set -eo pipefail

# Configure the project environment
REPO_ROOT="${LIMBERCLOUD_REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${REPO_ROOT}/scripts/nersc/load_environment.sh"
source "${REPO_ROOT}/scripts/nersc/modules/cpu.sh"
conda activate "${LIMBERCLOUD_CONDA_ENV}"

# Initialize the process
RUNTIME_ROOT="${LIMBERCLOUD_RUNTIME_ROOT:?Set LIMBERCLOUD_RUNTIME_ROOT to the external data/results root}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Run applications
python -u "${REPO_ROOT}/scripts/generate_config/cosmology.py" --folder="${RUNTIME_ROOT}"
