#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --job-name=dpg-evals
#SBATCH --time=01:00:00
#SBATCH --output=dpg_logs/dpg-slurm-%j.out
#SBATCH --error=dpg_logs/dpg-slurm-%j.err
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --signal=SIGUSR1@90
#SBATCH --exclude=gcn45

hostname
module purge
source $HOME/.bashrc

module load 2023
module load CUDA/12.1.1

conda activate dpg
export HF_HOME=/scratch-shared/dvarghese/models/

# Submit this script from benchmarking/dpg/. The repo root is put on
# PYTHONPATH so neobabel_generate_dpg.py can import the top-level
# `models`/`training` packages.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(cd ../.. && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CONFIG_FILE="$1"
echo "Config file: $CONFIG_FILE"

torchrun --nproc_per_node=4 neobabel_generate_dpg.py config="$CONFIG_FILE"