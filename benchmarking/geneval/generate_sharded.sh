#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --job-name=geneval-evals
#SBATCH --time=01:00:00
#SBATCH --output=geneval_logs/evals-slurm-%j.out
#SBATCH --error=geneval_logs/evals-slurm-%j.err
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

conda activate geneval
export HF_HOME=/scratch-shared/dvarghese/models/

# Submit this script from benchmarking/geneval/. The repo root is put on
# PYTHONPATH so generate.py can import the top-level `models`/`training` packages.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(cd ../.. && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

torchrun --nproc_per_node=4 generate.py config="$REPO_ROOT/configs/eval/neobabel_gen_eval_512x512.yaml"