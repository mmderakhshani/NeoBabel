#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:4
#SBATCH --time=30:00:00
#SBATCH --mem=700G
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=neobabel_pretraining_stage1
#SBATCH --output=logs/%x_%j_logs.log
#SBATCH --error=logs/%x_%j_errors.log

module load 2023
module load CUDA/12.1.1

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate neobabel

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

wandb login <add_wandb_api_key>
accelerate launch --config_file accelerate_configs/single_node/4_gpus_deepspeed_zero2.yaml --main_process_port=8888 training/train.py config=configs/neobabel_pretraining_stage1.yaml