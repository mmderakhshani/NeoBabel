#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --time=10:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=neobabel_pretraining_stage1_2_node
#SBATCH --output=logs/%x_%j_logs.log
#SBATCH --error=logs/%x_%j_errors.log

head_node_ip=$(getent hosts $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) | awk '{ print $1 }')
echo "Head node IP: $head_node_ip"

echo "Allocated nodes: "
srun hostname
echo ""

srun submit_job_multinode_2_runner.sh