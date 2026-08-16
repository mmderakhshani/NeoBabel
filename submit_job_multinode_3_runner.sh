#!/bin/bash

module load 2023
module load CUDA/12.1.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate neobabel

head_node_ip=$(getent hosts $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) | awk '{ print $1 }')

export NCCL_SOCKET_IFNAME="eno2np0"

# Repo root on PYTHONPATH so `training.*` imports resolve under accelerate launch
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Set the machine rank based on SLURM_NODEID
export MACHINE_RANK=$SLURM_NODEID
# Determine the appropriate YAML config file for each node 
CONFIG_FILE="4_gpus_node_${MACHINE_RANK}.yaml"
echo "Node ${MACHINE_RANK} using config file ${CONFIG_FILE}"

cmd="accelerate launch --config_file accelerate_configs/multi_nodes_3/${CONFIG_FILE} --main_process_ip $head_node_ip training/train.py config=configs/neobabel_pretraining_stage1.yaml"
echo "Command: " $cmd

$cmd