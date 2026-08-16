#!/bin/bash

module load 2023
module load CUDA/12.1.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate neobabel

head_node_ip=$(getent hosts $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) | awk '{ print $1 }')

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_DEBUG=ERROR
# export NCCL_SOCKET_IFNAME=ib0
export OMP_NUM_THREADS=1 # Set this to 1, use more workers # Previously set to 8 (num cpus / num gpus)


export NCCL_SOCKET_IFNAME="eno2np0"

# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO

# Repo root on PYTHONPATH so `training.*` imports resolve under accelerate launch
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Set the machine rank based on SLURM_NODEID
export MACHINE_RANK=$SLURM_NODEID
# Determine the appropriate YAML config file for each node 
CONFIG_FILE="4_gpus_node_${MACHINE_RANK}.yaml"
echo "Node ${MACHINE_RANK} using config file ${CONFIG_FILE}"

cmd="accelerate launch --config_file accelerate_configs/multi_nodes_2/${CONFIG_FILE} --main_process_ip $head_node_ip training/train.py config=configs/neobabel_pretraining_stage1.yaml"
echo "Command: " $cmd

$cmd