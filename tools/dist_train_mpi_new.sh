#!/bin/bash
source /mnt/cfs/algorithm/yunpeng.zhang/.bashrc
conda activate bevformer_a100
export PYTHONPATH="."

CONFIG=$1
nproc_per_node=$2
master_addr=$3
nnodes=$4
node_rank=$5
PORT=$6

export PYTHONPATH="$(dirname $0)/../":$PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --nnodes=$nnodes --node_rank=$node_rank --master_addr=$master_addr --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:7}