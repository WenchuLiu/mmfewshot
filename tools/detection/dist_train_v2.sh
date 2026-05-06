#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# 自动选择 launcher
if command -v torchrun &> /dev/null; then
    LAUNCHER="torchrun"
else
    LAUNCHER="python -m torch.distributed.launch"
fi

PYTHONPATH="$(dirname "$0")/../../":$PYTHONPATH \
$LAUNCHER \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch "${@:3}"
