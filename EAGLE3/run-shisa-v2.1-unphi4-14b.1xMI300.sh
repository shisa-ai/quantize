#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# Conda/Mamba environment - change this per machine if needed
ENV=${ENV:-quantize}

# GPU Configuration
# For 1xMI300:
NUM_GPUS=1
# For 8xMI300, uncomment:
# NUM_GPUS=8

# Batch size configuration
# For 1xMI300:
BATCH_SIZE=8
LEARNING_RATE=1e-4
# For 8xMI300, uncomment and adjust:
# BATCH_SIZE=16
# LEARNING_RATE=3e-4
# DRAFT_GLOBAL_BATCH_SIZE=32
# DRAFT_MICRO_BATCH_SIZE=4

# Trying to prevent crashes (uncomment if needed)
# export TORCH_COMPILE=0
# export TRITON_MAX_BLOCK_SIZE=64
# export TRITON_MAX_BLOCK_SIZE=32
# export TRITON_NUM_STAGES=1
# TORCHDYNAMO_VERBOSE=1

# Base arguments
ARGS=(
    --standalone
    --nproc_per_node $NUM_GPUS
    train_eagle3_online.py
    --target-model-path shisa-ai/shisa-v2.1-unphi4-14b
    --draft-model-config unphi4-14b-eagle3.json
    --train-data-path sft.shisa-v2.1.jsonl
    --dist-timeout 60
    --output-dir shisa-v2.1-unphi4-14b-eagle3
    --num-epochs 3
    --batch-size $BATCH_SIZE
    --learning-rate $LEARNING_RATE
    --max-length 2048
    --chat-template phi4
    --cache-dir $ROOT_DIR/cache
    --attention-backend sdpa
)

# For 8xMI300, add these arguments:
# ARGS+=(
#     --draft-global-batch-size $DRAFT_GLOBAL_BATCH_SIZE
#     --draft-micro-batch-size $DRAFT_MICRO_BATCH_SIZE
#     --log-steps 1
#     --report-to wandb
#     --wandb-project EAGLE3
#     --wandb-name "8xMI300 shisa-v2.1-unphi4-14b bs=$DRAFT_GLOBAL_BATCH_SIZE"
# )

# For more epochs on 8xMI300, uncomment:
# ARGS+=(--num-epochs 10)

mamba run -n $ENV torchrun "${ARGS[@]}"

# Attention backend notes:
# - MI300X: use sdpa (flex_attention may not work)
# - H100: can use flex_attention
#   --attention-backend flex_attention
#   --flex-block-size 64
