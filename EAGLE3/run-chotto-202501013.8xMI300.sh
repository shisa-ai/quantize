#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# Trying to prevent crashes
# export TORCH_COMPILE=0
# export TRITON_MAX_BLOCK_SIZE=64
 
# export TRITON_MAX_BLOCK_SIZE=32
# export TRITON_NUM_STAGES=1

# TORCHDYNAMO_VERBOSE=1

NUM_GPUS=8

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    train_eagle3_online.py \
    --target-model-path shisa-ai/chotto-14b-20251013-dpo \
    --draft-model-config unphi4-14b-eagle3.json \
    --train-data-path chotto-20251010.sft.jsonl \
    --dist-timeout 60 \
    --output-dir /data/outputs/chotto-14b-20251013-eagle3 \
    --num-epochs 10 \
    --batch-size 16 \
    --learning-rate 3e-4 \
    --max-length 2048 \
    --draft-global-batch-size 32 \
    --draft-micro-batch-size 4 \
    --log-steps 1 \
    --report-to wandb \
    --wandb-project EAGLE3 \
    --wandb-name "8xMI300 chotto-14-20251013-dpo bs=32" \
    --chat-template phi4 \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend sdpa

# If Flex Attention doesn't work...
# --attention-backend sdpa
#    --attention-backend flex_attention \
#    --flex-block-size 16
    # --flex-block-size 64
