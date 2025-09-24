#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# Trying to prevent crashes
# export TORCH_COMPILE=0
# export TRITON_MAX_BLOCK_SIZE=64
export TRITON_MAX_BLOCK_SIZE=32
export TRITON_NUM_STAGES=1
# TORCHDYNAMO_VERBOSE=1

NUM_GPUS=2

# shisa-ai/chotto-14b-20250922-FP8  - if necessary? \
#    --batch-size 4 \
#   --num-epochs 10 \
#    --train-data-path sft.shisa-v2.1-EAGLE3.jsonl \

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    train_eagle3_online.py \
    --target-model-path shisa-ai/chotto-14b-20250922 \
    --draft-model-config unphi4-eagle3.json \
    --train-data-path sft.shisa-v2.1.jsonl \
    --dist-timeout 60 \
    --output-dir chotto-14b-20250922-FP8-eagle3 \
    --num-epochs 10 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template phi4 \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend sdpa

# If Flex Attention doesn't work...
# --attention-backend sdpa
#    --attention-backend flex_attention \
#    --flex-block-size 16
    # --flex-block-size 64
