#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# Conda/Mamba environment - change this per machine if needed
ENV=${ENV:-quantize}

# GPU Configuration
NUM_GPUS=1

# =============================================================================
# Learning Rate / Batch Size Configurations
# EAGLE3 paper: LR 5e-5 at effective batch size 2, scale LR linearly with batch
# =============================================================================

# --- 1xMI300 (default) ---
LEARNING_RATE=2e-4
DRAFT_GLOBAL_BATCH_SIZE=8       # 4x paper batch → 4x LR

# --- 1xMI300 (paper baseline) ---
# LEARNING_RATE=5e-5
# DRAFT_GLOBAL_BATCH_SIZE=2

# --- 8xMI300X (conservative) ---
# NUM_GPUS=8
# LEARNING_RATE=2e-4
# DRAFT_GLOBAL_BATCH_SIZE=8

# --- 8xMI300X (higher throughput) ---
# NUM_GPUS=8
# LEARNING_RATE=4e-4
# DRAFT_GLOBAL_BATCH_SIZE=16

# --- 8xMI300X (max throughput) ---
NUM_GPUS=8
LEARNING_RATE=4e-4
DRAFT_GLOBAL_BATCH_SIZE=32

# Hmm, not faster...
# DRAFT_GLOBAL_BATCH_SIZE=64

# Target model inference batch size (increase for speedup if VRAM allows)
TARGET_BATCH_SIZE=32             # Try 8-16 on 8xMI300X for faster hidden state generation
# 8 is at 57GB, we can probably set to 16 maybe even 32 no problem
# 32 is at 60GB?
# only takes 5h/epoch...

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
    --learning-rate $LEARNING_RATE
    --batch-size $TARGET_BATCH_SIZE
    --draft-global-batch-size $DRAFT_GLOBAL_BATCH_SIZE
    --max-length 2048
    --chat-template phi4
    --cache-dir $ROOT_DIR/cache
    --attention-backend sdpa
)

# Optional: wandb logging
# ARGS+=(
#     --log-steps 1
#     --report-to wandb
#     --wandb-project EAGLE3
#     --wandb-name "shisa-v2.1-unphi4-14b-eagle3"
# )

mamba run -n $ENV torchrun "${ARGS[@]}"

# Attention backend notes:
# - MI300X: use sdpa (flex_attention may not work)
# - H100: can use flex_attention
#   --attention-backend flex_attention
#   --flex-block-size 64
