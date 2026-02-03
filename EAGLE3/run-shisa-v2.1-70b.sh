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

NUM_GPUS=${NUM_GPUS:-1}
TP_SIZE=${TP_SIZE:-1}
TARGET_DEVICE=${TARGET_DEVICE:-}
REPORT_TO=${REPORT_TO:-wandb}
WANDB_PROJECT=${WANDB_PROJECT:-EAGLE3}
WANDB_NAME=${WANDB_NAME:-shisa-v2.1-70b-eagle3}
WANDB_ENTITY=${WANDB_ENTITY:-augmxnt}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.85}
SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-4096}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-flashinfer}
SGLANG_TORCH_COMPILE=${SGLANG_TORCH_COMPILE:-1}
SGLANG_PIECEWISE_CUDA_GRAPH=${SGLANG_PIECEWISE_CUDA_GRAPH:-1}
SGLANG_PIECEWISE_TOKENS=${SGLANG_PIECEWISE_TOKENS:-"512 1024 2048 4096"}
SGLANG_FP32_LM_HEAD=${SGLANG_FP32_LM_HEAD:-0}
SGLANG_TRITON_ATTENTION_REDUCE_FP32=${SGLANG_TRITON_ATTENTION_REDUCE_FP32:-0}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-sdpa}
TARGET_MICRO_BATCH_SIZE=${TARGET_MICRO_BATCH_SIZE:-}
DRAFT_ACCUMULATION_STEPS=${DRAFT_ACCUMULATION_STEPS:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-sft.shisa-v2.1.jsonl}
LOSS_BACKEND=${LOSS_BACKEND:-triton}
MAX_NUM_STEPS=${MAX_NUM_STEPS:-}
LOG_INTERVAL=${LOG_INTERVAL:-50}
DEBUG_NAN=${DEBUG_NAN:-0}
NAN_ACTION=${NAN_ACTION:-raise}
NAN_CLAMP_VALUE=${NAN_CLAMP_VALUE:-1e4}
NAN_MASK_NONFINITE=${NAN_MASK_NONFINITE:-0}

export PYTHONPATH=$SCRIPT_DIR/SpecForge${PYTHONPATH:+:$PYTHONPATH}
export WANDB_ENTITY
TARGET_DEVICE_ARGS=()
if [[ -n "$TARGET_DEVICE" ]]; then
    TARGET_DEVICE_ARGS=(--target-device "$TARGET_DEVICE")
fi
TARGET_MICRO_BATCH_ARGS=()
if [[ -n "$TARGET_MICRO_BATCH_SIZE" ]]; then
    TARGET_MICRO_BATCH_ARGS=(--target-micro-batch-size "$TARGET_MICRO_BATCH_SIZE")
fi
MAX_NUM_STEPS_ARGS=()
if [[ -n "$MAX_NUM_STEPS" ]]; then
    MAX_NUM_STEPS_ARGS=(--max-num-steps "$MAX_NUM_STEPS")
fi

# torchrun \
#     --standalone \
#     --nproc_per_node $NUM_GPUS \
#     train_eagle3_online.py \
#     --target-model-path shisa-ai/shisa-v2.1-llama3.3-70b-FP8-dynamic \
#     --draft-model-config shisa-v2.1-llama3.3-70b-eagle3.json \
#     --train-data-path sft.shisa-v2.1.jsonl \
#     --dist-timeout 60 \
#     --output-dir shisa-v2.1-llama3.3-70b-FP8-dynamic-eagle3 \
#     --num-epochs 3 \
#     --batch-size 1 \
#     --learning-rate 1e-4 \
#     --max-length 4096 \
#     --chat-template llama3 \
#     --tp-size $TP_SIZE \
#     --cache-dir $ROOT_DIR/cache \
#     --attention-backend flex_attention \
#     --flex-block-size 128
# #    --attention-backend sdpa

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $SCRIPT_DIR/SpecForge/scripts/train_eagle3.py \
    --target-model-path shisa-ai/shisa-v2.1-llama3.3-70b-FP8-dynamic \
    --draft-model-config shisa-v2.1-llama3.3-70b-eagle3.json \
    --train-data-path $TRAIN_DATA_PATH \
    --dist-timeout 60 \
    --output-dir shisa-v2.1-llama3.3-70b-FP8-dynamic-eagle3 \
    --num-epochs 3 \
    --batch-size $BATCH_SIZE \
    --draft-accumulation-steps $DRAFT_ACCUMULATION_STEPS \
    --learning-rate $LEARNING_RATE \
    --max-length $MAX_LENGTH \
    --loss-backend $LOSS_BACKEND \
    --log-interval $LOG_INTERVAL \
    $( [[ "$DEBUG_NAN" == "1" ]] && echo --debug-nan ) \
    --nan-action $NAN_ACTION \
    --nan-clamp-value $NAN_CLAMP_VALUE \
    $( [[ "$NAN_MASK_NONFINITE" == "1" ]] && echo --nan-mask-nonfinite ) \
    --chat-template llama3 \
    --tp-size $TP_SIZE \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend $ATTENTION_BACKEND \
    --dataloader-num-workers $DATALOADER_NUM_WORKERS \
    --report-to $REPORT_TO \
    --wandb-project $WANDB_PROJECT \
    --wandb-name $WANDB_NAME \
    --sglang-mem-fraction-static $SGLANG_MEM_FRACTION_STATIC \
    --sglang-attention-backend $SGLANG_ATTENTION_BACKEND \
    --sglang-context-length $SGLANG_CONTEXT_LENGTH \
    --sglang-piecewise-cuda-graph-max-tokens $SGLANG_CONTEXT_LENGTH \
    $( [[ "$SGLANG_TORCH_COMPILE" == "1" ]] && echo --sglang-enable-torch-compile ) \
    $( [[ "$SGLANG_PIECEWISE_CUDA_GRAPH" == "1" ]] && echo --sglang-enable-piecewise-cuda-graph ) \
    $( [[ "$SGLANG_PIECEWISE_CUDA_GRAPH" == "1" ]] && echo --sglang-piecewise-cuda-graph-tokens $SGLANG_PIECEWISE_TOKENS ) \
    $( [[ "$SGLANG_FP32_LM_HEAD" == "1" ]] && echo --sglang-enable-fp32-lm-head ) \
    $( [[ "$SGLANG_TRITON_ATTENTION_REDUCE_FP32" == "1" ]] && echo --sglang-triton-attention-reduce-in-fp32 ) \
    "${TARGET_DEVICE_ARGS[@]}" \
    "${TARGET_MICRO_BATCH_ARGS[@]}" \
    "${MAX_NUM_STEPS_ARGS[@]}"

# If Flex Attention works for you, swap the backend:
# --attention-backend flex_attention \
#    --flex-block-size 128
