#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# Conda/Mamba environment - change this per machine if needed
ENV=${ENV:-quantize}

# =============================================================================
# EAGLE3 Training for Llama 3.3 70B (132GB weights in bf16)
# Hardware: 8x MI300X (192GB each) = 1536GB total VRAM
# =============================================================================
#
# =============================================================================
# PARAMETER REFERENCE - WHAT CONTROLS WHAT
# =============================================================================
#
# There are THREE separate "batch size" concepts. Understanding these is critical:
#
# +-------------------------+---------------------+----------------------------------+-------------+
# | Parameter               | Affects             | What it controls                 | Changes LR? |
# +-------------------------+---------------------+----------------------------------+-------------+
# | --draft-global-batch-size | TRAINING          | Effective batch for gradient     | YES         |
# |                         |                     | update (across all GPUs/accum)   |             |
# +-------------------------+---------------------+----------------------------------+-------------+
# | --draft-micro-batch-size | TRAINING (both!)   | Batch size for EACH forward pass | NO          |
# |   (default=1)           | - Target model fwd  | through target model (online     |             |
# |                         | - Draft model fwd   | hidden state generation) AND     |             |
# |                         |                     | draft model (training). Higher = |             |
# |                         |                     | faster but more VRAM.            |             |
# +-------------------------+---------------------+----------------------------------+-------------+
# | --batch-size            | EVAL ONLY           | Only used if --eval-data-path    | NO          |
# |                         |                     | is set. Does NOTHING for train!  |             |
# +-------------------------+---------------------+----------------------------------+-------------+
#
# The relationship:
#   draft_accumulation_steps = draft_global_batch_size / dp_size / draft_micro_batch_size
#
# Each gradient update requires (draft_accumulation_steps) forward passes through
# the target model, each processing (draft_micro_batch_size) samples.
#
# =============================================================================
# TENSOR PARALLELISM vs DATA PARALLELISM TRADEOFFS
# =============================================================================
#
# With NUM_GPUS=8:
#   dp_size = NUM_GPUS / tp_size
#
# +--------+--------+--------+-------------------+------------------+---------------+
# | tp_size| dp_size| Model/GPU | accum_steps     | Target fwd/step  | Parallelism   |
# |        |        |           | (gbs=32,mbs=1)  | per gradient     |               |
# +--------+--------+-----------+-----------------+------------------+---------------+
# | 1      | 8      | 132GB     | 32/8/1 = 4      | 4 (parallel)     | Data parallel |
# | 8      | 1      | 16.5GB    | 32/1/1 = 32     | 32 (sequential!) | Tensor par.   |
# +--------+--------+-----------+-----------------+------------------+---------------+
#
# EXPERIMENTAL RESULTS on 8xMI300X (192GB each):
#   - tp=1:           188GB/GPU, ~14h/epoch  <-- FASTEST (8-way data parallelism)
#   - tp=8, mbs=1:     95GB/GPU, ~40h/epoch  <-- 3x SLOWER (no data parallelism!)
#   - tp=8, mbs=4:    145GB/GPU, ~53h/epoch  <-- EVEN SLOWER (communication bound!)
#
# WHY TP=8 IS SLOWER:
#   With tp=8, dp_size=1, so there's NO data parallelism. All 8 GPUs work together
#   on ONE sample at a time via tensor parallelism. The 32 accumulation steps run
#   SEQUENTIALLY, not in parallel.
#
#   With tp=1, dp_size=8, each GPU processes different samples IN PARALLEL.
#   Only 4 accumulation steps per GPU, and all 8 GPUs work simultaneously.
#
# WHY INCREASING MICRO BATCH SIZE DOESN'T HELP TP=8:
#   TP=8 is COMMUNICATION-BOUND, not compute-bound. Every layer (80 layers for 70B)
#   requires all-reduce/all-gather operations across all 8 GPUs via Infinity Fabric.
#
#   | Config        | Micro BS | Accum Steps | Fwd Passes | Comm/Pass | Total Comm |
#   |---------------|----------|-------------|------------|-----------|------------|
#   | tp=8, mbs=1   | 1        | 32          | 32         | 1x        | 32x        |
#   | tp=8, mbs=4   | 4        | 8           | 8          | 4x        | 32x        |
#
#   Same total communication, but larger tensors per collective are LESS efficient
#   on Infinity Fabric. Plus higher memory pressure (145GB vs 95GB). Result: slower.
#
# RECOMMENDATION: Use tp=1 if the model fits (188GB/192GB is tight but works).
#   - TP=1 is 3-4x faster due to 8-way data parallelism
#   - Only use TP>1 if you truly need memory headroom (e.g., 405B model)
#   - For larger models, consider hybrid: tp=2,dp=4 or tp=4,dp=2 for balance
#
# =============================================================================
# LEARNING RATE SCALING
# =============================================================================
#
# EAGLE3 paper: LR 5e-5 at effective batch size 2
# Linear scaling rule: LR = 5e-5 * (draft_global_batch_size / 2)
#
# +------------------------+------------+------------------------------------------+
# | draft_global_batch_size| LR         | Notes                                    |
# +------------------------+------------+------------------------------------------+
# | 2                      | 5e-5       | Paper baseline                           |
# | 8                      | 2e-4       | 4x batch -> 4x LR                        |
# | 16                     | 4e-4       | 8x batch -> 8x LR                        |
# | 32                     | 8e-4       | Theoretical; 3e-4 to 5e-4 works better   |
# | 64                     | 1.6e-3     | Theoretical; likely need lower           |
# +------------------------+------------+------------------------------------------+
#
# ONLY draft_global_batch_size affects LR. Changes to tp_size, draft_micro_batch_size,
# or batch_size do NOT require LR changes (they affect speed/memory, not optimization).
#
# =============================================================================
# MEMORY ESTIMATION
# =============================================================================
#
# With tp=1 (full model on each GPU):
# +------------------------------------------+------------------+
# | Component                                | Per-GPU Memory   |
# +------------------------------------------+------------------+
# | Target Model (132GB, full)               | ~132 GB          |
# | Draft Model (1-layer EAGLE3 head)        | ~1-2 GB          |
# | Activations + Gradients                  | ~50-55 GB        |
# +------------------------------------------+------------------+
# | TOTAL (observed)                         | ~188 GB          |
# +------------------------------------------+------------------+
#
# With tp=8 (model sharded across 8 GPUs):
# +------------------------------------------+------------------+
# | Component                                | Per-GPU Memory   |
# +------------------------------------------+------------------+
# | Target Model (132GB / 8)                 | ~16.5 GB         |
# | Draft Model (1-layer EAGLE3 head)        | ~1-2 GB          |
# | Activations + Gradients                  | ~75 GB           |
# +------------------------------------------+------------------+
# | TOTAL (observed)                         | ~95 GB           |
# +------------------------------------------+------------------+
#
# =============================================================================
# CHAT TEMPLATE NOTES
# =============================================================================
#
# The --chat-template flag serves TWO purposes in SpecForge:
#
# 1. LOSS MASK GENERATION: The template's assistant_header and end_of_turn_token
#    are used to identify assistant response spans via regex matching.
#    Only tokens in assistant responses contribute to the loss.
#    See: SpecForge/specforge/data/preprocessing.py:56 (_apply_loss_mask_from_chat_template)
#
# 2. CONVERSATION FORMATTING: The actual tokenization uses the HuggingFace
#    tokenizer's built-in apply_chat_template() method, which reads the
#    chat_template from the model's tokenizer_config.json.
#    See: SpecForge/specforge/data/parse.py:83
#
# So --chat-template llama3 is correct for Llama-based models. The tokenizer
# will use whatever template is stored in the model config automatically.
# To verify: AutoTokenizer.from_pretrained("shisa-ai/shisa-v2.1-llama3.3-70b").chat_template
#
# =============================================================================

# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_GPUS=8

# --- Parallelism Strategy ---
# tp=1: Faster (data parallel), but needs 188GB/GPU
# tp=8: Slower (tensor parallel), but only needs 20GB/GPU for weights, 95GB/GPU total
TP_SIZE=1

# --- Training Batch Sizes ---
# draft_global_batch_size: Effective batch size for optimization (AFFECTS LR!)
# draft_micro_batch_size: Samples per forward pass (affects speed/memory, not LR)
DRAFT_GLOBAL_BATCH_SIZE=32
DRAFT_MICRO_BATCH_SIZE=1  # Increase to 4-8 if using tp=8 to speed up training

# --- Learning Rate ---
# Scale with draft_global_batch_size: LR = 5e-5 * (gbs / 2)
# gbs=32 theoretical=8e-4, but 3e-4 to 5e-4 is more stable
LEARNING_RATE=4e-4

# --- Eval Batch Size (only used if --eval-data-path is set) ---
# This does NOTHING for training speed! Don't waste time tuning it.
EVAL_BATCH_SIZE=8

# =============================================================================
# ALTERNATIVE CONFIGURATIONS (uncomment one block)
# =============================================================================

# --- Paper baseline (slow but safe) ---
# LEARNING_RATE=5e-5
# DRAFT_GLOBAL_BATCH_SIZE=2

# --- Conservative ---
# LEARNING_RATE=2e-4
# DRAFT_GLOBAL_BATCH_SIZE=8

# --- Aggressive (if training is stable) ---
# LEARNING_RATE=5e-4
# DRAFT_GLOBAL_BATCH_SIZE=32

# --- TP=8 config (if you need memory headroom) ---
# TP_SIZE=8
# DRAFT_MICRO_BATCH_SIZE=8  # Critical for tp=8 performance!
# LEARNING_RATE=3e-4
# DRAFT_GLOBAL_BATCH_SIZE=32

# =============================================================================

# Trying to prevent crashes (uncomment if needed)
# export TORCH_COMPILE=0
# export TRITON_MAX_BLOCK_SIZE=64
# export TRITON_MAX_BLOCK_SIZE=32
# export TRITON_NUM_STAGES=1
# TORCHDYNAMO_VERBOSE=1

# Base arguments
ARGS=(
    --resume
    --standalone
    --nproc_per_node $NUM_GPUS
    train_eagle3_online.py
    --target-model-path shisa-ai/shisa-v2.1-llama3.3-70b
    --draft-model-config llama3-70b-eagle3.json
    --train-data-path sft.shisa-v2.1.jsonl
    --dist-timeout 60
    --output-dir shisa-v2.1-llama3.3-70b-eagle3
    --num-epochs 3
    --learning-rate $LEARNING_RATE
    --batch-size $EVAL_BATCH_SIZE
    --draft-global-batch-size $DRAFT_GLOBAL_BATCH_SIZE
    --draft-micro-batch-size $DRAFT_MICRO_BATCH_SIZE
    --max-length 2048
    --chat-template llama3
    --cache-dir $ROOT_DIR/cache
    --attention-backend sdpa
    --tp-size $TP_SIZE
)

# Optional: wandb logging
# ARGS+=(
#     --log-steps 1
#     --report-to wandb
#     --wandb-project EAGLE3
#     --wandb-name "shisa-v2.1-llama3.3-70b-eagle3"
# )

mamba run -n $ENV torchrun "${ARGS[@]}"

# =============================================================================
# Attention backend notes:
# - MI300X: use sdpa (flex_attention may not work on ROCm)
# - H100: can use flex_attention for potential speedup
#   --attention-backend flex_attention
#   --flex-block-size 64
# =============================================================================
