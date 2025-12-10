#!/bin/bash
set -e  # Exit on error

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# Conda/Mamba environment - change this per machine if needed
ENV=${ENV:-quantize}

# =============================================================================
# EAGLE3 OFFLINE Training for Llama 3.1 405B
# Hardware: 8x MI300X (192GB each) = 1536GB total VRAM
# =============================================================================
#
# 405B is too large for online training (810GB weights in bf16).
# We use OFFLINE training which separates:
#   Stage 1: Generate hidden states using the 405B model (one-time, disk-intensive)
#   Stage 2: Train EAGLE3 head on cached hidden states (fast, no 405B in VRAM!)
#
# =============================================================================
# STORAGE REQUIREMENTS - CRITICAL!
# =============================================================================
#
# Offline training caches hidden states to disk. For 405B (hidden_size=16384):
#
# Per sample (seq_len=2048, bf16):
# +-------------------------+------------------------------------------+------------+
# | Component               | Shape                                    | Size       |
# +-------------------------+------------------------------------------+------------+
# | aux_hidden_state        | (seq_len, 3 × hidden_size) = (2048, 49152) | 200 MB   |
# | hidden_state            | (seq_len, hidden_size) = (2048, 16384)   | 67 MB      |
# | input_ids + loss_mask   | (seq_len,) × 2                           | 16 KB      |
# +-------------------------+------------------------------------------+------------+
# | TOTAL per sample        |                                          | ~267 MB    |
# +-------------------------+------------------------------------------+------------+
#
# Dataset storage estimates:
# +-----------------+------------------+
# | Dataset Samples | Storage Required |
# +-----------------+------------------+
# | 10,000          | ~2.6 TB          |
# | 50,000          | ~13 TB           |
# | 100,000         | ~26 TB           |
# +-----------------+------------------+
#
# Storage scales with hidden_size (NOT vocab_size - no logits stored!)
# For comparison: 70B (hidden_size=8192) needs ~133 MB/sample = half of 405B
#
# =============================================================================
# MEMORY REQUIREMENTS
# =============================================================================
#
# Stage 1 (Hidden State Generation) - requires full 405B model:
#   - 405B bf16 = ~810 GB weights
#   - TP=8 required: ~101 GB/GPU for weights
#   - Activations: ~50-80 GB/GPU depending on batch size
#   - Total: ~150-180 GB/GPU with batch_size=1-2
#
# Stage 2 (Offline Training) - NO target model needed!
#   - Only loads lm_head (~2 GB) + draft model (~2 GB)
#   - Can use full dp_size=8 for 8-way data parallelism
#   - ~20-40 GB/GPU - very lightweight!
#
# This is why offline training is so much faster for 405B:
#   - Stage 1: Run once, TP=8, slow but one-time cost
#   - Stage 2: Run many epochs, DP=8, fast iterations!
#
# =============================================================================
# PARAMETER REFERENCE (same as online training)
# =============================================================================
#
# +-------------------------+---------------------+----------------------------------+-------------+
# | Parameter               | Affects             | What it controls                 | Changes LR? |
# +-------------------------+---------------------+----------------------------------+-------------+
# | --draft-global-batch-size | TRAINING          | Effective batch for gradient     | YES         |
# |                         |                     | update (across all GPUs/accum)   |             |
# +-------------------------+---------------------+----------------------------------+-------------+
# | --draft-micro-batch-size | TRAINING           | Samples per forward pass         | NO          |
# |                         |                     | (only draft model in Stage 2!)   |             |
# +-------------------------+---------------------+----------------------------------+-------------+
# | --batch-size (Stage 1)  | HIDDEN STATE GEN    | Samples per 405B forward pass    | NO          |
# +-------------------------+---------------------+----------------------------------+-------------+
#
# =============================================================================

# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_GPUS=8

# --- Model Paths ---
TARGET_MODEL="shisa-ai/shisa-v2-llama3.1-405b"  # Your 405B model
DRAFT_CONFIG="llama3-405b-eagle3.json"           # or 405b-eagle3.json

# --- Data Paths ---
TRAIN_DATA="sft.shisa-v2.1.jsonl"
HIDDEN_STATES_DIR="$ROOT_DIR/cache/hidden_states/shisa-v2-405b"
OUTPUT_DIR="shisa-v2.1-llama3.1-405b-eagle3"
CACHE_DIR="$ROOT_DIR/cache"

# --- Training Hyperparameters ---
DRAFT_GLOBAL_BATCH_SIZE=32
DRAFT_MICRO_BATCH_SIZE=4   # Can be higher in Stage 2 (no 405B model in VRAM!)
LEARNING_RATE=3e-4
NUM_EPOCHS=3
MAX_LENGTH=2048
CHAT_TEMPLATE="llama3"

# --- Stage 1: Hidden State Generation ---
STAGE1_TP_SIZE=8           # Must be 8 to fit 405B
STAGE1_BATCH_SIZE=1        # Keep low due to memory constraints
STAGE1_MEM_FRAC=0.85       # GPU memory fraction for SGLang

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo "[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

check_disk_space() {
    local path="$1"
    local required_tb="$2"

    # Get available space in TB
    local available_kb=$(df -k "$path" 2>/dev/null | tail -1 | awk '{print $4}')
    local available_tb=$(echo "scale=2; $available_kb / 1024 / 1024 / 1024" | bc)

    log_info "Disk space check for $path:"
    log_info "  Available: ${available_tb} TB"
    log_info "  Required:  ~${required_tb} TB (estimated)"

    if (( $(echo "$available_tb < $required_tb" | bc -l) )); then
        log_error "Insufficient disk space! Need ~${required_tb} TB but only ${available_tb} TB available."
        return 1
    fi
    return 0
}

count_dataset_samples() {
    local data_path="$1"
    if [[ -f "$data_path" ]]; then
        wc -l < "$data_path"
    else
        echo "0"
    fi
}

count_hidden_states() {
    local hs_dir="$1"
    if [[ -d "$hs_dir" ]]; then
        find "$hs_dir" -name "*.ckpt" 2>/dev/null | wc -l
    else
        echo "0"
    fi
}

estimate_storage_tb() {
    local num_samples="$1"
    # 267 MB per sample for 405B
    echo "scale=2; $num_samples * 267 / 1024 / 1024" | bc
}

# =============================================================================
# SANITY CHECKS
# =============================================================================

run_sanity_checks() {
    log_info "Running sanity checks..."

    # Check training data exists
    if [[ ! -f "$TRAIN_DATA" ]]; then
        log_error "Training data not found: $TRAIN_DATA"
        exit 1
    fi

    # Count samples
    NUM_SAMPLES=$(count_dataset_samples "$TRAIN_DATA")
    log_info "Training data: $TRAIN_DATA ($NUM_SAMPLES samples)"

    # Estimate storage requirements
    REQUIRED_TB=$(estimate_storage_tb "$NUM_SAMPLES")
    log_info "Estimated storage for hidden states: ~${REQUIRED_TB} TB"

    # Check disk space for hidden states directory
    HIDDEN_STATES_PARENT=$(dirname "$HIDDEN_STATES_DIR")
    mkdir -p "$HIDDEN_STATES_PARENT"
    if ! check_disk_space "$HIDDEN_STATES_PARENT" "$REQUIRED_TB"; then
        log_error "Aborting due to insufficient disk space."
        log_info "Options:"
        log_info "  1. Free up disk space"
        log_info "  2. Use a smaller dataset"
        log_info "  3. Change HIDDEN_STATES_DIR to a path with more space"
        exit 1
    fi

    # Check draft config exists
    if [[ ! -f "$DRAFT_CONFIG" ]]; then
        log_error "Draft model config not found: $DRAFT_CONFIG"
        exit 1
    fi

    # Check GPU count
    DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || rocm-smi -l 2>/dev/null | grep -c "GPU" || echo "0")
    if [[ "$DETECTED_GPUS" -lt "$NUM_GPUS" ]]; then
        log_warn "Requested $NUM_GPUS GPUs but only detected $DETECTED_GPUS"
    fi

    log_info "Sanity checks passed!"
}

# =============================================================================
# STAGE DETECTION
# =============================================================================

detect_stage() {
    NUM_SAMPLES=$(count_dataset_samples "$TRAIN_DATA")
    NUM_HIDDEN_STATES=$(count_hidden_states "$HIDDEN_STATES_DIR")

    log_info "Stage detection:"
    log_info "  Training samples:     $NUM_SAMPLES"
    log_info "  Hidden states cached: $NUM_HIDDEN_STATES"

    # Check if Stage 1 is complete (allow 1% tolerance for edge cases)
    local threshold=$(echo "$NUM_SAMPLES * 0.99" | bc | cut -d. -f1)

    if [[ "$NUM_HIDDEN_STATES" -ge "$threshold" ]]; then
        log_info "  -> Stage 1 COMPLETE: Hidden states already generated"
        return 2  # Ready for Stage 2
    elif [[ "$NUM_HIDDEN_STATES" -gt 0 ]]; then
        log_info "  -> Stage 1 PARTIAL: $NUM_HIDDEN_STATES / $NUM_SAMPLES cached"
        log_info "     Will resume hidden state generation..."
        return 1  # Resume Stage 1
    else
        log_info "  -> Stage 1 NOT STARTED: No hidden states found"
        return 1  # Start Stage 1
    fi
}

# =============================================================================
# STAGE 1: HIDDEN STATE GENERATION
# =============================================================================

run_stage1() {
    log_info "=============================================="
    log_info "STAGE 1: Hidden State Generation"
    log_info "=============================================="
    log_info "This stage runs the 405B model to extract hidden states."
    log_info "This is a ONE-TIME cost - results are cached to disk."
    log_info ""
    log_info "Configuration:"
    log_info "  Target model:  $TARGET_MODEL"
    log_info "  TP size:       $STAGE1_TP_SIZE (required for 405B)"
    log_info "  Batch size:    $STAGE1_BATCH_SIZE"
    log_info "  Output dir:    $HIDDEN_STATES_DIR"
    log_info ""

    mkdir -p "$HIDDEN_STATES_DIR"

    # Note: prepare_hidden_states.py uses SGLang for efficient inference
    mamba run -n $ENV torchrun \
        --nproc_per_node=$NUM_GPUS \
        SpecForge/scripts/prepare_hidden_states.py \
        --data-path "$TRAIN_DATA" \
        --model-path "$TARGET_MODEL" \
        --cache-dir "$CACHE_DIR" \
        --output-path "$HIDDEN_STATES_DIR" \
        --chat-template "$CHAT_TEMPLATE" \
        --max-length $MAX_LENGTH \
        --enable-aux-hidden-states \
        --tp-size $STAGE1_TP_SIZE \
        --batch-size $STAGE1_BATCH_SIZE \
        --mem-frac $STAGE1_MEM_FRAC \
        --dist-timeout 3600

    log_info "Stage 1 complete!"
}

# =============================================================================
# STAGE 2: OFFLINE TRAINING
# =============================================================================

run_stage2() {
    log_info "=============================================="
    log_info "STAGE 2: Offline EAGLE3 Training"
    log_info "=============================================="
    log_info "Training draft model on cached hidden states."
    log_info "NO 405B model in VRAM - fast iterations with DP=$NUM_GPUS!"
    log_info ""
    log_info "Configuration:"
    log_info "  Draft config:          $DRAFT_CONFIG"
    log_info "  Global batch size:     $DRAFT_GLOBAL_BATCH_SIZE"
    log_info "  Micro batch size:      $DRAFT_MICRO_BATCH_SIZE"
    log_info "  Learning rate:         $LEARNING_RATE"
    log_info "  Epochs:                $NUM_EPOCHS"
    log_info "  Hidden states path:    $HIDDEN_STATES_DIR"
    log_info "  Output dir:            $OUTPUT_DIR"
    log_info ""

    mkdir -p "$OUTPUT_DIR"

    ARGS=(
        --standalone
        --nproc_per_node $NUM_GPUS
        SpecForge/scripts/train_eagle3_offline.py
        --target-model-path "$TARGET_MODEL"
        --draft-model-config "$DRAFT_CONFIG"
        --train-data-path "$TRAIN_DATA"
        --train-hidden-states-path "$HIDDEN_STATES_DIR"
        --output-dir "$OUTPUT_DIR"
        --num-epochs $NUM_EPOCHS
        --draft-global-batch-size $DRAFT_GLOBAL_BATCH_SIZE
        --draft-micro-batch-size $DRAFT_MICRO_BATCH_SIZE
        --learning-rate $LEARNING_RATE
        --max-length $MAX_LENGTH
        --chat-template "$CHAT_TEMPLATE"
        --cache-dir "$CACHE_DIR"
        --dist-timeout 60
        --draft-attention-backend sdpa
        --log-steps 10
        --resume
    )

    # Optional: wandb logging
    # ARGS+=(
    #     --report-to wandb
    #     --wandb-project EAGLE3
    #     --wandb-name "shisa-v2.1-llama3.1-405b-eagle3-offline"
    # )

    mamba run -n $ENV torchrun "${ARGS[@]}"

    log_info "Stage 2 complete!"
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    log_info "=============================================="
    log_info "EAGLE3 Offline Training Pipeline for 405B"
    log_info "=============================================="

    # Parse command line arguments
    FORCE_STAGE=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --stage1)
                FORCE_STAGE="1"
                shift
                ;;
            --stage2)
                FORCE_STAGE="2"
                shift
                ;;
            --check)
                run_sanity_checks
                detect_stage
                exit 0
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --stage1    Force run Stage 1 (hidden state generation)"
                echo "  --stage2    Force run Stage 2 (offline training)"
                echo "  --check     Run sanity checks and show stage status"
                echo "  --help      Show this help message"
                echo ""
                echo "Without options, automatically detects and runs the appropriate stage."
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Run sanity checks
    run_sanity_checks

    # Detect or force stage
    if [[ -n "$FORCE_STAGE" ]]; then
        log_info "Forcing Stage $FORCE_STAGE..."
        if [[ "$FORCE_STAGE" == "1" ]]; then
            run_stage1
        else
            run_stage2
        fi
    else
        detect_stage
        STAGE=$?

        if [[ $STAGE -eq 1 ]]; then
            run_stage1
            # After Stage 1, automatically run Stage 2
            log_info ""
            log_info "Stage 1 complete, proceeding to Stage 2..."
            log_info ""
            run_stage2
        else
            run_stage2
        fi
    fi

    log_info "=============================================="
    log_info "Pipeline complete!"
    log_info "=============================================="
}

# Run main
main "$@"
