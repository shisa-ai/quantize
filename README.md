# LLM Quantization Scripts

Scripts and tools for quantizing large language models, primarily for [Shisa AI](https://huggingface.co/shisa-ai) models. Includes vLLM/SGLang-compatible quantization (FP8, INT8, GPTQ) and llama.cpp GGUF formats.

## Quick Reference

| Tool | Formats | Best For |
|------|---------|----------|
| [llmcompressor](https://github.com/vllm-project/llm-compressor) | FP8, W8A8-INT8 | vLLM production deployment |
| [GPTQModel](https://github.com/ModelCloud/GPTQModel) | W4A16, W8A16 | Memory-constrained GPU inference |
| [ExLlamaV2](https://github.com/turboderp/exllamav2) | EXL2 | Fast local inference |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | GGUF (Q8, Q4, IQ4, IQ3, IQ2) | CPU/hybrid inference |
| [SpecForge](https://github.com/sgl-project/SpecForge) | EAGLE3 draft models | Speculative decoding |

## Calibration Dataset

Most scripts use [shisa-ai/shisa-v2-sharegpt](https://huggingface.co/datasets/shisa-ai/shisa-v2-sharegpt) as the calibration dataset. This is a bilingual Japanese/English dataset in ShareGPT format.

---

## Folders

### `shisa-asr/`

Phi4MM ASR decoder-only FP8 export tooling. The current script is pinned to
`shisa-ai/shisa-asr-v0.95b` and supports header-only scope inspection plus
`FP8_DYNAMIC` and `FP8_BLOCK` compressed-tensors exports while preserving the
multimodal towers and LoRA tensors.

- `quantize_decoder_fp8.py` - validated v0.95b exporter and vLLM target rewrite
- `README.md` - exact environment, reproduction commands, artifact checks, and
  publication TODOs

---

### `Mistral-Nemo-Japanese-Instruct-2408/`

Single-GPU quantization examples for a 12B parameter model. Good starting point for learning the quantization workflow.

**Scripts:**
- `quantize-llmcompressor-fp8-dynamic.py` - FP8 dynamic quantization
- `quantize-llmcompressor-w8a8-int8.py` - W8A8-INT8 quantization
- `quantize-gptqmodel-w4a16-g32.py` - GPTQ W4A16, group size 32
- `quantize-gptqmodel-w4a16-g32-noact.py` - GPTQ W4A16, group size 32, no activation order
- `quantize-gptqmodel-w4a16-g128.py` - GPTQ W4A16, group size 128
- `quantize-exllamav2-exl2.sh` - ExLlamaV2 EXL2 format

**Reference:** Original benchmarks at [shisa-ai/benchmarks](https://github.com/shisa-ai/benchmarks/tree/main/g5-vllm-vs-sglang)

---

### `shisa-v2-llama-3.1-405b/`

Multi-GPU quantization scripts for 405B parameter models. Tested on 8x H100/H200 systems.

**Quantization Scripts:**

| Script | Method | Hardware | Time |
|--------|--------|----------|------|
| `quantize-dynamic-fp8.py` | FP8 Dynamic | Single GPU | ~1h |
| `quantize-llmcompressor-w8a8-int8.py` | W8A8-INT8 + SmoothQuant | 8x H100 (125GB/GPU peak) | ~2 days |
| `quantize-gptqmodel-w4a16.py` | GPTQ W4A16/W8A16 | H200 (115GB peak) | ~12h |

**Usage Examples:**
```bash
# FP8 Dynamic (fast, minimal quality loss)
python quantize-dynamic-fp8.py

# W8A8-INT8 with SmoothQuant calibration
python quantize-llmcompressor-w8a8-int8.py \
    --model-id shisa-ai/shisa-v2-llama3.1-405b \
    --out-dir shisa-v2-llama3.1-405b-W8A8-INT8 \
    --calib-samples 2048 \
    --group-size 32 \
    --save-smoothquant

# GPTQ W4A16
python quantize-gptqmodel-w4a16.py \
    --model-id shisa-ai/shisa-v2-llama3.1-405b \
    --out-dir shisa-v2-llama3.1-405b-W4A16 \
    --calib-samples 2048 \
    --group-size 32 \
    --batch-size 1
```

**Subfolders:**

#### `gguf/`
GGUF conversion for llama.cpp. Uploads to [shisa-ai/shisa-v2-llama3.1-405b-GGUF](https://huggingface.co/shisa-ai/shisa-v2-llama3.1-405b-GGUF).

- `make-calibration_chat.py` - Generate calibration text from ShareGPT dataset
- `quantize-ggufs.sh` - Batch quantize to Q8_0, Q4_K_M, IQ4_XS, IQ3_M, IQ3_XS, IQ2_XXS
- `split-upload-to-hf.sh` - Split large GGUFs and upload to HuggingFace
- `imatrix.dat` - Importance matrix for quality quantization
- `calibration_chat.txt` - ~4000 samples calibration data

#### `sparse_logs/`
Quantization run logs for debugging.

**Other Files:**
- `debug-tokenizer.py` - Tokenizer debugging utility
- `gptq_log_*.log` - GPTQ quantization logs

---

### `EAGLE3/`

[EAGLE3](https://github.com/SafeAILab/EAGLE) speculative decoding draft model training using [SpecForge](https://github.com/sgl-project/SpecForge). Trains small "draft" models that predict multiple tokens for faster inference.

**Setup:**
```bash
# Install SpecForge
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge && pip install -v .
# or: pip install specforge
```

**Data Preparation:**
- `prepare_data.py` - Prepare training data
- `build_dataset.py` - Preprocess JSONL datasets, build SpecForge caches
- `generate-eagle-set.py` - Generate EAGLE-formatted training sets
- `count_tokens.py` - Token counting utility

**Training Scripts:**
- `train_eagle3_online.py` - Online EAGLE3 training (main training script)
- `run-chotto.sh` - Train draft model for chotto-14b
- `run-chotto.1xH100.sh` - Single H100 variant (uses FlexAttention)
- `run-shisa-v2.1-70b.sh` - Train draft model for shisa-v2.1-llama3.3-70b

**Config Files:**
- `unphi4-eagle3.json` - Draft model config for chotto/phi4-based models
- `shisa-v2.1-llama3.3-70b-eagle3.json` - Draft model config for 70B

**Environment Variables (run-shisa-v2.1-70b.sh):**
```bash
NUM_GPUS=1              # Number of GPUs
TP_SIZE=1               # Tensor parallelism
BATCH_SIZE=1            # Training batch size
MAX_LENGTH=4096         # Max sequence length
LEARNING_RATE=1e-4      # Learning rate
REPORT_TO=wandb         # Logging (wandb/tensorboard)
```

---

### `DeepSeek-V2.5-1210/`

Experimental/incomplete scripts for DeepSeek-V2.5 MoE model quantization (Dec 2024).

**Scripts:**
- `fp8.py` - FP8 quantization attempt
- `int8-w8a8-dynamic.py` - INT8 W8A8 dynamic quantization attempt

**Status:** Abandoned. For multi-GPU MoE conversion, refer to Llama-3.1-Tulu-3-405B examples.

---

### `cache/`

Shared cache directory for tokenized datasets, compiled kernels, and intermediate files. Used by EAGLE3 training scripts.

---

## Quantization Method Comparison

| Method | Bits | Quality | Speed | Memory | Use Case |
|--------|------|---------|-------|--------|----------|
| FP8 Dynamic | 8 | Excellent | Fast | ~50% | Quick deployment, minimal quality loss |
| W8A8-INT8 | 8 | Excellent | Fast | ~50% | Production vLLM/SGLang |
| W4A16 GPTQ | 4 | Good | Medium | ~25% | Memory-constrained GPUs |
| EXL2 | 2-8 | Variable | Fast | Variable | ExLlamaV2 local inference |
| GGUF Q8_0 | 8 | Excellent | Medium | ~50% | CPU inference |
| GGUF Q4_K_M | 4 | Good | Fast | ~25% | Balanced CPU inference |
| GGUF IQ4_XS | 4 | Good | Fast | ~25% | Quality-optimized 4-bit |
| GGUF IQ3_M | 3 | Moderate | Fast | ~20% | Aggressive compression |
| GGUF IQ2_XXS | 2 | Lower | Fast | ~15% | Maximum compression |

## Hardware Notes

- **Single GPU (12B models):** Any modern GPU with 24GB+ VRAM
- **405B FP8/W8A8:** 8x H100 80GB (peak ~125GB/GPU for W8A8-INT8)
- **405B GPTQ:** H200 141GB (peak ~115GB)
- **EAGLE3 training:** Single H100 for 14B target, multi-GPU for 70B+

---

## Related: SpinQuant Work

Additional quantization work using **SpinQuant with Cayley optimization** is in the [chotto-train](https://github.com/shisa-ai/chotto-train) repository:

**Location:** `/home/lhl/chotto-train/quantize/`

**Features:**
- **SpinQuant with learned Cayley rotations** - Better accuracy than static Hadamard transforms
- **Sequential onloading** - Quantize 405B+ models on a single GPU (~2GB VRAM)
- **vLLM compatible** - Outputs compressed-tensors format
- **Multiple schemes:** W8A8-INT8, W8A16, W4A16

**Key Scripts:**
- `quantize-llmcompressor.py` - Main quantization script with SpinQuant support
- `q-chotto-14b-20250922-spin.sh` - Production script with skip flags
- `debug-cayley.sh` - Fast iteration testing

**Quick Example:**
```bash
# SpinQuant with Cayley optimization (best accuracy)
python quantize-llmcompressor.py \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --out-dir llama-3.2-1b-spinquant-cayley \
  --use-spinquant \
  --spinquant-transform spinquant-learnable \
  --spinquant-learn \
  --spinquant-cayley-iters 100
```

**Comparison:**
| Method | Accuracy | Notes |
|--------|----------|-------|
| SpinQuant (Cayley) | Best | Learned rotations, ~3-4h on RTX 4090 |
| SpinQuant (Static) | Good | Fixed Hadamard, ~1h |
| SmoothQuant+GPTQ | Baseline | Standard approach |

See the [chotto-train quantize README](https://github.com/shisa-ai/chotto-train/tree/main/quantize) for full documentation.

> **Note:** This SpinQuant work will eventually be merged back into this repository.
