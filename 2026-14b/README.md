# 2026-14b Quantization Scripts

Scripts for quantizing 14B-class models using [llm-compressor](https://github.com/vllm-project/llm-compressor).

## Environment Setup

```bash
# Activate the llmcompressor environment
conda activate /home/lhl/.conda/envs/llmcompressor

# Verify installation
python -c "import llmcompressor; print(llmcompressor.__version__)"
# Expected: 0.9.0.1+
```

## quantize-fp8-block.py

Production script for FP8_BLOCK quantization with optional SpinQuant and AWQ modifiers.

### Basic Usage

```bash
# Basic FP8_BLOCK quantization (no transforms)
python quantize-fp8-block.py -m shisa-ai/chotto-14b-20260107-dpo

# With SpinQuant (offline rotations R1,R2 - no latency penalty)
python quantize-fp8-block.py -m shisa-ai/chotto-14b-20260107-dpo --spinquant

# With SpinQuant + AWQ smoothing
python quantize-fp8-block.py -m shisa-ai/chotto-14b-20260107-dpo --spinquant --awq
```

### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --model` | required | HuggingFace model ID or local path |
| `-o, --output-dir` | auto | Output directory (auto-generated from model name) |
| `--spinquant` | off | Enable SpinQuant transforms |
| `--spinquant-rotations` | `R1,R2` | Which rotations to apply (comma-separated) |
| `--spinquant-transform-type` | `hadamard` | Transform type: hadamard, random-hadamard, random-matrix |
| `--spinquant-transform-block-size` | `128` | Block size for rotation matrices |
| `--awq` | off | Use AWQ smoothing (slower, more accurate) |
| `--num-calibration-samples` | `256` | Number of calibration samples |
| `--max-seq-length` | `512` | Max sequence length for calibration |
| `--dataset` | `HuggingFaceH4/ultrachat_200k` | Calibration dataset |
| `--sglang-compat` | on | Strip fields incompatible with sglang |

---

## SpinQuant Guide

SpinQuant applies rotation transforms to reduce quantization loss. See [SpinQuant paper](https://arxiv.org/abs/2405.16406) and [QuaRot paper](https://arxiv.org/abs/2404.00456).

### Rotation Types

| Rotation | Type | Latency Cost | Description |
|----------|------|--------------|-------------|
| **R1** | Offline | None | Full hidden dimension rotation, fused into weights |
| **R2** | Offline | None | Head dimension rotation (attention), fused into weights |
| **R3** | Online | Yes | Runtime rotation on Q/K attention (requires vLLM hadacore) |
| **R4** | Online | Yes | Runtime rotation on MLP outputs |

### Best Quality (with latency penalty)

Use R1, R2, and R4 for maximum accuracy recovery:

```bash
python quantize-fp8-block.py -m your-model \
    --spinquant \
    --spinquant-rotations R1,R2,R4 \
    --spinquant-transform-type hadamard
```

R4 adds online rotation to MLP layers, which improves accuracy but requires hadacore kernel support in vLLM for efficient inference.

### Best Quality with NO Latency Penalty (Recommended)

Use only offline rotations (R1, R2) - these are fused into weights at save time:

```bash
python quantize-fp8-block.py -m your-model \
    --spinquant \
    --spinquant-rotations R1,R2 \
    --spinquant-transform-type hadamard
```

This is the **recommended configuration** for most use cases:
- No runtime overhead
- Significant accuracy improvement over no rotation
- Compatible with all vLLM backends

### Transform Type Selection

| Type | Performance | Size Requirement | Use Case |
|------|-------------|------------------|----------|
| `hadamard` | Fastest | Power of 2 | Default, use when possible |
| `random-hadamard` | Medium | More flexible | When hadamard size unavailable |
| `random-matrix` | Slowest | Any size | Fallback for odd dimensions |

### Block Size Considerations

- Default block size is **128**, which works for most models
- Block size must evenly divide both `hidden_size` and `head_dim`
- Smaller blocks = faster online rotation (if using R3/R4)
- Set `--spinquant-transform-block-size 0` to use model defaults

---

## W4A16 Quantization

For INT4 weight quantization (W4A16), use the llm-compressor examples directly or adapt the script:

### Quick W4A16 with SpinQuant

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SpinQuantModifier

MODEL_ID = "your-model-id"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# SpinQuant R1,R2 (offline, no latency) + W4A16
recipe = [
    SpinQuantModifier(
        rotations=["R1", "R2"],
        transform_block_size=128,
        transform_type="hadamard",
    ),
    QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
]

# Apply - uses datafree pipeline since SpinQuant doesn't need calibration
oneshot(model=model, recipe=recipe, pipeline="datafree")

# Save
SAVE_DIR = f"{MODEL_ID.split('/')[-1]}-spinquant-w4a16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

### W4A16 with GPTQ (Higher Accuracy)

GPTQ uses calibration data for better accuracy:

```python
from datasets import load_dataset
from llmcompressor.modifiers.quantization import GPTQModifier

# Load calibration data
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:512]")
ds = ds.shuffle(seed=42)

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
ds = ds.map(preprocess)

# GPTQ recipe - calibration-based
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
)
```

### W4A16 Calibration Best Practices

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Samples | 512+ | More samples = better accuracy, diminishing returns after 1024 |
| Seq Length | 2048 | Match your deployment context length |
| Dataset | Domain-matched | Use samples similar to deployment data |
| Chat Template | Yes | Apply model's chat template if instruction-tuned |

---

## Combining SpinQuant with GPTQ W4A16

For best W4A16 accuracy, combine SpinQuant transforms with GPTQ calibration:

```python
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier

recipe = [
    # SpinQuant first (offline rotations)
    SpinQuantModifier(
        rotations=["R1", "R2"],
        transform_block_size=128,
        transform_type="hadamard",
    ),
    # Then GPTQ quantization
    GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
]

oneshot(
    model=model,
    dataset=ds,  # calibration dataset
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
)
```

---

## Quality vs Latency Tradeoffs

| Configuration | Accuracy | Latency Impact | Recommended For |
|---------------|----------|----------------|-----------------|
| No rotation | Baseline | None | Quick testing |
| R1,R2 only | Good | **None** | Production (default) |
| R1,R2,R4 | Better | Small (hadacore) | When accuracy critical |
| R1,R2,R3,R4 | Best | Moderate | Research/benchmarking |

### vLLM Hadacore Support

For online rotations (R3, R4) to be efficient, vLLM must use hadacore kernels. Benchmark with:

```bash
python -m vllm.benchmarks.benchmark_latency --model path/to/model
```

Compare dense baseline vs quantized with transforms to verify no significant latency regression.

---

## References

- [SpinQuant Paper](https://arxiv.org/abs/2405.16406) - Learned rotation optimization
- [QuaRot Paper](https://arxiv.org/abs/2404.00456) - Outlier-free 4-bit inference
- [vLLM Office Hours #31](https://www.youtube.com/watch?v=WVenRmF4dPY) - LLM Compressor transform updates
- [llm-compressor examples](./llm-compressor/examples/) - Full example scripts
