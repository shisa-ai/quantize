# 2026-14b Quantization Scripts

Scripts for quantizing 14B-class models using [llm-compressor](https://github.com/vllm-project/llm-compressor).

See `SWEEP.md` for a step-by-step quantization + vLLM serving sweep focused on streaming performance.

## Environment Setup

Recommended (no shell activation required):

```bash
mamba run -n llmcompressor python -c "import llmcompressor; print(llmcompressor.__version__)"
# We test with: 0.9.0.1
```

Optional (interactive shells):

```bash
conda activate /home/lhl/.conda/envs/llmcompressor
```

---

## Speed-Focused Recommendations

These are the default schemes we target for **max speed** (latency/throughput) with vLLM:

| GPU / workload | Recommended | Notes |
|---|---|---|
| **Ampere (A100)**, high concurrency | **W8A8-INT8** | Best throughput/QPS; requires calibration (SmoothQuant + GPTQ). |
| **Ada / Hopper / Blackwell**, high concurrency | **W8A8-FP8_BLOCK** | Best throughput/QPS on FP8-capable GPUs; data-free by default (AWQ optional). |
| **Low concurrency / latency sensitive** | **W4A16 + SpinQuant (R1,R2 only)** | No runtime transforms (offline-only), good decode latency; GPTQ optional for accuracy. |

Notes:
- vLLM **does not support W8A8-INT8 on Blackwell-class GPUs** (see vLLM `docs/features/quantization/int8.md`). Use FP8 instead.
- “Best” is workload-dependent (context length, batch size, streaming decode vs prefill). Benchmark before standardizing.

## quantize-fp8-block.py

Production script for FP8_BLOCK quantization with optional SpinQuant and AWQ modifiers.

### Basic Usage

```bash
# Basic FP8_BLOCK quantization (no transforms, no calibration data)
mamba run -n llmcompressor python quantize-fp8-block.py -m shisa-ai/chotto-14b-20260107-dpo --datafree

# With SpinQuant (offline rotations R1,R2 - no latency penalty)
mamba run -n llmcompressor python quantize-fp8-block.py -m shisa-ai/chotto-14b-20260107-dpo --spinquant --datafree

# With SpinQuant + AWQ smoothing (requires calibration data)
mamba run -n llmcompressor python quantize-fp8-block.py -m shisa-ai/chotto-14b-20260107-dpo --spinquant --awq
```

### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --model` | required | HuggingFace model ID or local path |
| `-o, --output-dir` | auto | Output directory (auto-generated from model name) |
| `--datafree` | off | Skip calibration dataset and use the `datafree` pipeline |
| `--spinquant` | off | Enable SpinQuant transforms |
| `--spinquant-rotations` | `R1,R2` | Which rotations to apply (comma-separated) |
| `--spinquant-transform-type` | `hadamard` | Transform type: hadamard, random-hadamard, random-matrix |
| `--spinquant-transform-block-size` | `128` | Block size for rotation matrices |
| `--awq` | off | Use AWQ smoothing (slower, more accurate) |
| `--num-calibration-samples` | `256` | Number of calibration samples |
| `--max-seq-length` | `512` | Max sequence length for calibration |
| `--dataset` | `shisa-ai/shisa-v2.1-sharegpt` | Calibration dataset |
| `--sglang-compat` | on | Strip fields incompatible with sglang |

---

## SpinQuant Guide

SpinQuant applies rotation transforms to reduce quantization loss. See [SpinQuant paper](https://arxiv.org/abs/2405.16406) and [QuaRot paper](https://arxiv.org/abs/2404.00456).

### Rotation Types

| Rotation | Type | Latency Cost | Description |
|----------|------|--------------|-------------|
| **R1** | Offline | None | Full hidden dimension rotation, fused into weights |
| **R2** | Offline | None | Head dimension rotation (attention), fused into weights |
| **R3** | Online | Yes | Runtime rotation on attention Q/K (**not yet supported in vLLM**) |
| **R4** | Online | Yes | Runtime rotation on MLP activations (vLLM supports single-GPU; **no tensor-parallel**) |

### Best Quality (with latency penalty)

Use R1, R2, and R4 for maximum accuracy recovery:

```bash
mamba run -n llmcompressor python quantize-fp8-block.py -m your-model \
    --spinquant \
    --spinquant-rotations R1,R2,R4 \
    --spinquant-transform-type hadamard
```

R4 adds an online rotation on MLP activations. vLLM supports this on single-GPU (no tensor-parallel), and it is only efficient when hadacore kernels are available.

### Best Quality with NO Latency Penalty (Recommended)

Use only offline rotations (R1, R2) - these are fused into weights at save time:

```bash
mamba run -n llmcompressor python quantize-fp8-block.py -m your-model \
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
- Smaller blocks = faster online rotation (if using R4)
- Set `--spinquant-transform-block-size 0` to use model defaults

---

## W8A8-INT8 Quantization (Ampere High Concurrency)

For Ampere-class GPUs (e.g., A100) where FP8 kernels are not available, the best throughput/QPS at high concurrency is typically **W8A8-INT8** (SmoothQuant + GPTQ).

> vLLM does **not** support INT8 on Blackwell-class GPUs; use FP8 there.

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

MODEL_ID = "your-model-id"
DATASET_ID = "shisa-ai/shisa-v2.1-sharegpt"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

ds = load_dataset(DATASET_ID, split="train[:512]").shuffle(seed=42)

# If your dataset uses ShareGPT-style fields, map to role/content before applying the chat template.
messages_col = "messages" if "messages" in ds.column_names else "conversations"

def preprocess(example):
    msgs = example[messages_col]
    norm = []
    for m in msgs:
        role = m.get("role", m.get("from", "user"))
        content = m.get("content", m.get("value", ""))
        role = str(role).lower()
        if role in {"human", "user"}:
            role = "user"
        elif role in {"gpt", "assistant", "bot"}:
            role = "assistant"
        elif role != "system":
            role = "user"
        norm.append({"role": role, "content": str(content)})
    return {"text": tokenizer.apply_chat_template(norm, tokenize=False, add_generation_prompt=False)}

ds = ds.map(preprocess, remove_columns=ds.column_names)

recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
)

SAVE_DIR = f"{MODEL_ID.rstrip('/').split('/')[-1]}-w8a8-int8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

## W4A16 Quantization

For our production-style test checkpoint (W4A16 + SpinQuant R1,R2), use `quantize-w4a16.py`.

For INT4 weight quantization (W4A16), use the llm-compressor examples directly or adapt the script:

### Quick W4A16 with SpinQuant

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SpinQuantModifier

MODEL_ID = "your-model-id"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
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
ds = load_dataset("shisa-ai/shisa-v2.1-sharegpt", split="train[:512]")
ds = ds.shuffle(seed=42)

def preprocess(example):
    messages_col = "messages" if "messages" in example else "conversations"
    msgs = example[messages_col]
    norm = []
    for m in msgs:
        role = m.get("role", m.get("from", "user"))
        content = m.get("content", m.get("value", ""))
        role = str(role).lower()
        if role in {"human", "user"}:
            role = "user"
        elif role in {"gpt", "assistant", "bot"}:
            role = "assistant"
        elif role != "system":
            role = "user"
        norm.append({"role": role, "content": str(content)})
    return {
        "text": tokenizer.apply_chat_template(norm, tokenize=False, add_generation_prompt=False)
    }
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

## NVFP4 / NVFP4A16 Quantization (Blackwell)

For Blackwell-class GPUs, NVFP4 is NVIDIA’s microscaling FP4 format with block-wise scaling.

For our test checkpoints, use `quantize-nvfp4.py`:

```bash
# NVFP4A16 (FP4 weights only) - data-free
mamba run -n llmcompressor python quantize-nvfp4.py -m your-model --scheme NVFP4A16 --datafree

# NVFP4 (FP4 weights + activations) - requires calibration data
mamba run -n llmcompressor python quantize-nvfp4.py -m your-model --scheme NVFP4 --no-datafree \
  --dataset shisa-ai/shisa-v2.1-sharegpt --split train --num-calibration-samples 128 --max-seq-length 2048
```

## Serving Notes (vLLM, Streaming)

- **Blackwell (sm120)**: prefer **FP8_BLOCK** (or NVFP4/FP4 if you can use it). vLLM disables INT8 on Blackwell-class GPUs.
- **KV cache**: for long contexts and/or high concurrency, consider `kv_cache_dtype="fp8"` (and dataset-calibrated scales if accuracy matters).
- **Tail latency**: tune vLLM scheduling (`max_num_batched_tokens`, chunked prefill, prefix caching) and benchmark with `vllm bench latency`.

## Quality vs Latency Tradeoffs

| Configuration | Accuracy | Latency Impact | Recommended For |
|---------------|----------|----------------|-----------------|
| No rotation | Baseline | None | Quick testing |
| R1,R2 only | Good | **None** | Production (default) |
| R1,R2,R4 | Better | Small–Moderate | Single-GPU deployments when accuracy critical |
| R1,R2,R3,R4 | Best | N/A | R3 is not currently supported in vLLM |

### vLLM Hadacore Support

For online rotations (R4) to be efficient, vLLM must use hadacore kernels. Benchmark with:

```bash
vllm bench latency --model path/to/model --input-len 32 --output-len 128 --batch-size 1
```

Compare dense baseline vs quantized with transforms to verify no significant latency regression.

---

## References

- [SpinQuant Paper](https://arxiv.org/abs/2405.16406) - Learned rotation optimization
- [QuaRot Paper](https://arxiv.org/abs/2404.00456) - Outlier-free 4-bit inference
- [vLLM Office Hours #31](https://www.youtube.com/watch?v=WVenRmF4dPY) - LLM Compressor transform updates
- [llm-compressor examples](./llm-compressor/examples/) - Full example scripts
