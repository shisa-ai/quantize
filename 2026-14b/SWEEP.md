# Quantization + vLLM Sweep (Streaming / Max Speed)

This doc is the step-by-step “how we produce test checkpoints” + “what vLLM knobs we sweep” for our streaming app.

## Goals / Test Checkpoints

We standardize on these three quantized checkpoints for benchmarking:

1) **W8A8-FP8_BLOCK** (high concurrency / max throughput)  
2) **W4A16 + SpinQuant (R1,R2)** (low concurrency / low-latency decode; offline-only transforms)  
3) **NVFP4A16** (Blackwell-weight-only FP4 variant; compare vs W4A16 for latency/memory)

We also optionally test **EAGLE3 speculative decoding** on/off using our validated configs.

---

## Environment

We recommend running everything via `mamba run` to avoid shell state issues:

```bash
mamba run -n llmcompressor python -c "import llmcompressor; print(llmcompressor.__version__)"
```

---

## Step 0: Choose Base Model

Pick either a Hugging Face model id or local directory:

```bash
export BASE_MODEL="shisa-ai/chotto-14b-20260107-dpo"
# or: export BASE_MODEL=/path/to/local/model
```

---

## Step 1: Quantize (Generate Test Checkpoints)

### A) W8A8-FP8_BLOCK (data-free by default)

```bash
mamba run -n llmcompressor python quantize-fp8-block.py \
  -m "$BASE_MODEL" \
  --datafree
```

Optional (higher accuracy, slower quantization): add AWQ (requires calibration data):

```bash
mamba run -n llmcompressor python quantize-fp8-block.py \
  -m "$BASE_MODEL" \
  --awq \
  --dataset shisa-ai/shisa-v2.1-sharegpt \
  --split train \
  --num-calibration-samples 256 \
  --max-seq-length 512
```

### B) W4A16 + SpinQuant (R1,R2) (offline transforms, no runtime overhead)

Fast RTN-style W4A16 (data-free):

```bash
mamba run -n llmcompressor python quantize-w4a16.py \
  -m "$BASE_MODEL" \
  --spinquant \
  --spinquant-rotations R1,R2 \
  --datafree
```

Optional (higher accuracy, slower quantization): GPTQ W4A16 (requires calibration data):

```bash
mamba run -n llmcompressor python quantize-w4a16.py \
  -m "$BASE_MODEL" \
  --spinquant \
  --spinquant-rotations R1,R2 \
  --gptq \
  --dataset shisa-ai/shisa-v2.1-sharegpt \
  --split train \
  --num-calibration-samples 512 \
  --max-seq-length 2048
```

### C) NVFP4A16 (FP4 weights only; best first FP4 variant to try)

```bash
mamba run -n llmcompressor python quantize-nvfp4.py \
  -m "$BASE_MODEL" \
  --scheme NVFP4A16 \
  --datafree
```

Optional (more aggressive): NVFP4 (weights+activations; requires calibration)

```bash
mamba run -n llmcompressor python quantize-nvfp4.py \
  -m "$BASE_MODEL" \
  --scheme NVFP4 \
  --dataset shisa-ai/shisa-v2.1-sharegpt \
  --split train \
  --num-calibration-samples 128 \
  --max-seq-length 2048
```

---

## Step 2: vLLM Serving Knobs to Sweep

This is a practical “variants list” to combine with each test checkpoint.

### Core server flags (streaming-focused)

Recommended baseline starting point:

```bash
vllm serve /path/to/quantized-checkpoint \
  --stream-interval 1 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 1024
```

Notes:
- `--stream-interval 1` minimizes buffering (best UX); larger values can reduce CPU overhead.
- For *low concurrency* latency tests, sweep `--max-num-seqs 1/2/4` and keep `--max-num-batched-tokens` modest.
- For *higher concurrency* throughput tests, increase `--max-num-seqs` and `--max-num-batched-tokens`.

### Attention backend (important on Blackwell)

Sweep these (especially on Blackwell):

- `--attention-backend FLASHINFER`
  - Best starting point on SM120 Blackwell (RTX PRO 6000) and generally good on modern GPUs.
  - vLLM’s **TRTLLM attention** path is currently **SM100-only** (compute capability 10.x) and requires NVIDIA artifactory access for cubins.
    - On **SM120**, `-ac.use_trtllm_attention=...` is effectively a no-op (vLLM will fall back to native FlashInfer attention).
    - On **SM100**, you can force on/off:
      - `-ac.use_trtllm_attention=1` (force on)
      - `-ac.use_trtllm_attention=0` (force off)
- `--attention-backend TRITON_ATTN`
- (for A100/H100 baselines) `--attention-backend FLASH_ATTN`
  - Note: FlashAttention **does not** support `--kv-cache-dtype fp8` on non-Hopper, and vLLM defaults to FA2 on Blackwell.

Practical variant set:

```bash
--attention-backend FLASHINFER
--attention-backend TRITON_ATTN
--attention-backend FLASH_ATTN
```

If you use `--kv-cache-dtype fp8`, also consider sweeping whether FlashInfer quantizes Q to FP8:

```bash
-ac.disable_flashinfer_q_quantization=0   # default (Q is quantized when using fp8 KV where supported)
-ac.disable_flashinfer_q_quantization=1   # keep Q in model dtype (can help quality; may cost perf)
```

If you want to **disable FlashInfer prefill** (force an alternate prefill path), sweep:

```bash
-ac.disable_flashinfer_prefill=0  # default
-ac.disable_flashinfer_prefill=1
```

FlashAttention notes:
- vLLM defaults to FlashAttention v2 on Blackwell (FA3 is Hopper-focused).

### FP8 KV cache (memory + speed lever)

Sweep:

- KV dtype:
  - `--kv-cache-dtype auto` (baseline)
  - `--kv-cache-dtype fp8` (quantized KV)
  - (optional) `--kv-cache-dtype fp8_e4m3` / `fp8_e5m2` for stability experiments
- Scale calibration:
  - `--calculate-kv-scales` (on-the-fly warmup calibration)
  - omit it (use default scales)

Example:

```bash
vllm serve /path/to/model \
  --kv-cache-dtype fp8 \
  --calculate-kv-scales
```

Important behavior:
- If a checkpoint includes an llm-compressor `kv_cache_scheme`, vLLM forces `kv_cache_dtype="fp8"` and loads the saved scales (it also disables `--calculate-kv-scales` for that model).

#### Dataset-calibrated FP8 KV-cache scales (llm-compressor) = separate checkpoint

If you want **higher-quality FP8 KV-cache** than random-token calibration, llm-compressor can write **pre-calibrated scales into the checkpoint** (so vLLM loads them automatically and you can omit `--calculate-kv-scales`).

This produces a **new output directory** (a separate model checkpoint), not a sidecar file.

Reference implementations:
- vLLM docs: `vllm/docs/features/quantization/quantized_kvcache.md`
- llm-compressor examples: `llm-compressor/examples/quantization_kv_cache/`

For SM120 Blackwell + FlashInfer, use **per-tensor** (`strategy: tensor`) scales.
Per-attention-head (`strategy: attn_head`) scales are a Hopper/FlashAttention (FA3) feature.

### CUDA graphs / compilation

Decode-heavy streaming sometimes benefits from decode-only capture:

```bash
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

Sweep:
- default (v1 default is full+piecewise)
- `FULL_DECODE_ONLY`
- `NONE` (debug / if you hit memory or weird regressions)

### Prefix caching (production knob)

Sweep on/off:
- `--enable-prefix-caching` (helps repeated/system prompts, chat templates, RAG prefixes)

Note: some backend+mode combinations can disable prefix caching internally if incompatible.

### Chunked prefill (TTFT / fairness knob)

Chunked prefill can reduce **time-to-first-token** under load by interleaving long prefills.

Sweep:

```bash
--enable-chunked-prefill
--no-enable-chunked-prefill
```

Also consider (especially if you have long prompts):
- `--max-num-partial-prefills`
- `--max-long-partial-prefills`
- `--long-prefill-token-threshold`

### NVFP4 (Blackwell) runtime knobs (optional)

For NVFP4 / NVFP4A16 checkpoints, vLLM exposes an FP4 GEMM backend selector:

```bash
VLLM_NVFP4_GEMM_BACKEND=flashinfer-trtllm
VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass
VLLM_NVFP4_GEMM_BACKEND=flashinfer-cudnn
VLLM_NVFP4_GEMM_BACKEND=cutlass
VLLM_NVFP4_GEMM_BACKEND=marlin
```

If unset, vLLM auto-selects an available backend. Treat this as an extra sweep dimension for NVFP4 models.

---

## Step 3: EAGLE3 Speculative Decoding (On/Off)

We treat this as an additional boolean dimension in our sweeps.

On the vLLM server, pass `--speculative-config` as JSON.

Example (EAGLE3):

```bash
vllm serve /path/to/model \
  --speculative-config '{
    "model": "path/or/hf_id/of/eagle3-draft-model",
    "draft_tensor_parallel_size": 1,
    "num_speculative_tokens": 2,
    "method": "eagle3"
  }'
```

If you keep a validated config in a file:

```bash
vllm serve /path/to/model \
  --speculative-config "$(cat /path/to/eagle3_config.json)"
```

Notes:
- EAGLE/EAGLE3 draft models generally require `draft_tensor_parallel_size=1`.
- You can still use tensor parallelism for the main model.

---

## Suggested Minimal Sweep Matrix (What We Actually Combine)

Start small and expand:

**Models**
- FP8_BLOCK
- W4A16 + SpinQuant (R1,R2)
- NVFP4A16

**Per-model vLLM variants**
- Attention backend: `FLASHINFER` (SM100 only: also sweep `-ac.use_trtllm_attention=0/1`), `TRITON_ATTN`, (optional) `FLASH_ATTN` for A100/H100 baselines
- KV cache: `auto`, `fp8` (+ `--calculate-kv-scales` on/off)
- CUDAGraph: default, `FULL_DECODE_ONLY`
- Prefix caching: off, on
- Chunked prefill: off, on
- EAGLE3: off, on (validated configs)
- NVFP4 GEMM backend (NVFP4/NVFP4A16 only): auto, plus a few fixed `VLLM_NVFP4_GEMM_BACKEND=...` values
