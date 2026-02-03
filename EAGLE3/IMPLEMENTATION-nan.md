# NaN Investigation Notes (EAGLE3 / shisa-v2.1-70b)

This document captures the debugging work and conclusions around NaNs in EAGLE3
online training for `shisa-ai/shisa-v2.1-llama3.3-70b-FP8-dynamic`.

## Context
- Repo: `/home/lhl/quantize/EAGLE3`
- Target model: `shisa-ai/shisa-v2.1-llama3.3-70b-FP8-dynamic`
- Draft config: `shisa-v2.1-llama3.3-70b-eagle3.json`
- Dataset: `sft.shisa-v2.1.jsonl` (symlinked), plus sample `sft.shisa-v2.1.sample.jsonl`
- Hardware: RTX PRO 6000 Blackwell (sm120), 96GB, two GPUs
- Training runs used target model on `cuda:1`, draft on `cuda:0`

## Instrumentation added
These were added to make NaNs visible and to test mitigation:

- `--loss-backend {triton,torch}` to switch loss kernel.
- `--debug-nan` and `--nan-action {raise,skip,clamp}` with
  optional `--nan-clamp-value` and `--nan-mask-nonfinite`.
- Extra per-tensor nan stats (counts, min/max, shapes), and
  oob checks for `input_ids`.
- SGLang flags:
  - `--sglang-enable-fp32-lm-head`
  - `--sglang-triton-attention-reduce-in-fp32`
- Run script env knobs:
  `LOSS_BACKEND`, `MAX_NUM_STEPS`, `LOG_INTERVAL`, `DEBUG_NAN`,
  `NAN_ACTION`, `NAN_CLAMP_VALUE`, `NAN_MASK_NONFINITE`,
  `SGLANG_FP32_LM_HEAD`, `SGLANG_TRITON_ATTENTION_REDUCE_FP32`,
  `TRAIN_DATA_PATH`.

## Repro commands (representative)
Baseline with debug:
```
CUDA_VISIBLE_DEVICES=1,2 NUM_GPUS=1 TP_SIZE=1 TARGET_DEVICE=cuda:1 \
  SGLANG_ATTENTION_BACKEND=triton SGLANG_TORCH_COMPILE=0 SGLANG_PIECEWISE_CUDA_GRAPH=0 \
  LOSS_BACKEND=torch MAX_NUM_STEPS=10 LOG_INTERVAL=1 DEBUG_NAN=1 NAN_ACTION=raise \
  ./run-shisa-v2.1-70b.sh
```

Small dataset + shorter context:
```
CUDA_VISIBLE_DEVICES=1,2 NUM_GPUS=1 TP_SIZE=1 TARGET_DEVICE=cuda:1 \
  TRAIN_DATA_PATH=sft.shisa-v2.1.sample.jsonl \
  MAX_LENGTH=1024 SGLANG_CONTEXT_LENGTH=1024 \
  SGLANG_ATTENTION_BACKEND=triton SGLANG_TORCH_COMPILE=0 SGLANG_PIECEWISE_CUDA_GRAPH=0 \
  LOSS_BACKEND=torch MAX_NUM_STEPS=10 LOG_INTERVAL=1 DEBUG_NAN=1 NAN_ACTION=raise \
  ./run-shisa-v2.1-70b.sh
```

NaN clamping (keeps run alive but not necessarily valid):
```
CUDA_VISIBLE_DEVICES=1,2 NUM_GPUS=1 TP_SIZE=1 TARGET_DEVICE=cuda:1 \
  SGLANG_ATTENTION_BACKEND=triton SGLANG_TORCH_COMPILE=0 SGLANG_PIECEWISE_CUDA_GRAPH=0 \
  LOSS_BACKEND=torch MAX_NUM_STEPS=20 LOG_INTERVAL=1 DEBUG_NAN=1 \
  NAN_ACTION=clamp NAN_CLAMP_VALUE=1e4 \
  ./run-shisa-v2.1-70b.sh
```

## Findings
1) NaNs originate in target hidden states/logits (not the loss kernel).
   - Example (4096 context, step ~6):
     - `hidden_states` shape `(1, 2522, 24576)`, bf16,
       nan ~17,229,630, inf ~10, -inf ~12.
     - `target` logits shape `(1, 2522, 128256)`, fp32,
       nan ~262,144,000.
   - These appear before draft forward, so the loss kernel is not the cause.

2) Switching to torch loss does not fix NaNs.
   - `LOSS_BACKEND=torch` still shows NaNs in target outputs.

3) SGLang "fp32" knobs did not prevent NaNs.
   - `--sglang-enable-fp32-lm-head` and
     `--sglang-triton-attention-reduce-in-fp32` had no effect.

4) Context length matters.
   - At `MAX_LENGTH=1024` and `SGLANG_CONTEXT_LENGTH=1024`, no NaNs
     were observed in a 10-step run on a small sample dataset.
   - At `MAX_LENGTH=2048`, NaNs reappeared (hidden_states nan/inf).
   - At `MAX_LENGTH=4096`, NaNs reliably appeared within ~6 steps.

5) Clamping/masking can keep runs alive, but data quality is suspect.
   - `NAN_ACTION=clamp` keeps training from crashing, but large portions
     of hidden states/logits are invalid; accuracy becomes meaningless.
   - `NAN_MASK_NONFINITE=1` can zero out loss when non-finite dominates.

6) Target backend changes are not sufficient by themselves.
   - User reports NaNs with `--target-model-backend hf` as well.
   - `SGLANG_ATTENTION_BACKEND=triton` is faster, but still unstable.

## Conclusion
The NaNs are caused by the target model producing invalid hidden states and
logits at longer context lengths. This appears to be a numerical stability
issue tied to the FP8 dynamic target (and possibly to long contexts on sm120).
Loss kernel or LR changes do not address the root cause.

## Practical recommendations
- Keep context length at or below 1024 (or determine the highest stable
  length by binary search).
- If 4k context is required, use a higher-precision target (bf16/fp16),
  likely with TP across GPUs to fit.
- Treat clamping/masking as a temporary workaround, not a real fix.

## Files modified during investigation
- `SpecForge/scripts/train_eagle3.py`
- `SpecForge/specforge/core/eagle3.py`
- `SpecForge/specforge/args.py`
- `run-shisa-v2.1-70b.sh`
- `train_eagle3_online.py`
- `sft.shisa-v2.1.sample.jsonl`
