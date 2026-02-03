# EAGLE3 offline runbook (shisa-v2.1-llama3.3-70b-FP8-dynamic)

This document captures a repeatable offline workflow for hidden-state generation and
training. It is tuned for a 2xH100 instance and keeps commands explicit so you can
estimate cost and disk before committing a full run.

Assumptions:
- Dataset: sft.shisa-v2.1.jsonl (symlink ok)
- Chat template: llama3
- Target model: shisa-ai/shisa-v2.1-llama3.3-70b-FP8-dynamic
- Draft config: shisa-v2.1-llama3.3-70b-eagle3.json
- Cache dir: /home/lhl/quantize/EAGLE3/cache

Important note on FP8 stability:
- We observed non-finite hidden states with max_length >= 2048 on this FP8 target.
  If you see NaNs in offline generation, start at max_length=1024 or switch to a
  BF16/FP16 target model for longer contexts.

## Disk usage estimate (quick math + measured)

Offline files store per-sample tensors:
- aux_hidden_state: 3 layers concatenated (seq_len x 3*hidden_size)
- hidden_state (last layer): seq_len x hidden_size
- input_ids and loss_mask (small by comparison)

Approx per-token bytes:
per_token_bytes = (aux_layers + 1) * hidden_size * dtype_bytes
For Llama3.3 70B: aux_layers=3, hidden_size=8192, dtype_bytes=2 (bf16/fp16)
per_token_bytes = 65536 bytes (~64 KB)

Approx per-sample size (worst case, if seq_len hits max_length):
- max_length 1024 -> ~64 MB per sample
- max_length 2048 -> ~128 MB per sample
- max_length 4096 -> ~256 MB per sample

Total disk ~= per_sample_size * num_samples (plus ~5-10% file overhead).

### Measure with a small sample (recommended)

1) Count tokens (optional, to choose a sane max_length):
```bash
mamba run -n eagle python count_tokens.py \
  --data sft.shisa-v2.1.jsonl \
  --tokenizer shisa-ai/shisa-v2.1-llama3.3-70b-FP8-dynamic \
  --chat-template llama3 \
  --max-length 1024 \
  --num-proc 8
```

2) Generate a small sample and measure actual disk:
```bash
CUDA_VISIBLE_DEVICES=0,1 mamba run -n eagle torchrun --standalone --nproc_per_node 2 \
  SpecForge/scripts/prepare_hidden_states.py \
  --target-model-path shisa-ai/shisa-v2.1-llama3.3-70b-FP8-dynamic \
  --enable-aux-hidden-states \
  --data-path sft.shisa-v2.1.jsonl \
  --output-path cache/hidden_states/shisa-v2.1-70b_test \
  --chat-template llama3 \
  --max-length 1024 \
  --tp-size 2 \
  --batch-size 1 \
  --num-samples 200 \
  --cache-dir cache

du -sb cache/hidden_states/shisa-v2.1-70b_test
```

Estimate full size:
```bash
SAMPLES=200
TOTAL_SAMPLES=$(wc -l < sft.shisa-v2.1.jsonl)
BYTES=$(du -sb cache/hidden_states/shisa-v2.1-70b_test | awk '{print $1}')
PER_SAMPLE=$((BYTES / SAMPLES))
EST_TOTAL_GIB=$((PER_SAMPLE * TOTAL_SAMPLES / 1024 / 1024 / 1024))
echo "Estimated total: ${EST_TOTAL_GIB} GiB"
```

## Step 1: Generate hidden states (2xH100, TP=2)

This uses both GPUs to fit the 70B target (tensor parallel across 2 GPUs).
Start with max_length=1024 to avoid FP8 NaNs.

```bash
CUDA_VISIBLE_DEVICES=0,1 mamba run -n eagle torchrun --standalone --nproc_per_node 2 \
  SpecForge/scripts/prepare_hidden_states.py \
  --target-model-path shisa-ai/shisa-v2.1-llama3.3-70b-FP8-dynamic \
  --enable-aux-hidden-states \
  --data-path sft.shisa-v2.1.jsonl \
  --output-path cache/hidden_states/shisa-v2.1-70b_fp8_len1024 \
  --chat-template llama3 \
  --max-length 1024 \
  --tp-size 2 \
  --batch-size 1 \
  --build-dataset-num-proc 8 \
  --num-workers 4 \
  --num-io-threads 8 \
  --file-group-size 2000 \
  --cache-dir cache \
  --sglang-attention-backend triton \
  --sglang-mem-fraction-static 0.9 \
  --sglang-context-length 1024
```

Notes:
- Increase --batch-size to 2 if there is memory headroom; reduce if you OOM.
- The generator skips existing .ckpt files, so reruns resume automatically.
- If you see "NaN found... Skipping save", reduce max_length or use a BF16 target.

## Step 2: Train offline (draft-only + lm_head)

Offline training does not load the full target model; it loads the lm_head only.
Memory use is much lower than online training.

### 2a) Train on 1 GPU
```bash
CUDA_VISIBLE_DEVICES=0 mamba run -n eagle torchrun --standalone --nproc_per_node 1 \
  SpecForge/scripts/train_eagle3.py \
  --target-model-path shisa-ai/shisa-v2.1-llama3.3-70b-FP8-dynamic \
  --draft-model-config shisa-v2.1-llama3.3-70b-eagle3.json \
  --train-data-path sft.shisa-v2.1.jsonl \
  --train-hidden-states-path cache/hidden_states/shisa-v2.1-70b_fp8_len1024 \
  --output-dir outputs/shisa-v2.1-70b-eagle3-offline \
  --num-epochs 3 \
  --batch-size 2 \
  --learning-rate 1e-4 \
  --max-length 1024 \
  --chat-template llama3 \
  --cache-dir cache \
  --attention-backend sdpa \
  --dataloader-num-workers 8 \
  --report-to wandb \
  --wandb-project EAGLE3 \
  --wandb-name shisa-v2.1-70b-eagle3-offline
```

### 2b) Train on 2 GPUs (data parallel)
```bash
CUDA_VISIBLE_DEVICES=0,1 mamba run -n eagle torchrun --standalone --nproc_per_node 2 \
  SpecForge/scripts/train_eagle3.py \
  --target-model-path shisa-ai/shisa-v2.1-llama3.3-70b-FP8-dynamic \
  --draft-model-config shisa-v2.1-llama3.3-70b-eagle3.json \
  --train-data-path sft.shisa-v2.1.jsonl \
  --train-hidden-states-path cache/hidden_states/shisa-v2.1-70b_fp8_len1024 \
  --output-dir outputs/shisa-v2.1-70b-eagle3-offline \
  --num-epochs 3 \
  --batch-size 2 \
  --learning-rate 1e-4 \
  --max-length 1024 \
  --chat-template llama3 \
  --cache-dir cache \
  --attention-backend sdpa \
  --dataloader-num-workers 8 \
  --report-to wandb \
  --wandb-project EAGLE3 \
  --wandb-name shisa-v2.1-70b-eagle3-offline
```

Batch sizing notes:
- batch-size is per GPU. Effective global batch = batch-size * num_gpus.
- If you want to keep the same global batch when moving from 1 GPU to 2 GPUs,
  halve batch-size.
- If you want to push throughput, raise batch-size until you hit GPU memory.

## Scaling and cost-efficiency tips

- Start with --num-samples to measure throughput and disk, then extrapolate.
- Use local NVMe for cache/hidden states; many small .ckpt files stress network FS.
- For generation, TP is usually required for 70B; DP only helps if you have >2 GPUs.
- For training, TP is not needed; DP is the simplest way to scale.
- Keep max_length in training equal to max_length used for hidden states.
- If your data is preformatted, add --is-preformatted to both generation and training.
