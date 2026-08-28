# Shisa ASR Phi4MM FP8 Quantization

Reproducible decoder-only FP8 export for
[`shisa-ai/shisa-asr-v0.95b`](https://huggingface.co/shisa-ai/shisa-asr-v0.95b).
This restores the previously documented `shisa-asr/quantize_decoder_fp8.py`
location referenced by the published `shisa-asr-v0.93b-FP8` model card.

## Pinned source

```text
model: shisa-ai/shisa-asr-v0.95b
revision: faa3d244fe9490f3fa5d05acacafaa1f668e180d
published derivative repo: shisa-ai/shisa-asr-v0.95b-FP8
published HF revision: 52607d9751eca99c68d21e9995a3606e02fa2f36
visibility: private
```

## Scope

The default `FP8_DYNAMIC` recipe quantizes exactly 128 decoder base projections:

- `model.layers.*.self_attn.qkv_proj.base_layer`
- `model.layers.*.self_attn.o_proj.base_layer`
- `model.layers.*.mlp.gate_up_proj.base_layer`
- `model.layers.*.mlp.down_proj.base_layer`

Audio/image modules, embeddings, the LM head, norms, and speech/vision LoRA
weights remain BF16. The exporter restores source BF16 before serialization to
prevent PEFT from promoting ignored LoRA tensors to FP32. It then rewrites
compressed-tensors targets from the internal `.base_layer` names to the parent
module names expected by vLLM.

## Tested environment

```text
Python:              3.12
accelerate:          1.12.0
compressed-tensors:  0.14.0.1
huggingface-hub:      0.36.2
llmcompressor:        0.10.0.3
safetensors:          0.8.0
torch:                2.10.0+cu128
transformers:         4.57.6
```

Transformers 4.57.6 is intentional. Transformers 5.x meta-device construction
is incompatible with the checkpoint's Phi4MM Conformer constructor in this
export path.

## Inspect before export

The default command is header-only and does not load tensor data:

```bash
python shisa-asr/quantize_decoder_fp8.py \
  --model-path /path/to/shisa-asr-v0.95b \
  --scheme FP8_DYNAMIC \
  --report-json reports/shisa-asr-v0.95b-fp8-dynamic-dry-run.json
```

Continue only when the report says `safety_status=PASS`, `target_count=128`,
and `protected_target_matches=[]`.

## Export FP8_DYNAMIC

```bash
python shisa-asr/quantize_decoder_fp8.py \
  --run \
  --scheme FP8_DYNAMIC \
  --model-path /path/to/shisa-asr-v0.95b \
  --output-dir /path/to/shisa-asr-v0.95b-fp8-dynamic \
  --report-json reports/shisa-asr-v0.95b-fp8-dynamic-dry-run.json
```

Use `--force` only when intentionally replacing an existing output directory.
The script also accepts `--scheme FP8_BLOCK`; FP8_DYNAMIC is the selected RTX
4090/SM89 release candidate because the tested vLLM build lacked tuned block
configs for this model's matrix shapes.

## Validated artifact shape

The canonical FP8_DYNAMIC artifact produced on 2026-08-28 has:

- 2 safetensor shards
- 2,175 indexed tensors
- 128 `F8_E4M3` decoder target weights
- 128 BF16 scale tensors
- 1,919 original non-target tensors with source-identical dtype, shape, and values
- indexed weight size: 7,929,255,872 bytes (7.385 GiB)
- vLLM 0.26 detection: `quantization=compressed-tensors`
- selected kernel: `CutlassFP8ScaledMMLinearKernel` for `CompressedTensorsW8A8Fp8`

## Private publication

The canonical FP8_DYNAMIC artifact was published privately at:

```text
repo: shisa-ai/shisa-asr-v0.95b-FP8
revision: 52607d9751eca99c68d21e9995a3606e02fa2f36
files: 25
uploaded bytes: 7,959,087,632
```

The uploaded card, compressed-tensors config, provenance summary, file sizes,
and weight-shard LFS hashes were verified against the local artifact. Backend
validation should pin the exact revision above rather than `main`.

## Publication TODO

- [ ] Before production qualification or broader publication, run matched BF16
  versus FP8_DYNAMIC CHIME6 and JIA evaluations to provide strict evaluation
  parity with the `shisa-asr-v0.93b-FP8` model card.
- [x] Replace the copied base-model README with a quant-specific model card
  linking this script at exact Git commit `480a16e`.
- [x] Disclose the Earnings22 status inconsistency: the base card calls the
  benchmark finalized while its manifest/evaluator still identifies the
  model-assisted targets as pending final human review.
- [x] Upload privately and record the resulting Hugging Face commit SHA.
