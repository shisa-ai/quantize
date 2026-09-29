# Shisa ASR Phi4MM FP8 Quantization

Reproducible decoder-only FP8 export for the Shisa ASR Phi4MM checkpoints.

Current target: [`shisa-ai/shisa-asr-v0.97`](https://huggingface.co/shisa-ai/shisa-asr-v0.97).
Published lineage: [`shisa-ai/shisa-asr-v0.95b-FP8`](https://huggingface.co/shisa-ai/shisa-asr-v0.95b-FP8).

## Pinned sources

```text
# current target, not yet published
model: shisa-ai/shisa-asr-v0.97
revision: 7751a34a5d4f0868ce660320c830ddbbec9044cc

# published lineage
model: shisa-ai/shisa-asr-v0.95b
revision: faa3d244fe9490f3fa5d05acacafaa1f668e180d
published derivative repo: shisa-ai/shisa-asr-v0.95b-FP8
published HF revision: 52607d9751eca99c68d21e9995a3606e02fa2f36
visibility: private
```

v0.96 and v0.97 have no published FP8 derivative. The v0.95b, v0.93b, v0.9b,
and v0.1b checkpoints do.

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
backoff:             2.2.1
compressed-tensors:  0.14.0.1
huggingface-hub:      0.36.2
llmcompressor:        0.10.0.3
peft:                0.17.1
safetensors:          0.8.0
scipy:               1.18.1
torch:                2.10.0+cu128
torchvision:          0.25.0+cu128
transformers:         4.57.6
```

Transformers 4.57.6 is intentional. Transformers 5.x meta-device construction
is incompatible with the checkpoint's Phi4MM Conformer constructor in this
export path.

`peft`, `scipy`, `torchvision`, and `backoff` are required even though the
export is data-free and CPU-only: `get_class_from_dynamic_module` refuses to
load `modeling_phi4mm.py` without `peft`, and the model's remote code imports
`scipy` (`processing_phi4mm.py`), `torchvision` (`processing_phi4mm.py`), and
`backoff` (`speech_conformer_encoder.py`).

No GPU is required for the export. The script constructs the model on CPU with
`low_cpu_mem_usage=False` and patches `llmcompressor.pipelines.data_free.pipeline.dispatch_model`
to an identity function, so the model is never dispatched to CUDA. A GPU is
only relevant to downstream vLLM kernel selection.

## Inspect before export

The default command is header-only and does not load tensor data:

```bash
python shisa-asr/quantize_decoder_fp8.py \
  --model-path /path/to/shisa-asr-v0.97 \
  --model-id shisa-ai/shisa-asr-v0.97 \
  --model-revision 7751a34a5d4f0868ce660320c830ddbbec9044cc \
  --scheme FP8_DYNAMIC \
  --report-json reports/shisa-asr-v0.97-fp8-dynamic-dry-run.json
```

Continue only when the report says `safety_status=PASS`, `target_count=128`,
and `protected_target_matches=[]`.

`--model-id` and `--model-revision` affect report provenance only; they default
to the v0.95b values.

## Export FP8_DYNAMIC

```bash
python shisa-asr/quantize_decoder_fp8.py \
  --run \
  --scheme FP8_DYNAMIC \
  --model-path /path/to/shisa-asr-v0.97 \
  --model-id shisa-ai/shisa-asr-v0.97 \
  --model-revision 7751a34a5d4f0868ce660320c830ddbbec9044cc \
  --output-dir /path/to/shisa-asr-v0.97-fp8-dynamic \
  --report-json reports/shisa-asr-v0.97-fp8-dynamic-run.json
```

Use `--force` only when intentionally replacing an existing output directory.
The script also accepts `--scheme FP8_BLOCK`; FP8_DYNAMIC is the selected
RTX 4090/SM89 release candidate because the tested vLLM build lacked tuned
block configs for this model's matrix shapes.

## Verify an export

`validate_fp8_artifact.py` recomputes the integrity block from a source
checkpoint and its export, and can merge the result into the export's
`quantization_summary.json`. It compares dtype, shape, and values for every
non-target tensor.

```bash
python shisa-asr/validate_fp8_artifact.py \
  --source-path /path/to/shisa-asr-v0.97 \
  --artifact-path /path/to/shisa-asr-v0.97-fp8-dynamic \
  --summary-json /path/to/shisa-asr-v0.97-fp8-dynamic/quantization_summary.json \
  --output-repo-id shisa-ai/shisa-asr-v0.97-FP8 \
  --private \
  --report-json reports/shisa-asr-v0.97-fp8-dynamic-integrity.json
```

It exits non-zero if any non-target tensor differs. Use `--skip-value-check`
to record dtype and shape accounting without reading tensor values.

The committed `quantize_decoder_fp8.py` does not emit `tool_versions`,
`script`, `generated_at_utc`, or `artifact_integrity`; the published v0.95b-FP8
summary carries those fields, so this script is the way to reproduce them.

## Validated artifact shape

v0.97 (`shisa-asr-v0.97-fp8-dynamic`, exported 2026-09-29) and the published
v0.95b artifact are identical on every measured structural field:

| Field | v0.97 FP8_DYNAMIC | v0.95b-FP8 |
|---|---:|---:|
| safetensor shards | 2 | 2 |
| tensors per shard | 1672 / 503 | 1672 / 503 |
| indexed tensors | 2175 | 2175 |
| indexed parameters | 5575240512 | 5575240512 |
| indexed weight bytes | 7929255872 | 7929255872 |
| `F8_E4M3` target weights | 128 | 128 |
| BF16 scale tensors | 128 | 128 |
| BF16 tensors | 2047 | 2047 |
| non-target tensors checked | 1919 | 1919 |
| non-target mismatches | 0 | 0 |
| protected scale tensors | 0 | 0 |

Additional checks on the v0.97 export:

- `config.json` parses equal to the v0.95b-FP8 `config.json` (zero differing
  keys). The only non-quant differences from the v0.97 source are the
  `torch_dtype` to `dtype` key rename and the `transformers_version` string.
- vLLM 0.27.1 parses it as `compressed-tensors` / `float-quantized` with the
  same target regexes, the same schemes (8-bit float weights, per-channel; 8-bit
  float activations, per-token dynamic), and the same 876 ignore entries as the
  v0.95b-FP8 artifact.
- The published v0.95b artifact was additionally reported on vLLM 0.26 with
  `quantization=compressed-tensors` and the
  `CutlassFP8ScaledMMLinearKernel` kernel for `CompressedTensorsW8A8Fp8`. That
  runtime kernel check has not been reproduced for v0.97.

Local artifact path used for the v0.97 export:
`/home/morpheus/data/models/hf/shisa-ai/shisa-asr-v0.97-fp8-dynamic` (7.5 GiB).
The authoritative run record is the `quantization_summary.json` inside that
directory; `reports/shisa-asr-v0.97-fp8-dynamic-summary.json` is a copy of it.

## Private publication

The v0.95b artifact was published privately at:

```text
repo: shisa-ai/shisa-asr-v0.95b-FP8
revision: 52607d9751eca99c68d21e9995a3606e02fa2f36
files: 25
uploaded bytes: 7,959,087,632
```

The uploaded card, compressed-tensors config, provenance summary, file sizes,
and weight-shard LFS hashes were verified against the local artifact. Backend
validation should pin the exact revision above rather than `main`.

The v0.97 FP8 export has not been uploaded. The expected repo name following
this convention is `shisa-ai/shisa-asr-v0.97-FP8`.

## Publication TODO

- [ ] Commit the export before publishing so `script.git_revision` in the
  summary names a revision that contains the script that produced the artifact.
  The v0.97 summary currently records `git_dirty: true`.
- [ ] Run matched BF16 versus FP8_DYNAMIC CHIME6 and JIA evaluations before
  production qualification. The v0.95b artifact was published without them, so
  both carry the same gap.
- [ ] Reproduce the vLLM runtime kernel-selection check for v0.97 on the
  deployment GPU.
- [x] Replace the copied base-model README with a quant-specific model card
  linking this script at exact Git commit `480a16e` (v0.95b).
- [x] Disclose the Earnings22 status inconsistency: the base card calls the
  benchmark finalized while its manifest/evaluator still identifies the
  model-assisted targets as pending final human review (v0.95b).
- [x] Upload privately and record the resulting Hugging Face commit SHA
  (v0.95b).
