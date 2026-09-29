# Shisa ASR Phi4MM FP8 Quantization

Reproducible decoder-only FP8 export for the Shisa ASR Phi4MM checkpoints.

Current target: [`shisa-ai/shisa-asr-v0.97`](https://huggingface.co/shisa-ai/shisa-asr-v0.97).
Published lineage: [`shisa-ai/shisa-asr-v0.95b-FP8`](https://huggingface.co/shisa-ai/shisa-asr-v0.95b-FP8).

## Pinned sources

```text
# current target
model: shisa-ai/shisa-asr-v0.97
revision: 7751a34a5d4f0868ce660320c830ddbbec9044cc
published derivative repo: shisa-ai/shisa-asr-v0.97-FP8
published HF revision: 2fe198f37f84b1324886e91f622bee23ce8cfd44
visibility: private

# published lineage
model: shisa-ai/shisa-asr-v0.95b
revision: faa3d244fe9490f3fa5d05acacafaa1f668e180d
published derivative repo: shisa-ai/shisa-asr-v0.95b-FP8
published HF revision: 52607d9751eca99c68d21e9995a3606e02fa2f36
visibility: private
```

v0.96 has no published FP8 derivative. The v0.97, v0.95b, v0.93b, v0.9b, and
v0.1b checkpoints do.

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
summary carries those fields, so this script is the way to reproduce them. It
writes `script` for the exporter, following the published convention, and
`validator` for itself. Both record `git_dirty` so an uncommitted tree cannot
be mistaken for a reproducible revision.

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
  `CutlassFP8ScaledMMLinearKernel` kernel for `CompressedTensorsW8A8Fp8`. The
  v0.97 deployment reproduced this exactly on vLLM `0.26.0` and torch
  `2.11.0+cu130` (CUDA 13.0).

Local artifact path used for the v0.97 export:
`/home/morpheus/data/models/hf/shisa-ai/shisa-asr-v0.97-fp8-dynamic` (7.5 GiB).
The authoritative run record is the `quantization_summary.json` inside that
directory; `reports/shisa-asr-v0.97-fp8-dynamic-summary.json` is a copy of it.

## Matched runtime and accuracy evaluation

Measured 2026-09-29 on `aomori-gpu2`, comparing the new dev v0.97 backend
against the production v0.95b backend. The authoritative record is
`reports/shisa-asr-v0.97-fp8-dynamic-evals.json`, which carries the deployment
identities, the harness revision, and a SHA-256 for every `scores.json` and
`false_positives.csv` it summarizes.

| | v0.97-FP8 | v0.95b-FP8 baseline |
|---|---|---|
| deployment | `shisa-asr-v0.97-gpu5` | `shisa-asr-v0.95b-gpu4` |
| role | new dev ASR backend | current production ASR backend |
| artifact revision | `2fe198f37f84b1...` | `52607d9751eca9...` |
| GPU / port | GPU5 / 8004 | GPU4 / 8001 |
| process started | 2026-09-29 06:56 | 2026-09-08 11:24 |

Both deployments use the same `rtx5090-8x` profile launcher with identical
flags (`--gpu-memory-utilization 0.82`, `--max-num-seqs 8`, `--max-model-len
12800`, `--limit-mm-per-prompt '{"audio": 1}'`, `--enforce-eager`, TP1), the same
host, the same RTX 5090 model, and the same container runtime. The only
differences are the physical GPU index and the process age.

Startup memory accounting is identical:

| Quantity | v0.97-FP8 | v0.95b-FP8 |
|---|---:|---:|
| weight memory | `5.9 GiB` | `5.9 GiB` |
| available KV cache | `19.51 GiB` | `19.51 GiB` |
| KV cache capacity | `159,624 tokens` | `159,624 tokens` |
| selected kernel | `CutlassFP8ScaledMMLinearKernel` | `CutlassFP8ScaledMMLinearKernel` |

Harness: `shisa-multimodal-eval` at revision `1efa0a5`, `audio-evals/jia-test`
and `audio-evals/chime6`, reached through an SSH port forward. Reference text is
identical across both runs (1,299 characters for JIA-test, 8,715 for CHIME6),
every row has an empty `error` column, and both models scored 32/32 and 200/200
valid comparisons.

| Eval | v0.97 per-sample | v0.95b per-sample | v0.97 aggregate | v0.95b aggregate |
|---|---:|---:|---:|---:|
| JIA-test (32) | `0.115180` | `0.141873` | `0.121632` | `0.130100` |
| CHIME6 base (200) | `0.263867` | `0.282797` | `0.222490` | `0.236030` |
| CHIME6 hotwords (200) | `0.255802` | `0.264830` | `0.215032` | `0.221457` |

v0.97-FP8 is lower on all six measurements: JIA-test by `-0.008468` aggregate
and `-0.026693` per-sample; CHIME6 base by `-0.013540` and `-0.018930`; CHIME6
hotwords by `-0.006425` and `-0.009028`.

Hotword false positives on CHIME6 are unchanged: **17 of 200 for both**, 13 on
the same files. v0.97-FP8 introduced 4 (`OGG/3`, `OGG/72`, `OGG/89`,
`OGG/149`) and removed 4 (`OGG/38`, `OGG/39`, `OGG/68`, `OGG/108`). Matched
tokens are dominated by stopwords (`that`, `the`, `Shen`, `in`, `Here`) for both.

Hotword prompting helps both models on CHIME6, and helps v0.95b-FP8 more:
`-0.007458` aggregate for v0.97-FP8 against `-0.014573` for v0.95b-FP8. On the
only dataset here with a real distractor set, v0.97-FP8 lowers hotword-prompted
CER without lowering the false-positive count and is less responsive to hotword
prompting than v0.95b-FP8.

Not measured for v0.97: SPREDS, LibriVox, preference-test, and Earnings22. No
BF16 baseline was measured on this host, so the quantization delta remains
unquantified for v0.97, and the numbers above are not comparable to the
published v0.95b-FP8 card, which used a different harness configuration
(SPREDS base CER `0.044691` here versus `0.038981` on that card).

## Private publication

The v0.97 artifact was published privately at:

```text
repo: shisa-ai/shisa-asr-v0.97-FP8
revision: 2fe198f37f84b1324886e91f622bee23ce8cfd44
files: 25
visibility: private
uploaded bytes: 7,959,741,261
```

The uploaded card, `SHA256SUMS`, provenance summary, file sizes, and LFS
weight-shard hashes were verified against the local artifact. All four LFS
objects (both weight shards, `tokenizer.json`, `training_args.bin`) match the
local SHA-256 digests, and unauthenticated access is refused. Backend validation
should pin the exact revision above rather than `main`.

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

## Publication TODO

- [ ] Push the updated `shisa-ai/shisa-asr-v0.97-FP8` card. The local card at
  `/home/morpheus/data/models/hf/shisa-ai/shisa-asr-v0.97-fp8-dynamic/README.md`
  (SHA-256 `d0d3050eb1158c5b...`, 13,428 bytes) and its regenerated
  `SHA256SUMS` (24 of 24 entries verified) now record runtime kernel selection,
  GPU memory, and the matched FP8-versus-FP8 evaluation. The push is blocked:
  the stored Hugging Face token is read-only and the write token used for the
  original upload is no longer valid. The remote card still marks these as
  pending at revision `2fe198f`.
- [ ] Run a matched BF16 versus FP8_DYNAMIC evaluation and a throughput or
  latency benchmark. The RTX 5090 deployment ran with `--enforce-eager` and
  shared the host with live services, so no performance claim is available.
- [ ] Run SPREDS, LibriVox, preference-test, and Earnings22 against v0.97 to
  complete the coverage the v0.95b artifact already has. The preference test
  also needs `SHISA_API_KEY` before its judging stage can run.
- [ ] Run matched BF16 versus FP8_DYNAMIC CHIME6 and JIA evaluations before
  production qualification. The v0.95b artifact was published without them, so
  both carry the same gap.
- [x] Commit the export before publishing so `script.git_revision` in the
  summary names a revision that contains the script that produced the artifact.
  The v0.97 summary records revision `7bd2927` with `git_dirty: false`. The
  validator records both `script` and `validator` revisions with that flag so an
  uncommitted tree cannot be mistaken for a reproducible revision.
- [x] Upload privately and record the resulting Hugging Face commit SHA
  (v0.97).
- [x] Replace the copied base-model README with a quant-specific model card
  linking this script at exact Git commit `480a16e` (v0.95b).
- [x] Disclose the Earnings22 status inconsistency: the base card calls the
  benchmark finalized while its manifest/evaluator still identifies the
  model-assisted targets as pending final human review (v0.95b).
