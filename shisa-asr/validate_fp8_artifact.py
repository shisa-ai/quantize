#!/usr/bin/env python3
"""Purpose: verify a Shisa ASR Phi4MM FP8 export and record artifact integrity.
Date: 2026-09-29.

The published `shisa-asr-v0.95b-FP8` summary carries `tool_versions`,
`script`, `generated_at_utc`, `output_repo_id`, and `artifact_integrity`
blocks that `quantize_decoder_fp8.py` does not emit. This script recomputes
that integrity block from a source checkpoint and its FP8 export, and can
merge the result back into the export's `quantization_summary.json`.

Comparison is dtype, shape, and value equality for every non-target tensor.
Reading is shard-local and one tensor at a time so host memory stays bounded.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

PROTECTED_SUBSTRINGS = (
    "audio_embed",
    "image_embed",
    "vision",
    "embed_tokens",
    "lm_head",
    "lora_A",
    "lora_B",
)
TARGET_SUFFIXES = (
    ".self_attn.qkv_proj.base_layer.weight",
    ".self_attn.o_proj.base_layer.weight",
    ".mlp.gate_up_proj.base_layer.weight",
    ".mlp.down_proj.base_layer.weight",
)
TRACKED_TOOLS = (
    "accelerate",
    "compressed-tensors",
    "huggingface-hub",
    "llmcompressor",
    "safetensors",
    "torch",
    "transformers",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, help="Merge into this summary file.")
    parser.add_argument("--output-repo-id")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--skip-value-check", action="store_true")
    return parser.parse_args()


def read_index(model_path: Path) -> dict[str, Any]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"no model.safetensors.index.json in {model_path}")


def shard_names(weight_map: dict[str, str]) -> list[str]:
    return sorted(set(weight_map.values()))


def is_target_weight(name: str) -> bool:
    return name.endswith(TARGET_SUFFIXES)


def is_scale(name: str) -> bool:
    return name.endswith(".weight_scale")


def collect_dtype_counts(
    artifact_path: Path, weight_map: dict[str, str]
) -> tuple[Counter[str], dict[str, str]]:
    from safetensors import safe_open

    counts: Counter[str] = Counter()
    per_name: dict[str, str] = {}
    for shard in shard_names(weight_map):
        with safe_open(artifact_path / shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                dtype = str(handle.get_slice(name).get_dtype())
                counts[dtype] += 1
                per_name[name] = dtype
    return counts, per_name


def compare_non_targets(
    source_path: Path,
    artifact_path: Path,
    source_map: dict[str, str],
    artifact_map: dict[str, str],
) -> tuple[int, int, list[str]]:
    import torch
    from safetensors import safe_open

    non_targets = sorted(name for name in source_map if not is_target_weight(name))
    mismatches: list[str] = []
    by_source_shard: dict[str, list[str]] = {}
    for name in non_targets:
        by_source_shard.setdefault(source_map[name], []).append(name)

    for source_shard, names in sorted(by_source_shard.items()):
        with safe_open(source_path / source_shard, framework="pt", device="cpu") as src:
            for name in names:
                if name not in artifact_map:
                    mismatches.append(f"{name}: missing from artifact index")
                    continue
                with safe_open(
                    artifact_path / artifact_map[name], framework="pt", device="cpu"
                ) as dst:
                    source_tensor = src.get_tensor(name)
                    artifact_tensor = dst.get_tensor(name)
                    if source_tensor.dtype != artifact_tensor.dtype:
                        mismatches.append(
                            f"{name}: dtype {source_tensor.dtype} -> {artifact_tensor.dtype}"
                        )
                    elif tuple(source_tensor.shape) != tuple(artifact_tensor.shape):
                        mismatches.append(
                            f"{name}: shape {tuple(source_tensor.shape)} -> "
                            f"{tuple(artifact_tensor.shape)}"
                        )
                    elif not torch.equal(source_tensor, artifact_tensor):
                        mismatches.append(f"{name}: value mismatch")
                    del source_tensor, artifact_tensor
    return len(non_targets), len(mismatches), mismatches


def collect_tool_versions() -> dict[str, str]:
    tools: dict[str, str] = {}
    for name in TRACKED_TOOLS:
        try:
            tools[name] = version(name)
        except PackageNotFoundError:
            tools[name] = "not-installed"
    return tools


def git_state(repo_dir: Path) -> tuple[str | None, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, False


def script_block(relative_path: str, revision: str, dirty: bool) -> dict[str, Any]:
    return {
        "github_repo": "https://github.com/shisa-ai/quantize",
        "relative_path": relative_path,
        "git_revision": revision,
        "git_dirty": dirty,
        "github_url": (
            f"https://github.com/shisa-ai/quantize/blob/{revision}/{relative_path}"
        ),
    }


def main() -> None:
    args = parse_args()
    source_path = args.source_path.resolve()
    artifact_path = args.artifact_path.resolve()
    source_index = read_index(source_path)
    artifact_index = read_index(artifact_path)
    source_map: dict[str, str] = source_index["weight_map"]
    artifact_map: dict[str, str] = artifact_index["weight_map"]

    dtype_counts, name_dtypes = collect_dtype_counts(artifact_path, artifact_map)
    fp8_weights = sorted(
        name for name in artifact_map if is_target_weight(name) and not is_scale(name)
    )
    scale_tensors = sorted(name for name in artifact_map if is_scale(name))
    protected_scales = [
        name
        for name in scale_tensors
        if any(part in name for part in PROTECTED_SUBSTRINGS)
    ]

    if args.skip_value_check:
        checked, mismatches, examples = 0, -1, []
    else:
        checked, mismatches, examples = compare_non_targets(
            source_path, artifact_path, source_map, artifact_map
        )

    integrity = {
        "safetensor_shards": len(shard_names(artifact_map)),
        "indexed_tensors": len(artifact_map),
        "indexed_parameters": artifact_index.get("metadata", {}).get("total_parameters"),
        "indexed_weight_bytes": artifact_index.get("metadata", {}).get("total_size"),
        "bf16_tensors": dtype_counts.get("BF16", 0),
        "fp8_e4m3_target_weights": dtype_counts.get("F8_E4M3", 0),
        "bf16_scale_tensors": sum(
            1 for name in scale_tensors if name_dtypes.get(name) == "BF16"
        ),
        "original_non_target_tensors_checked": checked,
        "non_target_mismatches": mismatches,
        "protected_scale_tensors": len(protected_scales),
        "fp8_target_weight_names": len(fp8_weights),
        "dtype_counts": dict(sorted(dtype_counts.items())),
    }

    report: dict[str, Any] = {
        "source_path": str(source_path),
        "artifact_path": str(artifact_path),
        "source_indexed_tensors": len(source_map),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool_versions": collect_tool_versions(),
        "artifact_integrity": integrity,
        "mismatch_examples": examples[:20],
    }
    revision, dirty = git_state(Path(__file__).resolve().parent.parent)
    if revision:
        # `script` follows the published convention and names the exporter that
        # produced the artifact; `validator` names this verification script.
        report["script"] = script_block(
            "shisa-asr/quantize_decoder_fp8.py", revision, dirty
        )
        report["validator"] = script_block(
            "shisa-asr/validate_fp8_artifact.py", revision, dirty
        )
    if args.output_repo_id:
        report["output_repo_id"] = args.output_repo_id
        report["private"] = args.private

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.summary_json:
        summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
        summary.update(report)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if mismatches:
        raise SystemExit(f"artifact integrity failed: {mismatches} non-target mismatches")


if __name__ == "__main__":
    main()
