#!/usr/bin/env python3
"""Purpose: inspect or quantize the merged Phi4MM decoder while preserving multimodal modules.
Date: 2026-08-28.

Default mode is a header-only dry run. Pass --run to load the model and export a
compressed-tensors checkpoint. FP8_DYNAMIC is the lineage/control scheme;
FP8_BLOCK is the matched 128x128 candidate.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "shisa-ai/shisa-asr-v0.95b"
MODEL_REVISION = "faa3d244fe9490f3fa5d05acacafaa1f668e180d"
DEFAULT_MODEL_PATH = Path("/home/morpheus/data/models/hf/shisa-ai/shisa-asr-v0.95b")
DEFAULT_EXPECTED_TARGET_COUNT = 128
DEFAULT_TARGETS = [
    r"re:^model\.layers\.\d+\.self_attn\.qkv_proj\.base_layer$",
    r"re:^model\.layers\.\d+\.self_attn\.o_proj\.base_layer$",
    r"re:^model\.layers\.\d+\.mlp\.gate_up_proj\.base_layer$",
    r"re:^model\.layers\.\d+\.mlp\.down_proj\.base_layer$",
]
DEFAULT_IGNORE = [
    r"re:.*audio_embed.*",
    r"re:.*image_embed.*",
    r"re:.*vision.*",
    r"re:.*lora_[AB].*",
    r"re:.*embed_tokens.*",
    r"re:.*lm_head.*",
]
PROTECTED_SUBSTRINGS = (
    "audio_embed",
    "image_embed",
    "vision",
    "embed_tokens",
    "lm_head",
    "lora_A",
    "lora_B",
)
WEIGHT_ARTIFACT_RE = re.compile(r"^.*\.safetensors(?:\.index\.json)?$")
SKIP_COPY_NAMES = {"config.json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run", action="store_true", help="Run quantization after safety checks.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--scheme",
        choices=("FP8_DYNAMIC", "FP8_BLOCK"),
        default="FP8_DYNAMIC",
    )
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--expected-target-count", type=int, default=DEFAULT_EXPECTED_TARGET_COUNT)
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--ignore", nargs="+", default=list(DEFAULT_IGNORE))
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/shisa-asr-v0.95b-quant-dry-run.json"),
    )
    return parser.parse_args()


def pattern_matches(name: str, pattern: str) -> bool:
    if pattern.startswith("re:"):
        return re.match(pattern.removeprefix("re:"), name) is not None
    return name == pattern


def module_matches_policy(
    name: str, *, targets: Sequence[str], ignore: Sequence[str]
) -> bool:
    return any(pattern_matches(name, pattern) for pattern in targets) and not any(
        pattern_matches(name, pattern) for pattern in ignore
    )


def match_named_module_targets(
    names: Iterable[str], *, targets: Sequence[str], ignore: Sequence[str]
) -> list[str]:
    return sorted(
        name for name in names if module_matches_policy(name, targets=targets, ignore=ignore)
    )


def read_weight_names(model_path: Path) -> list[str]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return sorted(index.get("weight_map", {}))
    single_path = model_path / "model.safetensors"
    if not single_path.exists():
        raise FileNotFoundError("No model.safetensors or model.safetensors.index.json found")
    from safetensors import safe_open

    with safe_open(single_path, framework="pt", device="cpu") as handle:
        return sorted(handle.keys())


def module_names_from_weights(weight_names: Iterable[str]) -> list[str]:
    return sorted(name.removesuffix(".weight") for name in weight_names if name.endswith(".weight"))


def protected_target_matches(target_names: Iterable[str]) -> list[str]:
    return sorted(
        name for name in target_names if any(part in name for part in PROTECTED_SUBSTRINGS)
    )


def inspect_checkpoint(
    model_path: Path,
    *,
    targets: Sequence[str],
    ignore: Sequence[str],
    expected_target_count: int,
    scheme: str,
) -> dict[str, Any]:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    weight_names = read_weight_names(model_path)
    module_names = module_names_from_weights(weight_names)
    target_names = match_named_module_targets(module_names, targets=targets, ignore=ignore)
    protected = protected_target_matches(target_names)
    safety_errors: list[str] = []
    if config.get("model_type") != "phi4mm":
        safety_errors.append(f"unexpected model_type: {config.get('model_type')!r}")
    if config.get("torch_dtype") != "bfloat16":
        safety_errors.append(f"unexpected source dtype: {config.get('torch_dtype')!r}")
    if len(target_names) != expected_target_count:
        safety_errors.append(
            f"target count {len(target_names)} != expected {expected_target_count}"
        )
    if protected:
        safety_errors.append("quantization targets overlap protected multimodal modules")
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": str(model_path),
        "model_type": config.get("model_type"),
        "architecture": config.get("architectures"),
        "source_dtype": config.get("torch_dtype"),
        "hidden_size": config.get("hidden_size"),
        "intermediate_size": config.get("intermediate_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "weight_tensor_count": len(weight_names),
        "scheme": scheme,
        "targets": list(targets),
        "ignore": list(ignore),
        "expected_target_count": expected_target_count,
        "target_count": len(target_names),
        "target_names": target_names,
        "protected_target_matches": protected,
        "safety_errors": safety_errors,
        "safety_status": "PASS" if not safety_errors else "FAIL",
    }


def is_weight_artifact(name: str) -> bool:
    return WEIGHT_ARTIFACT_RE.match(name) is not None


def copy_snapshot_assets(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in source_dir.iterdir():
        if not child.is_file():
            continue
        if child.name in SKIP_COPY_NAMES or is_weight_artifact(child.name):
            continue
        destination = output_dir / child.name
        if not destination.exists():
            shutil.copy2(child, destination)


@dataclass
class SnapshotAssetCopier:
    source_dir: Path

    def save_pretrained(self, output_dir: str | Path) -> None:
        copy_snapshot_assets(self.source_dir, Path(output_dir))


def load_model_class(model_path: Path):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_cls = get_class_from_dynamic_module(
        "modeling_phi4mm.Phi4MMForCausalLM",
        str(model_path),
        local_files_only=True,
    )
    module = importlib.import_module(model_cls.__module__)
    if not hasattr(module.Phi4MMModel, "prepare_inputs_for_generation"):

        def _prepare_inputs_for_generation(self, input_ids=None, **kwargs):
            model_inputs = {}
            if input_ids is not None:
                model_inputs["input_ids"] = input_ids
            model_inputs.update(kwargs)
            return model_inputs

        module.Phi4MMModel.prepare_inputs_for_generation = _prepare_inputs_for_generation
    return model_cls


def validate_loaded_targets(
    model: Any,
    *,
    targets: Sequence[str],
    ignore: Sequence[str],
    expected_target_count: int,
    scheme: str,
) -> list[str]:
    named_modules = dict(model.named_modules())
    matched = match_named_module_targets(named_modules, targets=targets, ignore=ignore)
    if len(matched) != expected_target_count:
        raise ValueError(f"loaded target count {len(matched)} != expected {expected_target_count}")
    protected = protected_target_matches(matched)
    if protected:
        raise ValueError(f"protected modules selected: {protected[:8]}")
    if scheme == "FP8_BLOCK":
        bad_shapes = []
        for name in matched:
            weight = getattr(named_modules[name], "weight", None)
            shape = tuple(weight.shape) if weight is not None else ()
            if len(shape) != 2 or shape[0] % 128 or shape[1] % 128:
                bad_shapes.append((name, shape))
        if bad_shapes:
            raise ValueError(f"FP8_BLOCK target shapes are not 128x128 divisible: {bad_shapes[:8]}")
    return matched


def translate_target_for_vllm(target: str) -> str:
    if target.startswith("re:"):
        pattern = target.removeprefix("re:")
        pattern = pattern.replace(r"\.base_layer$", "$")
        pattern = pattern.replace(r"\.base_layer\.", r"\.")
        return f"re:{pattern}"
    return target.replace(".base_layer", "")


def rewrite_quantization_config_for_vllm(output_dir: Path) -> dict[str, Any]:
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, dict):
        raise TypeError("exported config.json has no quantization_config")
    translated_targets: list[str] = []
    for group in quant_config.get("config_groups", {}).values():
        translated = [translate_target_for_vllm(target) for target in group.get("targets", [])]
        group["targets"] = translated
        translated_targets.extend(translated)
    ignores = quant_config.get("ignore", [])
    filtered_ignores = [
        entry
        for entry in ignores
        if not any(pattern_matches(entry, target) for target in translated_targets)
    ]
    quant_config["ignore"] = filtered_ignores
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "translated_targets": translated_targets,
        "removed_ignore_count": len(ignores) - len(filtered_ignores),
    }


def validate_saved_quantization_config(output_dir: Path) -> dict[str, Any]:
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, dict):
        raise TypeError("exported config.json has no quantization_config")
    saved_targets: list[str] = []
    for group in quant_config.get("config_groups", {}).values():
        saved_targets.extend(group.get("targets", []))
    protected = [
        target
        for target in saved_targets
        if any(part in target for part in PROTECTED_SUBSTRINGS)
    ]
    if protected:
        raise ValueError(f"saved quantization config targets protected modules: {protected}")
    return {
        "quant_method": quant_config.get("quant_method"),
        "saved_targets": saved_targets,
        "protected_saved_targets": protected,
    }


def quantize_model(
    *,
    model_path: Path,
    output_dir: Path,
    report: dict[str, Any],
) -> None:
    import llmcompressor.pipelines.data_free.pipeline as data_free_pipeline
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    model_cls = load_model_class(model_path)
    # Phi4MM's Conformer constructor reads a derived scalar while building the
    # module tree, so Transformers' meta-device loading path is not safe here.
    # The checkpoint fits in host RAM; construct and load directly on CPU.
    model = model_cls.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        low_cpu_mem_usage=False,
        dtype=report["precision"],
    )
    # PEFT promotes adapter parameters to FP32 when constructing the model even
    # though every source checkpoint tensor is BF16. Restore source precision
    # before compression so ignored LoRA weights remain byte-equivalent BF16
    # rather than being needlessly serialized as FP32.
    if report["source_dtype"] != "bfloat16":
        raise ValueError(f"unsupported source dtype: {report['source_dtype']!r}")
    import torch

    model.to(dtype=torch.bfloat16)
    report["normalized_model_dtype"] = "bfloat16"
    loaded_targets = validate_loaded_targets(
        model,
        targets=report["targets"],
        ignore=report["ignore"],
        expected_target_count=report["expected_target_count"],
        scheme=report["scheme"],
    )
    recipe = [
        QuantizationModifier(
            targets=report["targets"],
            ignore=report["ignore"],
            scheme=report["scheme"],
        )
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    data_free_pipeline.dispatch_model = lambda model: model
    oneshot(
        model=model,
        processor=SnapshotAssetCopier(model_path),
        recipe=recipe,
        output_dir=str(output_dir),
        precision=report["precision"],
        save_compressed=True,
    )
    report["loaded_target_count"] = len(loaded_targets)
    report["vllm_target_rewrite"] = rewrite_quantization_config_for_vllm(output_dir)
    report["saved_quantization_config"] = validate_saved_quantization_config(output_dir)
    (output_dir / "quantization_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    model_path = args.model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    report = inspect_checkpoint(
        model_path,
        targets=args.targets,
        ignore=args.ignore,
        expected_target_count=args.expected_target_count,
        scheme=args.scheme,
    )
    report["precision"] = args.precision
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["safety_errors"]:
        raise SystemExit("Refusing to continue because checkpoint safety checks failed")
    if not args.run:
        return
    if args.output_dir is None:
        raise SystemExit("--output-dir is required with --run")
    output_dir = args.output_dir.resolve()
    if output_dir == model_path:
        raise SystemExit("--output-dir must differ from --model-path")
    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"Output exists; pass --force to replace it: {output_dir}")
        shutil.rmtree(output_dir)
    quantize_model(model_path=model_path, output_dir=output_dir, report=report)


if __name__ == "__main__":
    main()
