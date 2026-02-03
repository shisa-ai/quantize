#!/usr/bin/env python3
"""
Quantize a Hugging Face causal LM with LLM Compressor using FP8_BLOCK.

Defaults to QuantizationModifier PTQ. Optional flags enable AWQ and/or SpinQuant.

Example:
  python quantize-fp8-block.py -m shisa-ai/chotto-14b-20260107-dpo
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.utils import dispatch_for_generation


def _torch_dtype(value: str) -> str | torch.dtype:
    value = value.lower()
    if value == "auto":
        return "auto"
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _strip_sglang_incompatible_quant_args(config_path: str) -> bool:
    """
    sglang currently vendors / depends on compressed_tensors < 0.13.x, whose
    QuantizationArgs does not include `scale_dtype` / `zp_dtype`. Newer LLM
    Compressor writes those fields (often as null), which breaks strict pydantic
    validation in sglang. Stripping them is safe for FP8_BLOCK where they are
    unset anyway.
    """
    import json

    path = os.fspath(config_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qc = data.get("quantization_config")
    if not isinstance(qc, dict):
        return False

    changed = False

    def walk(x: Any) -> None:
        nonlocal changed
        if isinstance(x, dict):
            for k in ("scale_dtype", "zp_dtype"):
                if k in x:
                    x.pop(k, None)
                    changed = True
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(qc)

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
    return changed


def _parse_csv_list(value: str) -> list[str]:
    # Accept comma-separated and/or whitespace-separated values.
    tokens = []
    for part in value.split(","):
        tokens.extend(part.strip().split())
    return [t for t in (tok.strip() for tok in tokens) if t]


def _default_output_dir(
    model_id: str,
    *,
    use_awq: bool,
    use_spinquant: bool,
    spinquant_rotations: list[str],
) -> str:
    base = model_id.rstrip("/").split("/")[-1]
    suffix = ""
    if use_spinquant:
        suffix += "-spinquant" + "".join(spinquant_rotations)
    if use_awq:
        suffix += "-awq"
    return f"{base}-fp8-block{suffix}"


def _build_calibration_dataset(
    *,
    tokenizer: Any,
    dataset_id: str,
    split: str,
    num_samples: int,
    seed: int,
    use_chat_template: bool,
    messages_column: str,
    text_column: str,
) -> Any:
    ds = load_dataset(dataset_id, split=f"{split}[:{num_samples}]")
    ds = ds.shuffle(seed=seed)

    if use_chat_template:
        msg_col = messages_column
        if msg_col not in ds.column_names:
            for candidate in ("messages", "conversations", "conversation", "chat", "dialogue"):
                if candidate in ds.column_names:
                    msg_col = candidate
                    break
        if msg_col in ds.column_names:
            orig_cols = list(ds.column_names)

            def preprocess(example: dict[str, Any]) -> dict[str, str]:
                messages = example[msg_col]
                if isinstance(messages, str):
                    import json

                    try:
                        messages = json.loads(messages)
                    except json.JSONDecodeError:
                        pass

                normalized: list[dict[str, str]] = []
                if isinstance(messages, list):
                    for msg in messages:
                        if isinstance(msg, dict):
                            role = msg.get("role")
                            content = msg.get("content")
                            if role is None or content is None:
                                role = msg.get("from", role)
                                content = msg.get("value", content)
                            if role is None or content is None:
                                role = msg.get("speaker", role)
                                content = msg.get("text", content)
                        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                            role, content = msg
                        else:
                            continue

                        if role is None or content is None:
                            continue

                        role_str = str(role).strip().lower()
                        if role_str in {"human", "user"}:
                            role_str = "user"
                        elif role_str in {"gpt", "assistant", "bot"}:
                            role_str = "assistant"
                        elif role_str in {"system"}:
                            role_str = "system"
                        else:
                            role_str = "user"

                        normalized.append({"role": role_str, "content": str(content)})

                return {
                    "text": tokenizer.apply_chat_template(
                        normalized,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                }

            return ds.map(preprocess, remove_columns=orig_cols, desc="Applying chat template")

    # Fallback: use a plain text column.
    if text_column not in ds.column_names:
        raise ValueError(
            f"Dataset '{dataset_id}' split '{split}' has no '{messages_column}' column "
            f"and no '{text_column}' column."
        )
    if text_column == "text":
        return ds

    orig_cols = list(ds.column_names)

    def remap_text(example: dict[str, Any]) -> dict[str, str]:
        return {"text": example[text_column]}

    return ds.map(remap_text, remove_columns=orig_cols, desc=f"Using '{text_column}' as text")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quantize-fp8-block.py",
        description="FP8_BLOCK quantization with LLM Compressor (oneshot calibration).",
    )
    p.add_argument(
        "-m",
        "--model",
        help="Model id (Hugging Face) or local path. Required.",
    )
    p.add_argument(
        "--awq",
        action="store_true",
        help="Use AWQModifier (slower; includes smoothing). Default is QuantizationModifier PTQ.",
    )
    p.add_argument(
        "--datafree",
        action="store_true",
        help=(
            "Do not load calibration data and run the 'datafree' pipeline. "
            "Valid for recipes that do not require calibration data (e.g., FP8_BLOCK PTQ, "
            "SpinQuant R1/R2 + PTQ)."
        ),
    )
    p.add_argument(
        "--spinquant",
        action="store_true",
        help=(
            "Apply SpinQuant transforms before quantization. Note: R1/R2 are offline "
            "(no inference cost); R3/R4 are online (adds inference overhead)."
        ),
    )
    p.add_argument(
        "--spinquant-rotations",
        default="R1,R2",
        help="Comma-separated rotations to apply (default: R1,R2). Example: R1,R2,R4.",
    )
    p.add_argument(
        "--spinquant-transform-type",
        default="hadamard",
        choices=["hadamard", "random-hadamard", "random-matrix"],
        help="SpinQuant transform type (default: hadamard).",
    )
    p.add_argument(
        "--spinquant-transform-block-size",
        type=int,
        default=128,
        help=(
            "SpinQuant block size (default: 128). Set 0 to let SpinQuant pick a "
            "model-default block size."
        ),
    )
    p.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory (default: <model_name>-fp8-block[-awq]).",
    )
    p.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Overwrite output-dir if it already exists.",
    )
    p.add_argument(
        "--sglang-compat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Strip `scale_dtype`/`zp_dtype` from saved quantization_config for "
            "compatibility with sglang (default: enabled)."
        ),
    )

    # Calibration data
    p.add_argument("--dataset", default="shisa-ai/shisa-v2.1-sharegpt")
    p.add_argument("--split", default="train", help="Dataset split name.")
    p.add_argument("--num-calibration-samples", type=int, default=256)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If dataset has a messages column, apply tokenizer chat template into a text column.",
    )
    p.add_argument("--messages-column", default="messages")
    p.add_argument("--text-column", default="text")

    # Model loading
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32", "float16", "bfloat16", "float32"],
        help="Model dtype to load for calibration.",
    )
    p.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map (default: auto). Use 'none' to disable.",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model/tokenizer code from the hub.",
    )
    p.add_argument("--revision", default="main", help="Model revision (branch/tag/commit).")
    p.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not attempt to download from the Hugging Face Hub.",
    )

    # Quick sanity check generation
    p.add_argument("--skip-sample-generation", action="store_true")
    p.add_argument("--sample-prompt", default="Hello my name is")
    p.add_argument("--sample-max-new-tokens", type=int, default=100)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = _build_parser()

    if not argv:
        parser.print_help(sys.stderr)
        return 2

    args = parser.parse_args(argv)
    if not args.model:
        parser.print_help(sys.stderr)
        return 2

    if args.datafree and args.awq:
        print("error: --datafree is not compatible with --awq (AWQ requires calibration data).", file=sys.stderr)
        return 2

    spinquant_rotations = [r.upper() for r in _parse_csv_list(args.spinquant_rotations)]
    output_dir = args.output_dir or _default_output_dir(
        args.model,
        use_awq=args.awq,
        use_spinquant=args.spinquant,
        spinquant_rotations=spinquant_rotations,
    )
    if os.path.exists(output_dir) and os.listdir(output_dir) and not args.overwrite_output_dir:
        print(
            f"error: output dir '{output_dir}' already exists and is not empty. "
            f"Pass --overwrite-output-dir or choose --output-dir.",
            file=sys.stderr,
        )
        return 2

    device_map: str | None = None if str(args.device_map).lower() in {"none", "null"} else args.device_map

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=_torch_dtype(args.dtype),
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    ds = None
    if not args.datafree:
        ds = _build_calibration_dataset(
            tokenizer=tokenizer,
            dataset_id=args.dataset,
            split=args.split,
            num_samples=args.num_calibration_samples,
            seed=args.seed,
            use_chat_template=args.use_chat_template,
            messages_column=args.messages_column,
            text_column=args.text_column,
        )

    # MoE note: Qwen/Qwen3 MoE models (and some other MoEs) have router gate layers
    # that are sensitive and/or shape-incompatible with block quantization. LLM
    # Compressor examples typically leave these gates in full precision.
    ignore = ["lm_head", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"]

    recipe: list[Any] = []
    if args.spinquant:
        transform_block_size = (
            None if args.spinquant_transform_block_size == 0 else args.spinquant_transform_block_size
        )
        recipe.append(
            SpinQuantModifier(
                rotations=spinquant_rotations,
                transform_block_size=transform_block_size,
                transform_type=args.spinquant_transform_type,
            )
        )

    if args.awq:
        recipe.append(
            AWQModifier(
                ignore=ignore,
                scheme="FP8_BLOCK",
                targets=["Linear"],
                duo_scaling="both",
            )
        )
    else:
        recipe.append(
            QuantizationModifier(
                ignore=ignore,
                scheme="FP8_BLOCK",
                targets=["Linear"],
            )
        )

    oneshot_kwargs: dict[str, Any] = {}
    if args.datafree:
        oneshot_kwargs["pipeline"] = "datafree"

    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
        batch_size=args.batch_size,
        **oneshot_kwargs,
    )

    if not args.skip_sample_generation:
        print("\n\n========== SAMPLE GENERATION ==============")
        dispatch_for_generation(model)
        with torch.inference_mode():
            input_ids = tokenizer(args.sample_prompt, return_tensors="pt").input_ids.to(model.device)
            output = model.generate(input_ids, max_new_tokens=args.sample_max_new_tokens)
        print(tokenizer.decode(output[0]))
        print("==========================================\n\n")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    if args.sglang_compat:
        _strip_sglang_incompatible_quant_args(os.path.join(output_dir, "config.json"))
    print(f"Saved compressed model to: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
