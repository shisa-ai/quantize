#!/usr/bin/env python3
"""Utility to count tokens for EAGLE3 JSONL datasets."""
import argparse
import glob
import math
from pathlib import Path
from typing import List, Sequence

import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer

from specforge.data import build_eagle3_dataset

DEFAULT_MODEL = "shisa-ai/chotto-14b-20250922"
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count tokens in JSONL data prepared for EAGLE3 training."
    )
    parser.add_argument(
        "--data",
        nargs="+",
        default=None,
        help=(
            "One or more paths/globs to JSONL files. If omitted, all *.jsonl files "
            "next to this script are used."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_MODEL,
        help="Tokenizer or model path/repo id used by the target model.",
    )
    parser.add_argument(
        "--chat-template",
        default="phi4",
        help="Registered chat template name (matches training).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length applied during preprocessing.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=8,
        help="Number of processes for dataset.map inside SpecForge preprocessing.",
    )
    parser.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Set when the JSONL already stores flattened text under a 'text' column.",
    )
    parser.add_argument(
        "--is-vlm",
        action="store_true",
        help="Enable when counting multimodal datasets that need an AutoProcessor.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit the number of samples processed (after loading).",
    )
    parser.add_argument(
        "--report-samples",
        type=int,
        default=0,
        help="Print detailed counts for the first N samples.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional cache directory forwarded to build_eagle3_dataset.",
    )
    parser.add_argument(
        "--cache-key",
        type=str,
        default=None,
        help="Optional cache key forwarded to build_eagle3_dataset.",
    )
    return parser.parse_args()


def resolve_data_files(patterns: List[str] | None) -> List[str]:
    if not patterns:
        discovered = sorted(str(p) for p in SCRIPT_DIR.glob("*.jsonl"))
        if not discovered:
            raise FileNotFoundError(
                f"No JSONL files found in {SCRIPT_DIR}. Pass --data to override."
            )
        return discovered

    matched: List[str] = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if not expanded:
            raise FileNotFoundError(f"No files match pattern: {pattern}")
        matched.extend(sorted(expanded))
    return matched


def percentile(values: Sequence[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return float(ordered[lower])
    lower_val = ordered[lower]
    upper_val = ordered[upper]
    return float(lower_val + (upper_val - lower_val) * (pos - lower))


def flatten_length(tensor: torch.Tensor) -> int:
    if tensor.ndim == 0:
        return 0
    return int(tensor.shape[-1])


def collect_counts(dataset, report_limit: int) -> tuple[List[int], List[int]]:
    input_lengths: List[int] = []
    loss_lengths: List[int] = []
    for idx, example in enumerate(dataset):
        input_ids = example["input_ids"]
        loss_mask = example["loss_mask"]
        seq_len = flatten_length(input_ids)
        loss_tokens = int(loss_mask.sum().item())
        input_lengths.append(seq_len)
        loss_lengths.append(loss_tokens)
        if report_limit and idx < report_limit:
            print(
                f"sample {idx}: input_tokens={seq_len}, loss_tokens={loss_tokens}"
            )
    return input_lengths, loss_lengths


def main() -> None:
    args = parse_args()

    data_files = resolve_data_files(args.data)
    print("Files:")
    for path in data_files:
        print(f"  - {path}")

    raw_dataset = load_dataset("json", data_files=data_files, split="train")

    if args.limit is not None:
        capped = min(len(raw_dataset), args.limit)
        raw_dataset = raw_dataset.select(range(capped))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    processor = None
    if args.is_vlm:
        processor = AutoProcessor.from_pretrained(args.tokenizer)

    processed_dataset = build_eagle3_dataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        num_proc=args.num_proc,
        cache_dir=args.cache_dir,
        cache_key=args.cache_key,
        is_vlm=args.is_vlm,
        processor=processor,
        is_preformatted=args.is_preformatted,
    )

    input_lengths, loss_lengths = collect_counts(
        processed_dataset, args.report_samples
    )
    num_examples = len(input_lengths)

    total_input_tokens = sum(input_lengths)
    total_loss_tokens = sum(loss_lengths)

    avg_input = total_input_tokens / num_examples if num_examples else 0.0
    avg_loss = total_loss_tokens / num_examples if num_examples else 0.0

    print("=== Token Count Summary ===")
    print(f"examples processed: {num_examples}")
    print(f"total input tokens: {total_input_tokens:,}")
    print(f"total loss tokens: {total_loss_tokens:,}")
    print(f"average input tokens/example: {avg_input:.2f}")
    print(f"average loss tokens/example: {avg_loss:.2f}")
    if num_examples:
        print(
            "input length percentiles (tokens): "
            f"p50={percentile(input_lengths, 50):.1f}, "
            f"p90={percentile(input_lengths, 90):.1f}, "
            f"p95={percentile(input_lengths, 95):.1f}, "
            f"max={max(input_lengths)}"
        )
        print(
            "loss length percentiles (tokens): "
            f"p50={percentile(loss_lengths, 50):.1f}, "
            f"p90={percentile(loss_lengths, 90):.1f}, "
            f"p95={percentile(loss_lengths, 95):.1f}, "
            f"max={max(loss_lengths)}"
        )


if __name__ == "__main__":
    main()
