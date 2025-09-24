#!/usr/bin/env python3
"""Preprocess JSONL datasets for EAGLE3 training and populate SpecForge caches."""
import argparse
import glob
import hashlib
import os
from pathlib import Path
from typing import List

from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer

from specforge import AutoDraftModelConfig
from specforge.data import build_eagle3_dataset, generate_vocab_mapping_file
from specforge.utils import create_draft_config_from_target

DEFAULT_TARGET_MODEL = "shisa-ai/chotto-14b-20250922"
DEFAULT_DRAFT_CONFIG = "unphi4-eagle3.json"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = str((SCRIPT_DIR.parent / "cache").resolve())
DEFAULT_TRAIN_PATH = "sft.shisa-v2.1-EAGLE3.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize JSONL data and build cache/vocab files for EAGLE3 training."
    )
    parser.add_argument(
        "--data",
        nargs="+",
        default=None,
        help=(
            "One or more paths/globs to JSONL files. If omitted, every *.jsonl in this "
            "directory will be processed."
        ),
    )
    parser.add_argument(
        "--target-model-path",
        default=DEFAULT_TARGET_MODEL,
        help="Target model path or repo id (used for tokenizer loading).",
    )
    parser.add_argument(
        "--draft-model-config",
        default=DEFAULT_DRAFT_CONFIG,
        help="Draft model config JSON (auto-generated if missing).",
    )
    parser.add_argument(
        "--chat-template",
        default="phi4",
        help="Registered chat template name (must match training).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum length applied during preprocessing.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=8,
        help="Number of worker processes for dataset.map().",
    )
    parser.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Set when the JSONL already contains flattened text under 'text'.",
    )
    parser.add_argument(
        "--is-vlm",
        action="store_true",
        help="Enable when preprocessing multimodal conversations (loads AutoProcessor).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally truncate the dataset to the first N samples for quick runs.",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="Root cache directory (training passes the same path).",
    )
    parser.add_argument(
        "--train-data-path",
        default=DEFAULT_TRAIN_PATH,
        help="String used during training for --train-data-path (controls cache key).",
    )
    parser.add_argument(
        "--cache-key",
        default=None,
        help="Override automatic cache key (defaults to md5 of train path + template + max length + target).",
    )
    parser.add_argument(
        "--report-progress",
        action="store_true",
        help="Print dataset length after each stage for visibility.",
    )
    return parser.parse_args()


def resolve_data_files(patterns: List[str] | None) -> List[str]:
    if not patterns:
        discovered = sorted(str(p) for p in SCRIPT_DIR.glob("*.jsonl"))
        if not discovered:
            raise FileNotFoundError(
                f"No JSONL files found beside {SCRIPT_DIR}. Pass --data to specify inputs."
            )
        return discovered

    matched: List[str] = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if not expanded:
            raise FileNotFoundError(f"No files match pattern: {pattern}")
        matched.extend(sorted(expanded))
    return matched


def ensure_draft_config(path: str, target_model: str, cache_dir: str) -> str:
    config_path = Path(path)
    if config_path.exists():
        return str(config_path)
    auto_config_path = create_draft_config_from_target(
        target_model_path=target_model,
        cache_dir=cache_dir,
    )
    print(f"Draft config not found; auto-generated at {auto_config_path}")
    return auto_config_path


def compute_cache_key(args: argparse.Namespace) -> str:
    if args.cache_key:
        return args.cache_key
    cache_seed = f"{args.train_data_path}-{args.max_length}-{args.chat_template}-{args.target_model_path}"
    return hashlib.md5(cache_seed.encode()).hexdigest()


def main() -> None:
    args = parse_args()

    data_files = resolve_data_files(args.data)
    print("Input files:")
    for path in data_files:
        print(f"  - {path}")

    raw_dataset = load_dataset("json", data_files=data_files, split="train")
    if args.limit is not None:
        raw_dataset = raw_dataset.select(range(min(len(raw_dataset), args.limit)))
    if args.report_progress:
        print(f"Loaded {len(raw_dataset):,} rows from JSONL")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    processor = None
    if args.is_vlm:
        processor = AutoProcessor.from_pretrained(args.target_model_path)

    cache_key = compute_cache_key(args)
    processed_cache_dir = os.path.join(args.cache_dir, "processed_dataset")
    vocab_cache_dir = os.path.join(args.cache_dir, "vocab_mapping")
    os.makedirs(processed_cache_dir, exist_ok=True)
    os.makedirs(vocab_cache_dir, exist_ok=True)

    processed_dataset = build_eagle3_dataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        num_proc=args.num_proc,
        cache_dir=processed_cache_dir,
        cache_key=cache_key,
        is_vlm=args.is_vlm,
        processor=processor,
        is_preformatted=args.is_preformatted,
    )
    if args.report_progress:
        print(
            f"Processed dataset cached under {processed_cache_dir} with key {cache_key}; "
            f"{len(processed_dataset):,} items"
        )

    draft_config_path = ensure_draft_config(
        args.draft_model_config, args.target_model_path, args.cache_dir
    )
    draft_model_config = AutoDraftModelConfig.from_file(draft_config_path)
    vocab_mapping_path = generate_vocab_mapping_file(
        dataset=processed_dataset,
        target_vocab_size=draft_model_config.vocab_size,
        draft_vocab_size=draft_model_config.draft_vocab_size,
        cache_dir=vocab_cache_dir,
        cache_key=cache_key,
    )

    print("=== Preprocessing Complete ===")
    print(f"processed dataset cache : {os.path.join(processed_cache_dir, cache_key)}.pkl")
    print(f"vocab mapping cache     : {vocab_mapping_path}")
    print(f"cache key               : {cache_key}")


if __name__ == "__main__":
    main()
