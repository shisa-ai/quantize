import os
import json
import random
import time
import hashlib
from datasets import Features, Sequence, Value, load_dataset, concatenate_datasets, Dataset
from tqdm import tqdm

# ==============================================================================
# Main Execution Block - Configure Datasets Here
# ==============================================================================

OUTPUT = 'sft.shisa-v2.1-EAGLE3'

def main():
    """
    Main function to define datasets, process them, merge, shuffle, and save.
    Edit this function directly to add/configure datasets.
    """
    overall_start_time = time.time()
    processed_datasets = [] # List to hold the results of loading functions
    dataset_summaries = []  # Track per-dataset counts and sampling info

    print("--- Starting Dataset Processing ---")

    # --- Define Datasets to Process ---
    # Each call to a load_dataset_* function should return a Dataset
    # object with (at least) a 'conversations' column, or None if failed.

    datasets_config = [
        # 2025-08-12: 1.18K rows - Proper Politeness
        {
            "dataset_path": "shisa-ai/shisa-politeness-dataset",
            "field_messages": "conversations",
            "split": "train",
        },
        # 2025-08-07: 360 rows - 10X Shisa.AI ID, hard-coded behaviors
        {
            "dataset_path": "shisa-ai/shisa-hardcoded-set",
            "field_messages": "conversations",
            "split": "train",
        },
        # 2025-06-18: 12.8K/51.2K rows of Chotto formatted multi-turn translations
        {
            "dataset_path": "shisa-ai/chotto_translation_set_sft",
            "field_messages": "conversations",
            "split": "train[:10%]",
        },
        # 2025-06-17: 181K rows - Latest (+Shisa V2 405B) rejection sampled version of primary dataset
        {
            "dataset_path": "shisa-ai/shisa-v2.1-sharegpt",
            "field_messages": "conversations",
            "split": "train",
            "sample_pct": 16,  # Post-shuffle percent to avoid category bias
        },
        # Chinese instruction dataset (instruction/output format)
        {
            "dataset_path": "m-a-p/COIG-CQIA",
            "loader": "instruction_output_multi",
            "config_names": [
                "chinese_traditional",
                "coig_pc",
                "exam",
                "finance",
                "douban",
                "human_value",
                "logi_qa",
                "ruozhiba",
                "segmentfault",
                "wiki",
                "wikihow",
                "xhs",
                "zhihu",
            ],
            "instruction_field": "instruction",
            "input_field": "input",
            "output_field": "output",
            "split": "train",
            "sample_pct": 20,  # Post-shuffle percent to avoid category bias
        },
        # Multilingual sentences (language-code configs)
        {
            "dataset_path": "agentlans/high-quality-multilingual-sentences",
            "loader": "sentence_multi",
            "config_names": [
                "ar",
                "az",
                "bg",
                "bn",
                "ca",
                "cs",
                "da",
                "de",
                "el",
                "en",
                "es",
                "et",
                "fa",
                "fi",
                "fr",
                "he",
                "hi",
                "hu",
                "hy",
                "id",
                "is",
                "it",
                "ja",
                "ka",
                "kk",
                "ko",
                "lt",
                "lv",
                "mk",
                "ml",
                "mr",
                "ms",
                "ne",
                "nl",
                "no",
                "pl",
                "pt",
                "ro",
                "ru",
                "sk",
                "sl",
                "sq",
                "sr",
                "sv",
                "ta",
                "th",
                "tr",
                "uk",
                "ur",
                "vi",
                "zh",
            ],
            "text_field": "text",
            "split": "train",
            "sample_count_per_config": 200,
        },
        {
            "dataset_path": "shisa-ai/shisa-v2-roleplaying-sft",
            "field_messages": "conversations",
            "split": "train[:5%]",
        },
        {
            "dataset_path": "shisa-ai/translation_set_april_6",
            "field_messages": "conversations",
            "split": "train[:5%]",
        },
        {
            "dataset_path": "shisa-ai/rewild-set-deepseek-subset",
            "field_messages": "conversations",
            "split": "train[:8%]",
        },
        {
            "dataset_path": "shisa-ai/magpie-ultra-set",
            "field_messages": "conversations",
            "split": "train[:2%]",
        },
        {
            "dataset_path": "shisa-ai/magpie-advanced-questions-set",
            "field_messages": "conversations",
            "split": "train[:2%]",
        },
        {
            "dataset_path": "shisa-ai/japan-magpie-set",
            "field_messages": "conversations",
            "split": "train[:5%]",
        },
        {
            "dataset_path": "shisa-ai/shisa-v2-instruction-following-sft",
            "field_messages": "conversations",
            "split": "train[:10%]",
        },
    ]

    def apply_post_shuffle_sample(ds, sample_pct, dataset_name):
        if sample_pct is None:
            return ds
        try:
            pct = float(sample_pct)
        except (TypeError, ValueError):
            print(f"  WARNING: Invalid sample_pct for {dataset_name}: {sample_pct}. Skipping.")
            return ds
        if pct <= 0:
            print(f"  WARNING: sample_pct <= 0 for {dataset_name}. Skipping.")
            return ds
        if pct > 1:
            pct = pct / 100.0
        if pct > 1:
            print(f"  WARNING: sample_pct > 100 for {dataset_name}. Skipping.")
            return ds
        print(f"  Post-shuffle sampling {dataset_name}: {pct * 100:.2f}%")
        ds = ds.shuffle(seed=GLOBAL_SEED)
        target_size = int(len(ds) * pct)
        if target_size <= 0 and len(ds) > 0:
            target_size = 1
        return ds.select(range(target_size))

    for dsc in datasets_config:
        ds = None
        loader = dsc.get("loader", "conversation")
        if loader == "conversation":
            ds = load_dataset_conversation(
                dataset_path = dsc['dataset_path'],
                field_messages = dsc['field_messages'],
                split = dsc['split'],
                role_map={ # Map *source* role names to 'user', 'assistant', 'system'
                     "system": ["system"],
                     "user": ["user", "human"],
                     "assistant": ["gpt", "assistant", "model"]
                },
                # fields=['conversations', 'id'], # Optionally keep other source fields like 'id'
                # shuffle_seed=123, # Optional: shuffle only this dataset before merge
            )
        elif loader == "instruction_output":
            ds = load_dataset_instruction_output(
                dataset_path = dsc['dataset_path'],
                instruction_field = dsc.get("instruction_field", "instruction"),
                input_field = dsc.get("input_field", "input"),
                output_field = dsc.get("output_field", "output"),
                split = dsc['split'],
                # fields=['conversations', 'category'], # Optionally keep other fields
            )
        elif loader == "instruction_output_multi":
            ds = load_dataset_instruction_output_multi(
                dataset_path = dsc['dataset_path'],
                config_names = dsc.get("config_names", []),
                instruction_field = dsc.get("instruction_field", "instruction"),
                input_field = dsc.get("input_field", "input"),
                output_field = dsc.get("output_field", "output"),
                split = dsc['split'],
                # fields=['conversations', 'category'], # Optionally keep other fields
            )
        elif loader == "sentence_multi":
            ds = load_dataset_sentence_multi(
                dataset_path = dsc['dataset_path'],
                config_names = dsc.get("config_names", []),
                text_field = dsc.get("text_field", "text"),
                split = dsc['split'],
                sample_count_per_config = dsc.get("sample_count_per_config"),
            )
        else:
            print(f"  WARNING: Unknown loader '{loader}' for {dsc.get('dataset_path')}. Skipping.")
        if ds:
            ds = apply_post_shuffle_sample(ds, dsc.get("sample_pct"), dsc["dataset_path"])
            processed_datasets.append(ds)
            dataset_summaries.append({
                "dataset_path": dsc.get("dataset_path"),
                "config_names": dsc.get("config_names"),
                "split": dsc.get("split"),
                "sample_pct": dsc.get("sample_pct"),
                "sample_count_per_config": dsc.get("sample_count_per_config"),
                "rows_raw": len(ds),
                "rows_final": None,
            })



    def rebuild_dataset_with_clean_schema(dataset):
        """
        Completely rebuilds a dataset with a clean schema, removing any traces of turn_identifier
        by creating an entirely new dataset.
        """
        from datasets import Dataset
        import pandas as pd

        # Extract just the data we need with the exact structure we want
        clean_data = []

        for example in dataset:
            conversations = example.get(DEFAULT_CONVERSATION_FIELD, [])
            clean_conversations = []

            for turn in conversations:
                if isinstance(turn, dict) and "role" in turn and "content" in turn:
                    # Only keep the role and content fields, nothing else
                    clean_conversations.append({
                        "role": turn["role"],
                        "content": turn["content"]
                    })

            # Only add examples with non-empty conversations
            if clean_conversations:
                clean_data.append({DEFAULT_CONVERSATION_FIELD: clean_conversations})

        # Create a brand new dataset with the clean data
        df = pd.DataFrame(clean_data)
        return Dataset.from_pandas(df)

    print("Rebuilding datasets with clean schema...")
    clean_datasets = []

    for i, ds in enumerate(processed_datasets):
        print(f"Rebuilding dataset {i+1}...")
        clean_ds = rebuild_dataset_with_clean_schema(ds)
        print(f"  Dataset {i+1} rebuilt: {len(clean_ds)} examples")
        clean_datasets.append(clean_ds)
        if i < len(dataset_summaries):
            dataset_summaries[i]["rows_final"] = len(clean_ds)

    # Now try concatenation with the clean datasets
    print("\n--- Finalizing Dataset ---")
    print(f"Concatenating {len(clean_datasets)} clean datasets...")

    try:
        final_ds = concatenate_datasets(clean_datasets)
        print(f"Final dataset created with {len(final_ds)} examples.")
    except Exception as e:
        print(f"ERROR during concatenation: {e}")

    '''
    # --- Merge, Shuffle, Save ---
    print("\n--- Finalizing Dataset ---")

    if not processed_datasets:
        print("ERROR: No datasets were successfully processed. Exiting.")
        return

    # 1. Merge Datasets
    print(f"Concatenating {len(processed_datasets)} processed datasets...")
    try:
        # Ensure all datasets have the same features before concatenating
        # This relies on load_dataset_* functions correctly setting the 'fields'
        final_ds = concatenate_datasets(processed_datasets)
    except ValueError as e:
         print(f"\nERROR during concatenation: {e}")
         print("Ensure all loaded datasets have the same columns (check 'fields' argument).")
         print("Columns per dataset:")
         for i, ds in enumerate(processed_datasets):
             print(f"  Dataset {i+1} ({ds.builder_name if hasattr(ds, 'builder_name') else 'Unknown'}): {ds.column_names}")
         return
    print(f"Concatenated dataset size: {len(final_ds)} examples.")
    '''

    # 2. Global Shuffle (Optional but recommended)
    print(f"Shuffling the final dataset with seed {GLOBAL_SEED}...")
    final_ds = final_ds.shuffle(seed=GLOBAL_SEED)
    print("Shuffling complete.")

    # 2.5. Add unique IDs
    print("Adding unique IDs...")
    final_ds = final_ds.map(lambda example, idx: {**example, "id": str(idx)}, with_indices=True)
    print("ID generation complete.")

    # 3. Save Final Dataset
    output_filename_base = OUTPUT
    save_format = "jsonl"  # 'jsonl' or 'disk'
    num_shards = 1        # Use > 1 for multiple output files (esp. for jsonl)

    output_path = f"{output_filename_base}.{save_format}" if save_format == "jsonl" else output_filename_base
    print(f"Saving final dataset ({len(final_ds)} examples) as {save_format} to '{output_path}'...")

    try:
        if save_format == "jsonl":
            # Use datasets' built-in JSON writing with sharding support
            final_ds.to_json(
                output_path,
                lines=True,
                force_ascii=False,
                # num_shards=num_shards if num_shards > 1 else None # num_shards=None or 1 writes single file
            )
        elif save_format == "disk":
            final_ds.save_to_disk(output_path, num_shards=num_shards if num_shards > 1 else None)
        else:
            print(f"ERROR: Unknown save format: {save_format}")
            return # Exit if format is unknown
    except Exception as e:
        print(f"ERROR during saving: {e}")
        return # Exit on save error

    overall_time = time.time() - overall_start_time
    print(f"\n--- Script finished in {overall_time:.2f} seconds ---")
    print(f"Output saved to '{output_path}' {'(sharded)' if num_shards > 1 else ''}")

    # 4. Dataset Summary
    total_rows = len(final_ds)
    print("\n--- Dataset Summary ---")
    for entry in dataset_summaries:
        rows = entry["rows_final"] if entry["rows_final"] is not None else entry["rows_raw"]
        pct = (rows / total_rows * 100.0) if total_rows else 0.0
        sample_pct = entry.get("sample_pct")
        sample_note = f", sample_pct={sample_pct}" if sample_pct is not None else ""
        sample_count = entry.get("sample_count_per_config")
        count_note = (
            f", sample_count_per_config={sample_count}" if sample_count is not None else ""
        )
        config_names = entry.get("config_names") or []
        config_note = f", configs={','.join(config_names)}" if config_names else ""
        print(
            f"{entry['dataset_path']} | split={entry['split']}{sample_note}{count_note}{config_note} | "
            f"rows={rows} | {pct:.2f}%"
        )
    print(f"TOTAL | rows={total_rows} | 100.00%")


# ==============================================================================
# Configuration Constants
# ==============================================================================

GLOBAL_SEED = 42
DEFAULT_CONVERSATION_FIELD = "conversations" # Target field name
DEFAULT_SPLIT = "train"

# ==============================================================================
# Loading Functions for Specific Formats
# ==============================================================================

def _validate_and_select_columns(ds: Dataset, requested_fields: list, dataset_name: str) -> Dataset | None:
    """Internal helper to ensure final columns match requested fields."""
    final_columns = [f for f in requested_fields if f in ds.column_names]
    missing = set(requested_fields) - set(final_columns)
    if missing:
        print(f"  WARNING for {dataset_name}: Requested fields {missing} not found. Keeping only: {final_columns}")
    if not final_columns:
        print(f"  ERROR for {dataset_name}: No requested fields found in the processed dataset.")
        return None
    if set(final_columns) != set(ds.column_names):
        try:
            return ds.select_columns(final_columns)
        except Exception as e:
            print(f"  ERROR selecting columns {final_columns} for {dataset_name}: {e}")
            return None
    return ds # Columns already match

def load_dataset_conversation(
    dataset_path: str,
    field_messages: str = "conversations",
    role_map: dict = None, # Map source roles -> standard roles ('user', 'assistant', 'system')
    fields: list = None,   # List of *final* columns to keep (must include DEFAULT_CONVERSATION_FIELD)
    split: str = DEFAULT_SPLIT,
    shuffle_seed: int = None,
    **load_kwargs # Pass extra args like data_files to load_dataset
) -> Dataset | None:
    """
    Loads and converts a ShareGPT-like dataset to OpenAI format.
    Expected input: A field (`field_messages`) containing a list of turns.
    Each turn should have role ('role' or 'from') and content ('content' or 'value').
    """
    dataset_name = os.path.basename(dataset_path)
    print(f"Processing ShareGPT: {dataset_name} (Split: {split})")

    # Ensure the target conversation field is always included in the final output
    if fields is None:
        fields = [DEFAULT_CONVERSATION_FIELD]
    elif DEFAULT_CONVERSATION_FIELD not in fields:
        fields.append(DEFAULT_CONVERSATION_FIELD)

    # Invert role_map for efficient lookup: {'human': 'user', 'gpt': 'assistant', ...}
    if role_map is None: role_map = {} # Handle potential None
    inverted_role_map = {val.lower(): key for key, values in role_map.items() for val in values}
    # Add default mappings if not provided, useful for simple cases
    if 'user' not in inverted_role_map: inverted_role_map['user'] = 'user'
    if 'human' not in inverted_role_map: inverted_role_map['human'] = 'user'
    if 'assistant' not in inverted_role_map: inverted_role_map['assistant'] = 'assistant'
    if 'gpt' not in inverted_role_map: inverted_role_map['gpt'] = 'assistant'
    if 'system' not in inverted_role_map: inverted_role_map['system'] = 'system'


    try:
        ds = load_dataset(dataset_path, split=split, **load_kwargs)
    except Exception as e:
        print(f"  ERROR loading {dataset_name}: {e}")
        return None

    # Check if source message field exists
    if field_messages not in ds.column_names:
        print(f"  ERROR: Source field '{field_messages}' not found in {dataset_name}. Available: {ds.column_names}")
        return None

    def format_sharegpt_conversation(example):
        raw_conv = example.get(field_messages)
        processed_conv = []
        if isinstance(raw_conv, list):
            for turn in raw_conv:
                if not isinstance(turn, dict): continue # Skip invalid turns

                role_val = turn.get('role', turn.get('from'))
                content_val = turn.get('content', turn.get('value'))

                if role_val is not None and content_val is not None:
                    standard_role = inverted_role_map.get(str(role_val).lower())
                    if standard_role:
                        processed_conv.append({"role": standard_role, "content": str(content_val)})
                    # else: # Optionally warn about unmapped roles
                    #    print(f"  WARNING: Unmapped role '{role_val}' in {dataset_name}")

        result = {DEFAULT_CONVERSATION_FIELD: processed_conv}
        # Add other requested fields from the original example if they exist
        for f in fields:
            if f != DEFAULT_CONVERSATION_FIELD and f in example:
                result[f] = example[f]
        return result

    # Determine columns to remove: all except the source messages field and any other fields requested to be kept
    columns_to_remove = [
        col for col in ds.column_names
        if col != field_messages and col not in fields
    ]
    # Handle edge case: if field_messages is same as DEFAULT_CONVERSATION_FIELD and not in fields list initially
    if field_messages == DEFAULT_CONVERSATION_FIELD and field_messages not in fields:
         columns_to_remove = [col for col in ds.column_names if col not in fields] # Keep only final fields

    print(f"  Columns to remove for {dataset_name}: {columns_to_remove}")

    # columns_to_potentially_keep = [field_messages] + [f for f in fields if f != DEFAULT_CONVERSATION_FIELD]
    # columns_to_remove = [col for col in ds.column_names if col not in columns_to_potentially_keep]

    ds = ds.map(
        format_sharegpt_conversation,
        remove_columns=columns_to_remove,
        desc=f"Formatting {dataset_name}",
        # load_from_cache_file=False,
    )

    # Filter out examples with empty conversations after processing
    initial_count = len(ds)
    ds = ds.filter(lambda ex: len(ex.get(DEFAULT_CONVERSATION_FIELD, [])) > 0)
    if len(ds) < initial_count:
        print(f"  Filtered out {initial_count - len(ds)} examples with empty conversations.")

    if shuffle_seed is not None:
        print(f"  Shuffling {dataset_name} with seed {shuffle_seed}")
        ds = ds.shuffle(seed=shuffle_seed)

    # Validate and select final columns
    ds = _validate_and_select_columns(ds, fields, dataset_name)
    if ds is None: return None # Error during column selection

    print(f"  Finished {dataset_name}. Resulting examples: {len(ds)}")
    return ds


def load_dataset_instruction_output(
    dataset_path: str,
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
    fields: list = None,   # List of *final* columns to keep (must include DEFAULT_CONVERSATION_FIELD)
    split: str = DEFAULT_SPLIT,
    shuffle_seed: int = None,
    config_name: str | None = None,
    **load_kwargs
) -> Dataset | None:
    """
    Loads a dataset with instruction/output fields and converts it to OpenAI format.
    Expected input: columns for instruction and output, with optional input field.
    """
    dataset_name = os.path.basename(dataset_path)
    if config_name is None:
        config_name = load_kwargs.get("name")
    config_note = f", Config: {config_name}" if config_name else ""
    print(f"Processing Instruction/Output: {dataset_name} (Split: {split}{config_note})")

    if fields is None:
        fields = [DEFAULT_CONVERSATION_FIELD]
    elif DEFAULT_CONVERSATION_FIELD not in fields:
        fields.append(DEFAULT_CONVERSATION_FIELD)

    try:
        ds = load_dataset(dataset_path, split=split, **load_kwargs)
    except Exception as e:
        print(f"  ERROR loading {dataset_name}: {e}")
        return None

    missing = [f for f in [instruction_field, output_field] if f not in ds.column_names]
    if missing:
        print(f"  ERROR: Missing required fields {missing} in {dataset_name}. Available: {ds.column_names}")
        return None

    use_input = input_field in ds.column_names if input_field else False

    def format_instruction_output(example):
        instruction = example.get(instruction_field)
        output = example.get(output_field)
        if instruction is None or output is None:
            return {DEFAULT_CONVERSATION_FIELD: []}

        user_content = str(instruction)
        if use_input:
            input_val = example.get(input_field)
            if input_val is not None:
                input_text = str(input_val).strip()
                if input_text:
                    user_content = f"{user_content}\n\n{input_text}"

        conversations = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": str(output)},
        ]

        result = {DEFAULT_CONVERSATION_FIELD: conversations}
        for f in fields:
            if f != DEFAULT_CONVERSATION_FIELD and f in example:
                result[f] = example[f]
        return result

    columns_to_remove = [col for col in ds.column_names if col not in fields]
    ds = ds.map(
        format_instruction_output,
        remove_columns=columns_to_remove,
        desc=f"Formatting {dataset_name}",
    )

    initial_count = len(ds)
    ds = ds.filter(lambda ex: len(ex.get(DEFAULT_CONVERSATION_FIELD, [])) > 0)
    if len(ds) < initial_count:
        print(f"  Filtered out {initial_count - len(ds)} examples with empty conversations.")

    if shuffle_seed is not None:
        print(f"  Shuffling {dataset_name} with seed {shuffle_seed}")
        ds = ds.shuffle(seed=shuffle_seed)

    ds = _validate_and_select_columns(ds, fields, dataset_name)
    if ds is None:
        return None

    print(f"  Finished {dataset_name}. Resulting examples: {len(ds)}")
    return ds


def load_dataset_instruction_output_multi(
    dataset_path: str,
    config_names: list,
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
    fields: list = None,
    split: str = DEFAULT_SPLIT,
    shuffle_seed: int = None,
    **load_kwargs
) -> Dataset | None:
    """
    Loads multiple configs from the same dataset and concatenates them.
    """
    if not config_names:
        print(f"  ERROR: No config_names provided for {dataset_path}")
        return None

    datasets = []
    for config_name in config_names:
        ds = load_dataset_instruction_output(
            dataset_path=dataset_path,
            instruction_field=instruction_field,
            input_field=input_field,
            output_field=output_field,
            fields=fields,
            split=split,
            shuffle_seed=shuffle_seed,
            config_name=config_name,
            name=config_name,
            **load_kwargs,
        )
        if ds:
            datasets.append(ds)

    if not datasets:
        print(f"  ERROR: No configs loaded for {dataset_path}")
        return None
    if len(datasets) == 1:
        return datasets[0]
    try:
        return concatenate_datasets(datasets)
    except Exception as e:
        print(f"  ERROR concatenating configs for {dataset_path}: {e}")
        return None


def load_dataset_sentence_multi(
    dataset_path: str,
    config_names: list,
    text_field: str = "text",
    fields: list = None,
    split: str = DEFAULT_SPLIT,
    shuffle_seed: int = None,
    sample_count_per_config: int | None = None,
    **load_kwargs
) -> Dataset | None:
    """
    Loads sentence-only datasets for multiple language configs and converts them to OpenAI format.
    """
    if not config_names:
        print(f"  ERROR: No config_names provided for {dataset_path}")
        return None

    def fetch_sentence_samples_http(config_name: str, sample_count: int) -> Dataset | None:
        try:
            import urllib.request
            import urllib.parse
        except Exception as e:
            print(f"  ERROR: HTTP fallback imports failed for {dataset_path}/{config_name}: {e}")
            return None

        base_url = "https://datasets-server.huggingface.co"
        info_params = urllib.parse.urlencode({
            "dataset": dataset_path,
            "config": config_name,
        })
        info_url = f"{base_url}/info?{info_params}"
        try:
            with urllib.request.urlopen(info_url, timeout=30) as resp:
                info_data = json.load(resp)
        except Exception as e:
            print(f"  ERROR: HTTP info fetch failed for {dataset_path}/{config_name}: {e}")
            return None

        splits = info_data.get("dataset_info", {}).get("splits", {})
        split_info = splits.get(split, {})
        num_examples = split_info.get("num_examples")
        if not isinstance(num_examples, int) or num_examples <= 0:
            print(f"  ERROR: Invalid split size for {dataset_path}/{config_name}: {num_examples}")
            return None

        sample_count = min(sample_count, num_examples)
        seed = GLOBAL_SEED + int(hashlib.md5(config_name.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        max_offset = max(0, num_examples - sample_count)
        offset = rng.randint(0, max_offset) if max_offset > 0 else 0

        rows_params = urllib.parse.urlencode({
            "dataset": dataset_path,
            "config": config_name,
            "split": split,
            "offset": offset,
            "length": sample_count,
        })
        rows_url = f"{base_url}/rows?{rows_params}"
        try:
            with urllib.request.urlopen(rows_url, timeout=60) as resp:
                rows_data = json.load(resp)
        except Exception as e:
            print(f"  ERROR: HTTP rows fetch failed for {dataset_path}/{config_name}: {e}")
            return None

        rows = rows_data.get("rows", [])
        texts = []
        for row in rows:
            row_data = row.get("row", {})
            text = row_data.get(text_field)
            if text is not None:
                texts.append({text_field: text})

        if not texts:
            print(f"  ERROR: No rows returned for {dataset_path}/{config_name}")
            return None
        return Dataset.from_list(texts)

    if fields is None:
        fields = [DEFAULT_CONVERSATION_FIELD]
    elif DEFAULT_CONVERSATION_FIELD not in fields:
        fields.append(DEFAULT_CONVERSATION_FIELD)

    datasets = []
    for config_name in config_names:
        dataset_name = os.path.basename(dataset_path)
        print(f"Processing Sentences: {dataset_name} (Split: {split}, Config: {config_name})")
        try:
            ds = load_dataset(dataset_path, config_name, split=split, **load_kwargs)
        except Exception as e:
            print(f"  ERROR loading {dataset_name}/{config_name}: {e}")
            if sample_count_per_config is None:
                continue
            if os.environ.get("HF_DATASETS_OFFLINE") or os.environ.get("HF_HUB_OFFLINE"):
                print(
                    f"  ERROR: Offline mode enabled; cannot use HTTP fallback for "
                    f"{dataset_name}/{config_name}. Install python zstandard to load locally."
                )
                continue
            print(f"  Attempting HTTP fallback for {dataset_name}/{config_name}...")
            ds = fetch_sentence_samples_http(config_name, int(sample_count_per_config))
            if ds is None:
                continue

        if text_field not in ds.column_names:
            print(
                f"  ERROR: Source field '{text_field}' not found in {dataset_name}/{config_name}. "
                f"Available: {ds.column_names}"
            )
            continue

        def format_sentence(example):
            text = example.get(text_field)
            if text is None:
                return {DEFAULT_CONVERSATION_FIELD: []}
            user_content = f"LANG={config_name}"
            conversations = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": str(text)},
            ]
            result = {DEFAULT_CONVERSATION_FIELD: conversations}
            for f in fields:
                if f != DEFAULT_CONVERSATION_FIELD and f in example:
                    result[f] = example[f]
            return result

        columns_to_remove = [col for col in ds.column_names if col not in fields]
        ds = ds.map(
            format_sentence,
            remove_columns=columns_to_remove,
            desc=f"Formatting {dataset_name}/{config_name}",
        )

        initial_count = len(ds)
        ds = ds.filter(lambda ex: len(ex.get(DEFAULT_CONVERSATION_FIELD, [])) > 0)
        if len(ds) < initial_count:
            print(
                f"  Filtered out {initial_count - len(ds)} examples with empty conversations."
            )

        if sample_count_per_config is not None:
            target = min(int(sample_count_per_config), len(ds))
            if target > 0:
                seed = GLOBAL_SEED if shuffle_seed is None else shuffle_seed
                ds = ds.shuffle(seed=seed).select(range(target))
                print(f"  Sampled {target} rows from {dataset_name}/{config_name}")

        if shuffle_seed is not None and sample_count_per_config is None:
            print(f"  Shuffling {dataset_name}/{config_name} with seed {shuffle_seed}")
            ds = ds.shuffle(seed=shuffle_seed)

        ds = _validate_and_select_columns(ds, fields, f"{dataset_name}/{config_name}")
        if ds is None:
            continue

        print(f"  Finished {dataset_name}/{config_name}. Resulting examples: {len(ds)}")
        datasets.append(ds)

    if not datasets:
        print(f"  ERROR: No configs loaded for {dataset_path}")
        return None
    if len(datasets) == 1:
        return datasets[0]
    try:
        return concatenate_datasets(datasets)
    except Exception as e:
        print(f"  ERROR concatenating configs for {dataset_path}: {e}")
        return None



# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    # Set global random seed for reproducibility if needed by other libraries
    random.seed(GLOBAL_SEED)
    main()
