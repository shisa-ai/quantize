#!/usr/bin/env python
"""
quantize_llmcompressor_w8a8.py
--------------------------------
Quantise a Llama-3 405 B checkpoint to **W8 A8** (SmoothQuant + GPTQ) using
LLM Compressor, with correct ShareGPT → chat-template conversion.

Example
=======
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python quantize_llmcompressor_w8a8.py \
    --model-id  shisa-ai/shisa-v2-ubitus-llama3.1-405b \
    --out-dir   shisa-v2-ubitus-llama3.1-405b-W8A8-G32 \
    --calib-samples 512 \
    --group-size 32
"""

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier


# --------------------------------------------------------------------- #
#  ShareGPT → HF chat-list helper                                       #
# --------------------------------------------------------------------- #
def sharegpt_to_chat(conversations):
    """
    Convert ShareGPT's [{"from": "human"/"gpt", "value": "..."}] list into
    the [{"role": "user"/"assistant", "content": "..."}] format expected by
    tokenizer.apply_chat_template().
    """
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "system": "system",
        "user": "user",
        "assistant": "assistant",
    }
    out = []
    for turn in conversations:
        if "from" in turn and "value" in turn:
            role = role_map.get(turn["from"], turn["from"])
            out.append({"role": role, "content": turn["value"]})
        elif "role" in turn and "content" in turn:
            out.append(turn)  # already in the right structure
    return out


# --------------------------------------------------------------------- #
#  CLI                                                                  #
# --------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id",     required=True)
    ap.add_argument("--out-dir",      required=True)
    ap.add_argument("--dataset-id",   default="shisa-ai/shisa-v2-sharegpt")
    ap.add_argument("--dataset-split",default="train")
    ap.add_argument("--calib-samples",type=int, default=512)
    ap.add_argument("--max-seq-len",  type=int, default=2048)
    ap.add_argument("--group-size",   type=int, default=32)
    ap.add_argument("--sq-strength",  type=float, default=0.8)
    return ap.parse_args()


# --------------------------------------------------------------------- #
#  Main                                                                 #
# --------------------------------------------------------------------- #
def main():
    args = parse_args()

    print(f"[1/4] Loading tokenizer + model for {args.model_id}")
    # Try loading with fast tokenizer first (better chat template support)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            use_fast=True,  # Use fast tokenizer for better chat template support
            trust_remote_code=True,
        )
        print(f"✓ Loaded fast tokenizer: {type(tokenizer).__name__}")
    except Exception as e:
        print(f"Fast tokenizer failed: {e}, trying LlamaTokenizer...")
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            args.model_id,
            trust_remote_code=True,
        )
        print(f"✓ Loaded tokenizer: {type(tokenizer).__name__}")
    
    # Debug: print what we have
    print(f"  - Has apply_chat_template: {hasattr(tokenizer, 'apply_chat_template')}")
    print(f"  - Chat template exists: {tokenizer.chat_template is not None}")
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        print(f"  - Chat template length: {len(tokenizer.chat_template)} chars")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    # Verify chat template is accessible
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(
            "This tokenizer does not expose `apply_chat_template`. "
            "The tokenizer class might be incorrectly specified."
        )
    
    # Double-check the chat template is actually set
    if tokenizer.chat_template is None:
        raise RuntimeError(
            "Chat template is None despite being in config. "
            "Check the tokenizer_config.json file."
        )

    # -------------------------------------------------- #
    #  Build the calibration set                         #
    # -------------------------------------------------- #
    print(f"[2/4] Loading {args.calib_samples} ShareGPT rows …")
    ds = (
        load_dataset(args.dataset_id, split=args.dataset_split)
        .shuffle(seed=42)
        .select(range(args.calib_samples))
    )

    def preprocess(row):
        chat = sharegpt_to_chat(row["conversations"])
        text = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    ds = ds.map(preprocess)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding=False,
            max_length=args.max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    # -------------------------------------------------- #
    #  Define the compression recipe                     #
    # -------------------------------------------------- #
    recipe = [
        SmoothQuantModifier(smoothing_strength=args.sq_strength),
        GPTQModifier(
            targets="Linear",
            scheme="W8A8",
            group_size=args.group_size,
            ignore=["lm_head"],  # keep logits layer fp16 for safety
        ),
    ]

    # -------------------------------------------------- #
    #  One-shot SmoothQuant + GPTQ                       #
    # -------------------------------------------------- #
    print(
        f"[3/4] Compressing with SQ={args.sq_strength}, "
        f"W8A8 g={args.group_size} …"
    )
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.calib_samples,
    )

    # -------------------------------------------------- #
    #  Save                                              #
    # -------------------------------------------------- #
    print(f"[4/4] Saving compressed checkpoint to {args.out_dir}")
    model.save_pretrained(args.out_dir, save_compressed=True)
    tokenizer.save_pretrained(args.out_dir)
    print("✅  Done – your W8A8 model is ready for vLLM!")

if __name__ == "__main__":
    main()
