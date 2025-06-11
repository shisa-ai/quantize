#!/usr/bin/env python
"""
quantize_llmcompressor_w8a8.py
--------------------------------
Quantise a Llama-3 405 B checkpoint to **W8 A8** (SmoothQuant + GPTQ) using
LLM Compressor, with correct ShareGPT → chat-template conversion.

Example
=======
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python quantize_llmcompressor_w8a8.py \
    --model-id  shisa-ai/shisa-v2-ubitus-llama3.1-405b \
    --out-dir   shisa-v2-ubitus-llama3.1-405b-W8A8-G32 \
    --calib-samples 512 \
    --group-size 32
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map

# Set memory allocation strategy
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


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
    ap.add_argument("--damping-factor", type=float, default=0.01,
                    help="GPTQ damping factor (RedHat used 0.01)")
    ap.add_argument("--num-gpus",     type=int, default=None,
                    help="Number of GPUs to use (default: all available)")
    ap.add_argument("--dtype",        default="bfloat16",
                    choices=["float16", "bfloat16"],
                    help="Model dtype for loading (default: bfloat16 for Llama 3)")
    ap.add_argument("--save-smoothquant", action="store_true",
                    help="Save model after SmoothQuant (before GPTQ)")
    ap.add_argument("--w8a16", action="store_true",
                    help="Use W8A16 quantization instead of W8A8 (like RedHat)")
    return ap.parse_args()


# --------------------------------------------------------------------- #
#  Main                                                                 #
# --------------------------------------------------------------------- #
def main():
    args = parse_args()

    # Get number of available GPUs
    available_gpus = torch.cuda.device_count()
    num_gpus = args.num_gpus or available_gpus
    if num_gpus > available_gpus:
        raise ValueError(f"Requested {num_gpus} GPUs but only {available_gpus} available")
    
    print(f"[INFO] Using {num_gpus} out of {available_gpus} available GPUs")
    
    # Estimate memory requirements
    if "405b" in args.model_id.lower():
        model_size_gb = 810  # 405B params * 2 bytes (bf16)
        print(f"\n⚠️  405B Model Detected!")
        print(f"   - Model size: ~{model_size_gb}GB in bfloat16")
        print(f"   - Total GPU memory: {num_gpus * 144}GB ({num_gpus} x 144GB)")
        print(f"   - With Hessian reservation, some CPU offloading may occur")
        print(f"   - This is normal but will slow down quantization\n")

    # Set dtype
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print(f"[1/4] Loading tokenizer + model for {args.model_id}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            use_fast=True,  # Use fast tokenizer for better chat template support
            trust_remote_code=True,
        )
        print(f"✓ Loaded fast tokenizer: {type(tokenizer).__name__}")
    except Exception as e:
        print(f"Fast tokenizer failed: {e}, trying LlamaTokenizer...")
        tokenizer = LlamaTokenizer.from_pretrained(
            args.model_id,
            trust_remote_code=True,
        )
        print(f"✓ Loaded tokenizer: {type(tokenizer).__name__}")
    
    # Calculate device map with Hessian memory reservation
    print(f"[INFO] Calculating device map with Hessian memory reservation...")
    device_map = calculate_offload_device_map(
        args.model_id,
        num_gpus=num_gpus,
        reserve_for_hessians=True,  # Critical for GPTQ!
        torch_dtype=torch_dtype,
    )
    
    print(f"[INFO] Loading model with device map...")
    
    # Load model with the calculated device map
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    # Debug: print device distribution
    print("[INFO] Model device distribution:")
    device_counts = {}
    for name, device in model.hf_device_map.items():
        device_counts[device] = device_counts.get(device, 0) + 1
    
    # Sort with mixed types (int for GPUs, str for cpu/disk)
    for device in sorted(device_counts.keys(), key=lambda x: (isinstance(x, str), x)):
        count = device_counts[device]
        device_name = f"GPU {device}" if isinstance(device, int) else device
        print(f"  - {device_name}: {count} modules")
    
    # Warn if CPU offloading is happening
    if "cpu" in device_counts or "disk" in device_counts:
        print("\n⚠️  WARNING: Some parameters are offloaded to CPU/disk.")
        print("   This may significantly slow down quantization.")
        print("   Consider reducing --calib-samples or using fewer GPUs.")
        print("\n   For 405B models, you might also try:")
        print("   - Use fewer calibration samples (e.g., --calib-samples 512)")
        print("   - Use W8A16 instead of W8A8 (modify the recipe)")
        print("   - Run on a system with more GPUs or more GPU memory")

    # Debug: print tokenizer info
    print(f"  - Has apply_chat_template: {hasattr(tokenizer, 'apply_chat_template')}")
    print(f"  - Chat template exists: {tokenizer.chat_template is not None}")
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        print(f"  - Chat template length: {len(tokenizer.chat_template)} chars")

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
    # First apply SmoothQuant
    smoothquant_modifier = SmoothQuantModifier(
        smoothing_strength=args.sq_strength
    )
    
    # Optionally save after SmoothQuant
    if args.save_smoothquant:
        print(f"[3a/4] Applying SmoothQuant only (strength={args.sq_strength})...")
        oneshot(
            model=model,
            dataset=ds,
            recipe=[smoothquant_modifier],
            max_seq_length=args.max_seq_len,
            num_calibration_samples=args.calib_samples,
        )
        sq_dir = args.out_dir + "-smoothquant"
        print(f"[INFO] Saving SmoothQuant model to {sq_dir}")
        model.save_pretrained(sq_dir)
        tokenizer.save_pretrained(sq_dir)
        print("[INFO] SmoothQuant saved. Continuing with GPTQ...")

    # Full recipe with GPTQ (following RedHat's approach)
    quantization_scheme = "W8A16" if args.w8a16 else "W8A8"
    recipe = [
        smoothquant_modifier,
        GPTQModifier(
            targets="Linear",
            scheme=quantization_scheme,
            group_size=args.group_size,
            ignore=["lm_head"],  # keep logits layer fp16 for safety
            percdamp=args.damping_factor,  # RedHat used 0.01 (1%)
        ),
    ]

    # -------------------------------------------------- #
    #  One-shot compression                              #
    # -------------------------------------------------- #
    print(
        f"[3/4] Compressing with SQ={args.sq_strength}, "
        f"{quantization_scheme} g={args.group_size}, damping={args.damping_factor} …"
    )
    
    # Clear cache before GPTQ
    torch.cuda.empty_cache()
    
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.calib_samples,
        save_compressed=True,  # Following the example
        output_dir=args.out_dir,
    )

    # -------------------------------------------------- #
    #  Final save (if not already saved by oneshot)      #
    # -------------------------------------------------- #
    print(f"[4/4] Quantization complete! Output saved to {args.out_dir}")
    print(f"✅  Done – your {quantization_scheme} model is ready for vLLM!")

if __name__ == "__main__":
    main()
