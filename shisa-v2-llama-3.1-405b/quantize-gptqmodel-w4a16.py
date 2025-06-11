"""
quantize_w4a16_gptqmodel.py - FIXED VERSION WITH PROPER SHAREGPT HANDLING
Usage:
  python quantize_w4a16_gptqmodel.py \
      --model-id shisa-ai/shisa-v2-llama3.1-405b \
      --out-dir shisa-v2-llama3.1-405b-W4A16
"""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig

def convert_sharegpt_to_chat_format(conversations):
    """
    Convert ShareGPT format to standard chat format expected by chat templates.
    ShareGPT typically has: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
    Chat templates expect: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    chat_format = []
    
    # Handle both 'from'/'value' and 'role'/'content' formats
    for conv in conversations:
        if "from" in conv and "value" in conv:
            # ShareGPT format
            role_map = {
                "human": "user",
                "gpt": "assistant",
                "system": "system",
                "user": "user",  # Sometimes already in this format
                "assistant": "assistant"
            }
            role = role_map.get(conv["from"], conv["from"])
            chat_format.append({
                "role": role,
                "content": conv["value"]
            })
        elif "role" in conv and "content" in conv:
            # Already in chat format
            chat_format.append(conv)
        else:
            print(f"Warning: Unknown conversation format: {conv}")
            continue
    
    return chat_format

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--bits', type=int, default=4)
    ap.add_argument('--calib-samples', type=int, default=2048)
    ap.add_argument('--group-size', type=int, default=128)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-length', type=int, default=2048, help="Maximum sequence length for calibration")
    args = ap.parse_args()

    print(f"Loading tokenizer for: {args.model_id}")

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    print(f"Tokenizer loaded: {type(tok)}")
    
    # Set pad token if not already set
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        print(f"Set pad_token to: {tok.pad_token}")

    # --- prepare calibration data -----------------
    print(f"Loading {args.calib_samples} calibration samples...")
    ds = load_dataset("shisa-ai/shisa-v2-sharegpt", split="train") \
           .shuffle(seed=42).select(range(args.calib_samples))

    print("Processing calibration samples...")
    print("First sample structure for debugging:")
    if len(ds) > 0:
        print(f"Keys in dataset: {ds[0].keys()}")
        if "conversations" in ds[0]:
            print(f"First conversation sample: {ds[0]['conversations'][:2]}")  # Show first 2 turns

    calib = []
    skipped = 0
    
    for i, x in enumerate(ds):
        if i % 500 == 0:
            print(f"Processing sample {i}/{len(ds)}")
        
        try:
            # Convert ShareGPT format to chat format
            if "conversations" in x:
                chat_format = convert_sharegpt_to_chat_format(x["conversations"])
                
                # Skip empty or single-turn conversations
                if len(chat_format) < 2:
                    skipped += 1
                    continue
                
                # Apply chat template
                text = tok.apply_chat_template(
                    chat_format,
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                # Tokenize to check length
                tokens = tok(text, return_tensors="pt", truncation=True, max_length=args.max_length)
                
                # Skip if too short
                if tokens.input_ids.shape[1] < 10:
                    skipped += 1
                    continue
                
                calib.append(text)
            else:
                print(f"Warning: No 'conversations' field in sample {i}")
                skipped += 1
                
        except Exception as e:
            print(f"Warning: Error processing sample {i}: {e}")
            skipped += 1
            continue

    print(f"Processed {len(calib)} calibration samples (skipped {skipped})")
    
    if len(calib) == 0:
        raise ValueError("No valid calibration samples found! Check dataset format.")

    # --- load fp16 model + config -----------------
    print("Setting up quantization config...")
    qcfg = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=True,       # Improves quality
        sym=True,            # Symmetric quantization
        true_sequential=True # Sequential layer quantization
    )

    print(f"Loading model: {args.model_id}")
    model = GPTQModel.load(
        args.model_id,
        qcfg,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # --- quantize ---------------------------------
    print(f"Starting quantization with batch_size={args.batch_size}...")
    print(f"Using {len(calib)} calibration samples")
    
    model.quantize(calib, batch_size=args.batch_size)

    # --- save -------------------------------------
    print(f"Saving quantized model to: {args.out_dir}")
    model.save(args.out_dir)
    
    # Also save tokenizer
    print(f"Saving tokenizer to: {args.out_dir}")
    tok.save_pretrained(args.out_dir)
    
    print("Quantization complete!")

if __name__ == "__main__":
    main()
