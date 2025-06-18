from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_ID = "shisa-ai/shisa-v2-llama3.1-405b"
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

ds = (
    load_dataset('shisa-ai/shisa-v2-sharegpt', split='train')
    .shuffle(seed=42)
)


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

def to_chat_text(sample):
    # sample["conversation"] is assumed to be a list of {"role": "...", "value": "..."} dicts
    # Replace with the exact field names in your dataset.
    conv = convert_sharegpt_to_chat_format(sample['conversations'])
    return tok.apply_chat_template(conv, tokenize=False)

with open("calibration_chat.txt", "w", encoding="utf-8") as f:
    for i, s in enumerate(ds):
        f.write(to_chat_text(s) + "\n")
        if i >= 4000:          # ~1 M tokens for 405 B
            break
