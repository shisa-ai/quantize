from datasets import load_dataset
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig
import sys
import torch

# Model and dataset configuration
model_id = "cyberagent/Mistral-Nemo-Japanese-Instruct-2408"
quant_path = "Mistral-Nemo-Japanese-Instruct-2408-GPTQ-W4A16-g32-noact"
dataset_id = "augmxnt/ultra-orca-boros-en-ja-v1"
num_samples = 1024

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

def convert_sharegpt_to_chat(example):
    """Convert ShareGPT format to chat format with proper role mapping."""
    role_mapping = {
        'system': 'system',
        'human': 'user',
        'gpt': 'assistant'
    }
    
    formatted_conversations = [
        {
            "role": role_mapping[msg['from']],
            "content": msg['value']
        }
        for msg in example['conversations']
    ]
    
    # Apply the model's chat template and convert to tensor
    processed = tokenizer.apply_chat_template(
        formatted_conversations,
        tokenize=True,
        return_tensors="pt"
    )
    
    # Return as dictionary with 'input_ids' key
    return {"input_ids": processed[0]}  # Take first element to remove batch dimension

def load_calibration_dataset(debug=False):
    """Load and preprocess the calibration dataset."""
    if debug:
        # Load only first item for debugging
        dataset = load_dataset(dataset_id, split="train").select([0])
        example = dataset[0]
        
        print("\nOriginal ShareGPT format:")
        print(example)
        
        # Process single item for inspection
        processed = convert_sharegpt_to_chat(example)
        
        print("\nProcessed tensor shape:")
        print(processed["input_ids"].shape)
        
        print("\nDecoded back to text:")
        print(tokenizer.decode(processed["input_ids"]))
        
        print("\nExiting after debug output")
        sys.exit(0)
    
    # Load full dataset for actual quantization
    dataset = load_dataset(
        dataset_id,
        split="train"
    ).shuffle(seed=42).select(range(num_samples))
    
    # Apply preprocessing to dataset
    processed_dataset = dataset.map(
        convert_sharegpt_to_chat,
        remove_columns=dataset.column_names
    )
    
    return processed_dataset["input_ids"]

def main():
    # Check if debug mode is requested
    debug_mode = "--debug" in sys.argv
    
    # Load and preprocess calibration data
    calibration_data = load_calibration_dataset(debug=debug_mode)
    
    if not debug_mode:
        # Configure quantization
        quant_config = QuantizeConfig(
            bits=4,              # Quantization bits
            group_size=32,       # Group size for quantization
            desc_act=False,      # Marlin kernel may not suport
                                 # Note: May impact inference speed with current implementations
        )
        # Initialize model
        model = GPTQModel.load(model_id, quant_config)
        
        
        # Quantize model
        model.quantize(
            calibration_data,
            batch_size=8, # up to 33GB @ bs=8; on H100 could prrobably go up to bs=16?
        )
        
        # Save quantized model

        model.save_pretrained(quant_path)
        print(f"\nQuantized model saved to: {quant_path}")

if __name__ == "__main__":
    main()

    # test post-quant inference
    print('--- Testing:')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = GPTQModel.load(quant_path, device=device)
    result = model.generate("Uncovering deep insights begins with")[0]
