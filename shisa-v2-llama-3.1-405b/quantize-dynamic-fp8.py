from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot
import torch

# Load model.
model_name = "shisa-ai/shisa-v2-ubitus-llama3.1-405b"
# num_gpus = 8
# max_mem_per_gpu = "60GiB"  # leave headroom on 80GB GPUs
# max_mem = {str(i): max_mem_per_gpu for i in range(num_gpus)}
# max_mem["cpu"] = "500GiB"  # CPU RAM for offloading extra

config = AutoConfig.from_pretrained(model_name)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)


max_memory = {
      0: "120GiB",
      1: "120GiB",
      2: "120GiB",
      3: "120GiB",
      4: "120GiB",
      5: "120GiB",
      6: "120GiB",
      7: "120GiB",
      "cpu": "1500GiB",
}


'''
# https://chatgpt.com/c/67a74059-6344-8012-8ea5-a1e8dc6b827c
# This version... doesn't work so well

device_map = infer_auto_device_map(
    model,
    max_memory=max_memory,
    no_split_module_classes=["LlamaDecoderLayer"],
)

model = load_checkpoint_and_dispatch(
    model,
    model_name,
    device_map=device_map,
    dtype=torch.bfloat16,
    # dtype="auto",
    #torch_dtype="auto",
)
'''

device_map = infer_auto_device_map(
    model,
    max_memory=max_memory,
    no_split_module_classes=["LlamaDecoderLayer"],
    dtype="float16"  # Using fp16 for efficiency
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    torch_dtype="auto",
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with per channel via ptq
#   * quantize the activations to fp8 with dynamic per token
recipe = QuantizationModifier(
            targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
            )

# Apply quantization.
oneshot(model=model, recipe=recipe)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

# Save to disk in compressed-tensors format.
SAVE_DIR = model_name.split("/")[1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
