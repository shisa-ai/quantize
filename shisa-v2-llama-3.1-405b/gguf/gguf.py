git clone 



# https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md

python3 -m pip install -r requirements.txt
pip install -U transformers

python convert_hf_to_gguf.py /home/lhl/.cache/huggingface/hub/models--shisa-ai--shisa-v2-ubitus-llama3.1-405b/snapshots/71b83a7cb998c3a44f59c83a9928596ac348b9b

./llama-quantize ./models/mymodel/ggml-model-f16.gguf ./models/mymodel/ggml-model-Q4_K_M.gguf Q4_K_M

