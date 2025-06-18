# /data/llama.cpp/build/bin/llama-gguf-split --split-max-size 45G shisa-v2-llama3.1-405b-iq3_xs.gguf shisa-v2-llama3.1-405b-IQ3_XS
# for f in shisa-v2-llama3.1-405b-IQ3_XS-0000*; do huggingface-cli upload shisa-ai/shisa-v2-llama3.1-405b-GGUF "$f"; done

# /data/llama.cpp/build/bin/llama-gguf-split --split-max-size 45G shisa-v2-llama3.1-405b-iq3_xs.gguf shisa-v2-llama3.1-405b-IQ3_XS
# for f in shisa-v2-llama3.1-405b-IQ3_XS-0000*; do huggingface-cli upload shisa-ai/shisa-v2-llama3.1-405b-GGUF "$f"; done

# for f in shisa-v2-llama3.1-405b-IQ4_XS-0000*; do huggingface-cli upload shisa-ai/shisa-v2-llama3.1-405b-GGUF "$f"; done

# for f in shisa-v2-llama3.1-405b-IQ2_XXS-0000*; do huggingface-cli upload shisa-ai/shisa-v2-llama3.1-405b-GGUF "$f"; done

# /data/llama.cpp/build/bin/llama-gguf-split --split-max-size 45G shisa-v2-llama3.1-405b-Q4_K_M.gguf shisa-v2-llama3.1-405b-Q4_K_M
# for f in shisa-v2-llama3.1-405b-Q4_K_M-0000*; do huggingface-cli upload shisa-ai/shisa-v2-llama3.1-405b-GGUF "$f"; done

# Q8_0
/data/llama.cpp/build/bin/llama-gguf-split --split-max-size 45G shisa-v2-llama3.1-405b-Q8_0.gguf shisa-v2-llama3.1-405b-Q8_0
for f in shisa-v2-llama3.1-405b-Q8_0-0000*; do huggingface-cli upload shisa-ai/shisa-v2-llama3.1-405b-GGUF "$f"; done
